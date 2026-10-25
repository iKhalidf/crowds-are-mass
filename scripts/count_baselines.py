"""
count_baselines.py -- baselines built COUNT-FIRST, the mirror image of a
shape-only method.

    # a real, known count forecaster (Holt) with a sensible spatial prior
    python count_baselines.py --count holt --spatial last --horizon short

    # good count, deliberately bad shape
    python count_baselines.py --count holt --spatial uniform --horizon short

Emits an npz in the master-table schema for stratify_report.py.

THE FACTORISATION. Every model here predicts

    pred[t] = count_hat[t] * pattern        with sum(pattern) == 1

so the two failure modes are controlled independently: `--count` sets how many
people, `--spatial` sets where they go. Crossing them isolates what each
metric actually responds to, which no single-model baseline can do.

--count
    holt      Holt's linear trend exponential smoothing (Holt 1957) fitted
              per clip to the observed count series. This is the standard
              textbook method for a short univariate series with drift -- it
              tracks level AND slope, so unlike persistence it extrapolates a
              rising or falling crowd. alpha/beta are chosen per clip by
              in-sample SSE over a small grid, so no global tuning is needed
              and there is no train/test leakage: the fit uses only that
              clip's OWN observed frames.
    persist   repeat the last observed count (count-wise persistence)
    mean      mean of the observed counts
    oracle    the TRUE future count. Not a forecaster -- an upper bound on
              what perfect counting buys you, useful for reading how much of
              each metric is count error vs placement error.

--spatial
    last      last observed frame, normalised. Same placement as persistence,
              so `--count holt --spatial last` differs from persistence in
              COUNT ONLY, and any metric gap between them is pure count.
    mean      mean of the observed frames, normalised. Slightly smoother.
    uniform   flat over the whole 80x80 grid. Deliberately wrong: it puts
              people in walls, water, and sky. Count can be exact while the
              spatial distribution is as wrong as it is possible to be, which
              is the point of this row.

WHY `--spatial uniform` IS THE INTERESTING ONE. It is the exact complement of
a shape-preserving/count-destroying model: here count is right and shape is
destroyed. D_kappa should be dominated by its transport term, and a metric
that only compares normalised distributions should rate it terribly -- which
is the honest half of the story, and worth reporting alongside the cases
where the same metric is too forgiving.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_ORIG_CWD = os.getcwd()
from _paths import CROWDMAC, STREAMCDF, COUNTS

import utils_videomae
from datasets import build_test_dataset

from streamcdf.divergence import divergence_metrics
from streamcdf.transport import d_kappa_per_clip

DEN = 100.0
HORIZONS = {
    "short": dict(obs_frames=8, pred_frames=12),
    "mid":   dict(obs_frames=20, pred_frames=36),
    "long":  dict(obs_frames=36, pred_frames=76),
}

_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def holt_forecast(series, horizon):
    """
    Holt's linear trend method, per series, alpha/beta by in-sample SSE.

    series: (B, T_obs) tensor of counts. Returns (B, horizon).

        level_t = a*y_t + (1-a)*(level_{t-1} + trend_{t-1})
        trend_t = b*(level_t - level_{t-1}) + (1-b)*trend_{t-1}
        yhat_{t+k} = level_t + k*trend_t

    Fitting uses only the clip's own observed frames, so there is no
    train/test leakage and no global hyperparameter to tune.
    """
    B, T = series.shape
    out = torch.zeros(B, horizon, dtype=series.dtype)
    for b in range(B):
        y = series[b]
        best = (float("inf"), None, None)
        for al in _GRID:
            for be in _GRID:
                lev, tr = y[0].clone(), y[1] - y[0]
                sse = 0.0
                for t in range(1, T):
                    pred = lev + tr
                    sse = sse + (y[t] - pred) ** 2
                    prev = lev
                    lev = al * y[t] + (1 - al) * (lev + tr)
                    tr = be * (lev - prev) + (1 - be) * tr
                if sse < best[0]:
                    best = (float(sse), al, be)
        _, al, be = best
        lev, tr = y[0].clone(), y[1] - y[0]
        for t in range(1, T):
            prev = lev
            lev = al * y[t] + (1 - al) * (lev + tr)
            tr = be * (lev - prev) + (1 - be) * tr
        k = torch.arange(1, horizon + 1, dtype=series.dtype)
        out[b] = (lev + k * tr).clamp_min(0.0)
    return out


def build_pattern(obs_maps, mode):
    """obs_maps (B, T_obs, H, W) -> (B, H, W) normalised to sum 1 per clip."""
    B, T, H, W = obs_maps.shape
    if mode == "uniform":
        p = torch.ones(B, H, W, dtype=obs_maps.dtype, device=obs_maps.device)
    elif mode == "mean":
        p = obs_maps.mean(dim=1)
    else:                                   # "last"
        p = obs_maps[:, -1]
    s = p.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-9)
    return p / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", default="holt",
                    choices=["holt", "persist", "mean", "oracle"])
    ap.add_argument("--spatial", default="last",
                    choices=["last", "mean", "uniform"])
    ap.add_argument("--horizon", default="short", choices=list(HORIZONS))
    ap.add_argument("--dataset", nargs="+", default=["gcs"])
    ap.add_argument("--gcs_step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--fast", action="store_true",
                    help="TRAIN-precision D_kappa; NOT mixable with the "
                         "full-precision rows already in the master table")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OBS, PRED = HORIZONS[a.horizon]["obs_frames"], HORIZONS[a.horizon]["pred_frames"]

    cfg = OmegaConf.load(f"{CROWDMAC}/configs/config.yaml")
    cfg.dataset.datasets = a.dataset
    cfg.dataset.num_frames = OBS + PRED
    cfg.forecast.obs_frames = OBS
    cfg.forecast.pred_frames = PRED
    cfg.augmentation.train.RandomHorizontalFlip.apply = False
    cfg.augmentation.train.RandomVerticalFlip.apply = False
    utils_videomae.GCS_STEP = a.gcs_step

    test_set = build_test_dataset(cfg)
    loader = DataLoader(test_set, batch_size=a.batch, shuffle=False,
                        num_workers=0, drop_last=False)
    print(f"{len(test_set)} clips | {OBS}->{PRED} | "
          f"count={a.count} spatial={a.spatial}")

    gt_all, pr_all, ob_all, P_all, T_all = [], [], [], [], []
    t0 = time.time()
    for videos, _ in loader:
        v = videos.to(dev) * DEN
        obs_maps = v[:, 0, :OBS]                       # (B, OBS, H, W)
        target = v[:, :, OBS:]                          # (B, 1, PRED, H, W)

        obs_counts = obs_maps.sum(dim=(-1, -2)) / DEN   # (B, OBS) people
        true_counts = target[:, 0].sum(dim=(-1, -2)) / DEN

        if a.count == "holt":
            ch = holt_forecast(obs_counts.cpu(), PRED).to(dev)
        elif a.count == "persist":
            ch = obs_counts[:, -1:].expand(-1, PRED)
        elif a.count == "mean":
            ch = obs_counts.mean(dim=1, keepdim=True).expand(-1, PRED)
        else:                                            # oracle
            ch = true_counts

        pat = build_pattern(obs_maps, a.spatial)         # (B, H, W), sums to 1
        pred = (ch[:, :, None, None] * DEN) * pat[:, None]
        pred = pred.unsqueeze(1)                         # (B, 1, PRED, H, W)

        gt_all.append((target.sum(dim=(1, 3, 4)) / DEN).cpu())
        pr_all.append((pred.sum(dim=(1, 3, 4)) / DEN).cpu())
        ob_all.append((obs_maps.sum(dim=(-1, -2)) / DEN).cpu())
        P_all.append(pred.float())
        T_all.append(target.float())
    print(f"forecast in {time.time() - t0:.0f}s")

    gt = torch.cat(gt_all).numpy()
    pr = torch.cat(pr_all).numpy()
    ob = torch.cat(ob_all).numpy()
    P, T = torch.cat(P_all), torch.cat(T_all)

    print(f"solving D_kappa for {len(gt)} clips "
          f"({'TRAIN' if a.fast else 'EVAL'} precision) ...", flush=True)
    prec = dict(pool=2, eps=0.5, iters=30) if a.fast else {}
    D, tr, um = d_kappa_per_clip(P, T, den_factor=DEN, kappa=a.kappa,
                                 decompose=True, batch_chunk=a.chunk,
                                 progress=True, **prec)
    dv = divergence_metrics(P / DEN, T / DEN)

    print(f"\ncount MAE {np.abs(gt - pr).mean():.3f} | "
          f"persistence {np.abs(ob[:, -1:] - gt).mean():.3f}")
    print(f"D_kappa {D.mean():.2f} = transport {tr.mean():.2f} "
          f"+ {a.kappa:g} * unmatched {um.mean():.3f}   "
          f"(shape {100 * tr.mean() / D.mean():.1f}%)")
    print(f"ADJS {dv['adjs'].mean():.4f}  ADKL {dv['adkl'].mean():.4f}")

    tag = f"count-{a.count}-{a.spatial}"
    out = os.path.join(STREAMCDF, "runs",
                       f"{tag}_{'-'.join(a.dataset)}_{a.horizon}"
                       f"_s{a.gcs_step}_preds.npz")
    np.savez_compressed(
        out, gt=gt, pred=pr, obs=ob, d_kappa_per_clip=D,
        transport_per_clip=tr, unmatched_per_clip=um,
        **{f"{k}_per_clip": v for k, v in dv.items()},
        config=dict(model=tag, count_method=a.count, spatial=a.spatial,
                    horizon=a.horizon, kappa=a.kappa, dataset=list(a.dataset),
                    gcs_step=a.gcs_step, fast=bool(a.fast),
                    obs_frames=OBS, pred_frames=PRED))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
