"""Score the persistence forecast on real density-map clips.

Writes per-clip counts and raw transport/JS metrics in the shared NPZ schema.
Use --fast only for development; it changes transport solver precision.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from _paths import CROWDMAC, STREAMCDF, COUNTS

import utils_videomae
from datasets import build_test_dataset
from streamcdf.divergence import divergence_metrics
from streamcdf.transport import d_kappa_per_clip

DEN = 100.0
FPS = 1.25

HORIZONS = {
    "short": dict(obs_frames=8, pred_frames=12),
    "mid":   dict(obs_frames=20, pred_frames=36),
    "long":  dict(obs_frames=36, pred_frames=76),
}


def run(horizon, args, device):
    OBS, PRED = HORIZONS[horizon]["obs_frames"], HORIZONS[horizon]["pred_frames"]
    NUM = OBS + PRED

    # Persistence is deterministic: same test set in, same numbers out. There
    # is no reason to redo the 17k transport solves on every invocation.
    out_dir = f"{STREAMCDF}/runs"
    path = (f"{out_dir}/streamcdf_{'-'.join(args.dataset)}_{horizon}"
            f"_PERSISTENCE_s{args.gcs_step}_preds.npz")
    if os.path.exists(path) and not args.force:
        import numpy as _np
        z = _np.load(path, allow_pickle=True)
        if "transport_per_clip" in z.files:
            print(f"[{horizon}] cached, skipping: {path}")
            print(f"  (--force to recompute)")
            return path
        print(f"[{horizon}] cache exists but predates the shape/count split; "
              f"recomputing.")

    cfg = OmegaConf.load(f"{CROWDMAC}/configs/config.yaml")
    cfg.dataset.datasets = args.dataset
    cfg.dataset.num_frames = NUM
    cfg.forecast.obs_frames = OBS
    cfg.forecast.pred_frames = PRED
    cfg.augmentation.train.RandomHorizontalFlip.apply = False
    cfg.augmentation.train.RandomVerticalFlip.apply = False
    utils_videomae.GCS_STEP = args.gcs_step

    test_set = build_test_dataset(cfg)
    if len(test_set) == 0:
        print(f"[{horizon}] no clips (needs {NUM * args.gcs_step} consecutive "
              f"frames). skipping.")
        return
    loader = DataLoader(test_set, batch_size=args.batch, shuffle=False,
                        num_workers=0, persistent_workers=False, pin_memory=True)

    gt_all, pr_all, obs_all = [], [], []
    pred_all, target_all = [], []
    t0 = time.time()

    for videos, _ in loader:
        videos = videos.to(device) * DEN
        real_obs = videos[:, :, :OBS]
        target = videos[:, :, OBS:]

        # THE ONLY "MODEL": repeat the last observed frame across the horizon
        persist = real_obs[:, :, -1:].expand_as(target)

        gt_all.append((target.sum(dim=(1, 3, 4)) / DEN).cpu())
        pr_all.append((persist.sum(dim=(1, 3, 4)) / DEN).cpu())
        obs_all.append((real_obs.sum(dim=(1, 3, 4)) / DEN).cpu())
        pred_all.append(persist.detach())
        target_all.append(target.detach())

    gt = torch.cat(gt_all).numpy()
    pr = torch.cat(pr_all).numpy()
    ob = torch.cat(obs_all).numpy()

    print(f"  solving D_kappa for {len(gt)} clips at "
          f"{'TRAIN' if args.fast else 'EVAL'} precision ...", flush=True)
    # EVAL_CFG (pool=1, eps=0.25, 60 iters, fp64) is ~8x the work per frame of
    # TRAIN_CFG (pool=2, eps=0.5, 30 iters). 1448 clips x 12 frames = 17,376
    # transport problems either way; --fast trades the reference precision for
    # roughly an 8x speedup, which is fine for iterating and wrong for the
    # number you finally report.
    prec = dict(pool=2, eps=0.5, iters=30) if args.fast else {}
    d_kappa_clip, tr_clip, um_clip = d_kappa_per_clip(
        torch.cat(pred_all), torch.cat(target_all),
        den_factor=DEN, kappa=args.kappa, decompose=True,
        batch_chunk=args.chunk, progress=True, **prec)

    dv = divergence_metrics(torch.cat(pred_all) / DEN, torch.cat(target_all) / DEN)
    print(f"  ADJS      : {dv['adjs'].mean():.4f}  (CrowdMAC's metric -- "
          f"normalises each frame, so it cannot see count at all)")

    count_mae = np.abs(gt - pr).mean()
    std_ratio = 100 * pr.std(axis=1).mean() / max(1e-9, gt.std(axis=1).mean())
    dcorr = np.corrcoef(np.diff(gt, axis=1).flatten(),
                        np.diff(pr, axis=1).flatten())[0, 1]

    print(f"\n[{horizon}] persistence-only, n={len(gt)}, {time.time()-t0:.0f}s")
    print(f"  count MAE   : {count_mae:.3f}   (should be 0 by construction "
          f"once diffed against itself -- nonzero here because pred is the "
          f"REPEATED last obs frame vs a MOVING target, exactly what "
          f"'persistence' means)")
    print(f"  std ratio   : {std_ratio:.1f}%   (should be near 0 -- flat by construction)")
    print(f"  corr(d)     : {dcorr:+.3f}   (should be near 0 -- persistence "
          f"has no delta to correlate with)")
    print(f"  D_kappa     : {d_kappa_clip.mean():.2f} avg over {len(d_kappa_clip)} clips "
          f"(reference precision, matches stream_cdf_metric)")
    print(f"    transport : {tr_clip.mean():.2f}  (SHAPE half, people*pixels)")
    print(f"    unmatched : {um_clip.mean():.2f}  (COUNT half, people)")
    print(f"    share     : shape {100*tr_clip.mean()/d_kappa_clip.mean():.1f}%  "
          f"count {100*args.kappa*um_clip.mean()/d_kappa_clip.mean():.1f}%")

    out_dir = f"{STREAMCDF}/runs"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/streamcdf_{'-'.join(args.dataset)}_{horizon}_PERSISTENCE_s{args.gcs_step}_preds.npz"
    np.savez_compressed(
        path, gt=gt, pred=pr, obs=ob, d_kappa_per_clip=d_kappa_clip,
        transport_per_clip=tr_clip, unmatched_per_clip=um_clip,
        **{f'{k}_per_clip': v for k, v in dv.items()},
        config=dict(model="persistence_only", horizon=horizon,
                  gcs_step=args.gcs_step, kappa=args.kappa,
                  dataset=list(args.dataset)))
    print(f"  saved: {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["gcs"])
    ap.add_argument("--gcs_step", type=int, default=1)
    ap.add_argument("--horizon", nargs="+", default=["short"],
                    choices=["short", "mid", "long"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--chunk", type=int, default=256,
                    help="clips per Sinkhorn batch. The solve is parallel "
                         "across clips, so this is bounded by GPU memory, not "
                         "correctness. Raise it if you have headroom; lower it "
                         "on OOM.")
    ap.add_argument("--fast", action="store_true",
                    help="solve at TRAIN precision (pool=2) instead of the "
                         "EVAL reference. ~8x faster, NOT the reportable "
                         "number -- use for iteration only.")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the output npz already exists")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for h in a.horizon:
        run(h, a, dev)
