"""Evaluate a Stream-CDF checkpoint and save per-clip counts and metrics.

Architecture and gain settings are read from checkpoint args, with CLI fallbacks.
Evaluation uses raw full-resolution transport at reference precision by default.
Only load trusted checkpoints: PyTorch checkpoint metadata may contain objects.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

_ORIG_CWD = os.getcwd()
from _paths import CROWDMAC, STREAMCDF, COUNTS

import utils_videomae
from datasets import build_test_dataset
from streamcdf.model import StreamCDF, stream_cdf_metric
from streamcdf.divergence import divergence_metrics
from streamcdf.transport import attach_output_gain, d_kappa_per_clip

DEN = 100.0
HORIZONS = {
    "short": dict(obs_frames=8, pred_frames=12),
    "mid":   dict(obs_frames=20, pred_frames=36),
    "long":  dict(obs_frames=36, pred_frames=76),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--horizon", default=None,
                    help="overrides the horizon recorded in the checkpoint")
    ap.add_argument("--dataset", nargs="+", default=["gcs"])
    ap.add_argument("--gcs_step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    # fallbacks, used only when the checkpoint has no `config`
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc_tubelet", type=int, default=4)
    ap.add_argument("--dec_tubelet", type=int, default=2)
    ap.add_argument("--anchor_frames", type=int, default=4)
    ap.add_argument("--temporal", default="absolute")
    ap.add_argument("--out_act", default="softplus")
    ap.add_argument("--out_gain", type=float, default=10.0)
    ap.add_argument("--delta_bound", type=float, default=50.0)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--fast", action="store_true",
                    help="TRAIN-precision solve, ~8x faster, not reportable")
    a = ap.parse_args()
    # Resolve user-supplied checkpoints against the original working directory.
    a.ckpt = os.path.abspath(os.path.join(_ORIG_CWD, a.ckpt)) \
        if not os.path.isabs(a.ckpt) else a.ckpt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(a.ckpt, map_location=dev, weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    # train_v5.py saves hyperparameters under "args" (vars(argparse.Namespace)),
    # with horizon/epoch/D alongside it at the top level.
    saved = {}
    if isinstance(blob, dict):
        saved = dict(blob.get("args", {}))
        for k in ("horizon", "epoch", "D"):
            if k in blob:
                saved[k] = blob[k]
    if saved:
        print(f"checkpoint args: epoch={saved.get('epoch')} "
              f"D={saved.get('D')} horizon={saved.get('horizon')} "
              f"out_gain={saved.get('out_gain')} out_act={saved.get('out_act')}")
    else:
        print("checkpoint has no `args` -- using CLI fallbacks. Verify these "
              "match the run that produced it, or the eval is meaningless.")

    g = lambda k, d: saved.get(k, d)
    horizon = a.horizon or g("horizon", "short")
    OBS, PRED = HORIZONS[horizon]["obs_frames"], HORIZONS[horizon]["pred_frames"]
    kappa = g("kappa", a.kappa)

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
                        num_workers=0, persistent_workers=False, pin_memory=True)
    print(f"{len(test_set)} test clips | horizon={horizon} {OBS}->{PRED}")

    model = StreamCDF(
        obs_frames=OBS, pred_frames=PRED,
        enc_tubelet=g("enc_tubelet", a.enc_tubelet),
        dec_tubelet=g("dec_tubelet", a.dec_tubelet),
        anchor_frames=min(g("anchor_frames", a.anchor_frames), OBS),
        dim=g("dim", a.dim), depth=g("depth", a.depth),
        heads=g("heads", a.heads), dropout=0.0,
        temporal=g("temporal", a.temporal), out_act=g("out_act", a.out_act),
    ).to(dev)
    og = g("out_gain", a.out_gain)
    db = g("delta_bound", a.delta_bound)
    if og != 1.0:
        attach_output_gain(model, og, bound=(db if db and db > 0 else None))
    model.load_state_dict(state)
    model.eval()

    gt_all, pr_all, ob_all, pred_all, tgt_all = [], [], [], [], []
    with torch.no_grad():
        for videos, _ in loader:
            videos = videos.to(dev) * DEN
            real_obs = videos[:, :, :OBS]
            target = videos[:, :, OBS:]
            pred = model(real_obs, anchor_src=real_obs)
            gt_all.append((target.sum(dim=(1, 3, 4)) / DEN).cpu())
            pr_all.append((pred.sum(dim=(1, 3, 4)) / DEN).cpu())
            ob_all.append((real_obs.sum(dim=(1, 3, 4)) / DEN).cpu())
            pred_all.append(pred); tgt_all.append(target)

    gt = torch.cat(gt_all).numpy()
    pr = torch.cat(pr_all).numpy()
    ob = torch.cat(ob_all).numpy()
    print(f"solving D_kappa for {len(gt)} clips ...", flush=True)
    prec = dict(pool=2, eps=0.5, iters=30) if a.fast else {}
    D, tr, um = d_kappa_per_clip(torch.cat(pred_all), torch.cat(tgt_all),
                                 den_factor=DEN, kappa=kappa, decompose=True,
                                 batch_chunk=a.chunk, progress=True, **prec)

    dv = divergence_metrics(torch.cat(pred_all) / DEN, torch.cat(tgt_all) / DEN)
    print(f"ADJS {dv['adjs'].mean():.4f}  ADKL {dv['adkl'].mean():.4f}  "
          f"FDJS {dv['fdjs'].mean():.4f}   <- count-blind by construction")

    print(f"\ncount MAE {np.abs(gt - pr).mean():.3f} | "
          f"persistence {np.abs(ob[:, -1:] - gt).mean():.3f}")
    print(f"D_kappa {D.mean():.2f}  = transport {tr.mean():.2f} "
          f"+ {kappa:g} * unmatched {um.mean():.3f}")

    out = a.ckpt.replace(".pt", "") + "_preds.npz"
    np.savez_compressed(out, gt=gt, pred=pr, obs=ob, d_kappa_per_clip=D,
                        transport_per_clip=tr, unmatched_per_clip=um,
                        **{f'{k}_per_clip': v for k, v in dv.items()},
                        config=dict(saved, eval_only=True, kappa=kappa,
                                    horizon=horizon))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
