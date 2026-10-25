"""
crowdmac_eval.py -- evaluate a trained CrowdMAC checkpoint and emit an npz in
the SAME schema as Stream-CDF's eval_only.py / persistence_baseline.py, so it
drops straight into stratify_report.py as a third row of the master table.

    python crowdmac_eval.py --ckpt results/gcs_8_12/checkpoint-best.pth

Scores CrowdMAC on BOTH metric families:
    D_kappa (+ its shape/count split)   sees placement AND count
    ADJS / ADKL / FDJS ...              normalised, so count-blind

That is the whole point of the row. CrowdMAC early-stops on `adjs_min`, i.e.
it is model-selected on a metric that cannot see count, while training on MSE
against real density, which can. Whether that produces a good or bad D_kappa
is an open empirical question -- do not assume it loses.

INFERENCE PATH. CrowdMAC forecasts by masked completion, not a forward pass.
The test dataset already yields the correct `bool_masked_pos` via
masking_generator.test_masked_position_generator, which marks observed
tubelets 0 and predicted tubelets 1. This script uses the loader's mask
verbatim rather than rebuilding it -- a mask constructed even slightly
differently would silently produce a meaningless "forecast".

SCALE. CrowdMAC trains outputs against `labels * den_factor`, so the model's
predictions are already in people*den_factor units -- the same convention as
Stream-CDF's `videos * DEN`. generate_predicted_videos splices those outputs
into a clone of the RAW videos_patch, so the returned tensor is mixed-scale:
observed positions raw, predicted positions x den_factor. Only the horizon
slice is used here, and every position in it is a predicted one, so the slice
is uniformly x den_factor and directly comparable to Stream-CDF. den_factor
is 100 in configs/config.yaml, matching Stream-CDF's DEN.
"""

import argparse
import os
import sys

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, never a dependency
    def tqdm(x, **k):
        return x
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_ORIG_CWD = os.getcwd()
from _paths import CROWDMAC, STREAMCDF, COUNTS

import utils_videomae
from datasets import build_test_dataset
from metrics import generate_predicted_videos
from model import get_model

from streamcdf.divergence import divergence_metrics
from streamcdf.transport import d_kappa_per_clip


def strip_dataparallel(sd):
    """train_crowdmac.py saves model.state_dict() AFTER wrapping in
    torch.nn.DataParallel, so every key is prefixed 'module.'."""
    if not any(k.startswith("module.") for k in sd):
        return sd
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None,
                    help="hydra config; defaults to the checkpoint's own saved "
                         "cfg, falling back to configs/config.yaml")
    ap.add_argument("--obs_frames", type=int, default=8)
    ap.add_argument("--pred_frames", type=int, default=12)
    ap.add_argument("--dataset", nargs="+", default=["gcs"])
    ap.add_argument("--gcs_step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--chunk", type=int, default=64,
                    help="clips per Sinkhorn batch; lower on OOM")
    ap.add_argument("--fast", action="store_true",
                    help="TRAIN-precision D_kappa, ~8x faster, NOT reportable "
                         "and NOT mixable with full-precision rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ckpt = a.ckpt if os.path.isabs(a.ckpt) else \
        os.path.abspath(os.path.join(_ORIG_CWD, a.ckpt))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    saved_cfg = blob.get("args", None)
    print(f"checkpoint epoch {blob.get('epoch')}")

    # Prefer the cfg stored in the checkpoint: it is definitionally the one
    # that produced these weights. configs/config.yaml may have been edited
    # since, which would silently build a mismatched model.
    if a.config:
        cfg = OmegaConf.load(a.config)
    elif saved_cfg is not None:
        cfg = OmegaConf.create(saved_cfg) if not OmegaConf.is_config(saved_cfg) \
            else saved_cfg
        print("using the cfg saved inside the checkpoint")
    else:
        cfg = OmegaConf.load(f"{CROWDMAC}/configs/config.yaml")
        print("checkpoint stored no cfg; falling back to configs/config.yaml")

    OBS, PRED = a.obs_frames, a.pred_frames
    NUM = OBS + PRED
    cfg.dataset.datasets = a.dataset
    cfg.dataset.num_frames = NUM
    cfg.forecast.obs_frames = OBS
    cfg.forecast.pred_frames = PRED
    cfg.augmentation.train.RandomHorizontalFlip.apply = False
    cfg.augmentation.train.RandomVerticalFlip.apply = False
    utils_videomae.GCS_STEP = a.gcs_step

    den = float(cfg.model.den_factor)
    input_size = int(cfg.model.input_size)
    tublet = int(cfg.model.tublet_size)
    if NUM % tublet:
        raise SystemExit(f"num_frames {NUM} not divisible by tublet_size {tublet}")

    model = get_model(cfg)
    patch_size = model.encoder.patch_embed.patch_size[0]
    model.load_state_dict(strip_dataparallel(blob["model"]))
    model.to(dev).eval()
    print(f"den_factor {den:g} | input {input_size} | patch {patch_size} | "
          f"tublet {tublet} | {OBS}->{PRED}")

    test_set = build_test_dataset(cfg)
    # drop_last=False: train_crowdmac.py drops the last partial batch, which
    # would silently evaluate on fewer clips than Stream-CDF and break the
    # clip-for-clip comparison in stratify_report.py.
    loader = DataLoader(test_set, batch_size=a.batch, shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=False)
    print(f"{len(test_set)} test clips")

    gt_all, pr_all, ob_all, pred_all, tgt_all = [], [], [], [], []
    with torch.no_grad():
        for videos, bool_masked_pos in tqdm(loader, desc="crowdmac", unit="batch"):
            videos = videos.to(dev, non_blocking=True)
            bool_masked_pos = (bool_masked_pos.to(dev, non_blocking=True)
                               .flatten(1).to(torch.bool))
            B = videos.shape[0]

            videos_patch = rearrange(
                videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                p0=tublet, p1=patch_size, p2=patch_size)

            with torch.cuda.amp.autocast():
                outputs = model(videos, bool_masked_pos)

            predicted_videos = generate_predicted_videos(
                outputs, videos_patch, bool_masked_pos, B,
                input_size, patch_size, tublet, NUM)

            # horizon slices, both on the people*den scale (see module docstring)
            pred_h = predicted_videos[:, :, OBS:].float()
            tgt_h = videos[:, :, OBS:].float() * den
            obs_h = videos[:, :, :OBS].float() * den

            gt_all.append((tgt_h.sum(dim=(1, 3, 4)) / den).cpu())
            pr_all.append((pred_h.sum(dim=(1, 3, 4)) / den).cpu())
            ob_all.append((obs_h.sum(dim=(1, 3, 4)) / den).cpu())
            pred_all.append(pred_h)
            tgt_all.append(tgt_h)

    gt = torch.cat(gt_all).numpy()
    pr = torch.cat(pr_all).numpy()
    ob = torch.cat(ob_all).numpy()
    P, T = torch.cat(pred_all), torch.cat(tgt_all)

    print(f"solving D_kappa for {len(gt)} clips "
          f"({'TRAIN' if a.fast else 'EVAL'} precision) ...", flush=True)
    prec = dict(pool=2, eps=0.5, iters=30) if a.fast else {}
    D, tr, um = d_kappa_per_clip(P, T, den_factor=den, kappa=a.kappa,
                                 decompose=True, batch_chunk=a.chunk,
                                 progress=True, **prec)
    dv = divergence_metrics(P / den, T / den)

    print(f"\ncount MAE {np.abs(gt - pr).mean():.3f} | "
          f"persistence {np.abs(ob[:, -1:] - gt).mean():.3f}")
    print(f"D_kappa {D.mean():.2f} = transport {tr.mean():.2f} "
          f"+ {a.kappa:g} * unmatched {um.mean():.3f}")
    print(f"ADJS {dv['adjs'].mean():.4f}  ADKL {dv['adkl'].mean():.4f}  "
          f"FDJS {dv['fdjs'].mean():.4f}")

    out = a.out or os.path.join(
        STREAMCDF, "runs",
        f"crowdmac_{'-'.join(a.dataset)}_{OBS}_{PRED}_s{a.gcs_step}_preds.npz")
    np.savez_compressed(
        out, gt=gt, pred=pr, obs=ob, d_kappa_per_clip=D,
        transport_per_clip=tr, unmatched_per_clip=um,
        **{f"{k}_per_clip": v for k, v in dv.items()},
        config=dict(model="crowdmac", ckpt=ckpt, epoch=blob.get("epoch"),
                    obs_frames=OBS, pred_frames=PRED, kappa=a.kappa,
                    den_factor=den, dataset=list(a.dataset),
                    gcs_step=a.gcs_step, fast=bool(a.fast),
                    tublet_size=tublet, patch_size=patch_size,
                    input_size=input_size))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
