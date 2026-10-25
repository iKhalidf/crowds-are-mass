"""Train Stream-CDF using the supplied CrowdMAC density-map pipeline.

Defaults match the recorded GCS run: 8 observed and 12 forecast frames, 20 epochs,
absolute temporal embeddings, epsilon 2, debiased training, and gradient clip 750.
Use --dataset indoor --epochs 40 for the recorded indoor settings.
Raw evaluation costs and final checkpoints are written under runs/.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _paths import CROWDMAC, STREAMCDF, COUNTS

from omegaconf import OmegaConf
from datasets import build_train_dataset, build_test_dataset
import utils_videomae

from streamcdf.model import (StreamCDF, stream_cdf_loss, stream_cdf_metric,
                           TRAIN_CFG, EVAL_CFG)
from streamcdf.transport import (attach_output_gain, build_param_groups,
                              sinkhorn_divergence_loss, d_kappa_per_clip)
from streamcdf.convlstm import ConvLSTMForecaster

DEN = 100.0     # maps are stored scaled by this so the loss lands in a usable range
FPS = 1.25
SEC_PER_FRAME = 1.0 / FPS   # 0.8s

# ---------------------------------------------------------------------------
# Horizon presets.  (obs_frames, pred_frames) in frames at 1.25 fps, chosen so
# obs_frames % enc_tubelet(4) == 0 and pred_frames % dec_tubelet(2) == 0 for
# every window -- not a coincidence, the requested second ranges were chosen
# to land on tubelet boundaries.
#
#   short   context  6.4s ( 8f) -> horizon  9.6s (12f)   nowcast
#   mid     context 16.0s (20f) -> horizon 28.8s (36f)   = current v2 default
#   long    context 28.8s (36f) -> horizon 60.8s (76f)   far horizon, short ctx
#
# "long" inverts the v2 default (obs=76,pred=36): it asks the model to reach
# 60.8s out from only 28.8s of context, the more demanding regime.
# ---------------------------------------------------------------------------
HORIZONS = {
    "short": dict(obs_frames=8, pred_frames=12),
    "mid":   dict(obs_frames=20, pred_frames=36),
    "long":  dict(obs_frames=36, pred_frames=76),
}


def horizon_seconds(obs, pred):
    return obs * SEC_PER_FRAME, pred * SEC_PER_FRAME


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------
def spatial_ablation(obs):
    """Each frame -> a uniform blob of identical total mass.

    Spatial structure is destroyed, headcount is preserved exactly.  Applied to
    the ENCODER INPUT only; the residual anchor still sees the real observation,
    so the comparison isolates spatial information rather than also removing the
    model's starting point.
    """
    c = obs.sum(dim=(3, 4), keepdim=True)
    return c.expand_as(obs) / (obs.shape[3] * obs.shape[4])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, obs, device, ablate=False, verbose=False,
             return_preds=False, uniform_anchor=False, kappa=8.0):
    """Forecast quality (D_kappa, the SAME metric the model trains on) plus
    two collapse diagnostics that are independent of any loss.

    Only stream_cdf_metric is used for forecast quality -- no MSE, MAE, JS or
    KL is computed here. JS/KL are count-normalised and discard magnitude by
    construction (see the model's own header); using them for evaluation while
    training on D_kappa would score the model on a different quantity than the
    one it optimises.

    Diagnostics that catch a model scoring well while doing nothing:
      copy_d  distance to "repeat the last observed frame".  Near zero means
              the model has collapsed onto persistence.
      p_std   spread of predicted counts across the horizon against the truth's.
              Much smaller means the model is emitting a near-constant.

    Per-clip counts are accumulated rather than per-batch means, so the reported
    average is over clips and does not depend on batch size.
    """
    model.eval()
    d_all, tr_all, um_all = [], [], []
    gt_all, pr_all, obs_all = [], [], []
    pred_all, target_all, persist_all = [], [], []   # kept only if return_preds

    for videos, _ in loader:
        videos = videos.to(device) * DEN
        real_obs = videos[:, :, :obs]
        enc_in = spatial_ablation(real_obs) if ablate else real_obs

        anc_in = spatial_ablation(real_obs) if uniform_anchor else real_obs
        pred = model(enc_in, anchor_src=anc_in)
        target = videos[:, :, obs:]

        m = stream_cdf_metric(pred, target, den_factor=DEN, kappa=kappa)
        d_all.append(m["D"]); tr_all.append(m["transport"]); um_all.append(m["unmatched"])

        if return_preds:
            pred_all.append(pred.detach())
            target_all.append(target.detach())
            # persistence baseline, SPATIAL: last observed density map,
            # repeated across the horizon. This is what "persistence" means
            # everywhere else in this script (count_mae's persist=), just
            # kept as a full map instead of collapsed to a count, so it can
            # be scored by the real D_kappa and compared to the model in the
            # SAME units -- until now only the model's D_kappa existed, so
            # "does it beat persistence" could only be asked in count-MAE.
            last_frame = real_obs[:, :, -1:]                      # (B,1,1,H,W)
            persist_all.append(last_frame.expand_as(target).detach())

        gt_all.append((target.sum(dim=(1, 3, 4)) / DEN).cpu())
        pr_all.append((pred.sum(dim=(1, 3, 4)) / DEN).cpu())
        obs_all.append((real_obs.sum(dim=(1, 3, 4)) / DEN).cpu())

    model.train()

    gt = torch.cat(gt_all).numpy()
    pr = torch.cat(pr_all).numpy()
    ob = torch.cat(obs_all).numpy()

    per_clip = np.abs(gt - pr).mean(axis=1)   # people-MAE, diagnostic only
    out = dict(
        D=float(np.mean(d_all)), transport=float(np.mean(tr_all)),
        unmatched=float(np.mean(um_all)),
        count_mae=float(per_clip.mean()),
        final_mae=float(np.abs(gt[:, -1] - pr[:, -1]).mean()),
        copy_d=float(np.abs(pr - ob[:, -1:]).mean()),
        p_std=float(pr.std(axis=1).mean()),
        g_std=float(gt.std(axis=1).mean()),
        persist=float(np.abs(ob[:, -1:] - gt).mean()),
    )

    if verbose:
        k = 12
        print("      true:", " ".join(f"{v:6.1f}" for v in gt[0][:k]), "...")
        print("      pred:", " ".join(f"{v:6.1f}" for v in pr[0][:k]), "...")

    if return_preds:
        # ONE vectorised pass over the whole test set (chunked internally at
        # 64 clips for memory, not for correctness -- see d_kappa_per_clip's
        # docstring for why the old per-clip Python loop was ~50-100x slower
        # for identical numbers).
        d_kappa_clip, tr_clip, um_clip = d_kappa_per_clip(
            torch.cat(pred_all), torch.cat(target_all),
            den_factor=DEN, kappa=kappa, decompose=True)
        out.update(gt=gt, pred=pr, obs=ob, per_clip=per_clip,
                   d_kappa_per_clip=d_kappa_clip,
                   transport_per_clip=tr_clip,
                   unmatched_per_clip=um_clip)
    return out


# ---------------------------------------------------------------------------
# deterministic baselines
# ---------------------------------------------------------------------------
def load_counts(split):
    path = os.path.join(COUNTS, f"counts_{split}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_count_clips(blocks, obs, pred):
    """Sliding windows within blocks; clips never cross a block boundary."""
    n = obs + pred
    X, Y, B = [], [], []
    for bi, name in enumerate(sorted(blocks)):
        c = np.asarray(blocks[name], dtype=np.float64)
        for s in range(len(c) - n + 1):
            X.append(c[s:s + obs])
            Y.append(c[s + obs:s + n])
            B.append(bi)
    return np.asarray(X), np.asarray(Y), np.asarray(B), sorted(blocks)


def fit_baselines(obs, pred):
    """Persistence and Ridge from the CSV-derived count series.

    Deterministic by construction: no dataloader, no augmentation, no splatting.
    Returns (persistence MAE, ridge MAE, ridge coefficient checksum) or None if
    the count files have not been built yet.
    """
    from sklearn.linear_model import Ridge

    tr, te = load_counts("train"), load_counts("test")
    if tr is None or te is None:
        return None

    Xtr, Ytr, _, _ = build_count_clips(tr, obs, pred)
    Xte, Yte, _, _ = build_count_clips(te, obs, pred)

    R = Ridge(alpha=1.0).fit(Xtr, Ytr)
    return (float(np.abs(Xte[:, -1:] - Yte).mean()),
            float(np.abs(R.predict(Xte) - Yte).mean()),
            float(np.abs(R.coef_).sum()))


# ---------------------------------------------------------------------------
# clip -> block assignment
# ---------------------------------------------------------------------------
def match_blocks(obs_counts, gt_counts, blocks, clip_len):
    """Assign each evaluated clip to its source block.

    The test loader's clip order is scrambled, so index carries no block
    information.  Matching uses first differences of the clip's own count
    signature, which is invariant to the constant offset between density-map
    counts and CSV counts.

    clip_len = obs_frames + pred_frames for the horizon being evaluated (NOT a
    fixed constant -- the three horizon presets have different totals: 20, 56,
    112 frames. Using a fixed length here would silently mismatch two of the
    three windows).
    """
    if blocks is None:
        return None, None
    full = np.concatenate([obs_counts, gt_counts], axis=1)
    names = sorted(blocks)

    cands, meta = [], []
    for bi, nm in enumerate(names):
        v = np.asarray(blocks[nm], dtype=np.float64)
        for s in range(len(v) - clip_len + 1):
            cands.append(np.diff(v[s:s + clip_len]))
            meta.append(bi)
    cands = np.asarray(cands)
    meta = np.asarray(meta)

    D = np.diff(full, axis=1)
    assign, resid = [], []
    for i in range(len(D)):
        d = np.abs(cands - D[i]).max(axis=1)
        j = int(np.argmin(d))
        assign.append(meta[j])
        resid.append(float(d[j]))
    return np.asarray(assign), np.asarray(resid)


# ---------------------------------------------------------------------------
def run_training(horizon, args, device):
    """Train and evaluate ONE model for one horizon window. Called once per
    entry in --horizon (three times for --horizon all)."""

    OBS, PRED = HORIZONS[horizon]["obs_frames"], HORIZONS[horizon]["pred_frames"]
    NUM = OBS + PRED
    ctx_s, hor_s = horizon_seconds(OBS, PRED)
    kappa = args.kappa

    # ---- dataset -----------------------------------------------------
    cfg = OmegaConf.load(f"{CROWDMAC}/configs/config.yaml")
    cfg.dataset.datasets = args.dataset
    cfg.dataset.num_frames = NUM
    cfg.forecast.obs_frames = OBS
    cfg.forecast.pred_frames = PRED
    cfg.augmentation.train.RandomHorizontalFlip.apply = bool(args.hflip)
    cfg.augmentation.train.RandomVerticalFlip.apply = bool(args.vflip)
    utils_videomae.GCS_STEP = args.gcs_step

    train_set = build_train_dataset(cfg)
    test_set = build_test_dataset(cfg)
    if len(train_set) == 0 or len(test_set) == 0:
        print(f"\n[{horizon}] No clips: needs {NUM * args.gcs_step} "
              f"consecutive frames. Skipping this horizon.\n")
        return

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True,
                              persistent_workers=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False,
                             num_workers=0, persistent_workers=False, pin_memory=True)

    base = fit_baselines(OBS, PRED)
    if base is None:
        print(f"\ncounts_*.npz not found in {COUNTS}; run read_csv_counts.py "
              f"first. Baselines will be omitted.\n")
        persist_mae = ridge_mae = float("nan")
    else:
        persist_mae, ridge_mae, chk = base

    # ---- model ---------------------------------------------------------
    model = StreamCDF(
        obs_frames=OBS, pred_frames=PRED,
        enc_tubelet=args.enc_tubelet, dec_tubelet=args.dec_tubelet,
        anchor_frames=min(args.anchor_frames, OBS),
        dim=args.dim, depth=args.depth, heads=args.heads,
        dropout=args.dropout, temporal=args.temporal, out_act=args.out_act,
    ).to(device)
    if args.out_gain != 1.0:
        attach_output_gain(model, args.out_gain,
                           bound=(args.delta_bound if args.delta_bound > 0 else None))
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(
        build_param_groups(model, args.lr, head_lr_mult=args.head_lr_mult,
                           weight_decay=1e-4),
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Linear warmup on the HEAD group. The first few hundred steps are where a
    # high-gain head runs away: the net downward push from ~98% empty pixels is
    # at its largest when the prediction still carries the anchor's full mass.
    # Warmup lets the encoder start producing structure before the head is
    # allowed to move the output far.
    _head_lr = opt.param_groups[-1]["lr"]
    _gstep = [0]

    def _warm():
        _gstep[0] += 1
        if args.warmup_steps > 0 and _gstep[0] <= args.warmup_steps:
            opt.param_groups[-1]["lr"] = _head_lr * _gstep[0] / args.warmup_steps

    name = "-".join(args.dataset) + f"_{horizon}"
    if args.ablate_spatial:
        name += "_ablate"
    if args.arch != "streamcdf":
        name += f"_{args.arch}"
    if args.uniform_anchor:
        name += "_unifanchor"
    if args.temporal != "absolute":
        name += f"_{args.temporal}"
    if args.out_act != "relu":
        name += f"_{args.out_act}"
    if args.hflip or args.vflip:
        name += "_flip" + ("h" if args.hflip else "") + ("v" if args.vflip else "")
    # -- anti-collapse fix knobs (adjs) -----------------------------------
    # Encoded so a checkpoint/preds file is self-describing: without this,
    # "which config produced this npz?" is unanswerable after the fact, which
    # is exactly the ambiguity that came up comparing the surge-clip results.
    # Only appended when non-default, so old-style (pre-fix, --out_gain 1.0
    # --debias 0) runs keep their original filenames.
    if args.out_gain != 1.0:
        name += f"_gain{args.out_gain:g}"
    if args.head_lr_mult != 1.0:
        name += f"_hlrx{args.head_lr_mult:g}"
    if args.delta_bound and args.delta_bound > 0:
        name += f"_bound{args.delta_bound:g}"
    if args.warmup_steps != 500:
        name += f"_warm{args.warmup_steps}"
    if not args.debias:
        name += "_nodebias"
    if args.clip != 1.0:
        name += f"_clip{args.clip:g}"
    if args.tag:
        name += f"_{args.tag}"
    if kappa != 8.0:
        name += f"_k{kappa:g}"
    ckpt = f"{args.out}/streamcdf_{name}_s{args.gcs_step}_seed{args.seed}.pt"

    print(f"\n{'=' * 80}")
    print(f"  Stream-CDF | {'+'.join(args.dataset)} | seed {args.seed} "
          f"| horizon={horizon}")
    print(f"  {len(train_set)} train / {len(test_set)} test clips")
    print(f"  context {OBS} frames ({ctx_s:.1f}s) -> "
          f"horizon {PRED} frames ({hor_s:.1f}s)   @ {FPS} fps")
    print(f"  encoder {model.obs_chunks}x100 = {model.obs_chunks * 100} tokens")
    print(f"  decoder {model.pred_chunks}x100 = {model.pred_chunks * 100} queries")
    print(f"  {n_params / 1e6:.1f}M parameters | batch {args.batch} | {device}")
    print(f"  temporal={args.temporal} out_act={args.out_act}")
    print(f"  loss: D_kappa ONLY (kappa={kappa:g}, eps={args.eps:g}, "
          f"iters={args.iters}, pool={args.pool})  -- no MSE, no aux term")
    print(f"  augmentation: hflip={bool(args.hflip)} vflip={bool(args.vflip)}")
    if args.ablate_spatial:
        print("  SPATIAL ABLATION: encoder input is uniform-blob; anchor is real")
    print(f"  {'-' * 76}")
    if base:
        print(f"  persistence count MAE : {persist_mae:6.2f}   (reference only,"
              f" not part of the loss)")
        print(f"  Ridge       count MAE : {ridge_mae:6.2f}  (checksum {chk:.6f})")
    print(f"{'=' * 80}\n")

    # ---- training --------------------------------------------------------
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tot = t_tr = t_um = nb = 0
        dmax = -1e30          # signed delta range: runaway is always NEGATIVE
        dmin = 1e30
        gpre = []             # pre-clip grad norms : is the clip throttling us?

        for videos, _ in train_loader:
            videos = videos.to(device) * DEN
            real_obs, target = videos[:, :, :OBS], videos[:, :, OBS:]
            enc_in = spatial_ablation(real_obs) if args.ablate_spatial else real_obs

            anc_in = spatial_ablation(real_obs) if args.uniform_anchor else real_obs
            pred = model(enc_in, anchor_src=anc_in)
            _lossfn = sinkhorn_divergence_loss if args.debias else stream_cdf_loss
            _extra = dict(include_bb=args.include_bb) if args.debias else {}
            loss, transport, unmatched = _lossfn(
                pred, target, kappa=kappa, eps=args.eps, iters=args.iters,
                den_factor=DEN, pool=args.pool, **_extra)

            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            gpre.append(float(gn))
            _warm()
            opt.step()

            with torch.no_grad():
                _d = pred - model.make_anchor(anc_in, PRED)
                dmax = max(dmax, _d.max().item())
                dmin = min(dmin, _d.min().item())

            tot += loss.item()
            t_tr += transport
            t_um += unmatched
            nb += 1
        sched.step()

        if ep % 5 == 0 or ep == 1:
            m = evaluate(model, test_loader, OBS, device,
                         ablate=args.ablate_spatial, uniform_anchor=args.uniform_anchor,
                         kappa=kappa, verbose=(ep % 20 == 0 or ep == 1))
            star = ""
            if m["D"] < best:
                best = m["D"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "horizon": horizon, "epoch": ep, "D": best}, ckpt)
                star = " *"
            print(f"[{horizon}] ep {ep:>3} | train D {tot / nb:8.2f} "
                  f"(transport {t_tr / nb:7.2f} + k*unmatched {kappa * t_um / nb:7.2f}) "
                  f"| eval D {m['D']:8.2f} (shape {m['transport']:7.2f} "
                  f"count {m['unmatched']:6.2f}) "
                  f"| count MAE {m['count_mae']:6.2f} [P {persist_mae:.2f} "
                  f"R {ridge_mae:.2f}] | copy_d {m['copy_d']:.4f} "
                  f"| std p/g {m['p_std']:5.2f}/{m['g_std']:5.2f} "
                  f"| {time.time() - t0:.0f}s{star}")
            gp = sorted(gpre)
            print(f"           anchor: "
                  f"{[round(v, 3) for v in model.anchor_w.data.tolist()]}"
                  f" | delta [{dmin:8.2f}, {dmax:7.2f}]"
                  f" | grad norm p50 {gp[len(gp)//2]:8.3f}"
                  f" p90 {gp[int(0.9*len(gp))]:8.3f} (clip {args.clip:g},"
                  f" binding {100*sum(g > args.clip for g in gpre)/len(gpre):.0f}%)")
        else:
            gp = sorted(gpre)
            print(f"[{horizon}] ep {ep:>3} | train D {tot / nb:8.2f} "
                  f"(transport {t_tr / nb:7.2f} + k*unmatched {kappa * t_um / nb:7.2f}) "
                  f"| delta [{dmin:8.2f}, {dmax:7.2f}] | gnorm p50 {gp[len(gp)//2]:7.3f} "
                  f"| {time.time() - t0:.0f}s")

    # ---- final evaluation, measured fresh --------------------------------
    m = evaluate(model, test_loader, OBS, device,
                 ablate=args.ablate_spatial, uniform_anchor=args.uniform_anchor,
                 kappa=kappa, return_preds=True)
    final_ckpt = ckpt.replace(".pt", "_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args),
                "horizon": horizon, "epoch": args.epochs, "D": m["D"]}, final_ckpt)

    pred_path = final_ckpt.replace(".pt", "_preds.npz")
    np.savez_compressed(
        pred_path, gt=m["gt"], pred=m["pred"], obs=m["obs"],
        d_kappa_per_clip=m["d_kappa_per_clip"],
        transport_per_clip=m["transport_per_clip"],
        unmatched_per_clip=m["unmatched_per_clip"],
        # adjs: the run config, saved INSIDE the file so it survives a rename
        # or a copy off the checkpoint dir -- np.load(...)['adjs'].item()
        config=dict(kappa=kappa, out_gain=args.out_gain, head_lr_mult=args.head_lr_mult,
                  delta_bound=args.delta_bound, warmup_steps=args.warmup_steps,
                  debias=bool(args.debias), include_bb=bool(args.include_bb),
                  clip=args.clip, eps=args.eps, out_act=args.out_act,
                  dropout=args.dropout, lr=args.lr, epochs=args.epochs,
                  horizon=horizon, seed=args.seed))

    print(f"\n{'=' * 80}")
    print(f"  [{horizon}] final-epoch D_kappa : {m['D']:8.2f}   <- report this "
          f"(shape {m['transport']:.2f}, count {m['unmatched']:.2f} people)")
    print(f"  [{horizon}] best-epoch  D_kappa : {best:8.2f}   (test-selected, reference only)")
    print(f"  count MAE (diagnostic, not the loss) : {m['count_mae']:6.2f}")
    print(f"  persistence                          : {persist_mae:6.2f}")
    print(f"  Ridge                                : {ridge_mae:6.2f}")
    print(f"  predicted / true count std : {m['p_std']:.2f} / {m['g_std']:.2f}"
          f"   ({100 * m['p_std'] / max(1e-9, m['g_std']):.1f}% of truth)")
    print(f"{'=' * 80}")

    # ---- per-block breakdown ---------------------------------------------
    te = load_counts("test")
    bi, resid = match_blocks(m["obs"], m["gt"], te, clip_len=NUM)
    if bi is not None:
        names = sorted(te)
        pc = m["per_clip"]
        pers = np.abs(m["obs"][:, -1:] - m["gt"]).mean(axis=1)
        print(f"\n  [{horizon}] per-block count MAE   (clip->block match: max "
              f"residual {resid.max():.2f} people)")
        print(f"  {'block':<10}{'n':>5}{'lvl hi':>9}{'persist':>10}{'model':>9}"
              f"{'m-p':>8}")
        for k, nm in enumerate(names):
            sel = bi == k
            if sel.any():
                print(f"  {nm:<10}{int(sel.sum()):>5}"
                      f"{m['obs'][sel][:, -1].max():>9.0f}"
                      f"{pers[sel].mean():>10.2f}{pc[sel].mean():>9.2f}"
                      f"{pc[sel].mean() - pers[sel].mean():>+8.2f}")
        print(f"  {'ALL':<10}{len(pc):>5}{'':>9}{pers.mean():>10.2f}"
              f"{pc.mean():>9.2f}{pc.mean() - pers.mean():>+8.2f}")
        wins = int(sum((pc[bi == k].mean() < pers[bi == k].mean())
                       for k in range(len(names)) if (bi == k).any()))
        print(f"\n  beats persistence on {wins} / {len(names)} blocks")

    print(f"\n  saved {os.path.basename(final_ckpt)}")
    print(f"  saved {os.path.basename(pred_path)}  "
          f"(per-clip predictions, no re-inference needed)")


def main():
    ap = argparse.ArgumentParser(
        description="Train Stream-CDF with debiased unbalanced transport; evaluate raw D_kappa.")

    ap.add_argument("--dataset", nargs="+", default=["gcs"])
    ap.add_argument("--gcs_step", type=int, default=1)

    ap.add_argument("--horizon", nargs="+", default=["short"],
                    choices=["short", "mid", "long", "all"],
                    help="one or more of: "
                         "short (6.4s->9.6s, 8->12 frames @1.25fps), "
                         "mid   (16s->28.8s, 20->36 frames), "
                         "long  (28.8s->60.8s, 36->76 frames), "
                         "all   (train all three as separate models). "
                         "Overridden by --obs_frames/--pred_frames if both given.")
    ap.add_argument("--obs_frames", type=int, default=None,
                    help="override the horizon preset's context length")
    ap.add_argument("--pred_frames", type=int, default=None,
                    help="override the horizon preset's forecast length")

    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc_tubelet", type=int, default=4)
    ap.add_argument("--dec_tubelet", type=int, default=2)
    ap.add_argument("--anchor_frames", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--temporal", default="absolute",
                    choices=["absolute", "alibi"])
    ap.add_argument("--out_act", default="softplus", choices=["relu", "softplus"])

    # -- D_kappa is the ONLY loss and the ONLY eval metric --
    ap.add_argument("--kappa", type=float, default=8.0,
                    help="people <-> pixels exchange rate. A pixel misplaced "
                         "farther than 2*kappa is treated as absent -- set "
                         "from the camera's ground-plane scale.")
    ap.add_argument("--eps", type=float, default=2.0,
                    help="entropic blur of the training-time solve")
    ap.add_argument("--iters", type=int, default=TRAIN_CFG["iters"],
                    help="Sinkhorn iterations at TRAINING time")
    ap.add_argument("--pool", type=int, default=TRAIN_CFG["pool"],
                    help="mass-preserving spatial downsample before the OT "
                         "solve, speed lever for training only. Evaluation "
                         "always runs pool=1 (see stream_cdf.EVAL_CFG).")

    # -- fixes for the "never leaves the anchor" failure --
    ap.add_argument("--out_gain", type=float, default=10.0,
                    help="fixed multiplier on the head's delta. The loss is "
                         "1-Lipschitz in PEOPLE (|dD/dpred| <= kappa/DEN) "
                         "while the output lives at DEN=100, so a zero-init "
                         "head slews ~4e-3 density units/step against a target "
                         "needing O(10-50). This restores unit agreement. "
                         "Set 1.0 to reproduce the old flat behaviour.")
    ap.add_argument("--head_lr_mult", type=float, default=1.0,
                    help="extra lr on head.proj + anchor_w only. Total output "
                         "slew scales as out_gain * head_lr_mult; split the "
                         "factor between them rather than loading it all on "
                         "out_gain, which also amplifies UPSTREAM gradients.")
    ap.add_argument("--debias", type=int, default=1,
                    help="1 = train on the Sinkhorn divergence "
                         "S = D(a,b) - .5 D(a,a) - .5 D(b,b) instead of raw "
                         "D_kappa. Raw D_kappa is entropically biased: a "
                         "PERFECT prediction scores 10.51 and its gradient is "
                         "kappa/DEN at the optimum, so there is no annealing "
                         "and Adam hovers. S scores 0.0000 there.")
    ap.add_argument("--include_bb", type=int, default=1,
                    help="include the D(b,b) term. Constant in the parameters "
                         "(zero gradient), so 0 saves ~1/3 of loss time at the "
                         "cost of a per-batch offset in the LOGGED value.")
    ap.add_argument("--clip", type=float, default=750.0,
                    help="grad-norm clip. With out_gain>1 upstream gradients "
                         "are amplified; leaving this at 1.0 makes the clip "
                         "bind every step and silently re-throttles training. "
                         "Read the p90 in the epoch log and set it above that.")

    ap.add_argument("--delta_bound", type=float, default=50.0,
                    help="bound |delta| via tanh; 0 disables. Without it the "
                         "net downward gradient from ~98%% empty pixels drives "
                         "delta to large negative, softplus floors the output "
                         "at 0, its derivative vanishes and the model dies "
                         "permanently (observed: pred==0, grad norm 0).")
    ap.add_argument("--warmup_steps", type=int, default=500,
                    help="linear warmup on the HEAD lr group only")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--arch", default="streamcdf", choices=['streamcdf', 'convlstm'])
    ap.add_argument("--uniform_anchor", action="store_true", help="uniform residual anchor of the same mass")
    ap.add_argument("--ablate_spatial", action="store_true",
                    help="uniform-blob encoder input, mass preserved; the "
                         "anchor still sees the real observation")
    ap.add_argument("--hflip", action="store_true",
                    help="enable horizontal flip augmentation (off by default)")
    ap.add_argument("--vflip", action="store_true",
                    help="enable vertical flip. Physically meaningless for a "
                         "fixed camera; provided only to reproduce prior runs")
    ap.add_argument("--tag", default="", help="suffix for the checkpoint name")
    ap.add_argument("--out", default=f"{STREAMCDF}/runs")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No GPU detected; training on CPU will be slow.")

    # explicit --obs_frames/--pred_frames overrides any preset, and forces a
    # single custom run (doesn't make sense combined with --horizon all)
    if args.obs_frames is not None and args.pred_frames is not None:
        HORIZONS["custom"] = dict(obs_frames=args.obs_frames, pred_frames=args.pred_frames)
        run_list = ["custom"]
    else:
        run_list = list(HORIZONS) if "all" in args.horizon else list(dict.fromkeys(args.horizon))

    for horizon in run_list:
        run_training(horizon, args, device)


if __name__ == "__main__":
    main()
