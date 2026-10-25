"""Batched transport evaluation and Stream-CDF training helpers.

Evaluation reports raw transport plus kappa times unmatched mass. Training can
subtract self-comparisons using sinkhorn_divergence_loss. Output gain and bounded
residuals preserve the recorded model architecture while enabling optimization.
"""

import torch


from .model import flat_ot_loss, TRAIN_CFG, EVAL_CFG, _sep_softmax_transform


# ---------------------------------------------------------------------------
# FAST per-clip D_kappa -- replaces the 1448-call Python loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def flat_ot_values(pred, target, kappa=8.0, eps=0.5, iters=30,
                   den_factor=100.0, pool=2, pixel_size=1.0):
    """
    Same computation as flat_ot_loss, UNREDUCED: returns per-(clip,frame)
    tensors of shape (B, T) instead of collapsing everything to one scalar
    with .mean(). This is the entire fix for the hour-long eval.

    WHY THE LOOP WAS SLOW. flat_ot_loss already treats every frame as an
    independent transport problem: B and T fold into ONE axis M = B*T at the
    top (`a = pred.reshape(B*T, H, W)`) and the whole Sinkhorn loop runs on
    all M problems in parallel via broadcasting -- there is no coupling across
    clips or frames anywhere in the solve. Calling it once per clip (batch=1)
    therefore recomputed nothing extra mathematically, it just paid full
    Python + CUDA kernel-launch overhead 1448 separate times instead of once.
    At ~2-3s of pure launch/dispatch overhead per call, 1448 calls is the
    entire missing hour.

    This function inlines flat_ot_loss's tail (everything after the Sinkhorn
    loop) and skips only the final `.mean()`, reshaping the (B*T,) primal back
    to (B, T) and averaging over T alone. B stays batched -- call this on
    (e.g.) 64 or 128 clips at once, not 1448 individually.

    Math is copied verbatim from stream_cdf_v4.flat_ot_loss (same helper
    `_sep_softmax_transform`, same log-domain stabilisation) so the returned
    numbers are identical to what stream_cdf_metric reports, just unreduced.
    """
    B, _, T, H, W = pred.shape
    a = pred.reshape(B * T, H, W) / den_factor
    b = target.reshape(B * T, H, W) / den_factor
    if pool > 1:
        a = torch.nn.functional.avg_pool2d(a.unsqueeze(1), pool).squeeze(1) * (pool * pool)
        b = torch.nn.functional.avg_pool2d(b.unsqueeze(1), pool).squeeze(1) * (pool * pool)
        H, W = a.shape[-2:]

    px = pool * pixel_size
    ih = torch.arange(H, device=a.device, dtype=a.dtype)
    iw = torch.arange(W, device=a.device, dtype=a.dtype)
    dh = (ih[:, None] - ih[None, :]).abs() * px
    dw = (iw[:, None] - iw[None, :]).abs() * px

    tiny = 1e-30
    log_a = a.clamp_min(tiny).log()
    log_b = b.clamp_min(tiny).log()

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(iters):
        f = (-_sep_softmax_transform(g + eps * log_b, dh, dw, eps)).clamp(-kappa, kappa)
        g = (-_sep_softmax_transform(f + eps * log_a, dh, dw, eps)).clamp(-kappa, kappa)

    lu = f / eps + log_a
    lv = g / eps + log_b
    Mu = lu.amax(dim=(-1, -2), keepdim=True)
    Mv = lv.amax(dim=(-1, -2), keepdim=True)
    u = torch.exp(lu - Mu)
    v = torch.exp(lv - Mv)
    sh = (Mu + Mv).reshape(-1)

    Kr = torch.exp(-dh / eps)
    Kc = torch.exp(-dw / eps)

    def _unshift(x, ndim):
        s = sh.reshape((-1,) + (1,) * ndim)
        return torch.exp(x.clamp_min(tiny).log() + s)

    Kv = torch.einsum("hg,mgc->mhc", Kr, torch.einsum("mhc,cd->mhd", v, Kc))
    Ku = torch.einsum("hg,mgc->mhc", Kr, torch.einsum("mhc,cd->mhd", u, Kc))
    row = _unshift(u * Kv, 2)
    col = _unshift(v * Ku, 2)
    unmatched = ((a - row).abs().sum(dim=(-1, -2))
                 + (b - col).abs().sum(dim=(-1, -2)))       # (B*T,) people

    UKc = torch.einsum("mhc,cd->mhd", u, Kc)
    Rrow = torch.einsum("mhd,mgd->mhg", UKc, v)
    cost_r = _unshift((Rrow * (Kr * dh)[None]).sum(dim=(-1, -2)), 0)
    KrV = torch.einsum("hg,mgc->mhc", Kr, v)
    Rcol = torch.einsum("mhc,mhd->mcd", u, KrV)
    cost_c = _unshift((Rcol * (Kc * dw)[None]).sum(dim=(-1, -2)), 0)
    transport = cost_r + cost_c                             # (B*T,)
    D = (transport + kappa * unmatched)                     # (B*T,) exact primal

    return (D.reshape(B, T).mean(dim=1),
           transport.reshape(B, T).mean(dim=1),
           unmatched.reshape(B, T).mean(dim=1))              # each (B,)


def d_kappa_per_clip(pred, target, den_factor=100.0, batch_chunk=64,
                     decompose=False, progress=False, **cfg):
    """
    Drop-in replacement for looping stream_cdf_metric over single clips.

    pred, target: (N, 1, T, H, W), the FULL test set (or however many clips
    you have on hand). Processed in chunks of `batch_chunk` clips (default
    64) purely to bound GPU memory -- NOT because the solve needs it; the
    Sinkhorn loop is already embarrassingly parallel across clips.

    Runs at EVAL_CFG precision (fp64) by default, matching stream_cdf_metric.

    decompose=False (default): returns a (N,) float64 array of D_kappa.
    decompose=True:            returns (D, transport, unmatched), each (N,).

    WHY THE DECOMPOSITION MATTERS. D_kappa = transport + kappa * unmatched is
    the ONLY metric here that scores shape and count together; count-MAE is
    blind to WHERE mass sits, a pure-shape metric is blind to HOW MANY. The
    two halves are already computed inside flat_ot_values and were simply
    being thrown away, so splitting them costs nothing:

        transport  SHAPE error -- moving matched mass to where it belongs,
                   in people*pixels. Falls when placement improves.
        unmatched  COUNT error -- PEOPLE the two maps could not agree on.
                   Rises when totals get worse.

    Reporting both turns "D_kappa fell while count-MAE rose, so shape MUST
    have improved" from an inference into a measurement, and shows directly
    what exchange rate kappa is enforcing between them.
    """
    import sys as _sys, time as _time
    c = {**EVAL_CFG, **cfg}
    N = pred.shape[0]
    D_out, tr_out, um_out = [], [], []
    _t0 = _time.time()
    for s in range(0, N, batch_chunk):
        p = pred[s:s + batch_chunk].double()
        t = target[s:s + batch_chunk].double()
        d, tr, um = flat_ot_values(p, t, den_factor=den_factor, **c)
        D_out.append(d.cpu()); tr_out.append(tr.cpu()); um_out.append(um.cpu())
        if progress:
            done = min(s + batch_chunk, N)
            el = _time.time() - _t0
            _sys.stdout.write(f"\r  D_kappa {done}/{N} clips  "
                              f"{el:.0f}s elapsed, ~{el/done*(N-done):.0f}s left ")
            _sys.stdout.flush()
    if progress:
        _sys.stdout.write("\n")
    D = torch.cat(D_out).numpy().astype("float64")
    if not decompose:
        return D
    return (D,
            torch.cat(tr_out).numpy().astype("float64"),
            torch.cat(um_out).numpy().astype("float64"))


# ---------------------------------------------------------------------------
# FIX 1. Output gain -- the one that actually unblocks training
# ---------------------------------------------------------------------------
def attach_output_gain(model, gain=100.0, bound=None):
    """
    Multiply the head's delta by a FIXED constant `gain`, optionally BOUNDED.

    gain
      Raises d(output)/d(head weight) so the model can leave the anchor. A
      fixed constant, not a learned scalar: a learned gain initialised at 0
      times a zero-init weight is a product of two zeros, gradient 0 in both,
      and it never leaves the origin. gain * 0 == 0, so the exact
      anchor-at-init property survives (self-test 7 still passes).

    bound  -- THE RUNAWAY GUARD. delta = bound * tanh(gain * raw / bound)
      Without it, training diverges to a DEAD model, observed directly:
      ~98% of density pixels are empty and every empty pixel contributes
      dD/da = +kappa, so the summed gradient carries a large net DOWNWARD
      bias. The head's bias term moves all 6400 pixels with one parameter, so
      that push is concentrated rather than spread. At high gain delta runs to
      large negative, softplus(beta=10) floors the output at 0, its derivative
      vanishes, and the gradient dies permanently -- measured: pred == 0.0
      across the horizon, |delta|max 55.8, grad norm p50 0.000, no recovery.

      tanh bounds |delta| <= `bound` while staying EXACTLY linear with slope
      `gain` near 0 (tanh(x) ~ x), so early training is unchanged and only the
      runaway tail is clipped. tanh(0) = 0 keeps the anchor property.

      Set bound to a few times the largest plausible density change. Peak
      density is O(50) at den_factor 100, so bound=50 is a reasonable start.

    NOTE ON CALIBRATION. The free-tensor preflight gate OVERESTIMATES the gain
    needed here. It optimises 6400 INDEPENDENT pixels, so every pixel must be
    moved separately; the head expresses global modes ("scale everything down")
    through shared parameters, which is orders of magnitude cheaper. The gate
    calibrates fine spatial detail, not global count. Start ~10x below what the
    gate suggests and raise it while watching |delta|max.
    """
    head = model.head
    if getattr(head, "_gain_attached", False):
        raise RuntimeError("output gain already attached")
    inner = head.forward

    def forward_with_gain(tokens, _inner=inner, _g=float(gain), _b=bound):
        d = _inner(tokens) * _g
        if _b is not None:
            d = float(_b) * torch.tanh(d / float(_b))
        return d

    head.forward = forward_with_gain
    head._gain_attached = True
    head.gain = float(gain)
    head.bound = bound
    return model


# ---------------------------------------------------------------------------
# FIX 2. Debiased loss -- Sinkhorn divergence over D_kappa
# ---------------------------------------------------------------------------
def sinkhorn_divergence_loss(pred, target, den_factor=100.0, **cfg):
    """
    S(a,b) = D_k(a,b) - 0.5 D_k(a,a) - 0.5 D_k(b,b)

    WHY. flat_ot_loss returns the ENTROPIC D_kappa. Entropic regularisation is
    a bias, not noise: at the training config a PERFECT prediction scores
    D = 10.51, not 0, and dD/da = f does not vanish at a = b (measured:
    max|dD/dpred| = kappa/DEN = 0.08 at the exact optimum, 47.9% of pixels
    saturated at |f| = kappa). An objective whose gradient does not vanish at
    its own minimiser has no annealing: Adam hovers at fixed step size forever,
    which is precisely the +-1 jitter in the epoch table.

    The debiasing is the standard Sinkhorn divergence (Feydy et al. 2019). By
    the envelope theorem the gradient becomes

        dS/da = f_ab - f_aa

    which is EXACTLY 0 at a = b, is still bounded by 2*kappa, and still costs
    one inner product to backprop. S(a,b) >= 0 with equality iff a = b, so the
    reported number is now interpretable as an absolute error rather than an
    error plus an unknown blur floor.

    COST. Two extra solves. D(b,b) carries no gradient (target is a constant),
    so it runs under no_grad; D(a,a) does. ~2.4x the wall clock of the raw
    loss. If that is too expensive, drop the D(b,b) term -- it is constant in
    the parameters and changes only the reported VALUE, not the gradient --
    and set `include_bb=False`.

    Returns (loss, transport_people_px, unmatched_people) with the same
    signature as stream_cdf_loss, so it is a drop-in replacement. transport
    and unmatched are reported from the D(a,b) term only, so they keep their
    original meaning as diagnostics.
    """
    include_bb = cfg.pop("include_bb", True)
    self_iters = cfg.pop("self_iters", None)
    c = {**TRAIN_CFG, **cfg}
    c_self = {**c, "iters": self_iters or c["iters"]}

    d_ab, transport, unmatched = flat_ot_loss(pred, target,
                                              den_factor=den_factor, **c)
    d_aa, _, _ = flat_ot_loss(pred, pred, den_factor=den_factor, **c_self)
    if include_bb:
        with torch.no_grad():
            d_bb, _, _ = flat_ot_loss(target, target, den_factor=den_factor, **c_self)
    else:
        d_bb = torch.zeros((), device=d_ab.device, dtype=d_ab.dtype)

    loss = d_ab - 0.5 * d_aa - 0.5 * d_bb
    return loss, transport, unmatched


# ---------------------------------------------------------------------------
# FIX 3. Parameter groups
# ---------------------------------------------------------------------------
def build_param_groups(model, lr, head_lr_mult=1.0, weight_decay=1e-4):
    """
    Three groups:
      decay    -- weight matrices in the encoder / decoder
      no_decay -- biases, LayerNorms, learned queries, positional tables
      head     -- head.proj + anchor_w, NEVER decayed, optionally higher lr

    WHY head gets its own group: AdamW's decoupled weight decay applies a pull
    of -lr * wd * w every step. On a ZERO-INIT head that is a spring anchored
    at exactly the initialisation the model is trying to escape. It does not
    regularise anything -- there is nothing there yet to regularise -- it just
    subtracts from an already-marginal slew rate.

    head_lr_mult is the alternative to attach_output_gain. Use ONE of the two.
    """
    head_ids, head, decay, no_decay = set(), [], [], []
    for p in list(model.head.parameters()) + [model.anchor_w]:
        head_ids.add(id(p)); head.append(p)

    for name, p in model.named_parameters():
        if id(p) in head_ids or not p.requires_grad:
            continue
        if p.ndim <= 1 or "queries" in name or "spatial" in name or "temporal" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay,    "lr": lr,                "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr,                "weight_decay": 0.0},
        {"params": head,     "lr": lr * head_lr_mult, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# PREFLIGHT GATE -- run this before ever queueing a 40-epoch job
# ---------------------------------------------------------------------------
@torch.no_grad()
def _blobs(centres, H=80, W=80, sigma=3.0, den=100.0):
    ii = torch.arange(H).float()
    m = torch.zeros(H, W)
    for cy, cx, n in centres:
        g = torch.exp(-((ii[:, None] - cy) ** 2 + (ii[None, :] - cx) ** 2)
                      / (2 * sigma ** 2))
        m = m + g / g.sum() * n
    return (m * den).reshape(1, 1, 1, H, W)


def preflight_gate(loss_fn=None, lr=2.5e-4, steps=150, den_factor=100.0, **cfg):
    """
    The cheapest possible falsification of a training config.

    Optimise the PREDICTION TENSOR ITSELF with Adam at the lr you intend to
    use. 6400 free parameters, no encoder, no decoder, no zero-init head --
    the easiest version of the problem that exists. If D does not fall
    substantially here, it cannot fall for any architecture, and a 40-epoch
    cluster job will return a flat table.

    Prints the drop and the residual gradient at the optimum. A healthy config
    shows: drop > 80%, and max|grad| at the PERFECT prediction near 0.

    Runs in ~2 min on one CPU core. Run it every time you change kappa, eps,
    pool, den_factor or lr.
    """
    loss_fn = loss_fn or (lambda p, t, **k: flat_ot_loss(p, t, **k))
    c = {**TRAIN_CFG, **cfg}

    target = _blobs([(30, 25, 20), (45, 55, 25), (60, 30, 15)], den=den_factor)
    start = _blobs([(24, 19, 20), (39, 49, 25), (54, 24, 15)], den=den_factor)

    p = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    d0 = None
    for _ in range(steps):
        loss, _, _ = loss_fn(p, target, den_factor=den_factor, **c)
        if d0 is None:
            d0 = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p.clamp_(min=0)
    d1, _, _ = loss_fn(p, target, den_factor=den_factor, **c)
    d1 = d1.item()

    q = target.clone().requires_grad_(True)
    lp, _, _ = loss_fn(q, target, den_factor=den_factor, **c)
    lp.backward()

    drop = 100.0 * (d0 - d1) / max(abs(d0), 1e-9)
    print(f"  preflight  lr={lr:.1e} steps={steps}")
    print(f"    D {d0:9.3f} -> {d1:9.3f}   ({drop:5.1f}% drop)")
    print(f"    perfect prediction scores D = {lp.item():.4f}   "
          f"(should be ~0)")
    print(f"    max|dD/dpred| AT the optimum = {q.grad.abs().max():.3e}   "
          f"(should be ~0)")
    ok = drop > 60.0 and abs(lp.item()) < 0.05 * abs(d0)
    print(f"    VERDICT: {'PASS -- this config can train' if ok else 'FAIL -- do not queue a job with this config'}")
    return ok


if __name__ == "__main__":
    print("\n[A] current config (train_v4 defaults)")
    preflight_gate(lr=2.5e-4)

    print("\n[B] debiased loss, same lr -- isolates fix 2")
    preflight_gate(loss_fn=sinkhorn_divergence_loss, lr=2.5e-4)

    print("\n[C] debiased loss + eps matched to pooled pixel pitch + gain-equivalent lr")
    preflight_gate(loss_fn=sinkhorn_divergence_loss, lr=2.5e-2, eps=2.0)
