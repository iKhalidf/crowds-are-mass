"""
divergence_metrics.py -- CrowdMAC's ADJS/ADKL/FDJS family, PER CLIP.

CrowdMAC's metrics.py returns one scalar per batch (nn.KLDivLoss with
reduction="batchmean"), which cannot be sliced by stratum. This reimplements
the identical arithmetic while keeping the per-clip axis, and
`verify_against_crowdmac()` checks the two agree.

WHAT THESE METRICS ARE
    AD*  Average Displacement: mean over the horizon frames
    FD*  Final Displacement:   last frame only
    *KL / *rKL / *JS   KL(gt||pred), KL(pred||gt), Jensen-Shannon
(ADE/FDE naming from trajectory forecasting; the "displacement" is a
divergence between density maps, not a distance between points.)

THE COUNT-BLINDNESS, VERIFIED AGAINST THE REAL IMPLEMENTATION
kl_divergence() normalises each frame to sum 1 before comparing:
      p = p / sum(p, dim=(1,2))
Normalisation discards the total. Measured with CrowdMAC's own code, a
prediction equal to ground truth scaled by 100x -- 4,500 people where 45
exist, spatial pattern exact -- scores ADJS = -4.7e-07, i.e. zero to float
precision, identical to a perfect forecast. Same for all six metrics.

For JS specifically the invariance is not general: m = 0.5*(p+q) is formed
from the UNNORMALISED maps, so scaling q does move m. But whenever the
prediction is PROPORTIONAL to the truth, m stays proportional too and all
three normalised arguments coincide, giving exactly 0. Proportional-but-wrong
-count is precisely the crowd-safety failure case.

NOTE ON CrowdMAC's IMPLEMENTATION. kl_divergence does `p += eps` IN PLACE,
mutating its arguments; compute_eval_metrics calls it three times on the same
tensors, so the second and third calls see p+2eps, p+3eps. With eps=1e-10 on
densities of order 1e-2 this is numerically irrelevant, but it is why the
functions here take copies.
"""

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-10


def _norm(x, eps=EPS):
    """CrowdMAC's normalisation, exactly: add eps, normalise per frame over
    (H,W), then clamp. This is the step that discards count."""
    x = x + eps
    x = x / (x.sum(dim=(-1, -2), keepdim=True) + eps)
    return x.clamp(min=eps, max=1 - eps)


def _kl_perframe(p, q, eps=EPS):
    """KL(p || q) summed over pixels, per frame. Matches CrowdMAC's
    nn.KLDivLoss(reduction='batchmean')((q+eps).log(), p) before its division
    by the frame count."""
    pn, qn = _norm(p, eps), _norm(q, eps)
    return (pn * (pn.log() - (qn + eps).log())).sum(dim=(-1, -2))


def _js_perframe(p, q, eps=EPS):
    """CrowdMAC forms the mixture from the UNNORMALISED maps."""
    m = 0.5 * (p + q)
    return 0.5 * (_kl_perframe(p, m, eps) + _kl_perframe(q, m, eps))


@torch.no_grad()
def divergence_metrics(pred, target, per_clip=True, eps=EPS):
    """
    pred, target: (N, 1, T, H, W), HORIZON frames only, same scale convention
    as each other (they are normalised internally -- that is the point).

    Returns dict of (N,) arrays if per_clip else scalars:
        adkl, adrkl, adjs   mean over the T horizon frames
        fdkl, fdrkl, fdjs   final frame only
    """
    p = pred.squeeze(1).double()
    q = target.squeeze(1).double()
    kl = _kl_perframe(q, p, eps)      # KL(gt || pred)
    rkl = _kl_perframe(p, q, eps)
    js = _js_perframe(q, p, eps)
    out = dict(adkl=kl.mean(1), adrkl=rkl.mean(1), adjs=js.mean(1),
               fdkl=kl[:, -1], fdrkl=rkl[:, -1], fdjs=js[:, -1])
    out = {k: v.cpu().numpy() for k, v in out.items()}
    return out if per_clip else {k: float(v.mean()) for k, v in out.items()}


def verify_against_crowdmac(videos, predicted_videos, obs_frames=8, den_factor=100):
    """Diff this file against CrowdMAC's real compute_eval_metrics."""
    from metrics import compute_eval_metrics
    ref = compute_eval_metrics(videos.clone(),
                               predicted_videos.clone() / den_factor, obs_frames)
    names = ["adkl", "adrkl", "adjs", "fdkl", "fdrkl", "fdjs"]
    ref = {n: float(v) for n, v in zip(names, ref)}
    mine = divergence_metrics((predicted_videos / den_factor)[:, :, obs_frames:],
                              videos[:, :, obs_frames:], per_clip=False)
    print(f"{'metric':<8}{'crowdmac':>13}{'mine':>13}{'rel diff':>12}")
    ok = True
    for n in names:
        d = abs(ref[n] - mine[n]) / max(abs(ref[n]), 1e-12)
        ok &= d < 1e-3   # relative: CrowdMAC runs float32, this runs float64,
                          # which shows up on rKL where log() hits near-zero
                          # predicted pixels and the value is large
        print(f"{n:<8}{ref[n]:>13.6f}{mine[n]:>13.6f}{d:>12.2e}")
    print("MATCH" if ok else "MISMATCH")
    return ok
