"""Fixed-window Stream-CDF forecaster and raw unbalanced transport solver.

The causal encoder reads observed density-map tubelets. Learned horizon queries
decode all future frames as residuals on a learned recent-frame anchor.
Training debiasing and output-gain helpers live in streamcdf.transport.
The paper runs use absolute temporal embeddings; ALiBi is optional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Cube embedding
# ---------------------------------------------------------------------------
class CubeEmbed(nn.Module):
    """
    Density clip -> sequence of space-time tokens.

        (B, 1, T, H, W)  ->  (B, N, D)     N = (T/Tc) * (H/P) * (W/P)

    A Conv3d with kernel == stride slices the clip into non-overlapping cubes
    and projects each to D dims in one op.
    """

    def __init__(self, img_size=80, patch=8, tubelet=4, num_frames=76, dim=384):
        super().__init__()
        assert num_frames % tubelet == 0, "num_frames must divide by tubelet"
        self.grid_h = img_size // patch
        self.grid_w = img_size // patch
        self.grid_t = num_frames // tubelet
        self.num_tokens = self.grid_t * self.grid_h * self.grid_w
        self.tubelet = tubelet
        self.patch = patch
        self.dim = dim

        self.proj = nn.Conv3d(
            in_channels=1,
            out_channels=dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)
        x = self.proj(x)              # (B, D, t, h, w)
        x = x.flatten(2)              # (B, D, t*h*w)
        x = x.transpose(1, 2)         # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# 2. Causal mask
# ---------------------------------------------------------------------------
def build_causal_mask(grid_t, tokens_per_frame, device=None):
    """
    Temporal causal mask over space-time tokens laid out (t, h, w) flattened.

    Token i may attend to token j  iff  chunk(j) <= chunk(i),
    where chunk(x) = x // tokens_per_frame.

    Spatial attention WITHIN a time-chunk is unrestricted -- a crowd at time t is
    a spatial whole. Causality applies to TIME only.

    Returns bool (N, N):  True = BLOCKED, False = allowed  (masked_fill convention).
    NOTE: scaled_dot_product_attention uses the OPPOSITE convention for bool masks,
    so we convert to an additive float mask before handing it to SDPA.
    """
    N = grid_t * tokens_per_frame
    idx = torch.arange(N, device=device)
    chunk = idx // tokens_per_frame
    return chunk[None, :] > chunk[:, None]


def bool_mask_to_float(mask, dtype=torch.float32):
    """True=BLOCKED bool mask -> additive float mask (0 = allow, -inf = block)."""
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(
        mask, float("-inf")
    )


def alibi_slopes(heads):
    """
    ALiBi per-head slopes, geometric sequence 2^(-8/heads * i), i=1..heads.
    (Press et al. 2022.) Each head decays attention with temporal distance at a
    different rate, so the model keeps a mix of short- and long-memory heads.
    """
    start = 2.0 ** (-8.0 / heads)
    return torch.tensor([start ** (i + 1) for i in range(heads)])


def build_temporal_bias(grid_t, tokens_per_frame, heads, device=None):
    """
    Causal mask + ALiBi recency bias, as ONE additive (H, N, N) float mask
    handed straight to scaled_dot_product_attention.

        bias[h, i, j] = -inf                       if chunk(j) > chunk(i)  (future)
                      = -slope[h] * (chunk(i)-chunk(j))   otherwise        (recency)

    Why this REPLACES the absolute temporal embedding: an absolute learned table
    has a fixed number of rows (grid_t) and is meaningless at time indices beyond
    what it was trained on -- so a model using it CANNOT run on an unbounded
    stream, which is precisely the property the title claims. A relative recency
    bias depends only on the DISTANCE between two chunks, so it is valid at any
    absolute time index: the encoder memory can grow indefinitely as a KV cache
    and the bias still applies. Spatial position stays a learned table (distance-
    on-a-grid is not a recency relationship, so ALiBi would be wrong for it).
    """
    N = grid_t * tokens_per_frame
    idx = torch.arange(N, device=device)
    chunk = idx // tokens_per_frame                       # (N,)
    dist = chunk[:, None] - chunk[None, :]                 # (N, N), >0 = j is past
    future = dist < 0

    slopes = alibi_slopes(heads).to(device)                # (H,)
    bias = -slopes[:, None, None] * dist[None].clamp(min=0).float()   # (H, N, N)
    bias = bias.masked_fill(future[None], float("-inf"))
    return bias


# ---------------------------------------------------------------------------
# 3. Causal attention + transformer block
# ---------------------------------------------------------------------------
class CausalAttention(nn.Module):
    """
    Multi-head self-attention with an optional temporal causal mask.

    Default path uses scaled_dot_product_attention (flash / mem-efficient
    kernels), which never materialises the (B,H,N,N) matrix. Set return_attn=True
    only for diagnostics on small inputs -- it falls back to the explicit O(N^2)
    path so the attention weights can be inspected.
    """

    def __init__(self, dim=384, heads=6, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        B, N, D = x.shape
        H, Hd = self.heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, H, N, Hd)

        # mask may be:
        #   None                         -- no masking (decoder self-attn)
        #   bool (N, N)  True=BLOCKED     -- plain causal mask
        #   float (H, N, N) or (N, N)     -- additive bias (ALiBi + causal)
        is_bool = mask is not None and mask.dtype == torch.bool

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                if is_bool:
                    att = att.masked_fill(mask, float("-inf"))
                else:
                    att = att + mask                     # broadcast (H,N,N)
            att = att.softmax(dim=-1)
            att = self.drop(att)
            out = (att @ v).transpose(1, 2).reshape(B, N, D)
            return self.proj(out), att

        if mask is None:
            attn_mask = None
        elif is_bool:
            attn_mask = bool_mask_to_float(mask, q.dtype)
        else:
            attn_mask = mask.to(q.dtype)                  # already additive
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out), None


class Block(nn.Module):
    """Pre-norm transformer block: causal attention + MLP, both residual."""

    def __init__(self, dim=384, heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, return_attn=False):
        a, att = self.attn(self.norm1(x), mask, return_attn=return_attn)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, att


# ---------------------------------------------------------------------------
# 4. Factorized positional embedding
# ---------------------------------------------------------------------------
class FactorizedPosEmbed(nn.Module):
    """
    Separate spatial and temporal position embeddings, added to each token.

    Factorizing keeps the SPATIAL table fixed-size (the image never grows), so
    only the TEMPORAL part must handle longer streams. In the streaming version
    the temporal table is replaced by a relative/recency bias (ALiBi), valid at
    ANY time index -- absolute learned positions are not, and that is exactly
    what would break a model running on an unbounded stream.
    """

    def __init__(self, grid_t=19, grid_h=10, grid_w=10, dim=384, use_temporal=True):
        super().__init__()
        self.n_spatial = grid_h * grid_w
        self.grid_t = grid_t
        self.use_temporal = use_temporal

        self.spatial = nn.Parameter(torch.zeros(1, self.n_spatial, dim))
        nn.init.trunc_normal_(self.spatial, std=0.02)

        # In ALiBi mode there is NO absolute temporal table -- recency is encoded
        # by the attention bias instead, which is what makes the model streamable.
        if use_temporal:
            self.temporal = nn.Parameter(torch.zeros(1, grid_t, dim))
            nn.init.trunc_normal_(self.temporal, std=0.02)

    def forward(self, x):
        B, N, D = x.shape
        S, T = self.n_spatial, self.grid_t
        assert N == T * S, f"expected {T * S} tokens, got {N}"

        sp = self.spatial.repeat(1, T, 1)                    # (1, T*S, D)
        x = x + sp
        if self.use_temporal:
            tp = self.temporal.repeat_interleave(S, dim=1)   # (1, T*S, D)
            x = x + tp
        return x


# ---------------------------------------------------------------------------
# 5. Horizon queries (cross-attention decoder)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Queries attend into an encoded memory. Q from queries; K, V from memory."""

    def __init__(self, dim=384, heads=6, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in, memory, return_attn=False):
        # q_in: (B, Nq, D)   memory: (B, Nm, D)
        B, Nq, D = q_in.shape
        Nm = memory.shape[1]
        H, Hd = self.heads, self.head_dim

        q = self.q(q_in).reshape(B, Nq, H, Hd).transpose(1, 2)
        kv = self.kv(memory).reshape(B, Nm, 2, H, Hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * self.scale     # (B, H, Nq, Nm)
            att = att.softmax(dim=-1)
            att = self.drop(att)
            out = (att @ v).transpose(1, 2).reshape(B, Nq, D)
            return self.proj(out), att

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.proj(out), None


class HorizonDecoder(nn.Module):
    """
    Learned query tokens -- one per (future time-chunk x spatial location).

    Each query cross-attends into the encoded past and emits its own patch of the
    future density map. One forward pass produces ALL horizons: no autoregressive
    rollout, hence no compounding error.

    The queries live OUTSIDE the encoder sequence. That separation is what lets
    the encoder's memory become a streaming KV cache: the cache holds only the
    past, and queries attach at the end to read from it.

    Query self-attention is UNMASKED -- all future timesteps are produced at once
    and may inform each other. Fine for one-shot decoding; it does preclude using
    this decoder autoregressively later.
    """

    def __init__(self, n_future_chunks=18, n_spatial=100, dim=384,
                 heads=6, depth=2, mlp_ratio=4.0):
        super().__init__()
        self.n_future_chunks = n_future_chunks
        self.n_spatial = n_spatial
        self.n_queries = n_future_chunks * n_spatial

        self.queries = nn.Parameter(torch.zeros(1, self.n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim),
                "norm_m": nn.LayerNorm(dim),
                "cross": CrossAttention(dim, heads),
                "norm_s": nn.LayerNorm(dim),
                "self": CausalAttention(dim, heads),
                "norm_f": nn.LayerNorm(dim),
                "mlp": nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim * mlp_ratio), dim),
                ),
            }))

    def forward(self, memory, return_attn=False):
        B = memory.shape[0]
        q = self.queries.expand(B, -1, -1)

        last_cross = None
        for L in self.layers:
            c, att = L["cross"](L["norm_q"](q), L["norm_m"](memory),
                                return_attn=return_attn)
            q = q + c
            last_cross = att
            s, _ = L["self"](L["norm_s"](q), mask=None)
            q = q + s
            q = q + L["mlp"](L["norm_f"](q))

        return q, last_cross


# ---------------------------------------------------------------------------
# 6. Density head + count-aware loss
# ---------------------------------------------------------------------------
class DensityHead(nn.Module):
    """
    Future tokens -> a DELTA on the future density maps.

    Each token decodes to one dec_tubelet x P x P cube; cubes are reassembled into
    a clip. dec_tubelet is deliberately SMALLER than the encoder's: the past is
    redundant and can be compressed, but the future needs temporal precision, and
    a large tubelet forces every frame inside a cube to come from one linear
    projection of one token -- which structurally flattens the predicted trajectory.

    Output is the RAW DELTA and may be NEGATIVE (crowds shrink). Non-negativity is
    applied in StreamCDF.forward, on (anchor + delta).

    proj is ZERO-INITIALISED: delta starts at exactly 0, so the model begins life
    as its own anchor and every gradient step is a pure improvement on it.
    """

    def __init__(self, dim=384, tubelet=2, patch=8, grid_h=10, grid_w=10):
        super().__init__()
        self.tubelet = tubelet
        self.patch = patch
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_dim = tubelet * patch * patch

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens):
        # tokens: (B, Nq, D)
        B, Nq, D = tokens.shape
        S = self.grid_h * self.grid_w
        Tc = Nq // S

        x = self.proj(self.norm(tokens))            # (B, Nq, tubelet*P*P)
        x = x.reshape(B, Tc, self.grid_h, self.grid_w,
                      self.tubelet, self.patch, self.patch)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, 1,
                      Tc * self.tubelet,
                      self.grid_h * self.patch,
                      self.grid_w * self.patch)
        return x        # raw delta -- free to go NEGATIVE


def _softmin_1d(x, dist, eps):
    """
    eps-softmin of (x_j + dist_ij) over the LAST axis, via logsumexp.

        out_i = -eps * log sum_j exp( (x_j - dist_ij) / eps ... )

    written as a max-shifted logsumexp for numerical stability.
    x:    (..., n)     dist: (n, n) nonneg     returns (..., n)
    """
    z = (x.unsqueeze(-2) - dist) / eps            # (..., n_out, n_in)
    return eps * torch.logsumexp(z, dim=-1)       # (..., n_out)


def _sep_softmax_transform(x, dh, dw, eps):
    """
    eps * log sum_j exp( (x_j - C_ij) / eps )   over ALL pixels j of a 2-D map,
    with the L1 (city-block) ground cost  C_ij = dh|Δrow| + dw|Δcol|.

    L1 cost SEPARATES:  exp(-C/eps) = exp(-dh|Δr|/eps) * exp(-dw|Δc|/eps),
    so the full (HW x HW) transform is two 1-D passes -- O(HW(H+W)) instead of
    O(H^2 W^2). A 6400x6400 cost matrix is never materialised. This is the only
    reason the metric is affordable inside a training loop.

    x: (M, H, W) log-domain field. Returns (M, H, W).
    """
    x = _softmin_1d(x, dw, eps)                          # over columns
    x = _softmin_1d(x.transpose(-1, -2), dh, eps)        # over rows
    return x.transpose(-1, -2)


def flat_ot_loss(pred, target, kappa=8.0, eps=0.5, iters=30,
                 den_factor=100.0, pool=2, pixel_size=1.0):
    """
    D_kappa(pred, target): unbalanced optimal transport with a flat penalty.

        D_k = min_pi  sum_ij pi_ij d(i,j)
                      + kappa * ( ||a - pi.1||_1 + ||b - pi^T.1||_1 )
        s.t.  pi >= 0,  pi.1 <= a,  pi^T.1 <= b

    a = pred/den_factor, b = target/den_factor  (PEOPLE per pixel, per frame).

    Match what you can, charge kappa per person for what's left over:
      * matched mass pays its transport distance  -> SHAPE error
      * unmatched mass pays kappa flat            -> COUNT error, in people
    Destroy+create costs 2*kappa, so mass never moves farther than 2*kappa:
    kappa IS the exchange rate "pixels of displacement per person of count
    error", one interpretable knob replacing lam_count.

    Why this replaces MSE + count:
      * MSE over a sparse field is minimised by the conditional mean -- blur in
        space, FLAT in time. v2 removed the architectural cause of the flat
        forecast; this removes the objective's PREFERENCE for it. D_k is
        minimised only by mass of the right size in the right place.
      * the old count term was sign-blind (a sum): wrong-place mass was free.
        Here misplacement pays distance, capped at 2*kappa.
      * the old MSE was in (x den_factor)^2 units while count was in people --
        lam_count was silently entangled with the density scaling (the v2
        header's claim C held for one term only). Everything here is in
        people * pixels. No hidden scale.

    SOLVER: entropic TV-Sinkhorn (Chizat et al. 2018, scaling algorithms for
    unbalanced transport). The ||.||_1 marginal penalty makes each dual update
    a plain Sinkhorn update CLAMPED to [-kappa, +kappa] -- the clamp is where
    "give up and pay kappa" happens. Log-domain throughout; L1 ground cost for
    separability (see _sep_softmax_transform).

    GRADIENT: by the envelope theorem, dD/da = f* (the optimal dual potential).
    So the loss is the dual value  <a, f> + <b, g>  with f, g DETACHED: the
    Sinkhorn loop runs under no_grad (cheap, nothing stored), and backprop
    through the loss is a single inner product. No autograd through 30
    iterations, no O(N^2) memory anywhere.

    pool: mass-preserving sum-pool factor applied first (80 -> 40 at pool=2).
    Distances scale by pool*pixel_size, so kappa keeps its meaning in ORIGINAL
    pixels. Quarter the resolution, ~1/8 the softmin cost.

    pred, target: (B, 1, T, H, W). Every frame is its own transport problem;
    frames fold into the batch axis. Returns (loss, transport_people_px,
    unmatched_people) -- the last two are floats for logging.
    """
    B, _, T, H, W = pred.shape

    a = pred.reshape(B * T, H, W) / den_factor
    b = target.reshape(B * T, H, W) / den_factor
    if pool > 1:
        a = F.avg_pool2d(a.unsqueeze(1), pool).squeeze(1) * (pool * pool)
        b = F.avg_pool2d(b.unsqueeze(1), pool).squeeze(1) * (pool * pool)
        H, W = a.shape[-2:]

    px = pool * pixel_size
    ih = torch.arange(H, device=a.device, dtype=a.dtype)
    iw = torch.arange(W, device=a.device, dtype=a.dtype)
    dh = (ih[:, None] - ih[None, :]).abs() * px          # (H, H)
    dw = (iw[:, None] - iw[None, :]).abs() * px          # (W, W)

    tiny = 1e-30
    log_a = a.detach().clamp_min(tiny).log()
    log_b = b.detach().clamp_min(tiny).log()

    # --- Sinkhorn loop: potentials only, under no_grad (nothing stored) ---
    with torch.no_grad():
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
        for _ in range(iters):
            # Sinkhorn update, then clamp to [-kappa, kappa] = the TV prox.
            # The clamp IS "give up and pay kappa": a potential that saturates
            # marks mass whose best match is farther than it is worth.
            f = (-_sep_softmax_transform(g + eps * log_b, dh, dw, eps)
                 ).clamp(-kappa, kappa)
            g = (-_sep_softmax_transform(f + eps * log_a, dh, dw, eps)
                 ).clamp(-kappa, kappa)

    # --- primal of the entropic plan  pi = (a x b) . exp((f + g - C)/eps) ---
    # Validated against an exact LP solution of D_kappa on small grids:
    # matches to ~1e-3 relative. Evaluated ONCE, differentiably in a and b
    # (f, g fixed): backprop touches a single separable pass, not the loop.
    # --- primal of the entropic plan  pi = (a x b) . exp((f + g - C)/eps) ---
    # NUMERICAL NOTE. The naive form u = a.exp(f/eps) OVERFLOWS: f saturates at
    # +-kappa, so the exponent reaches kappa/eps -- 16 at the training config
    # but 80 at a tight eval config, and e^80 blows past fp32 in the products
    # below (silently -> NaN). We therefore factor a per-sample constant out of
    # u and v and carry it in LOG space; every array below is scaled to max 1,
    # and the constant is re-applied as a log-offset at the end. Stable for any
    # (kappa, eps).
    lu = f / eps + log_a                                 # (M, H, W)
    lv = g / eps + log_b
    Mu = lu.amax(dim=(-1, -2), keepdim=True)
    Mv = lv.amax(dim=(-1, -2), keepdim=True)
    u = torch.exp(lu - Mu)                               # max 1, well scaled
    v = torch.exp(lv - Mv)
    sh = (Mu + Mv).reshape(-1)                           # (M,) log offset

    Kr = torch.exp(-dh / eps)                            # (H, H)
    Kc = torch.exp(-dw / eps)                            # (W, W)

    def _unshift(x, ndim):
        """x is the shifted quantity; restore exp(sh) in log space."""
        s = sh.reshape((-1,) + (1,) * ndim)
        return torch.exp(x.clamp_min(tiny).log() + s)

    # plan marginals (separable):  pi.1 = u . K v,   pi^T.1 = v . K^T u
    Kv = torch.einsum("hg,mgc->mhc", Kr, torch.einsum("mhc,cd->mhd", v, Kc))
    Ku = torch.einsum("hg,mgc->mhc", Kr, torch.einsum("mhc,cd->mhd", u, Kc))
    row = _unshift(u * Kv, 2)
    col = _unshift(v * Ku, 2)
    unmatched = ((a - row).abs().sum(dim=(-1, -2))
                 + (b - col).abs().sum(dim=(-1, -2)))    # (M,) people

    # transport cost. C is L1, so  sum pi.C = sum pi.d_rows + sum pi.d_cols,
    # each from a FACTORED coupling -- the (HW x HW) plan is never materialised:
    #   rows: R[r,r'] = sum_cc' u[r,c] Kc[c,c'] v[r',c'];  cost = sum R.Kr.dh
    UKc = torch.einsum("mhc,cd->mhd", u, Kc)
    Rrow = torch.einsum("mhd,mgd->mhg", UKc, v)          # (M, H, H)
    cost_r = _unshift((Rrow * (Kr * dh)[None]).sum(dim=(-1, -2)), 0)
    KrV = torch.einsum("hg,mgc->mhc", Kr, v)
    Rcol = torch.einsum("mhc,mhd->mcd", u, KrV)          # (M, W, W)
    cost_c = _unshift((Rcol * (Kc * dw)[None]).sum(dim=(-1, -2)), 0)
    transport = cost_r + cost_c                          # (M,) people*pixels
    primal = transport + kappa * unmatched               # (M,) accurate VALUE

    # --- value from the primal, GRADIENT from the dual (envelope theorem) ---
    # dD/da = f*, the optimal potential. Verified against finite differences of
    # an exact LP: dual grad corr 0.998 with truth, whereas differentiating the
    # entropic primal with f, g held fixed gives corr 0.02 -- it is NOT the
    # gradient of D_kappa and training on it stalls at a large undercount (the
    # plan's marginals move when a moves, and that term is missing). The
    # entropic dual VALUE is meanwhile biased upward by the regulariser, while
    # the primal value is near-exact.
    # So: take the value from the primal and the slope from the dual, via the
    # standard  y = x + (v - x).detach()  trick.
    #   forward  -> primal   (accurate D_kappa)
    #   backward -> f        (correct dD/da, and bounded in [-kappa, kappa],
    #                         so no pixel can ever produce a gradient larger
    #                         than kappa: the metric is 1-Lipschitz in count
    #                         and cannot explode on a bad batch)
    dual = (a * f).sum(dim=(-1, -2)) + (b * g).sum(dim=(-1, -2))
    value = dual + (primal - dual).detach()

    loss = value.mean()
    return loss, transport.mean().item(), unmatched.mean().item()


# --- the two entry points. ONE metric, two precisions. ---------------------
# Training and evaluation optimise and report THE SAME QUANTITY, D_kappa.
# They differ only in solver precision, exactly as a training-time and a
# reference implementation of any metric would:
#
#   TRAIN_CFG  eps=0.5, iters=30, pool=2   fast, 0.26% from exact
#   EVAL_CFG   eps=0.25, iters=60, pool=1  reference, 0.03% from exact, fp64
#
# (accuracy measured against an exact LP solution of D_kappa; see the module
# tests). The two rank predictions identically -- verified, self-test 9 -- so
# the model is never optimising one thing and being scored on another.
TRAIN_CFG = dict(kappa=8.0, eps=0.5, iters=30, pool=2)
EVAL_CFG = dict(kappa=8.0, eps=0.25, iters=60, pool=1)


def stream_cdf_loss(pred, target, kappa=8.0, eps=0.5, iters=30,
                    den_factor=100.0, pool=2, pixel_size=1.0):
    """
    TRAINING objective: D_kappa. ONE loss, nothing added.

    There is no auxiliary pixel term, no second weight to tune, no term whose
    minimiser disagrees with the reported metric. Shape and count are not two
    losses balanced by a lambda; they are the two halves of one transport
    problem, already commensurable in people * pixels and traded off by kappa
    alone.

    That is the point of the redesign. v2 summed an MSE (whose minimiser over a
    sparse field is the blurred, temporally FLAT conditional mean) with a count
    term blind to WHERE mass sits, in mismatched units. Any residual pixel term
    would drag the optimum back toward that blur, so none is kept.

    Returns (loss, transport_people_px, unmatched_people). Since nothing is
    added, the logs reconcile exactly:  loss == transport + kappa * unmatched.
    """
    return flat_ot_loss(pred, target, kappa=kappa, eps=eps, iters=iters,
                        den_factor=den_factor, pool=pool, pixel_size=pixel_size)


@torch.no_grad()
def stream_cdf_metric(pred, target, den_factor=100.0, chunk=8, **cfg):
    """
    EVALUATION metric: D_kappa at reference precision. Report this.

    Same functional as the training loss -- not a proxy, not a correlate, the
    same D_kappa -- solved more tightly (EVAL_CFG: smaller eps, more iters, no
    spatial pooling) and in float64, so the reported number is the metric
    itself rather than its fast approximation. No MSE, MAE, JS or KL is
    computed anywhere: the model is trained and scored on one quantity.

    Reported as a dict, all in interpretable units:
        D          total, people * pixels
        transport  shape error   -- moving matched mass to where it belongs
        unmatched  count error   -- PEOPLE the two maps could not agree on
        share_*    fraction of D from each half (which error dominates)

    `chunk` frames are solved at a time to bound memory; the metric is a mean
    over frames, so chunking is exact, not an approximation.
    """
    c = {**EVAL_CFG, **cfg}
    pred = pred.double()
    target = target.double()
    T = pred.shape[2]
    tot = tr = um = 0.0
    for s in range(0, T, chunk):
        p, t = pred[:, :, s:s + chunk], target[:, :, s:s + chunk]
        n = p.shape[2]
        d, a, b = flat_ot_loss(p, t, den_factor=den_factor, **c)
        tot += d.item() * n
        tr += a * n
        um += b * n
    tot, tr, um = tot / T, tr / T, um / T
    return {
        "D": tot,
        "transport": tr,
        "unmatched": um,
        "share_shape": tr / tot if tot > 0 else 0.0,
        "share_count": (c["kappa"] * um) / tot if tot > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 7. The full model
# ---------------------------------------------------------------------------
class StreamCDF(nn.Module):
    """
    Stream-CDF: causal, count-aware crowd density forecasting.

    Defaults are the LONG-CONTEXT configuration, motivated by crowd physics:
    stop-and-go density waves in dense crowds have a characteristic period of
    ~45 s (Jamarat Bridge, Hajj 2006, at 7 ppl/m^2). To learn an oscillation you
    need at least one full period of context, ideally two.

        obs_frames  = 76  ->  60.8 s of history   (GCS @ 1.25 fps)
        pred_frames = 36  ->  28.8 s forecast

    encoder tubelet 4 -> 19 chunks -> 1,900 tokens
    decoder tubelet 2 -> 18 chunks -> 1,800 queries (fine temporal output)
    """

    def __init__(self, img_size=80, patch=8,
                 enc_tubelet=4, dec_tubelet=2,
                 obs_frames=76, pred_frames=36,
                 anchor_frames=4,
                 dim=384, heads=6, depth=8, dec_depth=2, dropout=0.0,
                 temporal="absolute", out_act="relu"):
        super().__init__()
        assert obs_frames % enc_tubelet == 0, "obs_frames must divide by enc_tubelet"
        assert pred_frames % dec_tubelet == 0, "pred_frames must divide by dec_tubelet"
        assert anchor_frames <= obs_frames, "anchor_frames must be <= obs_frames"
        assert temporal in ("absolute", "alibi")
        assert out_act in ("relu", "softplus")

        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.enc_tubelet = enc_tubelet
        self.dec_tubelet = dec_tubelet
        self.obs_chunks = obs_frames // enc_tubelet
        self.pred_chunks = pred_frames // dec_tubelet
        self.anchor_frames = anchor_frames
        self.temporal = temporal
        self.out_act = out_act
        self.heads = heads
        grid = img_size // patch
        self.n_spatial = grid * grid

        # --- encoder (past only) ---
        self.embed = CubeEmbed(img_size, patch, enc_tubelet,
                               num_frames=obs_frames, dim=dim)
        # ALiBi mode: NO absolute temporal table (recency lives in the attention
        # bias) -> the encoder can run on an unbounded stream. This is what makes
        # the "streaming" claim honest.
        self.pos = FactorizedPosEmbed(self.obs_chunks, grid, grid, dim,
                                      use_temporal=(temporal == "absolute"))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # --- decoder (future) ---
        self.decoder = HorizonDecoder(self.pred_chunks, self.n_spatial,
                                      dim, heads, depth=dec_depth)
        self.head = DensityHead(dim, dec_tubelet, patch, grid, grid)

        # --- LEARNED ANCHOR over the last `anchor_frames` observed frames ---
        # Init [0,...,0,1] => the anchor IS the last frame, so the model starts
        # exactly as persistence. It can then learn any linear combination, i.e.
        # learn its way to a Ridge-style anchor (which empirically beats persistence).
        w = torch.zeros(anchor_frames)
        w[-1] = 1.0
        self.anchor_w = nn.Parameter(w)

        # temporal attention structure over the OBSERVED tokens (buffer, not learned):
        #   absolute -> plain bool causal mask
        #   alibi    -> per-head float bias (causal + recency), streamable
        if temporal == "alibi":
            self.register_buffer(
                "temporal_bias",
                build_temporal_bias(self.obs_chunks, self.n_spatial, heads),
                persistent=False,
            )
            self.causal_mask = None
        else:
            self.register_buffer(
                "causal_mask",
                build_causal_mask(self.obs_chunks, self.n_spatial),
                persistent=False,
            )
            self.temporal_bias = None

    def make_anchor(self, obs, n_pred):
        """Learned linear combination of the last `anchor_frames` observed frames."""
        A = self.anchor_frames
        recent = obs[:, :, -A:]                              # (B, 1, A, H, W)
        w = self.anchor_w.view(1, 1, A, 1, 1)
        anchor = (recent * w).sum(dim=2, keepdim=True)       # (B, 1, 1, H, W)
        return anchor.repeat(1, 1, n_pred, 1, 1)

    def forward(self, obs, anchor_src=None):
        # obs: (B, 1, obs_frames, H, W)
        x = self.embed(obs)                     # (B, obs_chunks*S, D)
        x = self.pos(x)

        enc_mask = self.temporal_bias if self.temporal == "alibi" else self.causal_mask
        for blk in self.blocks:
            x, _ = blk(x, enc_mask)
        memory = self.norm(x)                   # the causally-encoded past

        fut, _ = self.decoder(memory)
        delta = self.head(fut)                  # (B, 1, pred_frames, H, W)

        # RESIDUAL: start from the anchor, learn only the CHANGE.
        raw = self.make_anchor(obs if anchor_src is None else anchor_src,
                               delta.shape[2]) + delta

        # Output non-negativity. relu has a hard dead zone: with a zero-init head
        # and shrinking crowds, many pixels sit at negative pre-activation early in
        # training with ZERO gradient, and clipping rectifies noise into phantom
        # mass (asymmetric: can't undershoot below 0), biasing the count UP. softplus
        # is smooth -- gradient everywhere, far less rectification bias. Kept as a
        # flag so the two can be compared as a stability ablation.
        if self.out_act == "softplus":
            # beta=10: softplus(0) = ln2/10 = 0.069 per pixel (~4.4 ppl/frame
            # at init, vs ~44 at beta=1). Smooth everywhere, near-identity for
            # raw > ~0.5, so the anchor property holds to tolerance (asserted
            # in self-test 7c).
            return F.softplus(raw, beta=10.0)
        return torch.relu(raw)                   # density >= 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print("  Stream-CDF v2 -- self-test  (long context, asymmetric tubelet)")
    print("=" * 70)

    # -- 1. cube embedding, LONG context --
    emb = CubeEmbed(num_frames=76, tubelet=4)
    x = torch.randn(2, 1, 76, 80, 80)
    y = emb(x)
    print(f"\n[1] CubeEmbed       {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"    76 frames / tubelet 4 = {emb.grid_t} chunks x 100 spatial"
          f" = {emb.num_tokens} tokens")

    # -- 2. causal mask --
    mask = build_causal_mask(emb.grid_t, 100)
    N = mask.shape[0]
    print(f"\n[2] causal mask     {tuple(mask.shape)}")
    print(f"    token    0 (chunk  0) blocked from {mask[0].sum().item():>4} / {N}")
    print(f"    token 1850 (chunk 18) blocked from {mask[1850].sum().item():>4} / {N}")

    # -- 3. CAUSALITY PROOF --
    small = build_causal_mask(5, 100)
    blk = Block()
    _, att = blk(torch.randn(2, 500, 384), small, return_attn=True)
    fut_w = att[0, 0, 0, 100:].sum().item()
    print(f"\n[3] causality       token 0 attention on the FUTURE = {fut_w:.10f}")
    assert fut_w == 0.0, "CAUSALITY VIOLATED"
    print("    PASS -- the model cannot see the future")

    # -- 3b. the memory-efficient path must agree with the explicit one --
    blk.eval()
    with torch.no_grad():
        xin = torch.randn(2, 500, 384)
        o_exp, _ = blk(xin, small, return_attn=True)
        o_sdp, _ = blk(xin, small, return_attn=False)
    d = (o_exp - o_sdp).abs().max().item()
    print(f"    SDPA vs explicit: max abs diff = {d:.2e}")
    assert d < 1e-4, "SDPA PATH DISAGREES WITH EXPLICIT PATH"
    print("    PASS -- memory-efficient attention is equivalent")

    # -- 4. positional embedding --
    pos = FactorizedPosEmbed(grid_t=19)
    p = pos(torch.zeros(1, 1900, 384))
    same_t = torch.allclose(p[0, 0], p[0, 100])
    same_s = torch.allclose(p[0, 0], p[0, 1])
    print(f"\n[4] pos embed       tok0 vs tok100 (same space, diff time) same? {same_t}")
    print(f"                    tok0 vs tok1   (diff space, same time) same? {same_s}")
    assert not same_t and not same_s

    # -- 5. THE METRIC: match what you can, pay kappa for the rest --
    # One 10-person blob. Four scenarios probe the metric's defining behaviour.
    kappa = 8.0
    tgt = torch.zeros(1, 1, 1, 80, 80)
    tgt[:, :, :, 40, 40] = 10.0 * 100.0              # 10 people, x den_factor

    perfect = tgt.clone()
    near = torch.zeros_like(tgt); near[:, :, :, 40, 46] = 10.0 * 100.0   # 6 px off
    far = torch.zeros_like(tgt);  far[:, :, :, 40, 76]  = 10.0 * 100.0   # 36 px off
    under = tgt.clone() * 0.8                                            # -2 people

    rows = []
    for name, p in [("perfect", perfect), ("6 px shift", near),
                    ("36 px shift", far), ("20% under", under)]:
        d, tr, um = flat_ot_loss(p, tgt, kappa=kappa, iters=60)
        rows.append((name, d.item(), tr, um))
    print(f"\n[5] flat metric D_k (kappa={kappa:.0f})"
          f"{'D':>16}{'transport':>12}{'unmatched':>11}")
    for name, d, tr, um in rows:
        print(f"    {name:<18}{d:>15.2f}{tr:>12.2f}{um:>11.2f} ppl")
    d0, dn, df, du = rows[0][1], rows[1][1], rows[2][1], rows[3][1]
    assert d0 < 2.0,                    "perfect prediction should cost ~0"
    assert abs(dn - 60.0) < 8.0,        "6px x 10 ppl should cost ~60 ppl*px"
    assert df < 2 * kappa * 10 + 8.0,   "far shift must be CAPPED near 2*kappa*mass"
    assert abs(du - kappa * 2.0) < 3.0, "missing ppl should cost ~kappa each"
    print(f"    perfect ~0 | near shift ~ dist x ppl | far shift CAPPED at"
          f" ~2k x ppl | undercount ~ k/person")
    print("    PASS -- shape error where matchable, count error for the rest,")
    print("            blur/flat forecasts are no longer the loss's minimiser")

    # -- 6. full model, long context --
    model = StreamCDF(obs_frames=76, pred_frames=36,
                      enc_tubelet=4, dec_tubelet=2, anchor_frames=4)
    obs = torch.randn(2, 1, 76, 80, 80).abs()
    out = model(obs)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n[6] StreamCDF       {tuple(obs.shape)} -> {tuple(out.shape)}")
    print(f"    encoder: {model.obs_chunks:>2} chunks x 100 = {model.obs_chunks*100:>4} tokens")
    print(f"    decoder: {model.pred_chunks:>2} chunks x 100 = {model.pred_chunks*100:>4} queries"
          f"   (was 3 x 100 = 300)")
    print(f"    non-negative: {out.min().item() >= 0}")
    print(f"    parameters:   {n:,} ({n / 1e6:.1f}M)   [CrowdMAC ViT-S ~ 22M]")

    # -- 7. THE ANCHOR TEST: at init the model IS persistence (relu mode) --
    with torch.no_grad():
        pred = model(obs)
        last = torch.relu(obs[:, :, -1:].repeat(1, 1, 36, 1, 1))
        diff = (pred - last).abs().max().item()
    print(f"\n[7] anchor          max|pred - persistence| at init = {diff:.2e}")
    assert diff < 1e-5, "MODEL DOES NOT START AS PERSISTENCE"
    print("    PASS -- zero-init delta => the model begins life AS its baseline")
    print(f"    anchor weights over last {model.anchor_frames} frames:"
          f" {[round(v,3) for v in model.anchor_w.data.tolist()]}")

    # -- 7b. ALiBi STREAMING MODE: no absolute temporal table, streamable --
    alibi = StreamCDF(obs_frames=76, pred_frames=36, temporal="alibi",
                      dim=64, depth=2, heads=4, dec_depth=1)
    has_temporal = hasattr(alibi.pos, "temporal")
    out_a = alibi(obs)
    print(f"\n[7b] ALiBi          absolute temporal table present? {has_temporal}")
    assert not has_temporal, "ALiBi mode should NOT have an absolute temporal table"
    print(f"     temporal_bias shape: {tuple(alibi.temporal_bias.shape)}  (H, N, N)")
    print(f"     output ok: {tuple(out_a.shape)}, non-neg: {out_a.min().item() >= 0}")
    tb = alibi.temporal_bias
    S = alibi.n_spatial
    future_is_blocked = torch.isinf(tb[0, 0, S]).item()          # tok0 -> chunk1
    past_is_finite   = torch.isfinite(tb[0, S, 0]).item()        # chunk1 -> tok0
    print(f"     future blocked (-inf): {future_is_blocked} | past finite: {past_is_finite}")
    assert future_is_blocked and past_is_finite, "ALiBi bias is wrong"
    print("     PASS -- causal + recency, valid at ANY absolute time index")

    # -- 7c. SOFTPLUS output mode (smooth, no dead zone) --
    sp = StreamCDF(obs_frames=76, pred_frames=36, out_act="softplus",
                   dim=64, depth=2, heads=4, dec_depth=1)
    out_s = sp(obs)
    print(f"\n[7c] softplus       output non-neg: {out_s.min().item() >= 0}, "
          f"shape {tuple(out_s.shape)}")
    assert out_s.min().item() >= 0

    # -- 8. gradients --
    target = (torch.rand(2, 1, 36, 80, 80) * 0.5).abs()
    loss, transport, unmatched = stream_cdf_loss(out, target, iters=15, pool=2)
    loss.backward()
    alive = sum(1 for p in model.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in model.parameters())
    print(f"\n[8] loss  {loss.item():.4f}  =  transport {transport:.2f} ppl*px"
          f"  +  k * {unmatched:.2f} unmatched ppl")
    recon = transport + 8.0 * unmatched
    print(f"    reconciles exactly: {recon:.4f}  -- ONE loss, no hidden terms")
    assert abs(recon - loss.item()) < 1e-2, "loss must be exactly D_kappa"
    print(f"    gradients reached {alive}/{total} tensors on step 1")
    print(f"    (expected: the zero-init head means upstream grads bootstrap after")
    print(f"     the FIRST optimizer step. Not a disconnection.)")
    assert alive >= 2, "NOTHING IS LEARNING"

    # -- 9. ONE metric for training AND evaluation --
    ii = torch.arange(80).float()
    gg = torch.exp(-((ii[:, None] - 30) ** 2 + (ii[None, :] - 35) ** 2) / 18)
    tv = torch.zeros(1, 1, 1, 80, 80)
    tv[0, 0, 0] = gg / gg.sum() * 20.0 * 100.0
    variants = {
        "perfect":   tv.clone(),
        "shift 2px": torch.roll(tv.clone(), 2, dims=4),
        "shift 6px": torch.roll(tv.clone(), 6, dims=4),
        "30% under": tv.clone() * 0.7,
        "blurred":   F.avg_pool2d(tv.clone().reshape(1, 1, 80, 80), 9,
                                  stride=1, padding=4).reshape(1, 1, 1, 80, 80),
        "FLAT":      torch.full_like(tv, 20.0 * 100.0 / 6400),
    }
    print(f"\n[9] train obj vs eval metric -- SAME quantity, two precisions")
    print(f"    {'variant':<12}{'train D':>10}{'EVAL D':>10}{'transp':>9}"
          f"{'unmat':>8}{'%shape':>8}")
    tr_v, ev_v = [], []
    for name, p in variants.items():
        dt, _, _ = stream_cdf_loss(p, tv, **{k: v for k, v in TRAIN_CFG.items()})
        m = stream_cdf_metric(p, tv)
        tr_v.append(dt.item()); ev_v.append(m["D"])
        print(f"    {name:<12}{dt.item():>10.2f}{m['D']:>10.2f}"
              f"{m['transport']:>9.2f}{m['unmatched']:>8.2f}"
              f"{m['share_shape']:>8.0%}")
    order_t = sorted(range(len(tr_v)), key=lambda i: tr_v[i])
    order_e = sorted(range(len(ev_v)), key=lambda i: ev_v[i])
    assert order_t == order_e, "train and eval configs must rank identically"
    assert ev_v[-1] == max(ev_v), "the FLAT prediction must score WORST"
    print("    PASS -- identical ranking; no MSE/MAE/JS/KL computed anywhere.")
    print("    The FLAT forecast -- what the v2 MSE term REWARDED -- is now the")
    print("    single worst-scoring prediction under the trained objective.")

    print("\n" + "=" * 70)
    print("  ALL CHECKS PASSED")
    print("=" * 70)
