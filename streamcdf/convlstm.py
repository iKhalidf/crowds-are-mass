"""
ConvLSTM encoder-decoder over density maps, as a drop-in alternative to StreamCDF.

Following the density stream of Niu et al. and the precipitation-nowcasting
architecture of Shi et al. Niu's full model is two-stream and takes RGB frames in the
first stream; this pipeline works from density maps only, so what is reproduced is the
recurrent auto-encoder over maps. Niu's network is global-residual, so the residual
anchor here is faithful rather than borrowed -- and it is the *same* anchor StreamCDF
uses, which makes the comparison one about the temporal model rather than the output
stage.

Interface matches StreamCDF exactly:
    model(obs, anchor_src=None) -> (B, 1, pred_frames, H, W)
so the existing train.py, loss, evaluation and strata apply unchanged.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        return o * torch.tanh(c), c

    def init_state(self, b, hw, device, dtype):
        z = torch.zeros(b, self.hid_ch, *hw, device=device, dtype=dtype)
        return z, z.clone()


class ConvLSTMForecaster(nn.Module):
    """
    Spatial encoder (80 -> 20), ConvLSTM over the observed frames, then the same
    ConvLSTM rolled forward for the horizon, decoded back to 80x80 and added to a
    residual anchor.
    """

    def __init__(self, obs_frames=76, pred_frames=36, hid=192, layers=2,
                 anchor_frames=4, out_act="relu", obs_stride=1, **kw):
        super().__init__()
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.anchor_frames, self.obs_stride = anchor_frames, obs_stride

        self.down = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.cells = nn.ModuleList(
            [ConvLSTMCell(64 if i == 0 else hid, hid) for i in range(layers)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hid, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        # zero-init the output conv so the model starts as exact persistence
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

        self.anchor_w = nn.Parameter(torch.zeros(anchor_frames))
        with torch.no_grad():
            self.anchor_w[-1] = 1.0
        self.out_act = out_act

    def make_anchor(self, src):
        w = self.anchor_w.view(1, 1, -1, 1, 1)
        a = (src[:, :, -self.anchor_frames:] * w).sum(dim=2, keepdim=True)
        return a.repeat(1, 1, self.pred_frames, 1, 1)

    def forward(self, obs, anchor_src=None):
        # obs: (B, 1, T, H, W)
        src = obs if anchor_src is None else anchor_src
        anchor = self.make_anchor(src)

        x = obs[:, :, ::self.obs_stride]                      # optional subsample
        B, _, T, H, W = x.shape
        feats = self.down(x.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W))
        _, C, h, w = feats.shape
        feats = feats.reshape(B, T, C, h, w)

        states = [c.init_state(B, (h, w), obs.device, feats.dtype) for c in self.cells]
        for t in range(T):                                    # read the observations
            inp = feats[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(inp, states[i])
                inp = states[i][0]

        outs = []
        inp_zero = torch.zeros_like(feats[:, 0])
        for _ in range(self.pred_frames):                     # roll the horizon
            inp = inp_zero
            for i, cell in enumerate(self.cells):
                states[i] = cell(inp, states[i])
                inp = states[i][0]
            outs.append(inp)
        y = torch.stack(outs, dim=1)                          # (B, Hp, hid, h, w)
        Bp, Hp = y.shape[0], y.shape[1]
        delta = self.up(y.reshape(Bp * Hp, *y.shape[2:]))
        delta = delta.reshape(Bp, Hp, 1, H, W).permute(0, 2, 1, 3, 4)

        out = anchor + delta
        return torch.relu(out) if self.out_act == "relu" else nn.functional.softplus(out)


# Approximate parameter counts, 2 layers:
#   hid=96  -> 1.4M      hid=160 -> 3.3M
#   hid=192 -> 4.6M      hid=224 -> 6.1M   (StreamCDF is 5.9M)
# Default 192 sits just under StreamCDF; use 224 for capacity parity if a reviewer
# might object that the baseline was undersized.
#
# Runtime: 76 observed steps x 2 layers = 152 cell calls per forward at 20x20. If that
# is too slow, obs_stride=4 subsamples the observations to 19 steps, matching
# StreamCDF's encoder tubelet -- but say so in the paper if you use it.


if __name__ == "__main__":
    m = ConvLSTMForecaster()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 1, 76, 80, 80).abs()
    with torch.no_grad():
        y = m(x, anchor_src=x)
    print(f"params {n/1e6:.2f}M   in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"persistence at init: max |out - last frame| = "
          f"{(y - x[:, :, -1:].repeat(1,1,36,1,1)).abs().max().item():.2e}")
