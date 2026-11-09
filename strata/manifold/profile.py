"""Profile-manifold-constrained feature extraction (along the hole): the depth encoding of the stations, the
random-Fourier latent field along the hole (depth_latent_field), the learned station context (ProfileDepthContext)
and the critic of profile windows (ProfilePatchDiscriminator).  The regional head that reduces the network's change
to an order-preserving warp, a level and a contrast of the background is RFMGGenTrainer._warp_background
(trainer.py)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..models.layers import _rnn_call


DEPTH_FEAT_DIM = 10


def depth_features(depth_m: torch.Tensor, d0: float, d1: float, spacing_med: float) -> torch.Tensor:
    """Per-station depth encoding: position u in the measured span, 4 Fourier pairs and the local
    log spacing ratio (the measured spacing pattern, e.g. 10 m above 320 m and 5 m below)."""
    u = ((depth_m - d0) / max(d1 - d0, 1e-9)).clamp(-0.5, 1.5)
    feats = [u]
    for k in range(1, 5):
        feats += [torch.sin(math.pi * k * u), torch.cos(math.pi * k * u)]
    if depth_m.size(1) > 1:
        dd = torch.diff(depth_m, dim=1)
        dd = torch.cat([dd[:, :1], 0.5 * (dd[:, 1:] + dd[:, :-1]), dd[:, -1:]], dim=1) if depth_m.size(1) > 2 else \
            torch.cat([dd, dd], dim=1)
    else:
        dd = torch.full_like(depth_m, spacing_med)
    feats.append(torch.log(dd.clamp_min(1e-6) / max(spacing_med, 1e-6)))
    return torch.stack(feats, dim=-1)


class ProfileDepthContext(nn.Module):
    """Learned depth context: a bidirectional GRU runs along the stations of a profile window over
    (anchor condition, depth encoding, alpha, profile code) and returns a per-station latent offset
    for the generator's z pathway.  Zero-initialised output = identity at the start of training."""

    def __init__(self, cond_dim: int, z_dim: int, code_dim: int = 16, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.inp = nn.Sequential(nn.Linear(cond_dim + DEPTH_FEAT_DIM + 1 + code_dim, 2 * hidden), nn.GELU(),
                                 nn.Linear(2 * hidden, 2 * hidden), nn.GELU())
        self.gru = nn.GRU(2 * hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, z_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c_seq: torch.Tensor, dfeat: torch.Tensor, alpha_seq: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        W = c_seq.size(1)
        x = torch.cat([c_seq, dfeat.to(c_seq.dtype), alpha_seq.to(c_seq.dtype),
                       code.unsqueeze(1).expand(-1, W, -1).to(c_seq.dtype)], dim=-1)
        h, _ = _rnn_call(self.gru, self.inp(x))
        return self.out(h)


class ProfilePatchDiscriminator(nn.Module):
    """Profile critic on windows of consecutive stations.  Input channels: normalised log amplitude,
    its first and second depth differences (spacing-normalised) and the depth position.  A 2-D conv
    stem compresses the gate axis, a dilated 1-D stack along depth gives a ~13-station receptive
    field, a minibatch-stddev channel exposes collapse, and per-station patch logits are averaged."""

    def __init__(self, seq_len: int, ch: int = 48) -> None:
        super().__init__()
        c2 = 2 * ch
        self.stem = nn.Sequential(
            nn.Conv2d(4, ch, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, c2, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)), nn.LeakyReLU(0.2),
            nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)), nn.LeakyReLU(0.2))
        t_out = seq_len
        for _ in range(4):
            t_out = (t_out + 1) // 2
        self.flat = nn.Conv1d(c2 * t_out, 128, kernel_size=1)
        self.depth = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
                                   nn.LeakyReLU(0.2))
        self.head = nn.Conv1d(129, 1, kernel_size=1)

    @staticmethod
    def make_input(y: torch.Tensor, depth_m: torch.Tensor, d0: float, d1: float, spacing_med: float) -> torch.Tensor:
        """y: (B, W, T) normalised; depth_m: (B, W) metres -> (B, 4, W, T)."""
        B, W, T = y.shape
        if W > 1:
            dd = torch.diff(depth_m, dim=1).clamp_min(1e-6) / max(spacing_med, 1e-6)
            g = torch.diff(y, dim=1) / dd.unsqueeze(-1)
            g = torch.cat([g, g[:, -1:]], dim=1)
            g2 = torch.diff(g, dim=1)
            g2 = torch.cat([g2[:, :1], g2], dim=1)
        else:
            g = torch.zeros_like(y)
            g2 = torch.zeros_like(y)
        u = ((depth_m - d0) / max(d1 - d0, 1e-9)).unsqueeze(-1).expand(-1, -1, T)
        return torch.stack([y, 10.0 * g, 30.0 * g2, u.to(y.dtype)], dim=1)

    def forward(self, x4: torch.Tensor, return_features: bool = False):
        h = self.stem(x4)                                # (B, c2, W, T')
        B, C, W, Tp = h.shape
        h = h.permute(0, 1, 3, 2).reshape(B, C * Tp, W)
        h = self.depth(self.flat(h))                    # (B, 128, W)
        feat = h.mean(dim=2)
        if B > 1:
            mb = h.float().std(dim=0, unbiased=False).mean().to(h.dtype).expand(B, 1, W)
        else:
            mb = h.new_zeros(B, 1, W)
        logits = self.head(torch.cat([h, mb], dim=1)).mean(dim=2)
        return (logits, feat) if return_features else logits


def fourier_params_t(B: int, n_channels: int, bands: Sequence[Tuple[float, float]], probs: Sequence[float], K: int,
                     gen: Optional[torch.Generator], device: torch.device) -> Dict[str, torch.Tensor]:
    """Random Fourier parameters of a multi-scale depth field (wavelengths in metres, phases)."""
    pr = torch.tensor(list(probs), dtype=torch.float32, device=device)
    band = torch.multinomial(pr / pr.sum(), B * n_channels, replacement=True, generator=gen).view(B, 1, n_channels)
    lo = torch.tensor([b[0] for b in bands], dtype=torch.float32, device=device)[band]
    hi = torch.tensor([b[1] for b in bands], dtype=torch.float32, device=device)[band]
    u = torch.rand(B, K, n_channels, device=device, generator=gen)
    lam = torch.exp(torch.log(lo) + u * (torch.log(hi) - torch.log(lo)))
    phase = torch.rand(B, K, n_channels, device=device, generator=gen) * (2.0 * math.pi)
    return {"lam": lam, "phase": phase}


def fourier_field(depth_m: torch.Tensor, fp: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate a random-Fourier depth field at physical depths (B, W) -> (B, W, C), unit variance."""
    lam, phase = fp["lam"], fp["phase"]
    K = lam.shape[1]
    arg = (2.0 * math.pi) * depth_m.float()[:, :, None, None] / lam[:, None] + phase[:, None]
    return torch.cos(arg).sum(dim=2) * math.sqrt(2.0 / K)


def depth_latent_field(depth_m: torch.Tensor, n_channels: int, bands: Sequence[Tuple[float, float]],
                       probs: Sequence[float], K: int = 8, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Multi-scale depth-correlated Gaussian field, unit variance per channel: each channel draws a
    depth-scale band (long / mid / short wavelengths in metres, anchored to the measured span and
    spacing) and K random Fourier components (log-uniform wavelength, uniform phase).  Evaluated on
    the physical depth of every station, so one profile has one coherent field."""
    return fourier_field(depth_m, fourier_params_t(depth_m.shape[0], n_channels, bands, probs, K, gen, depth_m.device))
