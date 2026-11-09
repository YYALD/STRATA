"""Plain DCGAN-1D generator and critic of the GAN / CGAN ablation baselines."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _AblBaselineGenerator(nn.Module):
    """Vanilla DCGAN-1D generator with resize-convolution and a fixed triangular low-pass."""

    def __init__(self, z_dim: int, seq_len: int, cond_dim: int = 0, base_ch: int = 128, conv_k: int = 5,
                 smooth_k: int = 7):
        super().__init__()
        self.z_dim, self.seq_len, self.cond_dim, self.base_ch = int(z_dim), int(seq_len), int(cond_dim), int(base_ch)
        self.l0 = max(4, self.seq_len // 16)
        self.proj = nn.Linear(self.z_dim + self.cond_dim, self.base_ch * self.l0)
        self.bn0 = nn.BatchNorm1d(self.base_ch)
        chans = [self.base_ch, 64, 32, 16, 8]
        pad = int(conv_k // 2)
        blocks: List[nn.Module] = []
        for i in range(len(chans) - 1):
            blocks += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                       nn.Conv1d(chans[i], chans[i + 1], kernel_size=int(conv_k), padding=pad),
                       nn.BatchNorm1d(chans[i + 1]), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv1d(chans[-1], 1, kernel_size=int(conv_k), padding=pad)
        w = np.bartlett(int(smooth_k) + 2)[1:-1].astype(np.float32)
        self.register_buffer("smooth_w", torch.from_numpy(w / max(float(w.sum()), 1e-8)).view(1, 1, -1))
        self.smooth_pad = int(smooth_k) // 2

    def forward(self, z: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cond_dim > 0 and c is not None:
            z = torch.cat([z, c], dim=1)
        h = torch.relu(self.bn0(self.proj(z).view(z.shape[0], self.base_ch, self.l0)))
        h = self.head(self.body(h))
        if h.shape[-1] != self.seq_len:
            h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        h = torch.sigmoid(h)
        h = F.conv1d(F.pad(h, (self.smooth_pad, self.smooth_pad), mode="replicate"), self.smooth_w)
        return h.transpose(1, 2)


class _AblBaselineDiscriminator(nn.Module):
    def __init__(self, seq_len: int, cond_dim: int = 0, base_ch: int = 64):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.cond_proj = nn.Linear(self.cond_dim, int(seq_len)) if self.cond_dim > 0 else None
        chans = [1 + (1 if self.cond_dim > 0 else 0), int(base_ch), int(base_ch) * 2, int(base_ch) * 4]
        blocks: List[nn.Module] = []
        for i in range(len(chans) - 1):
            blocks += [nn.Conv1d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(chans[-1], 1)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x.transpose(1, 2)
        if self.cond_proj is not None and c is not None:
            h = torch.cat([h, self.cond_proj(c).unsqueeze(1)], dim=1)
        return self.head(self.body(h).mean(dim=2))
