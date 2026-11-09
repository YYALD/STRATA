"""Critics of latent sequences (ConditionalDiscriminator) and of single decays (SequencePatchDiscriminator)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiScaleDiscBlock(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int = 64) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=5, padding=2), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3), nn.LeakyReLU(0.2))])
        self.res = nn.Conv1d(in_ch, branch_ch * 3, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.out_ch = branch_ch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.cat([b(x) for b in self.branches], dim=1) + self.res(x))


class ConditionalDiscriminator(nn.Module):
    """Latent-space critic with label map and minibatch-stddev channel."""

    def __init__(self, hidden_dim: int = 64, cond_dim: int = 64, seq_len: int = 256, branch_ch: int = 64) -> None:
        super().__init__()
        self.label_proj = nn.Linear(cond_dim, seq_len)
        self.in_proj = nn.Conv1d(hidden_dim + 1, branch_ch, kernel_size=1)
        self.block1 = MultiScaleDiscBlock(branch_ch, branch_ch=branch_ch)
        self.block2 = MultiScaleDiscBlock(self.block1.out_ch, branch_ch=branch_ch)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.block2.out_ch + 1, 128),
                                  nn.LeakyReLU(0.2), nn.Linear(128, 1))

    def forward(self, h: torch.Tensor, c: torch.Tensor, return_features: bool = False):
        x = torch.cat([h.transpose(1, 2), self.label_proj(c).unsqueeze(1)], dim=1)
        x = self.block2(self.block1(self.in_proj(x)))
        feat = x.mean(dim=2)
        if x.size(0) > 1:
            mb = x.float().std(dim=0, unbiased=False).mean().to(x.dtype).expand(x.size(0), 1, x.size(2))
        else:
            mb = x.new_zeros(x.size(0), 1, x.size(2))
        score = self.head(torch.cat([x, mb], dim=1))
        return (score, feat) if return_features else score


class SequencePatchDiscriminator(nn.Module):
    """Signal-space critic with projection on (condition, alpha, amplitude statistics)."""

    def __init__(self, seq_len: int = 256, cond_dim: int = 64, branch_ch: int = 64, projection: bool = True) -> None:
        super().__init__()
        self.projection = projection
        extra = 4
        self.label_proj = nn.Linear(cond_dim + 1 + extra, seq_len)
        self.in_proj = nn.Conv1d(2, branch_ch, kernel_size=1)
        self.block1 = MultiScaleDiscBlock(branch_ch, branch_ch=branch_ch)
        self.block2 = MultiScaleDiscBlock(self.block1.out_ch, branch_ch=branch_ch)
        self.body = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.block2.out_ch, 128),
                                  nn.LeakyReLU(0.2))
        self.out = nn.Linear(128, 1)
        self.proj = nn.Linear(cond_dim + 1 + extra, 128)

    @staticmethod
    def _amplitude_stats(x: torch.Tensor) -> torch.Tensor:
        s = x[..., 0]
        k1 = min(1, s.shape[1] - 1)
        return torch.stack([s[:, 0].abs(), s[:, :min(3, s.shape[1])].abs().max(1).values,
                            s[:, :min(8, s.shape[1])].abs().max(1).values, s[:, 0] / (s[:, k1].abs() + 1e-6)], dim=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, return_features: bool = False):
        cond = torch.cat([c, alpha, self._amplitude_stats(x)], dim=1)
        h = torch.cat([x.transpose(1, 2), self.label_proj(cond).unsqueeze(1)], dim=1)
        feat = self.body(self.block2(self.block1(self.in_proj(h))))
        score = self.out(feat)
        if self.projection:
            score = score + (feat * self.proj(cond)).sum(dim=1, keepdim=True) / math.sqrt(max(feat.shape[1], 1))
        return (score, feat) if return_features else score
