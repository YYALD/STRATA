"""Generator in the latent space (BridgeResidualGenerator) with its fast FiLM decay memory, and the supervisor of the
latent dynamics."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from ..common import autocast
from ..config import _RFMG_ABLATION_FLAGS
from .layers import CrossAttention, FiLM, PositionalEncoding, _rnn_call


class Supervisor(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 1, module_name: str = "lstm") -> None:
        super().__init__()
        rnn_cls = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers=max(1, num_layers), batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y, _ = _rnn_call(self.rnn, h)
        return self.proj(y)


class FastFiLMDecayMemory(nn.Module):
    """Vectorised gate-level FiLM decay memory: [f,i,g,o] = Gate(x) + FiLM(c); m = f*P(x) + i*DWConv(g); y =
    o*tanh(m)."""

    def __init__(self, input_size: int = 256, hidden_size: int = 64, cond_dim: int = 64, kernel_size: int = 5):
        super().__init__()
        pad = int(kernel_size) // 2
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size * 4)
        self.c_gate = nn.Sequential(nn.Linear(cond_dim, max(64, hidden_size)), nn.GELU(),
                                    nn.Linear(max(64, hidden_size), hidden_size * 8))
        self.dw_memory = nn.Conv1d(hidden_size, hidden_size, kernel_size=int(kernel_size), padding=pad,
                                   groups=hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.c_gate[-1].weight)
        nn.init.zeros_(self.c_gate[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            xf = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            cf = torch.nan_to_num(cond.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            base = self.x_proj(xf)
            gates = self.x_gate(xf)
            gamma, beta = self.c_gate(cf).unsqueeze(1).chunk(2, dim=-1)
            if not _RFMG_ABLATION_FLAGS.get("no_film"):
                gates = gates * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * torch.tanh(beta)
            f_pre, i_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
            local = self.dw_memory(torch.tanh(g_pre).transpose(1, 2)).transpose(1, 2)
            mem = torch.sigmoid(f_pre) * base + torch.sigmoid(i_pre) * local
            y = torch.sigmoid(o_pre) * torch.tanh(self.norm(mem))
        return y.to(dtype=x.dtype)


def _v42_signal_amp_token_torch(x: torch.Tensor) -> torch.Tensor:
    """Six-dimensional amplitude / decay token of a normalised curve."""
    with autocast(enabled=False):
        s = (x[..., 0] if x.dim() == 3 else x).float()
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        T = s.shape[1]
        e = max(4, T // 8)
        m0 = max(e + 1, T // 3)
        m1 = max(m0 + 1, 2 * T // 3)
        l0 = max(1, int(0.67 * T))
        eps = 1.0e-6
        early, mid, late = s[:, :e].abs(), s[:, m0:m1].abs(), s[:, l0:].abs()
        early_peak = early.max(1).values + eps
        early_mean, mid_mean, late_mean = early.mean(1) + eps, mid.mean(1) + eps, late.mean(1) + eps
        peak_idx = torch.argmax(early, dim=1).float() / max(e - 1, 1)
        tok = torch.stack([torch.log1p(early_peak), torch.log1p(early_mean), torch.log1p(mid_mean),
                           torch.log1p(late_mean), torch.log(early_mean / mid_mean), torch.log(mid_mean / late_mean)], 1)
        tok[:, 5] = 0.5 * tok[:, 5] + 0.5 * peak_idx
        return torch.nan_to_num(tok, nan=0.0, posinf=5.0, neginf=-5.0)


class BridgeResidualGenerator(nn.Module):
    """Generator in the latent space: a shared mix of the anchor, measured and simulated conditions, alpha and the
    latent field z; bidirectional LSTMs with FiLM on the anchor condition, FiLM injection of z and cross-attention; a
    fast FiLM decay memory; a residual path on the anchor latent blended with a free path.  z carries the latent
    field along the hole and the learned depth context."""

    def __init__(self, seq_len: int = 256, cond_dim: int = 64, z_dim: int = 32, hidden_dim: int = 64,
                 dropout: float = 0.2, shared_dim: int = 512, static_dim: int = 512, z_proj_dim: int = 128,
                 temporal_hidden: int = 128, time_embed_dim: int = 32) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.shared = nn.Sequential(nn.Linear(cond_dim * 3 + 1 + z_dim, shared_dim), nn.LayerNorm(shared_dim),
                                    nn.LeakyReLU(0.2))
        self.static_branch = nn.Sequential(nn.Linear(shared_dim, static_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                                           nn.Linear(static_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2))
        self.z_seq_proj = nn.Linear(z_dim, z_proj_dim)
        self.pos = PositionalEncoding(z_proj_dim, max_len=max(seq_len, 512))
        self.time_embed = nn.Parameter(torch.randn(1, seq_len, time_embed_dim) * 0.02)
        self.alpha_proj = nn.Linear(1, time_embed_dim)
        self.temporal_in = nn.Linear(z_proj_dim + shared_dim + time_embed_dim, 256)
        self.temporal_rnn1 = nn.LSTM(256, temporal_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.film1 = FiLM(cond_dim, temporal_hidden * 2)
        self.temporal_rnn2 = nn.LSTM(temporal_hidden * 2, temporal_hidden, num_layers=1, batch_first=True,
                                     bidirectional=True)
        self.film2 = FiLM(cond_dim, temporal_hidden * 2)
        nh = 4
        while (temporal_hidden * 2) % nh != 0 and nh > 1:
            nh //= 2
        self.cross_attn1 = CrossAttention(temporal_hidden * 2, cond_dim, n_heads=nh)
        self.cross_attn2 = CrossAttention(temporal_hidden * 2, cond_dim, n_heads=nh)
        self.temp_drop = nn.Dropout(dropout)
        self.temp_proj = nn.Linear(temporal_hidden * 2, 256)
        self.delta_head = nn.Sequential(nn.Linear(256 + 256 + hidden_dim, 512), nn.LeakyReLU(0.2), nn.LayerNorm(512),
                                        nn.Dropout(dropout), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.LayerNorm(256),
                                        nn.Linear(256, hidden_dim))
        self.gate_head = nn.Sequential(nn.Linear(256 + 256 + hidden_dim + 1, 256), nn.LeakyReLU(0.2),
                                       nn.LayerNorm(256), nn.Linear(256, hidden_dim), nn.Sigmoid())
        self.gate_min = 0.15
        self.creative_head = nn.Sequential(nn.Linear(256 + 256, 256), nn.LeakyReLU(0.2), nn.LayerNorm(256),
                                           nn.Linear(256, hidden_dim), nn.Tanh())
        # still learnable.
        self.creative_alpha = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.z_film1 = FiLM(z_dim, temporal_hidden * 2)
        self.z_film2 = FiLM(z_dim, temporal_hidden * 2)
        for zm in (self.z_film1, self.z_film2):
            nn.init.zeros_(zm.mlp[-1].weight)
            nn.init.zeros_(zm.mlp[-1].bias)
        self.v42_amp_mlp = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, cond_dim), nn.LayerNorm(cond_dim))
        self.v42_amp_alpha = nn.Parameter(torch.tensor(-1.20))
        fast_hidden = max(32, min(64, hidden_dim))
        self.v43_fast_film_gate_branch = FastFiLMDecayMemory(256, fast_hidden, cond_dim, 5)
        self.v43_fast_film_gate_proj = nn.Linear(fast_hidden, 256)
        self.v43_film_gate_alpha = nn.Parameter(torch.tensor(-2.35))

    def condition_with_amp_token(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tok = _v42_signal_amp_token_torch(x).to(device=c.device, dtype=c.dtype)
        return c + torch.sigmoid(self.v42_amp_alpha).to(c.dtype) * self.v42_amp_mlp(tok)

    def forward(self, z_seq: torch.Tensor, c_anchor: torch.Tensor, c_real: torch.Tensor, c_sim: torch.Tensor,
                alpha: torch.Tensor, h_anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _RFMG_ABLATION_FLAGS.get("no_condition"):
            c_anchor, c_real, c_sim = torch.zeros_like(c_anchor), torch.zeros_like(c_real), torch.zeros_like(c_sim)
        T = z_seq.size(1)
        z_global = z_seq.mean(dim=1)
        shared = self.shared(torch.cat([c_anchor, c_real, c_sim, alpha, z_global], dim=1))
        static = self.static_branch(shared)
        zt = self.pos(self.z_seq_proj(z_seq))
        shared_t = shared.unsqueeze(1).expand(-1, T, -1)
        alpha_t = self.alpha_proj(alpha).unsqueeze(1).expand(-1, T, -1)
        t_in = self.temporal_in(torch.cat([zt, shared_t, self.time_embed[:, :T].expand(zt.size(0), -1, -1) + alpha_t], 2))
        t1, _ = _rnn_call(self.temporal_rnn1, t_in)
        t1 = self.z_film1(self.film1(t1, c_anchor), z_global)
        t1 = t1 + self.cross_attn1(t1, c_anchor)
        t2, _ = _rnn_call(self.temporal_rnn2, t1)
        t2 = self.z_film2(self.film2(t2, c_anchor), z_global)
        t2 = t2 + self.cross_attn2(t2, c_anchor)
        temporal = self.temp_proj(self.temp_drop(t2))
        gm = self.v43_fast_film_gate_branch(temporal, c_anchor)
        temporal = temporal + torch.sigmoid(self.v43_film_gate_alpha).to(temporal.dtype) * self.v43_fast_film_gate_proj(gm)
        static_t = static.unsqueeze(1).expand(-1, T, -1)
        alpha_seq = alpha.unsqueeze(1).expand(-1, T, -1)
        delta_t = self.delta_head(torch.cat([temporal, static_t, h_anchor], dim=2))
        gate_t = self.gate_min + (1.0 - self.gate_min) * self.gate_head(torch.cat([temporal, static_t, h_anchor, alpha_seq], 2))
        h_residual = h_anchor + gate_t * delta_t
        h_creative = self.creative_head(torch.cat([temporal, static_t], dim=2))
        blend = torch.sigmoid(self.creative_alpha).clamp(0.05, 0.60)
        return h_residual + blend * h_creative, gate_t, delta_t
