"""Decay-manifold-constrained feature extraction (in time): the decay encoder (Embedder), the condition extractor
(TEMConditionExtractor) and its supervision heads, the decoder with a diffusion-consistent head (Recovery), which
preserves the decay features of the physical diffusion process, and the credibility gate on the manifold of the
anchor conditions (_FinCMG with its memory bank)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import autocast
from ..models.layers import InceptionTimeLite, PositionalEncoding, TransformerBlock, _rfft_power, _rnn_call


class Embedder(nn.Module):
    """Decay encoder: two Transformer self-attention layers and a recurrent stack map a decay to a latent sequence in
    (0, 1).  A direct value path, h(t) = sigmoid(W lstm(t) + V x(t)), keeps the amplitude of every gate in the
    latent."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, module_name: str = "lstm",
                 n_heads: int = 4, attn_layers: int = 2, value_gain: float = 8.0) -> None:
        super().__init__()
        while hidden_dim % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=1024)
        self.attn_blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads=n_heads, dropout=0.1)
                                          for _ in range(attn_layers)])
        self.attn_norm = nn.LayerNorm(hidden_dim)
        rnn_cls = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.value_out = nn.Linear(input_dim, hidden_dim)
        with torch.no_grad():
            self.value_out.weight.mul_(float(value_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        for blk in self.attn_blocks:
            h = blk(h, causal=False)
        h, _ = _rnn_call(self.rnn, self.attn_norm(h))
        return torch.sigmoid(self.proj[0](h) + self.value_out(x))


class Recovery(nn.Module):
    """Decoder: a recurrent stack with a zero-initialised convolutional residual head.  With monotone_head the output
    is read as a start level and decay increments per gate, which preserves the decay features of the diffusion
    process."""

    def __init__(self, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 2, module_name: str = "lstm",
                 residual_enabled: bool = True, residual_channels: int = 32, residual_layers: int = 3,
                 residual_kernel: int = 3, residual_scale: float = 0.10, monotone_head: bool = False) -> None:
        super().__init__()
        rnn_cls = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        if self.residual_enabled:
            k = int(max(3, residual_kernel))
            pad = k // 2
            ch = int(max(8, residual_channels))
            nl = int(max(2, residual_layers))
            layers: List[nn.Module] = [nn.Conv1d(hidden_dim + output_dim, ch, k, padding=pad), nn.GELU()]
            for _ in range(nl - 2):
                layers += [nn.Conv1d(ch, ch, k, padding=pad), nn.GELU()]
            out_conv = nn.Conv1d(ch, output_dim, k, padding=pad)
            nn.init.zeros_(out_conv.weight)
            nn.init.zeros_(out_conv.bias)
            layers.append(out_conv)
            self.residual_head = nn.Sequential(*layers)
        else:
            self.residual_head = None
        self.monotone_head = bool(monotone_head)
        self.mono_bias = nn.Parameter(torch.tensor(0.0))
        self.mono_step = 1.0          # mean decrement softplus(0)*step/T ~ 0.69 of the range over T gates
        self.floor_beta = 100.0

    def _monotone(self, r: torch.Tensor) -> torch.Tensor:
        # start level is linear (0.5 + r0): in the log domain it needs no bound, and a sigmoid start put the strongest
        # (shallow) curves at y0 ~ 0.95 in its flat region -> early-gate bias -0.13 decades on the high-amplitude curves
        with autocast(enabled=False):
            r = torch.nan_to_num(r.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            T = r.size(1)
            y0 = 0.5 + r[:, :1, :]
            inc = F.softplus(r[:, 1:, :] + self.mono_bias) * (self.mono_step / max(T, 1))
            y = torch.cat([y0, y0 - torch.cumsum(inc, dim=1)], dim=1)
            return F.softplus(self.floor_beta * y) / self.floor_beta   # smooth, monotone floor at 0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y, _ = _rnn_call(self.rnn, h)
        base = self.proj(y)
        if self.residual_enabled and self.residual_head is not None:
            feat = torch.cat([h, base], dim=2).transpose(1, 2)
            r = torch.tanh(self.residual_head(feat).transpose(1, 2)) * self.residual_scale
            base = base + r
        if self.monotone_head:
            return self._monotone(base)
        return base


def _rfmg_cond_feat_guard(feat: torch.Tensor, bound: float = 1.0e3) -> torch.Tensor:
    """Bounds the hand-crafted ratio features before they enter the fp16 MLPs."""
    return torch.nan_to_num(feat, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


class TEMConditionExtractor(nn.Module):
    """Multi-stream condition vector: hand-crafted time / frequency / morphology streams, InceptionTime deep stream
    and a Transformer sequence stream, fused to cond_dim."""

    def __init__(self, seq_len: int = 256, out_dim: int = 64, inception_base_ch: int = 32, deep_dim: int = 128,
                 branch_dim: int = 32, n_heads: int = 4, transformer_layers: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.deep_stream = InceptionTimeLite(in_ch=1, base_ch=inception_base_ch, out_dim=deep_dim)
        self.time_mlp = nn.Sequential(nn.Linear(22, branch_dim * 2), nn.GELU(), nn.Linear(branch_dim * 2, branch_dim))
        self.freq_mlp = nn.Sequential(nn.Linear(10, branch_dim * 2), nn.GELU(), nn.Linear(branch_dim * 2, branch_dim))
        self.morph_mlp = nn.Sequential(nn.Linear(12, branch_dim * 2), nn.GELU(), nn.Linear(branch_dim * 2, branch_dim))
        tfm = 128
        nh = n_heads
        while tfm % nh != 0 and nh > 1:
            nh //= 2
        self.seq_proj = nn.Linear(1, tfm)
        self.seq_pos = PositionalEncoding(tfm, max_len=max(seq_len, 512))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(tfm, n_heads=nh, dropout=0.1)
                                                 for _ in range(transformer_layers)])
        self.seq_pool = nn.Sequential(nn.LayerNorm(tfm), nn.Linear(tfm, deep_dim))
        self.fusion = nn.Sequential(nn.Linear(branch_dim * 3 + deep_dim * 2, 128), nn.LayerNorm(128), nn.GELU(),
                                    nn.Linear(128, out_dim))

    def _handcrafted_time(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            s = x[..., 0].float()
            T = s.shape[1]
            early = s[:, : max(8, T // 8)]
            mid = s[:, T // 3: 2 * T // 3]
            late = s[:, -max(8, T // 8):]
            q = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75], device=s.device), dim=1)
            k1, k3 = min(1, T - 1), min(3, T - 1)
            feat = torch.stack([
                s.mean(1), s.std(1), (s ** 2).mean(1), s.abs().mean(1), s.max(1).values - s.min(1).values,
                q[0], q[1], q[2], early.mean(1), mid.mean(1), late.mean(1), (s[:, -1] - s[:, 0]) / max(T - 1, 1),
                (early ** 2).mean(1), (late ** 2).mean(1), s[:, 0].abs(), s[:, :min(3, T)].abs().max(1).values,
                s[:, :min(8, T)].abs().max(1).values, (s[:, :min(8, T)] ** 2).mean(1),
                (s[:, 0] / (s[:, k1].abs() + 1e-6)).clamp(-1e3, 1e3), (s[:, 0] / (s[:, k3].abs() + 1e-6)).clamp(-1e3, 1e3),
                s[:, :min(8, T)].max(1).values - s[:, :min(8, T)].min(1).values,
                (late.abs().mean(1) / (early.abs().mean(1) + 1e-6)).clamp(-1e3, 1e3)], dim=1)
        return feat

    def _handcrafted_freq(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            s = x[..., 0].float()
            n = int(s.shape[1])
            n_fft = 1 << (max(1, n) - 1).bit_length()
            if n_fft != n:
                s = F.pad(s, (0, n_fft - n))
            power = _rfft_power(s).clamp_min(1e-8)
            k = power.shape[1]
            idx = torch.arange(k, device=power.device, dtype=power.dtype).unsqueeze(0)
            ps = power.sum(1).clamp_min(1e-8)
            centroid = (power * idx).sum(1) / ps
            spread = (((idx - centroid.unsqueeze(1)) ** 2) * power).sum(1) / ps
            low = power[:, : max(1, k // 6)].mean(1)
            lowmid = power[:, max(1, k // 6): max(2, k // 3)].mean(1)
            mid = power[:, max(2, k // 3): max(3, 2 * k // 3)].mean(1)
            high = power[:, max(3, 2 * k // 3):].mean(1)
            roll = (torch.cumsum(power, 1) >= (0.85 * ps).unsqueeze(1)).float().argmax(1).float() / max(k - 1, 1)
            flat = torch.exp(torch.log(power).mean(1)) / power.mean(1).clamp_min(1e-8)
            return torch.stack([low, lowmid, mid, high, low / high.clamp_min(1e-8), lowmid / mid.clamp_min(1e-8),
                                centroid, spread, roll, flat], dim=1)

    def _handcrafted_morph(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            s = x[..., 0].float()
            d1 = s[:, 1:] - s[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            T = s.shape[1]
            q1, q2 = max(4, T // 4), max(8, T // 2)
            early, mid, late = s[:, :q1], s[:, q1:q2], s[:, q2:]
            return torch.stack([
                early.mean(1), mid.mean(1), late.mean(1), d1[:, : max(4, d1.shape[1] // 4)].mean(1),
                d1[:, d1.shape[1] // 3: 2 * d1.shape[1] // 3].mean(1), d1[:, -max(4, d1.shape[1] // 4):].mean(1),
                d2.mean(1), d2.abs().mean(1), ((d1[:, 1:] * d1[:, :-1]) < 0).float().mean(1), (d1 < 0).float().mean(1),
                (early.abs().mean(1) / mid.abs().mean(1).clamp_min(1e-8)).clamp(-1e3, 1e3),
                (s.abs().max(1).values / late.abs().mean(1).clamp_min(1e-8)).clamp(-1e3, 1e3)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_mlp(_rfmg_cond_feat_guard(self._handcrafted_time(x)))
        f_feat = self.freq_mlp(_rfmg_cond_feat_guard(self._handcrafted_freq(x)))
        m_feat = self.morph_mlp(_rfmg_cond_feat_guard(self._handcrafted_morph(x)))
        d_feat = self.deep_stream(x)
        h = self.seq_pos(self.seq_proj(x))
        for blk in self.transformer_blocks:
            h = blk(h, causal=False)
        seq_feat = self.seq_pool(h.mean(dim=1))
        c = self.fusion(torch.cat([t_feat, f_feat, m_feat, d_feat, seq_feat], dim=1))
        return torch.nan_to_num(c, nan=0.0, posinf=1.0e2, neginf=-1.0e2)


class ConditionSupervisionHead(nn.Module):
    """Stage-0B heads: domain (measured / simulated / bridge), alpha and TEM physics statistics."""

    def __init__(self, cond_dim: int, stat_dim: int) -> None:
        super().__init__()
        self.domain = nn.Linear(cond_dim, 3)
        self.alpha = nn.Sequential(nn.Linear(cond_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.stats = nn.Sequential(nn.Linear(cond_dim, 128), nn.GELU(), nn.Linear(128, stat_dim))

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.domain(c), self.alpha(c), self.stats(c)


class _ManifoldMemoryBank(nn.Module):
    """Field-anchor memory bank for DLLR (differentiable local linear reconstruction):
    pi_M(c) = sum_{i in N_K(c)} softmax(-|c - z_i|^2 / tau) z_i  over measured-domain anchors."""

    def __init__(self, cond_dim: int, capacity: int = 512, tau: float = 0.10):
        super().__init__()
        self.capacity = capacity
        self.tau = tau
        self.register_buffer("bank", torch.zeros(capacity, cond_dim))
        self.register_buffer("bank_phi5", torch.zeros(capacity, 5))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, c_real: torch.Tensor, phi5_real: Optional[torch.Tensor] = None) -> None:
        B = min(c_real.size(0), self.capacity)
        ptr = int(self.ptr.item())
        slots = (ptr + torch.arange(B, device=c_real.device)) % self.capacity
        self.bank[slots] = c_real[:B].detach().float()
        if phi5_real is not None and phi5_real.shape[0] >= B:
            self.bank_phi5[slots] = phi5_real[:B].detach().float()
        self.ptr.fill_((ptr + B) % self.capacity)
        self.filled.fill_(min(int(self.filled.item()) + B, self.capacity))

    def dllr_project(self, c_fake: torch.Tensor, K: int = 8, exclude_self: bool = False) -> torch.Tensor:
        n = int(self.filled.item())
        if n < 2:
            return c_fake
        anchors = self.bank[:n].to(c_fake.dtype)
        sq = ((c_fake.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(2)
        if exclude_self:   # leave-self-out: a query's own bank entries do not reconstruct it
            sq = torch.where(sq <= 0.0, torch.full_like(sq, 1e30), sq)
        d, idx = sq.topk(min(K, n), dim=1, largest=False)
        w = F.softmax(-d / self.tau, dim=1)
        return (w.unsqueeze(2) * anchors[idx]).sum(1)


class _FinCMG(nn.Module):
    """Credibility gate on the condition manifold: gamma = sigma(-v / tau_v) sigma((delta - rho) / tau_U), the
    credibility of a condition (v = d_M / tau_M + d_P / tau_P: its distance to the manifold of the anchor conditions
    and its physical inconsistency) times a trust region on its DLLR residual delta = |c - pi(c)| / |c|.  A condition
    within rho of the manifold is left alone (its diversity stays), one beyond it is moved toward its projection
    pi(c), to c + gamma (pi(c) - c); rho is learned as the 90 % quantile of the anchor conditions' own leave-self-out
    residual."""

    def __init__(self, cond_dim: int, hidden: int = 64, bank_capacity: int = 512, dllr_K: int = 8,
                 dllr_tau: float = 0.10, form: str = "v20"):
        super().__init__()
        self.dllr_K = dllr_K
        self.memory_bank = _ManifoldMemoryBank(cond_dim, bank_capacity, dllr_tau)
        self.d_M_net = nn.Sequential(nn.Linear(cond_dim * 2 + 10, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.d_P_net = nn.Sequential(nn.Linear(cond_dim * 2 + 5 + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.log_tau_M = nn.Parameter(torch.tensor(float(np.log(0.30))))
        self.log_tau_P = nn.Parameter(torch.tensor(float(np.log(0.30))))
        for m in (self.d_M_net, self.d_P_net):
            nn.init.normal_(m[-1].weight, std=0.01)
            nn.init.constant_(m[-1].bias, 0.0)
        self.form = str(form).strip().lower()
        if self.form == "eaai":
            self.rho_net = nn.Sequential(nn.Linear(cond_dim + 5, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            nn.init.normal_(self.rho_net[-1].weight, std=0.01)
            nn.init.constant_(self.rho_net[-1].bias, -2.0)          # softplus(-2) = 0.13 relative residual at start
            self.log_tau_U = nn.Parameter(torch.tensor(float(np.log(0.05))))

    def _credibility(self, c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real) -> Tuple[torch.Tensor, torch.Tensor]:
        """Credibility of each condition and its DLLR residual."""
        tau_M = torch.exp(self.log_tau_M).clamp(0.01, 1.0)
        tau_P = torch.exp(self.log_tau_P).clamp(0.01, 1.0)
        d_M_raw = self.d_M_net(torch.cat([c_fake, c_real.detach(), phi5_fake, phi5_real.detach()], 1)).squeeze(1)
        proj = self.memory_bank.dllr_project(c_fake, K=self.dllr_K)
        resid = (c_fake - proj).norm(dim=1) / (c_fake.norm(dim=1) + 1e-6)
        d_M = torch.sigmoid(d_M_raw + 0.5 * resid)
        d_P = torch.sigmoid(self.d_P_net(torch.cat([c_fake - c_sim.detach(), c_real.detach() - c_sim.detach(),
                                                    phi5_fake - phi5_real.detach(), alpha.view(-1, 1)], 1))).squeeze(1)
        return torch.sigmoid(-d_M / tau_M - d_P / tau_P), resid

    def _rho(self, c: torch.Tensor, phi5: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.rho_net(torch.cat([c, phi5], 1))).squeeze(1)

    def _compute_gate(self, c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real) -> torch.Tensor:
        cred, resid = self._credibility(c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real)
        if self.form != "eaai":
            return cred
        tau_U = torch.exp(self.log_tau_U).clamp(0.005, 0.5)
        return cred * torch.sigmoid((resid - self._rho(c_fake, phi5_fake)) / tau_U)

    def forward(self, c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real) -> torch.Tensor:
        return self._compute_gate(c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real).unsqueeze(1)

    def gate_loss(self, c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real) -> torch.Tensor:
        B = c_fake.size(0)
        if B < 2:
            return c_fake.new_tensor(0.0)

        def _cred(*args) -> torch.Tensor:          # the contrast trains the credibility
            return self._credibility(*args)[0]
        r_pos = _cred(c_fake, c_real, c_sim, alpha, phi5_fake, phi5_real)
        perm = torch.randperm(B, device=c_fake.device)
        r_perm = _cred(c_fake[perm], c_real, c_sim, alpha, phi5_fake[perm], phi5_real)
        r_sim = _cred(c_sim.detach(), c_real, c_sim, alpha, phi5_real, phi5_real)
        noise = torch.randn_like(c_fake) * c_fake.detach().std().clamp(min=1e-4) * 2
        r_noise = _cred(c_fake.detach() + noise, c_real, c_sim, alpha, phi5_fake, phi5_real)
        eps = 1e-6
        contrast = (-torch.log(r_pos + eps) - 0.5 * torch.log(1 - r_perm + eps) - 0.3 * torch.log(1 - r_sim + eps)
                    - 0.2 * torch.log(1 - r_noise + eps)).mean()
        bank_align = F.smooth_l1_loss(self.memory_bank.dllr_project(c_real, K=self.dllr_K), c_real.detach())
        tau_reg = F.relu(0.02 - torch.exp(self.log_tau_M)) + F.relu(0.02 - torch.exp(self.log_tau_P))
        if self.form == "eaai":
            # trust radius = the 90 % quantile of the anchors' leave-self-out DLLR residual (the batch's anchors were
            # just written to the bank: their own entries are excluded, other windows' copies of a station stay)
            with torch.no_grad():
                cr = c_real.detach().float()
                proj_r = self.memory_bank.dllr_project(cr, K=self.dllr_K, exclude_self=True)
                res_r = (cr - proj_r).norm(dim=1) / (cr.norm(dim=1) + 1e-6)
                q90 = torch.quantile(res_r, 0.9)
            rho_r = self._rho(c_real.detach(), phi5_real.detach())
            tau_reg = tau_reg + F.smooth_l1_loss(rho_r.float(), q90.expand_as(rho_r).float(), beta=0.05) \
                + F.relu(0.005 - torch.exp(self.log_tau_U))
        loss = torch.nan_to_num(contrast + 0.5 * bank_align + tau_reg, nan=0.0, posinf=30.0)
        return 3.0 * torch.tanh(loss / 3.0)


class _FinFiLMMixer(nn.Module):
    def __init__(self, hidden: int, cond: int):
        super().__init__()
        self.proj = nn.Linear(cond, 2 * hidden)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.proj(c).chunk(2, -1)
        if h.dim() == 3:
            g, b = g.unsqueeze(1), b.unsqueeze(1)
        return h * (1.0 + 0.5 * torch.tanh(g)) + 0.5 * torch.tanh(b)
