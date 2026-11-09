"""DDPM-P, profile denoising diffusion (Ho et al. 2020; cosine schedule, Nichol & Dhariwal 2021): an epsilon-prediction
1-D U-Net over 64-point depth windows on the 2 m grid, T = 1000 steps, EMA weights (0.999), deterministic DDIM
sampling in 50 steps (Song et al. 2021) of whole profiles.  cddpm_p.py is the same model conditioned on the
anomaly-weight map and depth.

    python methods/ddpm_p.py --profile examples/data/profile.csv --gates examples/data/gate_times.csv --out runs/ddpm_p
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import _cond_channels, _dnorm, _tt, _windows, pchip_depth, run_method, weight_map


# ---- diffusion schedule, network, training and sampling --------------------------------------------------------------
DIFF_T = 1000               # diffusion steps of the DDPM baselines (training)
DIFF_SAMPLE_STEPS = 50      # DDIM steps (eta 0) at generation


def _cosine_alpha_bar(T: int = DIFF_T, s: float = 0.008) -> np.ndarray:
    """alpha_bar(k), k = 1..T, of the cosine schedule (Nichol & Dhariwal 2021)."""
    t = np.linspace(0.0, 1.0, T + 1)
    f = np.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    return np.clip(f[1:] / f[0], 1e-5, 1.0)


class _TimeRes(nn.Module):
    """Residual 1-D conv block, scale / shift from the diffusion-time embedding (FiLM)."""

    def __init__(self, a: int, b: int, t_dim: int):
        super().__init__()
        self.c1, self.n1 = nn.Conv1d(a, b, 5, padding=2), nn.GroupNorm(8, b)
        self.c2, self.n2 = nn.Conv1d(b, b, 5, padding=2), nn.GroupNorm(8, b)
        self.t = nn.Linear(t_dim, 2 * b)
        self.skip = nn.Conv1d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x)))
        sc, sh = self.t(te).unsqueeze(-1).chunk(2, dim=1)
        return F.silu(self.n2(self.c2(h)) * (1.0 + sc) + sh) + self.skip(x)


class DiffUNet1D(nn.Module):
    """Epsilon-prediction U-Net over depth (gates + condition channels), three levels; length a multiple of 8."""

    def __init__(self, g: int, cond: int = 0, ch: Sequence[int] = (64, 128, 256), t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.tm = nn.Sequential(nn.Linear(t_dim, 4 * t_dim), nn.SiLU(), nn.Linear(4 * t_dim, t_dim))
        self.e1, self.e2 = _TimeRes(g + cond, ch[0], t_dim), _TimeRes(ch[0], ch[1], t_dim)
        self.e3, self.mid = _TimeRes(ch[1], ch[2], t_dim), _TimeRes(ch[2], ch[2], t_dim)
        self.u3, self.d3 = nn.ConvTranspose1d(ch[2], ch[2], 2, stride=2), _TimeRes(2 * ch[2], ch[1], t_dim)
        self.u2, self.d2 = nn.ConvTranspose1d(ch[1], ch[1], 2, stride=2), _TimeRes(2 * ch[1], ch[0], t_dim)
        self.u1, self.d1 = nn.ConvTranspose1d(ch[0], ch[0], 2, stride=2), _TimeRes(2 * ch[0], ch[0], t_dim)
        self.out = nn.Conv1d(ch[0], g, 1)

    def forward(self, x, t, c=None):
        half = self.t_dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
        a = 1000.0 * t.float()[:, None] * fr[None]
        te = self.tm(torch.cat([a.sin(), a.cos()], 1))
        if c is not None:
            x = torch.cat([x, c], 1)
        e1 = self.e1(x, te)
        e2 = self.e2(F.avg_pool1d(e1, 2), te)
        e3 = self.e3(F.avg_pool1d(e2, 2), te)
        b = self.mid(F.avg_pool1d(e3, 2), te)
        y = self.d3(torch.cat([self.u3(b), e3], 1), te)
        y = self.d2(torch.cat([self.u2(y), e2], 1), te)
        y = self.d1(torch.cat([self.u1(y), e1], 1), te)
        return self.out(y)


def train_ddpm_p(L, z, grid, iters, seed, dev, say, events=None, z0=0.0, z1=1.0):
    """DDPM on depth windows of the training profile (2 m grid, gates as channels, standardised per gate, the level
    jitter of gan_p); with events: conditioned on the anomaly-weight map and depth (CDDPM-P).  The EMA weights (0.999)
    generate."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = L.mean(0), L.std(0) + 1e-6
    Lg = pchip_depth(z, (L - mu) / sd, grid)
    cond = (_cond_channels(weight_map(events, grid), _dnorm(grid, z0, z1)) if events is not None
            else np.zeros((0, grid.size)))
    net = DiffUNet1D(L.shape[1], cond.shape[0]).to(dev)
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    ab = _tt(_cosine_alpha_bar(), dev)
    tag = "cddpm_p" if events is not None else "ddpm_p"
    for it in range(iters):
        X, E = _windows(Lg, cond, 64, rng)
        x0 = _tt(X + rng.normal(0, 0.02, size=(64, 1, 1)) / sd.mean(), dev)
        k = torch.randint(0, DIFF_T, (x0.shape[0],), device=dev)
        a = ab[k].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        pred = net(a.sqrt() * x0 + (1.0 - a).sqrt() * eps, (k.float() + 1.0) / DIFF_T,
                   _tt(E, dev) if cond.shape[0] else None)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
        if (it + 1) % max(1, iters // 5) == 0:
            say(f"[{tag}] iter {it + 1}/{iters} loss {loss.item():.4f}")
    return ema, {"mu": mu, "sd": sd}


def ddim_sample(net, shape, cond, steps: int, g, dev) -> torch.Tensor:
    """Deterministic DDIM (eta 0) from Gaussian noise over `steps` of the DIFF_T schedule; x0 clipped to +-6 sd."""
    ab = _tt(_cosine_alpha_bar(), dev)
    ks = np.unique(np.linspace(DIFF_T - 1, 0, steps).round().astype(int))[::-1]
    x = torch.randn(shape, generator=g, device=dev)
    for i, k in enumerate(ks):
        a = ab[int(k)]
        a_prev = ab[int(ks[i + 1])] if i + 1 < len(ks) else torch.ones((), device=dev)
        eps = net(x, torch.full((shape[0],), (float(k) + 1.0) / DIFF_T, device=dev), cond)
        x0 = ((x - (1.0 - a).sqrt() * eps) / a.sqrt()).clamp(-6.0, 6.0)
        x = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
    return x


def gen_ddpm_p(net, st, grid, zq, n, seed, dev, layouts=None, z0=0.0, z1=1.0, native=None):
    """Whole profiles on the 2 m grid (padded to a multiple of 8), then the measured stations."""
    g = torch.Generator(device=dev).manual_seed(seed)
    n_pad = int(math.ceil(grid.size / 8) * 8)
    grid_p = np.concatenate([grid, grid[-1] + 2.0 * np.arange(1, n_pad - grid.size + 1)])
    G_ = st["mu"].size
    n = len(layouts) if layouts is not None else n
    out = np.zeros((n, zq.size, G_))
    ys = []
    with torch.no_grad():
        for i0 in range(0, n, 64):
            b = min(64, n - i0)
            c = (_tt(np.stack([_cond_channels(weight_map(l, grid_p), _dnorm(grid_p, z0, z1))
                               for l in layouts[i0:i0 + b]]), dev) if layouts is not None else None)
            y = ddim_sample(net, (b, G_, n_pad), c, DIFF_SAMPLE_STEPS, g, dev).cpu().numpy().transpose(0, 2, 1)
            out[i0:i0 + b] = pchip_depth(grid, y[:, :grid.size], zq) * st["sd"] + st["mu"]
            ys.append(y[:, :grid.size] * st["sd"] + st["mu"])
    if native is not None:
        native.update({"depth_m": np.asarray(grid, float), "L": np.concatenate(ys)})
    return out


INFO = "DDPM-P (profile denoising diffusion, cosine schedule, DDIM sampling)"


def train_and_generate(d, dev, say):
    """Trained on depth windows of the training profile; d["n"] whole profiles (2 m grid and output stations)."""
    nat = {}
    G, st = train_ddpm_p(d["L"], d["zt"], d["grid"], d["iters"], d["seed"], dev, say)
    return G, gen_ddpm_p(G, st, d["grid"], d["zq"], d["n"], d["gs"], dev, native=nat), None, nat


if __name__ == "__main__":
    run_method("ddpm_p", INFO, train_and_generate)
