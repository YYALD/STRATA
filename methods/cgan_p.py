"""CGAN-P, profile conditional GAN (pix2pix, Isola et al. 2017): a 1-D U-Net maps the anomaly-weight map and depth,
plus noise channels, to a profile window on the 2 m grid; PatchGAN critic (least-squares loss) and an L1 term.  Whole
profiles are generated from drawn anomaly layouts, which are also their labels; dropout stays on at generation as
the noise source of pix2pix.

    python methods/cgan_p.py --profile examples/data/profile.csv --events examples/data/events.json --out runs/cgan_p
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import WIN, ConvCritic, _cond_channels, _dnorm, _tt, _windows, pchip_depth, run_method, weight_map


# ---- network, training and generation --------------------------------------------------------------------------------
class UNetGen(nn.Module):
    """pix2pix-style 1-D U-Net: (cond + noise channels, n) -> (gates, n), n divisible by 8."""

    def __init__(self, c_in: int, g: int = 31, ch: Sequence[int] = (64, 128, 192)):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv1d(c_in, ch[0], 5, padding=2), nn.LeakyReLU(0.2))
        self.d2 = nn.Sequential(nn.Conv1d(ch[0], ch[1], 5, stride=2, padding=2), nn.BatchNorm1d(ch[1]), nn.LeakyReLU(0.2))
        self.d3 = nn.Sequential(nn.Conv1d(ch[1], ch[2], 5, stride=2, padding=2), nn.BatchNorm1d(ch[2]), nn.LeakyReLU(0.2))
        self.b = nn.Sequential(nn.Conv1d(ch[2], ch[2], 5, stride=2, padding=2), nn.BatchNorm1d(ch[2]), nn.ReLU(),
                               nn.Dropout(0.5))
        self.u3 = nn.Sequential(nn.ConvTranspose1d(ch[2], ch[2], 4, stride=2, padding=1), nn.BatchNorm1d(ch[2]), nn.ReLU())
        self.u2 = nn.Sequential(nn.ConvTranspose1d(2 * ch[2], ch[1], 4, stride=2, padding=1), nn.BatchNorm1d(ch[1]), nn.ReLU())
        self.u1 = nn.Sequential(nn.ConvTranspose1d(2 * ch[1], ch[0], 4, stride=2, padding=1), nn.BatchNorm1d(ch[0]), nn.ReLU())
        self.out = nn.Conv1d(2 * ch[0], g, 5, padding=2)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        b = self.b(e3)
        y = self.u3(b)
        y = self.u2(torch.cat([y, e3], 1))
        y = self.u1(torch.cat([y, e2], 1))
        return self.out(torch.cat([y, e1], 1))


def train_cgan_p(L, z, grid, z0, z1, events, iters, seed, dev, say):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = L.mean(0), L.std(0) + 1e-6
    Lg = pchip_depth(z, (L - mu) / sd, grid)
    cond = _cond_channels(weight_map(events, grid), _dnorm(grid, z0, z1))
    G = UNetGen(cond.shape[0] + 4, g=L.shape[1]).to(dev)
    D = ConvCritic(L.shape[1] + cond.shape[0], patch=True).to(dev)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    for it in range(iters):
        X, E = _windows(Lg, cond, 32, rng)
        real, c = _tt(X, dev), _tt(E, dev)
        fake = G(torch.cat([c, torch.randn(32, 4, WIN, device=dev)], 1))
        pr, pf = D(torch.cat([real, c], 1)), D(torch.cat([fake.detach(), c], 1))
        lD = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
        oD.zero_grad(); lD.backward(); oD.step()
        pg = D(torch.cat([fake, c], 1))
        lG = 0.5 * F.mse_loss(pg, torch.ones_like(pg)) + 100.0 * F.l1_loss(fake, real)
        oG.zero_grad(); lG.backward(); oG.step()
        if (it + 1) % max(1, iters // 5) == 0:
            say(f"[cgan_p] iter {it + 1}/{iters} D {lD.item():.4f} G {lG.item():.4f}")
    return G, {"mu": mu, "sd": sd}


def gen_cgan_p(G, st, grid, z0, z1, zq, layouts, seed, dev, native=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    # the dropout masks (pix2pix's test-time noise) come from the global RNG: seeded, so a rerun with the same seed
    # generates the same profiles
    torch.manual_seed(int(seed))
    n_pad = int(math.ceil(grid.size / 8) * 8)
    grid_p = np.concatenate([grid, grid[-1] + 2.0 * np.arange(1, n_pad - grid.size + 1)])
    out = np.zeros((len(layouts), zq.size, st["mu"].size))
    ys = []
    G.train()                                     # pix2pix keeps dropout on at test time as its noise source
    for mod in G.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.eval()
    with torch.no_grad():
        for i0 in range(0, len(layouts), 64):
            lay = layouts[i0:i0 + 64]
            c = np.stack([_cond_channels(weight_map(l, grid_p), _dnorm(grid_p, z0, z1)) for l in lay])
            x = torch.cat([_tt(c, dev), torch.randn(len(lay), 4, n_pad, generator=g, device=dev)], 1)
            y = G(x).cpu().numpy().transpose(0, 2, 1)[:, :grid.size]
            out[i0:i0 + len(lay)] = pchip_depth(grid, y, zq) * st["sd"] + st["mu"]
            ys.append(y * st["sd"] + st["mu"])
    if native is not None:
        native.update({"depth_m": np.asarray(grid, float), "L": np.concatenate(ys)})
    return out


INFO = "CGAN-P (pix2pix profile CGAN)"


def train_and_generate(d, dev, say):
    """Trained on depth windows of the training profile and its anomaly-weight map; one whole profile per drawn
    layout."""
    nat = {}
    G, st = train_cgan_p(d["L"], d["zt"], d["grid"], d["z0"], d["z1"], d["ev"], d["iters"], d["seed"], dev, say)
    Pm = gen_cgan_p(G, st, d["grid"], d["z0"], d["z1"], d["zq"], d["layouts"], d["gs"], dev, native=nat)
    return G, Pm, d["layouts"], nat


if __name__ == "__main__":
    run_method("cgan_p", INFO, train_and_generate)
