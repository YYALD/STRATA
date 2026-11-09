"""GAN-P, profile GAN: a fully convolutional WGAN-GP over 64-point (126 m) depth windows of the training profile on a
2 m depth grid, the gates as channels.  It generates whole profiles on the grid, which are read at the output stations
by shape-preserving interpolation along depth.

    python methods/gan_p.py --profile examples/data/profile.csv --gates examples/data/gate_times.csv --out runs/gan_p
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

from common import WIN, ConvCritic, _tt, _windows, gradient_penalty, pchip_depth, run_method


# ---- network, training and generation --------------------------------------------------------------------------------
class ConvGen(nn.Module):
    """Fully convolutional profile generator: latent map (zc, n/8) -> (gates, n)."""

    def __init__(self, zc: int = 32, g: int = 31, ch: Sequence[int] = (128, 96, 64)):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv1d(zc, ch[0], 3, padding=1), nn.LeakyReLU(0.2)]
        c_in = ch[0]
        for c in ch:
            layers += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(c_in, c, 5, padding=2),
                       nn.BatchNorm1d(c), nn.LeakyReLU(0.2)]
            c_in = c
        layers += [nn.Conv1d(c_in, g, 5, padding=2)]
        self.net = nn.Sequential(*layers)
        self.zc = zc

    def forward(self, zmap):
        return self.net(zmap)


def train_gan_p(L, z, grid, iters, seed, dev, say):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = L.mean(0), L.std(0) + 1e-6
    Lg = pchip_depth(z, (L - mu) / sd, grid)
    G, D = ConvGen(g=L.shape[1]).to(dev), ConvCritic(L.shape[1]).to(dev)
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
    for it in range(iters):
        for _ in range(5):
            X, _ = _windows(Lg, np.zeros((0, Lg.shape[0])), 32, rng)
            real = _tt(X + rng.normal(0, 0.02, size=(32, 1, 1)) / sd.mean(), dev)
            fake = G(torch.randn(32, G.zc, WIN // 8, device=dev)).detach()
            lD = D(fake).mean() - D(real).mean() + 10.0 * gradient_penalty(D, real, fake)
            oD.zero_grad(); lD.backward(); oD.step()
        lG = -D(G(torch.randn(32, G.zc, WIN // 8, device=dev))).mean()
        oG.zero_grad(); lG.backward(); oG.step()
        if (it + 1) % max(1, iters // 5) == 0:
            say(f"[gan_p] iter {it + 1}/{iters} critic {lD.item():.4f} gen {lG.item():.4f}")
    return G, {"mu": mu, "sd": sd}


def gen_gan_p(G, st, grid, zq, n, seed, dev, native=None):
    """native: dict that receives the whole profiles on the 2 m grid (log10, the generator's own output)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    n_lat = int(math.ceil(grid.size / 8))
    out = np.zeros((n, zq.size, st["mu"].size))
    ys = []
    G.eval()
    with torch.no_grad():
        for i0 in range(0, n, 64):
            b = min(64, n - i0)
            y = G(torch.randn(b, G.zc, n_lat, generator=g, device=dev)).cpu().numpy().transpose(0, 2, 1)[:, :grid.size]
            out[i0:i0 + b] = pchip_depth(grid, y, zq) * st["sd"] + st["mu"]
            ys.append(y * st["sd"] + st["mu"])
    if native is not None:
        native.update({"depth_m": np.asarray(grid, float), "L": np.concatenate(ys)})
    return out


INFO = "GAN-P (profile WGAN-GP)"


def train_and_generate(d, dev, say):
    """Trained on depth windows of the training profile; d["n"] whole profiles (2 m grid and output stations)."""
    nat = {}
    G, st = train_gan_p(d["L"], d["zt"], d["grid"], d["iters"], d["seed"], dev, say)
    return G, gen_gan_p(G, st, d["grid"], d["zq"], d["n"], d["gs"], dev, native=nat), None, nat


if __name__ == "__main__":
    run_method("gan_p", INFO, train_and_generate)
