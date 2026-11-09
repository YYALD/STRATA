"""CGAN-T, trace conditional GAN (Mirza & Osindero 2014): one decay curve given its depth and anomaly weight, BCE loss.
Every station of a generated profile is drawn independently given its condition, so nothing constrains the profile
along depth; the conditions come from a drawn anomaly layout, which is also the profile's label.

    python methods/cgan_t.py --profile examples/data/profile.csv --events examples/data/events.json --out runs/cgan_t
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from common import MLP, _dnorm, _tt, run_method, weight_map


# ---- training and generation -----------------------------------------------------------------------------------------
def train_cgan_t(L, zn, w, iters, seed, dev, say):
    torch.manual_seed(seed)
    mu, sd = L.mean(0), L.std(0) + 1e-6
    X = _tt((L - mu) / sd, dev)
    C = _tt(np.stack([zn, w], 1), dev)
    G, D = MLP(16 + 2, L.shape[1]).to(dev), MLP(L.shape[1] + 2, 1).to(dev)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for it in range(iters):
        idx = torch.randint(0, X.size(0), (64,), device=dev)
        real, c = X[idx], C[idx]
        fake = G(torch.cat([torch.randn(64, 16, device=dev), c], 1))
        lD = (bce(D(torch.cat([real, c], 1)), torch.full((64, 1), 0.9, device=dev))
              + bce(D(torch.cat([fake.detach(), c], 1)), torch.zeros(64, 1, device=dev)))
        oD.zero_grad(); lD.backward(); oD.step()
        lG = bce(D(torch.cat([fake, c], 1)), torch.ones(64, 1, device=dev))
        oG.zero_grad(); lG.backward(); oG.step()
        if (it + 1) % max(1, iters // 5) == 0:
            say(f"[cgan_t] iter {it + 1}/{iters} D {lD.item():.4f} G {lG.item():.4f}")
    return G, {"mu": mu, "sd": sd}


def gen_cgan_t(G, st, zq_n, zq, layouts, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    n_st = zq.size
    out = np.zeros((len(layouts), n_st, st["mu"].size))
    G.eval()
    with torch.no_grad():
        for i0 in range(0, len(layouts), 64):
            lay = layouts[i0:i0 + 64]
            c = np.concatenate([np.stack([zq_n, weight_map(l, zq)], 1) for l in lay])
            x = torch.cat([torch.randn(len(lay) * n_st, 16, generator=g, device=dev), _tt(c, dev)], 1)
            out[i0:i0 + len(lay)] = G(x).cpu().numpy().reshape(len(lay), n_st, -1) * st["sd"] + st["mu"]
    return out


INFO = "CGAN-T (trace CGAN | depth, weight)"


def train_and_generate(d, dev, say):
    """Trained on the curves of the training profile with their depth and anomaly weight; one profile per drawn
    layout."""
    zt, zq, z0, z1 = d["zt"], d["zq"], d["z0"], d["z1"]
    G, st = train_cgan_t(d["L"], _dnorm(zt, z0, z1), weight_map(d["ev"], zt), d["iters"], d["seed"], dev, say)
    return G, gen_cgan_t(G, st, _dnorm(zq, z0, z1), zq, d["layouts"], d["gs"], dev), d["layouts"], {}


if __name__ == "__main__":
    run_method("cgan_t", INFO, train_and_generate)
