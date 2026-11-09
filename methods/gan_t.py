"""GAN-T, trace GAN: an unconditional WGAN-GP on single decay curves of the training profile.  A generated profile is a
set of generated curves given to the output stations in the order of their earliest-gate amplitude (the training
profile's order); nothing constrains it along the profile.

    python methods/gan_t.py --profile examples/data/profile.csv --gates examples/data/gate_times.csv --out runs/gan_t
"""
from __future__ import annotations

import numpy as np
import torch

from common import MLP, _tt, gradient_penalty, run_method


# ---- training and generation -----------------------------------------------------------------------------------------
def train_gan_t(L, iters, seed, dev, say):
    torch.manual_seed(seed)
    mu, sd = L.mean(0), L.std(0) + 1e-6
    X = _tt((L - mu) / sd, dev)
    G, D = MLP(16, L.shape[1]).to(dev), MLP(L.shape[1], 1).to(dev)
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
    for it in range(iters):
        for _ in range(5):
            real = X[torch.randint(0, X.size(0), (64,), device=dev)]
            fake = G(torch.randn(64, 16, device=dev)).detach()
            lD = D(fake).mean() - D(real).mean() + 10.0 * gradient_penalty(D, real, fake)
            oD.zero_grad(); lD.backward(); oD.step()
        lG = -D(G(torch.randn(64, 16, device=dev))).mean()
        oG.zero_grad(); lG.backward(); oG.step()
        if (it + 1) % max(1, iters // 5) == 0:
            say(f"[gan_t] iter {it + 1}/{iters} critic {lD.item():.4f} gen {lG.item():.4f}")
    return G, {"mu": mu, "sd": sd}


def gen_gan_t(G, st, n_st, rank_order, n, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    out = np.zeros((n, n_st, st["mu"].size))
    G.eval()
    with torch.no_grad():
        for i0 in range(0, n, 64):
            b = min(64, n - i0)
            C = G(torch.randn(b * n_st, 16, generator=g, device=dev)).cpu().numpy().reshape(b, n_st, -1) * st["sd"] + st["mu"]
            for k in range(b):
                out[i0 + k, rank_order] = C[k][np.argsort(-C[k][:, 0])]
    return out


INFO = "GAN-T (trace WGAN-GP, rank-assembled)"


def train_and_generate(d, dev, say):
    """Trained on the curves of the training profile; d["n"] profiles on the output stations."""
    G, st = train_gan_t(d["L"], d["iters"], d["seed"], dev, say)
    return G, gen_gan_t(G, st, d["zq"].size, d["rank_order"], d["n"], d["gs"], dev), None, {}


if __name__ == "__main__":
    run_method("gan_t", INFO, train_and_generate)
