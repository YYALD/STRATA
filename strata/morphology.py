"""Anomaly-morphology prior: anomaly shapes as patches over (offset / width along the hole, time), the source shares
of the shape bank, physical shape filters and the beta-VAE of learned shapes."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _parse_mix


# A labelled anomaly of the "morphology" kind is a patch M(u, t) in u = (z - centre) / sigma along the hole and the
# measured time grid, max |M| = 1 over the measured window, scaled to the drawn |peak| in decades.
MORPH_U_HALF, MORPH_U_POINTS = 8.0, 161
MORPH_U = np.linspace(-MORPH_U_HALF, MORPH_U_HALF, MORPH_U_POINTS)
MORPH_VAE_U_HALF, MORPH_VAE_NU, MORPH_VAE_NT = 6.0, 49, 32
MORPH_VAE_LATENT, MORPH_VAE_BETA, MORPH_VAE_PIX_SD = 8, 2.0, 0.05
MORPH_PHYS_KERNELS = 2048
MORPH_PHYS_SHARES = (0.35, 0.50, 0.15)            # hole through a conductor / beside the hole / near-hole zone
MORPH_SIM_TAUS = (1.0 / 6.0, 0.5, 5.0 / 6.0)      # a simulated shape's time constants: quantiles of its tau range
MORPH_EXAMPLE_SHARE = 0.30                        # measured + simulated examples in a training batch
MORPH_FILTER = {"outside_u": 4.0, "outside_energy_max": 0.10, "time_peak_rel": 0.5, "sign_change_max": 2,
                "plausibility_quantile": 0.99, "plausibility_factor": 1.25}


def morphology_mix(spec: str, n_meas: int, n_sim: int, n_learned: int) -> Dict[str, float]:
    """Source shares of the morphology bank over the sources present (auto: measured 0.25, simulated 0.25, learned
    0.5)."""
    s_ = str(spec or "auto").strip().lower()
    base = {"measured": 0.25, "sim": 0.25, "learned": 0.5}
    if s_ not in ("", "auto"):
        kv = {k.strip().lower(): v for k, v in _parse_mix(s_, "measured:0.25,sim:0.25,learned:0.5")}
        if not kv or set(kv) - set(base):
            raise ValueError(f"--morphology_mix expects 'measured:a,sim:b,learned:c', got {spec!r}")
        base = {k: float(kv.get(k, 0.0)) for k in base}
    have = {"measured": n_meas > 0, "sim": n_sim > 0, "learned": n_learned > 0}
    out = {k: (max(v, 0.0) if have[k] else 0.0) for k, v in base.items()}
    tot = sum(out.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in out.items()}


def _interp_matrix(xq: np.ndarray, xk: np.ndarray, outside: str = "hold") -> np.ndarray:
    """(len(xq), len(xk)) linear-interpolation weights onto xq (xk ascending); outside the knots the end value is
    held (hold) or 0 (zero)."""
    xq, xk = np.asarray(xq, dtype=np.float64).reshape(-1), np.asarray(xk, dtype=np.float64).reshape(-1)
    Wm = np.zeros((xq.size, xk.size))
    j = np.clip(np.searchsorted(xk, xq, side="right") - 1, 0, max(xk.size - 2, 0))
    if xk.size == 1:
        Wm[:, 0] = 1.0
    else:
        w = np.clip((xq - xk[j]) / np.maximum(xk[j + 1] - xk[j], 1e-12), 0.0, 1.0)
        r = np.arange(xq.size)
        Wm[r, j] += 1.0 - w
        Wm[r, j + 1] += w
    if outside == "zero":
        Wm[(xq < xk[0] - 1e-9) | (xq > xk[-1] + 1e-9)] = 0.0
    return Wm


def _normalise_patches(X: np.ndarray) -> np.ndarray:
    """Every patch of X (n, ...) divided by its largest |value| (sign kept); empty patches stay 0."""
    X = np.asarray(X, dtype=np.float64)
    m = np.abs(X.reshape(X.shape[0], -1)).max(axis=1) if X.size else np.zeros(0)
    return X / np.where(m > 0, m, 1.0).reshape((-1,) + (1,) * (X.ndim - 1))


def morphology_filters(M: np.ndarray, uv: np.ndarray) -> Dict[str, np.ndarray]:
    """Physical admissibility of morphologies M (n, NU, NT), max |M| = 1: localised along the hole (energy beyond |u|
    > 4 at most 10 %), one maximum of the time envelope max_u |M| above half its peak (an eddy-current system peaks
    once relative to the background), at most two sign changes along the hole at the peak time (the axial field of a
    body beside the hole has two; more is not one conductor)."""
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    E = M ** 2
    out_e = E[:, np.abs(uv) > MORPH_FILTER["outside_u"], :].sum(axis=(1, 2)) / np.maximum(E.sum(axis=(1, 2)), 1e-12)
    env = np.abs(M).max(axis=1)
    envs = env.copy()
    if env.shape[1] > 2:
        envs[:, 1:-1] = 0.25 * env[:, :-2] + 0.5 * env[:, 1:-1] + 0.25 * env[:, 2:]
    mx = envs.max(axis=1, keepdims=True)
    left = np.concatenate([np.full((n, 1), -np.inf), envs[:, :-1]], axis=1)
    right = np.concatenate([envs[:, 1:], np.full((n, 1), -np.inf)], axis=1)
    peaks = ((envs >= left) & (envs > right) & (envs >= MORPH_FILTER["time_peak_rel"] * mx)).sum(axis=1)
    col = M[np.arange(n), :, env.argmax(axis=1)]
    sg = np.where(np.abs(col) > 0.05 * np.maximum(np.abs(col).max(axis=1, keepdims=True), 1e-12), np.sign(col), 0.0)
    nsc = np.array([int(np.sum(np.diff(r_[r_ != 0]) != 0)) for r_ in sg], dtype=np.int64)
    ok = (out_e <= MORPH_FILTER["outside_energy_max"]) & (peaks == 1) & (nsc <= MORPH_FILTER["sign_change_max"])
    return {"outside_energy": out_e, "time_peaks": peaks, "sign_changes": nsc, "ok": ok}


class MorphologyVAE(nn.Module):
    """beta-VAE of anomaly morphologies: patches (1, MORPH_VAE_NU along the hole, MORPH_VAE_NT log-time nodes) in
    [-1, 1] -> MORPH_VAE_LATENT latent factors -> patches."""

    def __init__(self, latent: int = MORPH_VAE_LATENT, ch: int = 32) -> None:
        super().__init__()
        self.latent, self.ch = int(latent), int(ch)
        self.enc = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(ch, 2 * ch, 4, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(2 * ch, 4 * ch, 4, stride=2, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d((6, 4)), nn.Flatten(),
                                 nn.Linear(4 * ch * 24, 256), nn.GELU(), nn.Linear(256, 2 * self.latent))
        self.dec_fc = nn.Sequential(nn.Linear(self.latent, 256), nn.GELU(), nn.Linear(256, 4 * ch * 12 * 8), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(4 * ch, 2 * ch, 4, stride=2, padding=1), nn.GELU(),
                                 nn.ConvTranspose2d(2 * ch, ch, 4, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(ch, 1, 3, padding=1))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return h[:, :self.latent], h[:, self.latent:].clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.dec(self.dec_fc(z).view(-1, 4 * self.ch, 12, 8))
        return torch.tanh(F.interpolate(y, size=(MORPH_VAE_NU, MORPH_VAE_NT), mode="bilinear", align_corners=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        return self.decode(mu + torch.randn_like(mu) * torch.exp(0.5 * lv)), mu, lv
