"""TEM physics descriptor of a decay (levels, log-log slopes, curvature and roughness on the measured grid) and the
field-admissible manifold (FAM) of the measured background decays."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .common import autocast


PHYS_NAMES = ("L_first", "L_q25", "L_mid", "L_q75", "L_last", "slope_early", "slope_mid", "slope_late",
              "curvature", "roughness")


class PhysicsDescriptor:
    """Linear functionals of L(x) (levels, least-squares log-log slopes, mean curvature) + a roughness
    term.  The same definitions are used by the losses (torch) and the QC metrics (numpy)."""

    def __init__(self, grid_ms: np.ndarray) -> None:
        x = np.log10(np.asarray(grid_ms, dtype=np.float64))
        T = x.size
        tau = (x - x[0]) / max(x[-1] - x[0], 1e-12)
        M = np.zeros((T, 9), dtype=np.float64)

        def _at(q: float) -> np.ndarray:
            w = np.zeros(T)
            pos = np.interp(q, tau, np.arange(T))
            i0 = int(np.floor(pos))
            i1 = min(i0 + 1, T - 1)
            f = pos - i0
            w[i0] += 1 - f
            w[i1] += f
            return w

        def _slope(lo: float, hi: float) -> np.ndarray:
            sel = (tau >= lo - 1e-12) & (tau <= hi + 1e-12)
            if sel.sum() < 3:
                sel = np.argsort(np.abs(tau - 0.5 * (lo + hi)))[:3]
                m = np.zeros(T, dtype=bool)
                m[sel] = True
                sel = m
            xc = x[sel] - x[sel].mean()
            w = np.zeros(T)
            w[sel] = xc / max(float((xc ** 2).sum()), 1e-12)
            return w

        M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 4] = _at(0.0), _at(0.25), _at(0.5), _at(0.75), _at(1.0)
        M[:, 5], M[:, 6], M[:, 7] = _slope(0.0, 0.15), _slope(0.4, 0.6), _slope(0.8, 1.0)
        # mean second derivative d2L/dx2 over tau in [0.2, 0.9] (non-uniform central differences)
        c = np.zeros(T)
        idx = [k for k in range(1, T - 1) if 0.2 <= tau[k] <= 0.9]
        for k in idx:
            h0, h1 = x[k] - x[k - 1], x[k + 1] - x[k]
            c[k - 1] += 2.0 / (h0 * (h0 + h1))
            c[k] += -2.0 / (h0 * h1)
            c[k + 1] += 2.0 / (h1 * (h0 + h1))
        M[:, 8] = c / max(len(idx), 1)
        self.M = M
        self.x = x
        self.T = T
        h = np.diff(x)
        self.d2 = np.zeros((T, T - 2))
        for k in range(1, T - 1):
            h0, h1 = h[k - 1], h[k]
            self.d2[k - 1, k - 1] = 2.0 / (h0 * (h0 + h1))
            self.d2[k, k - 1] = -2.0 / (h0 * h1)
            self.d2[k + 1, k - 1] = 2.0 / (h1 * (h0 + h1))
        self._torch_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def numpy(self, L: np.ndarray) -> np.ndarray:
        L2 = np.asarray(L, dtype=np.float64).reshape(-1, self.T)
        lin = L2 @ self.M
        dd = L2 @ self.d2
        sm = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), np.ones(5) / 5, mode="valid"), 1, dd)
        rough = np.log10(np.sqrt(np.mean((dd - sm) ** 2, axis=1)) + 1e-4)
        return np.concatenate([lin, rough[:, None]], axis=1).astype(np.float32)

    def torch(self, L: torch.Tensor) -> torch.Tensor:
        key = f"{L.device}"
        if key not in self._torch_cache:
            self._torch_cache[key] = (torch.tensor(self.M, dtype=torch.float32, device=L.device),
                                      torch.tensor(self.d2, dtype=torch.float32, device=L.device))
        M, d2 = self._torch_cache[key]
        with autocast(enabled=False):
            L2 = L.float().reshape(-1, self.T)
            lin = L2 @ M
            dd = L2 @ d2
            sm = F.avg_pool1d(F.pad(dd.unsqueeze(1), (2, 2), mode="replicate"), 5, stride=1).squeeze(1)
            rough = torch.log10(torch.sqrt(((dd - sm) ** 2).mean(dim=1) + 1e-12) + 1e-4)
            return torch.cat([lin, rough.unsqueeze(1)], dim=1)


class FieldAdmissibleManifold:
    """Field-admissible manifold (FAM) of the measured background curves in the ten PhysicsDescriptor features on the
    measured gates (levels, log-log slopes, mean curvature, roughness): the union of balls B(a_m, r_m) about the
    robustly standardised anchor descriptors a_m (median, 1.4826 MAD floored at 5 % of the p5-p95 range + 1e-3, the
    standardisation of the physics guidance), r_m = max(kappa d_k(a_m), r_floor) with d_k the distance to the k-th
    nearest OTHER anchor and r_floor half the median nearest-neighbour spacing.  A curve is field-admissible when its
    descriptor psi lies in the union: excess(psi) = min_m (|psi - a_m| - r_m) <= 0.  The leave-one-out occupancy
    (each anchor tested against the other anchors' balls) states how well the radii cover the measured curves
    themselves."""

    def __init__(self, psi_anchor: np.ndarray, k: int = 5, kappa: float = 1.25,
                 device: Optional[torch.device] = None, block: int = 1024) -> None:
        X = np.asarray(psi_anchor, dtype=np.float64)
        X = X[np.all(np.isfinite(X), axis=1)]
        n = int(X.shape[0])
        if n < 2:
            raise ValueError(f"needs at least 2 finite anchor curves, got {n}")
        self.med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.med), axis=0)
        self.scale = np.maximum(mad, 0.05 * (np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0)) + 1e-3)
        Z = (X - self.med) / self.scale
        self.n, self.k, self.kappa = n, int(max(1, min(int(k), n - 1))), float(kappa)
        dk, d1 = np.empty(n), np.empty(n)
        for i0 in range(0, n, block):            # k-th and first nearest OTHER anchor, in blocks (long flight lines)
            Dm = np.sqrt(np.maximum(((Z[i0:i0 + block, None, :] - Z[None, :, :]) ** 2).sum(-1), 0.0))
            Dm[np.arange(Dm.shape[0]), np.arange(i0, i0 + Dm.shape[0])] = np.inf
            Ds = np.sort(Dm, axis=1)
            dk[i0:i0 + block], d1[i0:i0 + block] = Ds[:, self.k - 1], Ds[:, 0]
        self.r_floor = float(0.5 * np.median(d1))
        self.radius = np.maximum(self.kappa * dk, self.r_floor)
        inside = np.zeros(n, dtype=bool)
        for i0 in range(0, n, block):
            Dm = np.sqrt(np.maximum(((Z[i0:i0 + block, None, :] - Z[None, :, :]) ** 2).sum(-1), 0.0))
            Dm[np.arange(Dm.shape[0]), np.arange(i0, i0 + Dm.shape[0])] = np.inf
            inside[i0:i0 + block] = ((Dm - self.radius[None, :]) <= 0.0).any(axis=1)
        self.loo_occupancy = float(inside.mean())
        self.Z = Z
        dev = device if device is not None else torch.device("cpu")
        self.t_med = torch.tensor(self.med, dtype=torch.float32, device=dev)
        self.t_scale = torch.tensor(self.scale, dtype=torch.float32, device=dev)
        self.t_Z = torch.tensor(Z, dtype=torch.float32, device=dev)
        self.t_Z2 = (self.t_Z * self.t_Z).sum(dim=1)
        self.t_radius = torch.tensor(self.radius, dtype=torch.float32, device=dev)

    def excess_torch(self, psi: torch.Tensor) -> torch.Tensor:
        """excess (N,) of raw descriptors psi (N, 10) (PhysicsDescriptor units): < 0 inside the manifold, in robust sd.
        Differentiable (the nearest ball's centre direction); float32 outside autocast."""
        with torch.autocast(device_type=psi.device.type, enabled=False):
            z = (psi.float() - self.t_med) / self.t_scale
            d2 = (z * z).sum(dim=1, keepdim=True) + self.t_Z2.view(1, -1) - 2.0 * z @ self.t_Z.t()
            d = torch.sqrt(d2.clamp_min(1e-12))
            return (d - self.t_radius.view(1, -1)).min(dim=1).values

    def excess_numpy(self, psi: np.ndarray, block: int = 4096) -> np.ndarray:
        z = (np.asarray(psi, dtype=np.float64) - self.med) / self.scale
        out = np.empty(z.shape[0])
        for i0 in range(0, z.shape[0], block):
            d = np.sqrt(np.maximum(((z[i0:i0 + block, None, :] - self.Z[None, :, :]) ** 2).sum(-1), 0.0))
            out[i0:i0 + block] = (d - self.radius[None, :]).min(axis=1)
        return out

    def summary(self) -> Dict[str, Any]:
        return {"anchors": int(self.n), "descriptors": int(self.Z.shape[1]), "k": int(self.k), "kappa": self.kappa,
                "radius_median": float(np.median(self.radius)), "radius_min": float(self.radius.min()),
                "radius_max": float(self.radius.max()), "r_floor": self.r_floor,
                "loo_occupancy": self.loo_occupancy, "descriptor_median": self.med.tolist(),
                "descriptor_scale": self.scale.tolist()}
