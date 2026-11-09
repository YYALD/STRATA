"""Anchored time grid that holds every measured gate, shape-preserving (PCHIP) log-log resampling, interpolation and
smoothing along the profile."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ..common import array_digest


def build_anchored_time_grid(t_meas_ms: np.ndarray, T: int, t_ext_ms: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """T-point grid over [t_first, t_last] of the measured gates (over [t_first, t_ext] with a late-time extension).

    T >= n_gates: every measured gate time is a grid node (exact values); the T-n extra nodes are shared out over the
    gate intervals in proportion to their log-length (largest remainder) and placed log-uniformly inside each
    interval -> near log-uniform grid, lossless for the gates. t_ext > t_last adds the extension interval [t_last,
    t_ext] as one more interval of the share-out (its last node is t_ext). T <  n_gates: log-uniform grid with exact
    endpoints (gates are not all embedded; reported)."""
    t = np.unique(np.asarray(t_meas_ms, dtype=np.float64))
    n = int(t.size)
    T = int(T)
    lt = np.log10(t)
    ext = float(t_ext_ms) > float(t[-1]) * (1.0 + 1e-9)
    lx = math.log10(float(t_ext_ms)) if ext else float(lt[-1])
    if T >= n + int(ext):
        seg = np.diff(lt)
        if ext:
            seg = np.append(seg, lx - lt[-1])
        extra = T - n - int(ext)
        quota = extra * seg / max(seg.sum(), 1e-30)
        base = np.floor(quota).astype(np.int64)
        rem = int(extra - base.sum())
        if rem > 0:
            order = np.argsort(-(quota - base), kind="mergesort")
            base[order[:rem]] += 1
        if ext and base[-1] < min(2, extra):
            # the extension keeps >= 2 inner nodes (taken from the gate intervals with the most nodes)
            need = int(min(2, extra) - base[-1])
            for _ in range(need):
                k = int(np.argmax(base[:-1]))
                if base[k] <= 0:
                    break
                base[k] -= 1
                base[-1] += 1
        pts: List[float] = [lt[0]]
        gate_index = [0]
        for k in range(n - 1):
            m = int(base[k])
            if m > 0:
                pts.extend(list(lt[k] + seg[k] * (np.arange(1, m + 1) / (m + 1.0))))
            pts.append(lt[k + 1])
            gate_index.append(len(pts) - 1)
        if ext:
            m = int(base[n - 1])
            if m > 0:
                pts.extend(list(lt[-1] + seg[n - 1] * (np.arange(1, m + 1) / (m + 1.0))))
            pts.append(lx)
        grid = 10.0 ** np.asarray(pts, dtype=np.float64)
        gate_index_arr = np.asarray(gate_index, dtype=np.int64)
        grid[gate_index_arr] = t  # exact measured values at the embedded nodes
        if ext:
            grid[-1] = float(t_ext_ms)
        embedded = True
    else:
        grid = 10.0 ** np.linspace(lt[0], lx, T)
        grid[0], grid[-1] = t[0], (float(t_ext_ms) if ext else t[-1])
        gate_index_arr = np.full(n, -1, dtype=np.int64)
        gate_index_arr[0] = 0
        if not ext:
            gate_index_arr[-1] = T - 1
        embedded = False
    steps = np.diff(np.log10(grid))
    info = {"target_len": T, "n_measured_gates": n, "all_gates_embedded": bool(embedded),
            "t_first_ms": float(grid[0]), "t_last_ms": float(grid[-1]),
            "log10_step_min": float(steps.min()), "log10_step_max": float(steps.max()),
            "log10_step_mean": float(steps.mean()), "grid_sha16": array_digest(grid)}
    if ext:
        info["late_time_extension"] = {"t_last_measured_ms": float(t[-1]), "t_ext_ms": float(t_ext_ms),
                                       "decades": float(lx - lt[-1]),
                                       "n_nodes_beyond_last_gate": int(np.sum(grid > t[-1] * (1.0 + 1e-9)))}
        if not embedded:
            info["late_time_extension"]["warning"] = (
                f"{n} gates need target_len >= {n + 1} to stay grid nodes with a late-time extension; at {T} the grid is "
                f"log-uniform and the measured gates are interpolated (see max_readback_error_decades)")
    return grid, gate_index_arr, info


def _pchip_edge_slope(h0: float, h1: float, m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    flip = np.sign(d) != np.sign(m0)
    over = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
    d = np.where(flip, 0.0, d)
    d = np.where((~flip) & over, 3.0 * m0, d)
    return d


def pchip_slopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone slopes for many curves sharing knots x. Y: (N, n)."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    h = np.diff(x)
    delta = np.diff(Y, axis=1) / h[None, :]
    N, n = Y.shape
    m = np.zeros_like(Y)
    if n == 2:
        m[:] = delta[:, :1]
        return m
    w1 = (2.0 * h[1:] + h[:-1])[None, :]
    w2 = (h[1:] + 2.0 * h[:-1])[None, :]
    d0, d1 = delta[:, :-1], delta[:, 1:]
    same = (d0 * d1) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = (w1 + w2) / (w1 / d0 + w2 / d1)
    m[:, 1:-1] = np.where(same, hm, 0.0)
    m[:, 0] = _pchip_edge_slope(h[0], h[1], delta[:, 0], delta[:, 1])
    m[:, -1] = _pchip_edge_slope(h[-1], h[-2], delta[:, -1], delta[:, -2])
    return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)


def pchip_eval(x: np.ndarray, Y: np.ndarray, m: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the PCHIP interpolant inside [x0, x_last] (xq outside is clamped to the range)."""
    x = np.asarray(x, dtype=np.float64)
    xq = np.clip(np.asarray(xq, dtype=np.float64), x[0], x[-1])
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    h = x[k + 1] - x[k]
    s = (xq - x[k]) / h
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    return (h00[None, :] * Y[:, k] + (h10 * h)[None, :] * m[:, k]
            + h01[None, :] * Y[:, k + 1] + (h11 * h)[None, :] * m[:, k + 1])


def resample_loglog(t_src_ms: np.ndarray, L_src: np.ndarray, t_dst_ms: np.ndarray,
                    extrapolation: str = "loglinear", edge_frac: float = 0.10,
                    slope_bounds: Tuple[float, float] = (-8.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Re-sample log10-amplitude curves L_src (N, n) given at t_src onto t_dst (shape-preserving PCHIP in
    log10 t).  Outside the source span: log-log linear continuation with the end slope fitted by least
    squares on the outer edge_frac of the source gates (bounded), or constant ('edge').
    Returns (L_dst (N, T), native_mask (T,) 1 inside the source span, info)."""
    xs = np.log10(np.asarray(t_src_ms, dtype=np.float64))
    xq = np.log10(np.asarray(t_dst_ms, dtype=np.float64))
    L = np.asarray(L_src, dtype=np.float64)
    m = pchip_slopes(xs, L)
    out = pchip_eval(xs, L, m, xq)
    inside = (xq >= xs[0] - 1e-12) & (xq <= xs[-1] + 1e-12)
    info = {"n_left_extrapolated": int(np.sum(xq < xs[0] - 1e-12)),
            "n_right_extrapolated": int(np.sum(xq > xs[-1] + 1e-12)),
            "left_extrapolation_decades": float(max(0.0, xs[0] - xq.min())),
            "right_extrapolation_decades": float(max(0.0, xq.max() - xs[-1])), "mode": extrapolation}
    k = max(3, int(round(edge_frac * xs.size)))
    for side in ("left", "right"):
        sel = xq < xs[0] - 1e-12 if side == "left" else xq > xs[-1] + 1e-12
        if not np.any(sel):
            continue
        if extrapolation == "edge":
            out[:, sel] = (L[:, :1] if side == "left" else L[:, -1:])
            continue
        xx = xs[:k] if side == "left" else xs[-k:]
        yy = L[:, :k] if side == "left" else L[:, -k:]
        xc = xx - xx.mean()
        slope = ((yy - yy.mean(axis=1, keepdims=True)) * xc[None, :]).sum(axis=1) / max(float((xc ** 2).sum()), 1e-30)
        slope = np.clip(slope, slope_bounds[0], slope_bounds[1])
        x0 = xs[0] if side == "left" else xs[-1]
        y0 = L[:, 0] if side == "left" else L[:, -1]
        out[:, sel] = y0[:, None] + slope[:, None] * (xq[sel] - x0)[None, :]
        info[f"{side}_slope_median"] = float(np.median(slope))
    return out, inside.astype(np.float64), info


def depth_interp_profile(depth_knots: np.ndarray, L_knots: np.ndarray, depth_q: np.ndarray) -> np.ndarray:
    """Shape-preserving interpolation of a (D, T) log-amplitude profile along depth at depth_q (any shape)."""
    dq = np.asarray(depth_q, dtype=np.float64)
    flat = dq.reshape(-1)
    LT = np.asarray(L_knots, dtype=np.float64).T          # (T, D)
    m = pchip_slopes(depth_knots, LT)
    out = pchip_eval(depth_knots, LT, m, flat).T           # (Q, T)
    return out.reshape(dq.shape + (LT.shape[0],))


def _box_smooth_cols(A: np.ndarray, radius: int) -> np.ndarray:
    """Edge-padded moving average along axis 0 (O(N))."""
    if radius <= 0:
        return A.copy()
    pad = np.pad(A, ((radius, radius), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, A.shape[1])), pad]), axis=0)
    w = 2 * radius + 1
    return (c[w:] - c[:-w]) / w


def gaussian_smooth_depth(A: np.ndarray, depth: np.ndarray, sigma_m: float) -> np.ndarray:
    """Gaussian-like smoothing along depth (three box passes) on a uniform resampling of depth."""
    A = np.asarray(A, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    n_u = int(min(max(4 * d.size, 256), 4096))
    du = np.linspace(d[0], d[-1], n_u)
    Au = np.stack([np.interp(du, d, A[:, j]) for j in range(A.shape[1])], axis=1)
    step = (d[-1] - d[0]) / max(n_u - 1, 1)
    # three box passes of radius r have variance r(r+1) samples^2 -> r ~ sigma/step - 0.5
    r = int(max(0, round(sigma_m / max(step, 1e-12) - 0.5)))
    S = Au
    for _ in range(3):
        S = _box_smooth_cols(S, r)
    return np.stack([np.interp(d, du, S[:, j]) for j in range(A.shape[1])], axis=1)


def robust_depth_trend(L: np.ndarray, depth: np.ndarray, sigma_m: float, iters: int = 3,
                       c: float = 2.5) -> np.ndarray:
    """Background (trend) of a log-amplitude profile along depth: Gaussian smoothing with Cauchy
    re-weighting so localized anomalies do not leak into the background."""
    L = np.asarray(L, dtype=np.float64)
    trend = gaussian_smooth_depth(L, depth, sigma_m)
    for _ in range(int(iters)):
        r = L - trend
        mad = 1.4826 * np.median(np.abs(r - np.median(r, axis=0, keepdims=True)), axis=0, keepdims=True) + 1e-6
        w = 1.0 / (1.0 + (r / (c * mad)) ** 2)
        num = gaussian_smooth_depth(w * L, depth, sigma_m)
        den = gaussian_smooth_depth(w, depth, sigma_m)
        trend = num / np.maximum(den, 1e-6)
    return trend
