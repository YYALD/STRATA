"""Stations and times without a measurement: the anchor rule beyond the measured stations, its spread calibrated on
the stations themselves (pseudo hold-out), the late-time extension and the virtual stations beyond the measured span."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .time_grid import depth_interp_profile


# A measured station is a hard constraint of the anchor.
PROFILE_EXTRAP_RULES = ("clamp", "trend10", "trend20", "trend30", "trend50")
TIME_EXTRAP_RULES = ("loglin10", "loglin20", "loglin30")


def _extrap_side(kd: np.ndarray, kL: np.ndarray, dq: np.ndarray, rule: str, side: str) -> np.ndarray:
    """(len(dq), T) values beyond one end of the known stations kd (ascending): 'clamp' continues the end station's
    curve (level continuation), 'trendNN' continues it with the linear profile-axis trend fitted per time sample on the
    outer NN % of the stations (>= 3)."""
    kd = np.asarray(kd, dtype=np.float64)
    kL = np.asarray(kL, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64).reshape(-1)
    e = -1 if side == "high" else 0
    y0 = kL[e]
    if rule == "clamp" or kd.size < 3:
        return np.repeat(y0[None, :], dq.size, axis=0)
    n = min(kd.size, max(3, int(round(int(rule[5:]) / 100.0 * kd.size))))
    xs, ys = (kd[-n:], kL[-n:]) if side == "high" else (kd[:n], kL[:n])
    xc = xs - xs.mean()
    sl = ((ys - ys.mean(axis=0)) * xc[:, None]).sum(axis=0) / max(float((xc * xc).sum()), 1e-30)
    return y0[None, :] + sl[None, :] * (dq - kd[e])[:, None]


def anchor_profile_on(kd: np.ndarray, kL: np.ndarray, dq: np.ndarray,
                      rules: Sequence[str] = ("clamp", "clamp")) -> np.ndarray:
    """Anchor profile (len(dq), T) at the 1-D positions dq: PCHIP along the profile axis between the known stations
    kd, the per-side extrapolation rule (low end, high end) beyond them."""
    dq = np.asarray(dq, dtype=np.float64).reshape(-1)
    kd = np.asarray(kd, dtype=np.float64)
    out = depth_interp_profile(kd, kL, dq)
    lo, hi = dq < kd[0], dq > kd[-1]
    if np.any(lo):
        out[lo] = _extrap_side(kd, kL, dq[lo], rules[0], "low")
    if np.any(hi):
        out[hi] = _extrap_side(kd, kL, dq[hi], rules[1], "high")
    return out


def _fit_power_rms(dist: np.ndarray, err: np.ndarray, n_bins: int = 8) -> Dict[str, Any]:
    """RMS(dist) = a_t dist^gamma from the errors err (N, T) at the distances dist (N,): RMS per quantile bin of dist and
    time sample, gamma pooled over the time samples (log-log least squares with one intercept per sample, clipped to
    [0.5, 1.5]), a_t smoothed over 5 neighbouring time samples."""
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64).reshape(dist.size, -1)
    ok = dist > 0
    dist, err = dist[ok], err[ok]
    Tn = int(err.shape[1])
    if dist.size < 4:
        return {"a": np.zeros(Tn), "gamma": 1.0, "bins": []}
    edges = np.unique(np.quantile(dist, np.linspace(0.0, 1.0, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, max(edges.size - 2, 0))
    cen: List[float] = []
    rms: List[np.ndarray] = []
    cnt: List[int] = []
    for k in range(max(edges.size - 1, 1)):
        s = b == k
        if int(s.sum()) >= 2:
            cen.append(float(np.exp(np.mean(np.log(dist[s])))))
            rms.append(np.sqrt(np.mean(err[s] ** 2, axis=0)))
            cnt.append(int(s.sum()))
    if len(cen) < 2:
        g = 1.0
        a = np.sqrt(np.mean(err ** 2, axis=0)) / max(float(np.mean(dist)), 1e-9)
    else:
        lx = np.log(np.asarray(cen))
        LY = np.log(np.maximum(np.stack(rms), 1e-9))                     # (bins, T)
        xc = lx - lx.mean()
        g = float(np.sum(xc[:, None] * (LY - LY.mean(axis=0))) / max(float(np.sum(xc * xc)) * Tn, 1e-12))
        g = float(np.clip(g, 0.5, 1.5))
        a = np.exp(np.mean(LY - g * lx[:, None], axis=0))
    k = min(5, Tn)
    if k > 1:
        a = np.convolve(np.pad(a, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return {"a": np.asarray(a, dtype=np.float64), "gamma": g,
            "bins": [{"distance": c, "n": n, "rms_median_decades": float(np.median(r))} for c, r, n in zip(cen, rms, cnt)]}


def calibrate_unmeasured(kd: np.ndarray, kL: np.ndarray, frac_max: float = 0.25,
                         rules: Sequence[str] = PROFILE_EXTRAP_RULES) -> Dict[str, Any]:
    """Rules and spreads for positions without a measured station, from the known stations themselves (pseudo hold-out).
    Ends: hide the h outermost stations (h = 1 .. frac_max x D) at each end, extrapolate them with every rule from the
    rest; the rule with the lowest RMS error over all h is that end's rule, and its errors against the distance to the
    remaining end give RMS = a_t dist^gamma.  Gaps: hide interior blocks of h stations (all start positions, thinned to
    <= 48 per h), interpolate across them (PCHIP) and fit the errors against the bridge distance d1 d2 / (d1 + d2)."""
    kd = np.asarray(kd, dtype=np.float64)
    kL = np.asarray(kL, dtype=np.float64)
    D, Tn = int(kd.size), int(kL.shape[1])
    H = max(1, int(round(float(frac_max) * D)))
    out: Dict[str, Any] = {"n_known": D, "max_hidden_stations": H}
    for side in ("low", "high"):
        best = None
        table: Dict[str, float] = {}
        for rule in rules:
            dists, errs = [], []
            for h in range(1, H + 1):
                if D - h < 3:
                    break
                if side == "high":
                    kk, hh, edge = slice(0, D - h), slice(D - h, D), kd[D - h - 1]
                else:
                    kk, hh, edge = slice(h, D), slice(0, h), kd[h]
                dists.append(np.abs(kd[hh] - edge))
                errs.append(_extrap_side(kd[kk], kL[kk], kd[hh], rule, side) - kL[hh])
            if not dists:
                continue
            dist, err = np.concatenate(dists), np.concatenate(errs)
            table[rule] = float(np.sqrt(np.mean(err ** 2)))
            if best is None or table[rule] < best[1] - 1e-12:
                best = (rule, table[rule], dist, err)
        if best is None:
            out[side] = {"rule": "clamp", "rms_decades": float("nan"), "rules": table, "a": np.zeros(Tn), "gamma": 1.0,
                         "bins": []}
        else:
            out[side] = {"rule": best[0], "rms_decades": best[1], "rules": table, **_fit_power_rms(best[2], best[3])}
    dists, errs = [], []
    for h in range(1, H + 1):
        starts = np.arange(1, D - h)              # block [s, s + h) keeps a known station on each side
        if starts.size == 0:
            break
        for s in starts[:: max(1, int(math.ceil(starts.size / 48)))]:
            keep = np.ones(D, dtype=bool)
            keep[s:s + h] = False
            d1, d2 = kd[~keep] - kd[s - 1], kd[s + h] - kd[~keep]
            dists.append(d1 * d2 / np.maximum(d1 + d2, 1e-12))
            errs.append(depth_interp_profile(kd[keep], kL[keep], kd[~keep]) - kL[~keep])
    if dists:
        e_all = np.concatenate(errs)
        out["gap"] = {"rule": "pchip", "rms_decades": float(np.sqrt(np.mean(e_all ** 2))),
                      **_fit_power_rms(np.concatenate(dists), e_all)}
    else:
        out["gap"] = {"rule": "pchip", "rms_decades": float("nan"), "a": np.zeros(Tn), "gamma": 1.0, "bins": []}
    return out


def extend_late_time(lt_k: np.ndarray, Lk: np.ndarray, lt_q: np.ndarray, rule: str = "loglin10") -> np.ndarray:
    """Decays continued beyond the last kept time (log10 ms lt_k ascending, Lk (N, n) log10 amplitude): per curve the
    log-log slope fitted on the last NN % of the kept times (>= 3), clipped to the p5..p95 band of the curves' slopes and
    to <= 0 (a decay does not rise), continued linearly in log t from the last kept value."""
    lt_k = np.asarray(lt_k, dtype=np.float64)
    Lk = np.asarray(Lk, dtype=np.float64)
    n = min(lt_k.size, max(3, int(round(int(rule[6:]) / 100.0 * lt_k.size))))
    xs, ys = lt_k[-n:], Lk[:, -n:]
    xc = xs - xs.mean()
    sl = ((ys - ys.mean(axis=1, keepdims=True)) * xc[None, :]).sum(axis=1) / max(float((xc * xc).sum()), 1e-30)
    if sl.size >= 5:
        lo, hi = np.percentile(sl, [5, 95])
        sl = np.clip(sl, lo - 0.5, hi + 0.5)
    sl = np.minimum(sl, 0.0)
    return Lk[:, -1:] + sl[:, None] * (np.asarray(lt_q, dtype=np.float64).reshape(1, -1) - lt_k[-1])


def calibrate_late_extension(t_ms: np.ndarray, L: np.ndarray, ext_decades: float,
                             rules: Sequence[str] = TIME_EXTRAP_RULES) -> Dict[str, Any]:
    """Late-time extension rule and spread from the measured gates (pseudo hold-out): hide the gates after each cut in the
    last h_max = max(ext, 0.3) decades (at most half the measured window, >= 4 gates kept), continue every curve with each
    rule from the gates before the cut, keep the rule with the lowest RMS error, and fit RMS(dlog t) = a dlog t^gamma to
    its errors (one curve for all stations: a, gamma scalars)."""
    lt = np.log10(np.asarray(t_ms, dtype=np.float64))
    L = np.asarray(L, dtype=np.float64)
    W = float(lt[-1] - lt[0])
    hmax = min(max(float(ext_decades), 0.3), 0.5 * W)
    cuts = [c for c in lt[:-1] if c >= lt[-1] - hmax - 1e-12 and int(np.sum(lt <= c + 1e-12)) >= 4]
    best = None
    table: Dict[str, float] = {}
    for rule in rules:
        dists, errs = [], []
        for c in cuts:
            keep = lt <= c + 1e-12
            e = extend_late_time(lt[keep], L[:, keep], lt[~keep], rule) - L[:, ~keep]
            dists.append(np.broadcast_to(lt[~keep] - c, e.shape).reshape(-1))
            errs.append(e.reshape(-1, 1))
        if not dists:
            continue
        dist, err = np.concatenate(dists), np.concatenate(errs)
        table[rule] = float(np.sqrt(np.mean(err ** 2)))
        if best is None or table[rule] < best[1] - 1e-12:
            best = (rule, table[rule], dist, err)
    if best is None:
        # no cut keeps >= 4 gates: no pseudo hold-out is possible; the spread falls back to half the p5-p95 range of the
        # curves' own late slopes per decade of extension (an uncalibrated but non-zero uncertainty, reported)
        n = min(lt.size, max(3, int(round(0.1 * lt.size))))
        xc = lt[-n:] - lt[-n:].mean()
        sl = ((L[:, -n:] - L[:, -n:].mean(axis=1, keepdims=True)) * xc[None, :]).sum(axis=1) / max(float((xc * xc).sum()), 1e-30)
        a_fb = float(0.5 * (np.percentile(sl, 95) - np.percentile(sl, 5))) if sl.size >= 2 else 0.0
        return {"rule": "loglin10", "rms_decades": float("nan"), "rules": table, "a": a_fb, "gamma": 1.0, "bins": [],
                "hidden_decades_max": hmax,
                "fallback": f"{lt.size} gates: no pseudo hold-out cut keeps >= 4 gates; spread = half the p5-p95 range of "
                            f"the curves' late slopes ({a_fb:.4f} decades per decade), not calibrated"}
    fit = _fit_power_rms(best[2], best[3])
    return {"rule": best[0], "rms_decades": best[1], "rules": table, "a": float(fit["a"][0]), "gamma": fit["gamma"],
            "bins": fit["bins"], "hidden_decades_max": hmax}


def extension_stations(depth: np.ndarray, frac: float, side: str, nonnegative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Virtual stations beyond the measured span: frac x span per extended end ('low', 'high', 'both'), spaced like the
    measured stations at that end (median of its three outer spacings, at least the profile's median spacing, so a densely
    sampled end does not multiply the stations); nonnegative keeps borehole depths >= 0.
    Returns (sorted axis = measured + virtual stations, known mask)."""
    d = np.asarray(depth, dtype=np.float64)
    if float(frac) <= 0 or d.size < 2:
        return d.copy(), np.ones(d.size, dtype=bool)
    Lx = float(frac) * float(d[-1] - d[0])
    sp_med = float(np.median(np.diff(d)))
    lo_part, hi_part = np.zeros(0), np.zeros(0)
    if side in ("low", "both"):
        sp = max(float(np.median(np.diff(d[:4]))), sp_med)
        n = max(1, int(round(Lx / max(sp, 1e-9))))
        lo_part = d[0] - sp * np.arange(n, 0, -1, dtype=np.float64)
        if nonnegative:
            lo_part = lo_part[lo_part >= 0.0]
    if side in ("high", "both"):
        sp = max(float(np.median(np.diff(d[-4:]))), sp_med)
        n = max(1, int(round(Lx / max(sp, 1e-9))))
        hi_part = d[-1] + sp * np.arange(1, n + 1, dtype=np.float64)
    ax = np.concatenate([lo_part, d, hi_part])
    known = np.zeros(ax.size, dtype=bool)
    known[lo_part.size: lo_part.size + d.size] = True
    return ax, known
