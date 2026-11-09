"""Laws of the synthetic anomalies: time constants, amplitudes, counts, lateral correlation, template widths, the
scale ranges of the measured anomalies and the axial kernel of a conductor beside the hole."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..common import _parse_mix
from .contract import local_structure_residual
from .time_grid import robust_depth_trend


def anomaly_tau_range_log10(grid_ms: np.ndarray, spec: str = "auto") -> Tuple[float, float]:
    """log10 range (ms) of the anomalies' eddy-current time constants.  auto = [t_first, 2 t_last] of the
    measured window: from responses that have decayed after the first gates (poor conductors) to responses
    still rising at the last gate (good conductors)."""
    s = str(spec or "auto").strip().lower()
    if s == "auto":
        lo, hi = float(grid_ms[0]), 2.0 * float(grid_ms[-1])
    else:
        parts = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
        if len(parts) != 2 or not (0.0 < parts[0] <= parts[1]):
            raise ValueError(f"--anomaly_tau_range expects 'lo,hi' in ms with 0 < lo <= hi or auto, got {spec!r}")
        lo, hi = parts
    return float(np.log10(lo)), float(np.log10(hi))


def lateral_correlation_law(L: np.ndarray, x: np.ndarray, t_ms: np.ndarray) -> Dict[str, Any]:
    """Lateral scale of a line profile versus time: per time sample (<= 32 samples), the residual from a robust trend
    (sigma = span / 4) is resampled on a uniform axis and its correlation length is the e^-1/2 crossing of the
    autocorrelation; l(t) = l1 (t / t1)^beta is fitted in log-log with beta in [0, 0.5] (the footprint of the induced
    currents widens at most like the diffusion length, ~ sqrt t)."""
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t_ms = np.asarray(t_ms, dtype=np.float64)
    span = float(x[-1] - x[0])
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    R = L - robust_depth_trend(L, x, 0.25 * span)
    xu = np.arange(x[0], x[-1] + 0.5 * dx, dx)
    cols = np.unique(np.linspace(0, L.shape[1] - 1, min(32, L.shape[1])).round().astype(np.int64))
    ell = []
    for j in cols:
        r = np.interp(xu, x, R[:, j])
        r = r - r.mean()
        den = float(r @ r)
        if den <= 1e-20 or r.size < 4:
            ell.append(float("nan"))
            continue
        if r.size <= 4096:
            ac = np.correlate(r, r, "full")[r.size - 1:] / den
        else:   # long unstacked lines: the same autocorrelation through the FFT (O(N log N))
            nf = 1 << int(math.ceil(math.log2(2 * r.size)))
            ac = np.fft.irfft(np.abs(np.fft.rfft(r, nf)) ** 2, nf)[:r.size] / den
        below = np.nonzero(ac < math.exp(-0.5))[0]
        ell.append(float(max(int(below[0]) if below.size else r.size - 1, 1)) * dx)
    ell_a = np.asarray(ell)
    lt = np.log10(t_ms[cols])
    ok = np.isfinite(ell_a) & (ell_a > 0)
    if int(ok.sum()) >= 3:
        xc = lt[ok] - lt[ok].mean()
        beta = float(np.sum(xc * (np.log10(ell_a[ok]) - np.log10(ell_a[ok]).mean())) / max(float(np.sum(xc * xc)), 1e-12))
        beta = float(np.clip(beta, 0.0, 0.5))
        l1 = float(10.0 ** np.mean(np.log10(ell_a[ok]) - beta * (lt[ok] - lt[0])))
    else:
        beta, l1 = 0.0, float(max(0.1 * span, dx))
    return {"l1_m": l1, "beta": beta, "t1_ms": float(t_ms[0]),
            "samples": [{"t_ms": float(t_ms[c]), "length_m": float(v)} for c, v in zip(cols, ell_a)]}


def anomaly_amp_range(L: np.ndarray, depth: np.ndarray, spec: str = "auto", family: str = "borehole",
                      t_ms: Optional[np.ndarray] = None, visible: bool = False) -> Tuple[float, float]:
    """|peak amplitude| range (decades) of the random profile anomalies. auto, borehole family: anchored to the
    measured profile's own local depth structure — the lower end is its p95 (a barely visible departure), the upper
    end three times its strongest local feature, clipped to [0.10, 0.50] decades."""
    s = str(spec or "auto").strip().lower()
    if s != "auto":
        parts = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
        if len(parts) != 2 or not (0.0 < parts[0] <= parts[1]):
            raise ValueError(f"--anomaly_amp_range expects 'lo,hi' with 0 < lo <= hi (decades) or auto, got {spec!r}")
        return float(parts[0]), float(parts[1])
    if family == "buried" and np.asarray(depth).size >= 8:
        L = np.asarray(L, dtype=np.float64)
        tt = np.asarray(t_ms, dtype=np.float64) if t_ms is not None else np.logspace(0.0, 1.0, L.shape[1])
        law = lateral_correlation_law(L, depth, tt)
        r = np.abs(L - robust_depth_trend(L, np.asarray(depth, dtype=np.float64), 2.0 * law["l1_m"]))
        hi = float(np.clip(np.percentile(r, 99), 0.10, 0.50))
        lo = float(np.clip(np.percentile(r, 50), 0.01, hi / 4.0))
        return lo, hi
    r = np.abs(local_structure_residual(np.asarray(L, dtype=np.float64), depth))
    if r.size == 0:
        return (0.1, 1.0) if visible else (0.02, 0.2)
    if visible:
        lo = float(np.clip(r.max(), 0.05, 0.20))
        return lo, float(np.clip(10.0 * lo, 0.5, 1.0))
    hi = float(np.clip(3.0 * r.max(), 0.10, 0.50))
    lo = float(np.clip(np.percentile(r, 95), 0.01, hi / 4.0))
    return lo, hi


def borehole_physical_anomalies(cfg: "AppConfig", family: str) -> bool:
    """True when the synthetic anomalies follow the physical borehole law (anomaly family borehole,
    --borehole_anomaly_model physical, or measured = that law calibrated on the measured anomalies)."""
    return family == "borehole" and str(cfg.profile.borehole_anomaly_model).strip().lower() in ("physical", "measured")


V19_ANOMALY_COUNT_PROBS = "0:0.2,1:0.35,2:0.25,3:0.12,4:0.08"
MEASURED_LIKE_SIZE_RANGE = (0.75, 1.5)      # along-hole size factor of a measured anomaly shape at another depth
SEPARATION_CANDIDATES = 32                  # candidate centres per synthetic anomaly for the separation rule
SEPARATION_CORE_SIGMAS = 2.0                # core of an anomaly: +- 2 x its width (sigma / distance)
UNION_REFERENCE_PROFILES = 24              # random anchor profiles in the physics-descriptor reference
UNION_REFERENCE_PROFILES_MEASURED = 96     # measured model: up to five strong anomalies per profile - with 24 anchors
                                           # the envelope (0.5-99.5 %) left out anomaly-rich profiles' curves
UNDULATION_LOCAL_POINTS = 64                # points of the local axis of the undulation check (<= width / 8)
UNDULATION_REDRAWS = 4                      # redraws of amplitude / tau for a synthetic anomaly that does not make the
                                            # profile rise against its trend by half the visibility floor


def measured_anomaly_mode(cfg: "AppConfig", family: str) -> str:
    """--measured_anomaly_mode resolved: auto = count for the borehole family, all for the buried family."""
    m = str(cfg.profile.measured_anomaly_mode or "auto").strip().lower()
    if m == "auto":
        m = "count" if str(family) == "borehole" else "all"
    return m


def anomaly_count_law(spec: str, max_anomalies: int, mode: str) -> List[Tuple[int, float]]:
    """(count, weight) pairs of the number of anomalies per profile.  auto: measured_anomaly_mode count = every count
    0 .. max_anomalies equally often (the count includes the measured anomalies). An explicit 'k:w,...' list is used
    as given; in count mode its counts must not exceed max_anomalies."""
    m = str(mode or "count").strip().lower()
    kmax = int(max_anomalies)
    if kmax < 0:
        raise ValueError(f"--max_anomalies expects >= 0, got {max_anomalies}")
    s = str(spec or "auto").strip().lower()
    if s in ("", "auto"):
        if m == "count":
            return [(k, 1.0) for k in range(kmax + 1)]
        s = V19_ANOMALY_COUNT_PROBS
    cnt = sorted((int(float(k)), float(w)) for k, w in _parse_mix(s, "0:1.0"))
    if any(k < 0 for k, _ in cnt) or sum(max(w, 0.0) for _, w in cnt) <= 0:
        raise ValueError(f"--anomaly_count_probs expects 'count:weight,...' with counts >= 0 and a positive total, got {spec!r}")
    if m == "count" and any(k > kmax and w > 0 for k, w in cnt):
        raise ValueError(f"--anomaly_count_probs {spec!r} has counts above --max_anomalies {kmax} (the count includes the "
                         f"measured anomalies; raise --max_anomalies or drop those counts)")
    return cnt


def template_width_sigma(A: np.ndarray, x: np.ndarray) -> float:
    """Gaussian-equivalent width (FWHM / 2.355, m) of an anomaly template A (D, T) along x at the time sample of its
    largest |value|: the same-signed part around the peak, half-maximum crossings interpolated linearly."""
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if A.size == 0 or not np.any(A != 0):
        return float("nan")
    i, g = np.unravel_index(int(np.argmax(np.abs(A))), A.shape)
    p = np.clip(A[:, g] * np.sign(A[i, g]), 0.0, None)
    half = 0.5 * p[i]
    lo, hi = i, i
    while lo > 0 and p[lo - 1] >= half:
        lo -= 1
    while hi < p.size - 1 and p[hi + 1] >= half:
        hi += 1

    def _cross(a: int, b: int) -> float:
        ya, yb = p[a] - half, p[b] - half
        return float(x[a] + (x[b] - x[a]) * ya / (ya - yb)) if ya != yb else float(x[a])

    left = _cross(lo, lo - 1) if lo > 0 else float(x[0])
    right = _cross(hi, hi + 1) if hi < p.size - 1 else float(x[-1])
    return float(max(right - left, 1e-6) / 2.355)


def measured_anomaly_scale_ranges(runs: Sequence[Dict[str, Any]], spec: str, a_vis: float) -> List[Tuple[float, float]]:
    """Amplitude factor range (lo, hi) of each measured anomaly.  'lo,hi' = the same range for all; auto = from half
    its measured amplitude up to max(2, the factor that lifts its peak to twice the visibility floor a_vis of the
    synthetic anomalies): a slight measured anomaly is also shown clearly in part of the profiles, its shape always
    the measured one."""
    s = str(spec or "auto").strip().lower()
    if s in ("", "auto"):
        return [(0.5, float(max(2.0, 2.0 * float(a_vis) / max(abs(float(r_["peak_decades"])), 1e-9)))) for r_ in runs]
    sc = [float(v) for v in s.replace(";", ",").split(",") if v.strip()]
    if len(sc) != 2 or not 0.0 < sc[0] <= sc[1]:
        raise ValueError(f"--measured_anomaly_scale expects 'lo,hi' with 0 < lo <= hi or auto, got {spec!r}")
    return [(sc[0], sc[1]) for _ in runs]


def dipole_axial_kernel(u: torch.Tensor, dip: torch.Tensor) -> torch.Tensor:
    """Along-hole profile of the axial field of a conductor beside the hole (dipole at distance r, moment tilted by
    dip from the hole axis), normalised by r^3: u = (z - z0) / r,
        K(u) = [cos(dip) (2 u^2 - 1) - 3 u sin(dip)] / (1 + u^2)^(5/2) * exp(-u^2 / 8).
    dip 0 (flat body beside the hole): three peaks, the centre negative (-1 at u = 0, crossovers at u = +-1/sqrt 2, side
    lobes +0.20 at u = +-1.22); dip 90 deg: a positive / negative pair; in between asymmetric (forward models of plates
    beside / at the edge of a hole: single peak when the hole cuts the body, a negative-centred three-peak anomaly at the
    edge of a flat body, positive / negative pairs for dipping bodies).  The Gaussian taper (sigma 2 r) keeps the label a
    compact anomaly: the dipole tail beyond ~2 r is a few per cent of the centre."""
    u2 = u * u
    return (torch.cos(dip) * (2.0 * u2 - 1.0) - 3.0 * u * torch.sin(dip)) / (1.0 + u2).pow(2.5) * torch.exp(-u2 / 8.0)
