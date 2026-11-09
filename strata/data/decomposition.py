"""The measured profile as geological background plus anomalies: evidence for the decay constraint, the smooth
background, the two-sided decay trend and its undulations (the measured anomalies), anomaly templates and
inventories."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anomaly_laws import lateral_correlation_law
from .readers import _strict_float


# A profile without anomalies is the response of the host geology: smooth along the hole / line (it changes with the
# distance to the transmitter and with the layering, over many stations) and a smooth decay in time.
DECAY_MONOTONE_MIN = 0.99          # share of the measured gate pairs that must not rise beyond the noise (hard)
_DECAY = {"data_monotone": True}   # running minimum over time in the data-level fits of this run (hard constraint)


def _tmono_np(A: np.ndarray, axis: int = 1) -> np.ndarray:
    """Running minimum along time when the run's decay constraint is hard, else A."""
    return np.minimum.accumulate(A, axis=axis) if _DECAY["data_monotone"] else A


def decay_constraint_evidence(meas: "MeasuredData", cfg: "AppConfig", verbose: bool = True) -> Dict[str, Any]:
    """Whether the time-monotone constraint applies to the measured quantity of this survey (decay_constraint.json).
    A non-increasing decay is a property of a positive amplitude of a step-off response in a conductive, non-polarisable
    host (the coaxial |dB/dt| or |B|), not of TEM data in general: signed components change sign with the source - receiver
    - conductor geometry, induced polarisation reverses late-time signs, the in-line / transverse borehole components are
    often not monotone.  The test runs on the measured gates of the measured window (log10): a rise between consecutive
    gates counts when it exceeds the gate-to-gate noise, 3 sqrt(2) x the robust scale of the second time difference /
    sqrt(6).
      auto : hard when the values are positive and >= DECAY_MONOTONE_MIN of the gate pairs pass; soft when positive and
             fewer pass; off for signed data (or --signal_quantity signed);
      hard / soft / off : as forced (hard on signed data is refused; on positive data that fail the test it is applied as
             requested and reported)."""
    spec = str(cfg.generation.decay_constraint or "auto").strip().lower()
    qty = str(cfg.generation.signal_quantity or "auto").strip().lower()
    gi = np.asarray(meas.gate_index)
    kn = meas.known_time_mask()
    cols = gi[kn[gi]] if bool(np.all(gi >= 0)) else np.nonzero(kn)[0]
    rep: Dict[str, Any] = {"requested": spec, "signal_quantity": qty, "positive": bool(meas.positive),
                           "n_stations": int(meas.D), "n_gates": int(cols.size)}
    frac = float("nan")
    if meas.positive and cols.size >= 3:
        X = np.asarray(meas.L[:, cols], dtype=np.float64)
        d1 = np.diff(X, axis=1)
        d2 = X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]
        sig = 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6.0)
        tol = 3.0 * math.sqrt(2.0) * sig
        rise = d1 > tol
        frac = float(1.0 - rise.mean())
        rep.update({"gate_pairs": int(d1.size), "noise_decades": sig, "rise_tolerance_decades": tol,
                    "non_increasing_fraction": frac, "strict_non_increasing_fraction": float(np.mean(d1 <= 0.0)),
                    "largest_rise_decades": float(d1.max()), "stations_with_a_rise": int(rise.any(axis=1).sum())})
    signed = (not meas.positive) or qty == "signed"
    ok_hard = bool(not signed and np.isfinite(frac) and frac >= DECAY_MONOTONE_MIN)
    if spec == "auto":
        mode = "hard" if ok_hard else ("off" if signed else "soft")
        why = ("auto: positive values, %.2f%% of the gate pairs non-increasing within the noise (>= %.0f%%)"
               % (100.0 * frac, 100.0 * DECAY_MONOTONE_MIN) if ok_hard else
               "auto: signed data - no time constraint" if signed else
               "auto: positive values, but only %.2f%% of the gate pairs non-increasing within the noise (< %.0f%%): "
               "hinge penalty instead of a projection" % (100.0 * frac, 100.0 * DECAY_MONOTONE_MIN))
    elif spec in ("hard", "soft", "off"):
        if spec == "hard" and signed:
            raise ValueError("--decay_constraint hard: the measured quantity is signed - a running minimum over time is not "
                             "a constraint of signed data (use soft or off)")
        mode = spec
        why = "forced by --decay_constraint" + ("" if (spec != "hard" or ok_hard) else
                                                 " (the measured gates do not pass the test: applied as requested)")
    else:
        raise ValueError(f"--decay_constraint expects auto, hard, soft or off, got {spec!r}")
    rep.update({"mode": mode, "reason": why, "threshold_fraction": DECAY_MONOTONE_MIN,
                "applies_to": "positive, non-increasing amplitude decays (dB/dt or B of a step-off source in a "
                              "non-polarisable host); not signed components, not polarisable ground",
                "hard": "running minimum over time in every anchor / generated profile + the monotone recovery head",
                "soft": "no projection; hinge on rises in time (lambda_decay_soft)", "off": "no time constraint"})
    if verbose:
        ev = (f"{rep['non_increasing_fraction']:.2%} of {rep['gate_pairs']} gate pairs non-increasing within "
              f"{rep['rise_tolerance_decades']:.4f} decades (strict {rep['strict_non_increasing_fraction']:.2%}, largest rise "
              f"{rep['largest_rise_decades']:+.4f})" if "gate_pairs" in rep else "signed data")
        print(f"[DECAY] time-monotone constraint: {mode} | {ev} | {why}")
    return rep


def background_scale(L: np.ndarray, x: np.ndarray, t_ms: np.ndarray, survey_mode: str,
                     spec: str = "auto") -> Tuple[float, str]:
    """Smoothing scale (Gaussian sigma, m) of the background along the profile.  auto: borehole = 2 x the median station
    spacing (near-hole anomalies span 1-3 stations, the host response varies over tens of stations); airborne / ground =
    2 x the measured line's lateral correlation length l1 (conductor footprints span ~l1, the regional geology varies
    over kilometres)."""
    s_ = str(spec or "auto").strip().lower()
    x = np.asarray(x, dtype=np.float64)
    if s_ not in ("", "auto"):
        v = _strict_float(s_.replace("m", ""))
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"--background_sigma expects auto or a length in m > 0, got {spec!r}")
        return float(v), f"--background_sigma {v:g} m"
    if str(survey_mode) == "borehole" or x.size < 8:
        sp = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        return 2.0 * sp, f"borehole: 2 x the median station spacing ({sp:.4g} m)"
    law = lateral_correlation_law(np.asarray(L, dtype=np.float64), x, np.asarray(t_ms, dtype=np.float64))
    return 2.0 * float(law["l1_m"]), f"line: 2 x the measured lateral correlation length l1 ({law['l1_m']:.4g} m)"


def background_decomposition(L: np.ndarray, x: np.ndarray, t_ms: np.ndarray, sigma_m: float, deg: int = 2,
                             iters: int = 8, c: float = 4.685, w_time_dec: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth geological background of a log-amplitude profile L (D, T) along x (D,): robust local polynomial regression
    (degree 2, Gaussian kernel sigma_m) per time sample with Tukey biweight down-weighting of the anomalies (scale = MAD of
    the residual per time sample), started from the residual to a running repeated-median line (robust to compact
    anomalies).  The
    weights are smoothed along log time (Gaussian, w_time_dec decades) so neighbouring times share them: per-time weights
    alone made the background decays wiggle (time curvature 10 x the measured one on both local profiles).  The background
    is projected onto non-increasing decays.  Returns (background, weights)."""
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    D, T = L.shape
    if D < 4:
        return _tmono_np(L.copy(), axis=1), np.ones_like(L)
    xs = (x - x.mean()) / max(float(np.ptp(x)), 1e-9)
    K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / max(float(sigma_m), 1e-9)) ** 2)
    K[K < 1e-8] = 0.0
    lt = np.log10(np.asarray(t_ms, dtype=np.float64))
    if w_time_dec > 0 and T > 1:
        Kt = np.exp(-0.5 * ((lt[:, None] - lt[None, :]) / float(w_time_dec)) ** 2)
        Kt /= Kt.sum(axis=1, keepdims=True)
    else:
        Kt = np.eye(T)
    q = int(deg) + 1
    # robust start: the first weights come from the residual to a running repeated-median line along the profile
    # (Siegel's estimator, 50 % breakdown, over 2 m + 1 stations, m >= 3 and >= 1.5 sigma_m / spacing; the ends padded
    # by odd reflection, which keeps a trend at the end station).
    sp = float(np.median(np.diff(x))) if D > 1 else 1.0
    m = int(min(max(3, math.ceil(1.5 * float(sigma_m) / max(sp, 1e-9))), D - 1))
    Lp = np.concatenate([2.0 * L[:1] - L[m:0:-1], L, 2.0 * L[-1:] - L[-2:-m - 2:-1]], axis=0)
    xp = np.concatenate([2.0 * x[:1] - x[m:0:-1], x, 2.0 * x[-1:] - x[-2:-m - 2:-1]])
    off_diag = ~np.eye(2 * m + 1, dtype=bool)
    rm = np.empty_like(L)
    for i in range(D):
        xw, Lw = xp[i:i + 2 * m + 1], Lp[i:i + 2 * m + 1]
        dx = xw[None, :] - xw[:, None]
        dx = np.where(off_diag, dx, np.nan)
        slope = np.nanmedian(np.nanmedian((Lw[None, :, :] - Lw[:, None, :]) / dx[..., None], axis=1), axis=0)
        rm[i] = np.median(Lw - slope[None, :] * (xw - x[i])[:, None], axis=0)
    r = L - rm
    sc = 1.4826 * np.median(np.abs(r), axis=0, keepdims=True) + 1e-9
    u = r / (c * sc)
    w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0) @ Kt.T
    B = L.copy()
    for _ in range(int(iters)):
        B = np.empty_like(L)
        for i in range(D):
            nz = np.nonzero(K[i])[0]
            V = np.stack([(xs[nz] - xs[i]) ** k for k in range(q)], axis=1)          # (n, q)
            Wk = K[i, nz][:, None] * w[nz]                                           # (n, T)
            A = np.einsum("np,nt,nq->tpq", V, Wk, V) + 1e-12 * np.eye(q)[None]
            b = np.einsum("np,nt,nt->tp", V, Wk, L[nz])
            B[i] = np.linalg.solve(A, b[..., None])[..., 0, 0]
        r = L - B
        sc = 1.4826 * np.median(np.abs(r), axis=0, keepdims=True) + 1e-9
        u = r / (c * sc)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0) @ Kt.T
    return _tmono_np(B, axis=1), w


def local_poly_smoother_np(x: np.ndarray, sigma_m: float, deg: int = 2) -> np.ndarray:
    """(D, D) rows of the Gaussian local-polynomial smoother (degree deg, kernel sigma_m) on the stations x: S @ Y is the
    smooth of Y along the profile.  Degree 2 = the order of background_decomposition: quadratics are reproduced exactly,
    so a smooth curved background leaves no residual (a local-linear smoother leaves -(sigma^2 / 2) B'')."""
    x = np.asarray(x, dtype=np.float64)
    u = (x[None, :] - x[:, None]) / max(float(sigma_m), 1e-9)                       # (target, source) in sigmas
    K = np.exp(-0.5 * u * u)
    q = int(deg) + 1
    V = np.stack([u ** k for k in range(q)], axis=-1)                                # (D, D, q)
    M = np.einsum("ij,ijp,ijq->ipq", K, V, V) + 1e-12 * np.eye(q)[None]
    c = np.linalg.solve(M, np.repeat(np.eye(q)[:1], x.size, axis=0)[..., None])[..., 0]   # (D, q) = M_i^-1 e1
    return K * np.einsum("ijp,ip->ij", V, c)


def highpass_profile_np(Y: np.ndarray, x: np.ndarray, sigma_m: float, S: Optional[np.ndarray] = None) -> np.ndarray:
    """Part of Y (..., D, T) along the profile axis x (D,) that the Gaussian local-quadratic smoother of scale sigma_m
    does not keep: the anomaly-scale content (a smooth background, linear and quadratic trends give ~0).  S = a
    precomputed local_poly_smoother_np(x, sigma_m)."""
    Y = np.asarray(Y, dtype=np.float64)
    S = local_poly_smoother_np(x, sigma_m) if S is None else S
    return Y - np.einsum("ij,...jt->...it", S, Y)


def _decay_trend_1d(b: np.ndarray, s: np.ndarray, min_rise: float = 0.0) -> np.ndarray:
    """Decaying trend of b along the increasing distance s: b where it decays; where it rises (stations above its
    running minimum) by more than min_rise, from the station a before the rise the lowest supporting chord: the
    straight line to the station k with the steepest chord slope (b[k] - b[a]) / (s[k] - s[a]), k up to one rise
    width beyond the rise's end.  The chord lies on or below b, decays, and passes under the rise as the trend
    continuing beneath a response; the search stops one rise width on, so a decay steepening further down (a hole's
    bottom roll-off) is no reason to cut below the profile.  A smaller rise, or one with no lower station within
    reach: the running minimum."""
    run = np.minimum.accumulate(b)
    act = (b - run) > 1e-12
    out = run.copy()
    i = 1
    while i < b.size:
        if not act[i]:
            i += 1
            continue
        a = i - 1
        j = i
        while j + 1 < b.size and act[j + 1]:
            j += 1
        e = min(j + 1, b.size - 1)                       # the first station back at the running minimum (or the end)
        if float((b[i:j + 1] - run[i:j + 1]).max()) <= min_rise:
            i = j + 1
            continue
        reach = int(np.searchsorted(s, s[e] + (s[e] - s[a]), side="right"))
        c = (b[a + 1:reach] - b[a]) / np.maximum(s[a + 1:reach] - s[a], 1e-12)
        k = int(np.argmin(c))
        if c[k] < 0.0:
            ks = a + 1 + k
            out[a:ks + 1] = b[a] + c[k] * (s[a:ks + 1] - s[a])
            i = max(ks, j) + 1
        else:
            i = j + 1
    return np.minimum.accumulate(np.minimum(out, b))


def trend_minorant(B: np.ndarray, x: np.ndarray, min_rise: float = 0.0) -> np.ndarray:
    """Per time sample the overall decay trend of B (D, T) along the profile x (D,): B itself where it decays in the
    trend's direction (down the profile where B at the last station <= at the first: every borehole and time so far;
    else up), and under every rise higher than min_rise the lowest supporting chord (_decay_trend_1d).  The trend
    never rises; Liuyuan 5-8 ms: the 150-235 m late-time response is passed by a line of -0.0004 decades / m to the
    hole's end, as the trend continues beneath it."""
    B = np.asarray(B, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(B)
    up = B[-1] > B[0]
    for g in range(B.shape[1]):
        if up[g]:
            out[::-1, g] = _decay_trend_1d(B[::-1, g], x[-1] - x[::-1], min_rise)
        else:
            out[:, g] = _decay_trend_1d(B[:, g], x - x[0], min_rise)
    return out


def purify_background(B: np.ndarray, x: np.ndarray, sigma_m: float, eps: float,
                      max_passes: int = 200, decay_along: bool = False) -> Tuple[np.ndarray, int, float]:
    """The background anchor must itself pass the purity test the generated profiles are held to.  The robust fit
    keeps part of the measured content at wavelengths of a few sigma (it is not a fixed point of the smoother), and
    such a remnant sits at the same place in every generated profile.  The fit is smoothed again with the
    local-quadratic smoother (quadratics, hence the regional trend and its curvature, are kept exactly) until max
    |high-pass| <= eps, then projected onto non-increasing decays.  Returns (background, passes, final max
    |high-pass|). decay_along (background_trend monotone): the background is the overall decay trend of the profile,
    with no rise and no dip along it."""
    B = np.asarray(B, dtype=np.float64).copy()
    if B.shape[0] < 5:
        return _tmono_np(B, axis=1), 0, 0.0
    S = local_poly_smoother_np(x, sigma_m)
    if decay_along:
        B = trend_minorant(B, x, min_rise=eps)
    n = 0
    while float(np.abs(B - S @ B).max()) > eps and n < int(max_passes):
        B = S @ B
        if decay_along:
            B = trend_minorant(B, x, min_rise=eps)
        n += 1
    B = _tmono_np(B, axis=1)
    return B, n, float(np.abs(B - S @ B).max())


def measured_anomaly_templates(R: np.ndarray, x: np.ndarray, runs: Sequence[Dict[str, Any]],
                               taper_m: float) -> np.ndarray:
    """The measured anomalies as templates (n_runs, D, T): the residual R = measured - background (D, T) of each
    inventory run, window 1 over the run's stations and a cosine taper over taper_m beyond it; where windows overlap
    they are normalised, so the templates add up to R inside every run (the measured shape, exactly) and to 0 away
    from them."""
    R = np.asarray(R, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not runs:
        return np.zeros((0,) + R.shape)
    if all(r_.get("partition") == "ramp" for r_ in runs):
        # undulation inventory: window 1 over the anomaly's part of its run, 0 outside the run (the residual is 0 there:
        # the trend is the measured profile), a cosine ramp across a cut between two parts (ramp_partition): the
        # templates add up to R on every run
        return ramp_partition(x, runs)[:, :, None] * R[None]
    W = []
    for r_ in runs:
        dist = np.maximum(np.maximum(float(r_["from_m"]) - x, x - float(r_["to_m"])), 0.0)
        tap = max(float(taper_m), 1e-9)
        W.append(np.where(dist <= 0.0, 1.0, np.where(dist < tap, np.cos(0.5 * math.pi * dist / tap) ** 2, 0.0)))
    W = np.stack(W)
    W = W / np.maximum(W.sum(axis=0, keepdims=True), 1.0)
    return W[:, :, None] * R[None]


def measured_anomaly_inventory(R: np.ndarray, x: np.ndarray, t_ms: np.ndarray, floor: float,
                               noise_rule: str = "mad") -> Dict[str, Any]:
    """Anomalies of the measured profile = runs of stations where the residual from the background exceeds the
    threshold max(4 x the station-scale noise of the residual, floor) at some time sample.  Reported, not used to
    generate. noise_rule "quiet": the noise from the quietest quarter of the station-to-station changes (their 25th
    percentile / 0.3186, the 25th percentile of |Z| for Gaussian noise), not their MAD: the MAD counts the anomalies'
    own flanks once they cover half the stations (Liuyuan: 4 x MAD = 0.025 decades at 3.7 ms left out a 6-station
    late feature of +0.024 at 105-130 m)."""
    R = np.asarray(R, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t_ms = np.asarray(t_ms, dtype=np.float64)
    if R.shape[0] < 3:
        return {"threshold_decades": float("nan"), "noise_decades_median": float("nan"), "anomalies": []}
    dR = np.diff(R, axis=0)
    dev = np.abs(dR - np.median(dR, axis=0))
    if str(noise_rule) == "quiet":
        noise = np.percentile(dev, 25, axis=0) / 0.3186394 / math.sqrt(2.0)       # P(|Z| <= 0.3186) = 1/4
    else:
        noise = 1.4826 * np.median(dev, axis=0) / math.sqrt(2.0)
    thr = np.maximum(4.0 * noise, float(floor))
    hit = np.abs(R) > thr[None, :]
    st = hit.any(axis=1)
    runs: List[Dict[str, Any]] = []
    i = 0
    while i < st.size:
        if not st[i]:
            i += 1
            continue
        j = i
        while j + 1 < st.size and st[j + 1]:
            j += 1
        seg = R[i:j + 1]
        k = np.unravel_index(int(np.argmax(np.abs(seg))), seg.shape)
        runs.append({"from_m": float(x[i]), "to_m": float(x[j]), "peak_decades": float(seg[k]),
                     "peak_time_ms": float(t_ms[k[1]]), "peak_at_m": float(x[i + k[0]]),
                     "time_samples_above": int(hit[i:j + 1].any(axis=0).sum())})
        i = j + 1
    return {"threshold_decades_median": float(np.median(thr)), "noise_decades_median": float(np.median(noise)),
            "cells_above_fraction": float(hit.mean()), "anomalies": runs}


# ---------------------------------------------------------------------------------
# "0 anomalies" = the measured profile's own decay trend; the measured anomalies = its undulations
# ---------------------------------------------------------------------------------
# A profile without anomalies decays along the hole; an undulation (the decay rising against its trend) is an anomaly.
UNDULATION_SPLIT_SADDLE = 0.5      # two maxima of a run's envelope are two anomalies when the saddle between them is lower
                                   # than half the lower maximum (topographic prominence of at least half its height)
UNDULATION_TIME_DILATE_DEC = 0.2   # an undulation is bridged at the times its rise is significant, +- this (log time) ...
UNDULATION_TIME_RAMP_DEC = 0.2     # ... and ramped to the measured profile over this


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Index ranges (first, last) of the runs of True in a 1-D boolean array."""
    out: List[Tuple[int, int]] = []
    m = np.asarray(mask, dtype=bool)
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < m.size and m[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def measured_station_noise(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Station-scale noise of a measured profile L (D, T) along x (D,), per time sample: each station's residual from
    the straight line through its two neighbours, from the quietest quarter of the stations (25th percentile of |r -
    median| / 0.3186, the 25th percentile of |Z|), divided by the factor that line puts on white noise, sqrt(1 +
    (h0^2 + h1^2) / (h0 + h1)^2)."""
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if L.shape[0] < 3:
        return np.zeros(L.shape[1])
    h0, h1 = np.diff(x)[:-1], np.diff(x)[1:]
    r = L[1:-1] - (h1[:, None] * L[:-2] + h0[:, None] * L[2:]) / (h0 + h1)[:, None]
    fac = float(np.median(np.sqrt(1.0 + (h0 ** 2 + h1 ** 2) / (h0 + h1) ** 2)))
    return np.percentile(np.abs(r - np.median(r, axis=0)), 25, axis=0) / 0.3186394 / fac


def ramp_partition(x: np.ndarray, parts: Sequence[Dict[str, Any]]) -> np.ndarray:
    """(n, D) windows of the parts of undulation runs: 1 over a part, 0 outside its run, and at a cut between two
    parts of one run a cosine ramp 1 -> 0 over +-ramp m around the saddle, the neighbour's window its complement, so
    the windows of a run add up to 1 on it."""
    x = np.asarray(x, dtype=np.float64)

    def _down(c: float, h: float) -> np.ndarray:
        if h <= 0:
            return (x <= c).astype(np.float64)
        u = np.clip((x - c) / h, -1.0, 1.0)
        return 0.5 * (1.0 - np.sin(0.5 * math.pi * u))

    W = []
    for r_ in parts:
        w = ((x >= float(r_["run_from_m"]) - 1e-9) & (x <= float(r_["run_to_m"]) + 1e-9)).astype(np.float64)
        if r_.get("cut_left_m") is not None:
            w = w * (1.0 - _down(float(r_["cut_left_m"]), float(r_["ramp_left_m"])))
        if r_.get("cut_right_m") is not None:
            w = w * _down(float(r_["cut_right_m"]), float(r_["ramp_right_m"]))
        W.append(w)
    return np.stack(W) if W else np.zeros((0, x.size))


def _significance_weight(sig: np.ndarray, lt: np.ndarray, idx_k: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Time weight (T,) of a bridge: 1 within UNDULATION_TIME_DILATE_DEC of log time of a significant sample (sig
    over the measured samples idx_k), ramped to 0 over UNDULATION_TIME_RAMP_DEC; a late-time extension holds the last
    value."""
    w = np.zeros(lt.size)
    for j in idx_k[sig]:
        w = np.maximum(w, np.clip(1.0 - (np.abs(lt - lt[j]) - UNDULATION_TIME_DILATE_DEC) / UNDULATION_TIME_RAMP_DEC,
                                  0.0, 1.0))
    if np.any(~known) and idx_k.size:
        w[~known] = w[idx_k[-1]]
    return w


def _pav_nonincreasing(y: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit of the sequence y (pool adjacent violators)."""
    vals: List[float] = []
    wts: List[float] = []
    cnts: List[int] = []
    for v in np.asarray(y, dtype=np.float64):
        vals.append(float(v))
        wts.append(1.0)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnts.append(c1 + c2)
    return np.repeat(np.asarray(vals, dtype=np.float64), np.asarray(cnts, dtype=np.int64))


def trimmed_isotonic_trend(L: np.ndarray, x: np.ndarray, thr: Any, iters: int = 10,
                           exclude: Optional[np.ndarray] = None, start: Optional[np.ndarray] = None,
                           start_thr: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided overall decay trend of L (D, T) along x (D,): per time sample the least-squares non-increasing fit
    (in the trend's direction: down the profile where L at the last station <= at the first), refitted without the
    cells that depart from it by more than thr (T,) - rises AND dips - until the trimmed set repeats; trimmed cells
    take the fit interpolated linearly along the profile; exclude (D, T) = cells always left out.  start (D, T): a
    robust smooth reference (the two-sided Tukey fit at sigma_B, background_decomposition); the first fit leaves out
    the cells departing from it by more than max(thr, start_thr)."""
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    D, T = L.shape
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float64), (T,))
    M = L.copy()
    trimmed = np.zeros((D, T), dtype=bool)
    if D < 4:
        return M, trimmed
    for g in range(T):
        up = bool(L[-1, g] > L[0, g])
        y = L[::-1, g] if up else L[:, g]
        s_ = (x[-1] - x[::-1]) if up else (x - x[0])
        fix = np.zeros(D, dtype=bool) if exclude is None else np.asarray(exclude[:, g], dtype=bool)
        fix = fix[::-1] if up else fix
        inc = ~fix
        if int(inc.sum()) < 3:
            trimmed[:, g] = fix[::-1] if up else fix
            continue
        if start is not None:
            ref = start[::-1, g] if up else start[:, g]
            inc0 = (np.abs(y - ref) <= max(float(thr[g]), float(start_thr))) & ~fix
            if int(inc0.sum()) >= 3:
                inc = inc0
        f = np.interp(s_, s_[inc], _pav_nonincreasing(y[inc]))
        for _ in range(int(iters)):
            new = (np.abs(y - f) <= thr[g]) & ~fix
            if int(new.sum()) < 3 or np.array_equal(new, inc):
                break
            inc = new
            f = np.interp(s_, s_[inc], _pav_nonincreasing(y[inc]))
        M[:, g] = f[::-1] if up else f
        trimmed[:, g] = (~inc)[::-1] if up else ~inc
    return M, trimmed


def dips_beside_rises(dip: np.ndarray, rise: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Stations (D,) whose dip cells (D, T) are an undulation: the runs of dip stations (a dip cell at some measured
    time) that overlap or touch a run of rise stations (a rise cell at some measured time)."""
    kn = np.asarray(known, dtype=bool)
    fd = (np.asarray(dip, dtype=bool) & kn[None, :]).any(axis=1)
    fr = (np.asarray(rise, dtype=bool) & kn[None, :]).any(axis=1)
    keep = np.zeros(fd.size, dtype=bool)
    for a, b in _true_runs(fd):
        if fr[max(a - 1, 0):b + 2].any():
            keep[a:b + 1] = True
    return keep


def measured_trend_undulations(L: np.ndarray, x: np.ndarray, t_ms: np.ndarray, known: np.ndarray, B_shape: np.ndarray,
                               tol: float, two_sided: bool = False, sigma_m: Optional[float] = None,
                               eps: Optional[float] = None, B_start: Optional[np.ndarray] = None,
                               start_thr: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The overall decay trend of a measured profile L (D, T) and its undulations (background_trend measured).
      undulation: per time sample the profile rises above its lowest supporting chord (trend_minorant, no smoothing) by
          more than thr = max(4 x the station noise (measured_station_noise), tol) at some measured time; the runs of such
          stations are the undulation runs (a rise covers the hump up to where the decay comes back down to its trend, the
          chord ends there; a dip whose recovery rises is covered too, both its flanks; a decay that only steepens and
          flattens again is part of the trend);
      trend: the measured profile itself off the runs; under a run the shape of the smooth decay trend B_shape tied
          to the measured values at the stations just outside it (their two mismatches blended linearly along the
          run; a run at an end of the profile: the mismatch of its one outside station), used at the times where the
          undulation is there and ramped to the measured profile elsewhere (_significance_weight), so the bridge's
          own error stays out of the templates.  Two passes: the first weights a run by its own significant rises;
          the run is then split into its anomalies (undulation_inventory) and the second weights each part by its own
          rises, the parts' weights blended across a cut with the templates' partition (ramp_partition) - Liuyuan's
          155-196 m part rises from 1.2 ms, the 196-245 m part from 0.6 ms: with one weight per run the first part
          was bridged from 0.25 ms and its template held up to 0.030 decades of the bridge's error before 0.49 ms.
          Then the chord minorant (no rise) and a decay in time; a run grows over the stations next to it where that
          minorant cut the measured profile (its residual stays in a template).
      two_sided: besides the rises (as above) the dips are undulations: cells more than thr below the decay trend
          fitted without the rises and the dips (trimmed_isotonic_trend, started from the robust two-sided fit
          B_start with start_thr = half the visibility floor), so the near lobe of a conductor beside the hole is an
          anomaly and not trend; the chord's rise beside a dip (its recovery) joins the dip's run and keeps ~0
          residual there.  A dip is an undulation only beside a rise against the decay (its recovery or the hump it
          flanks; dips_beside_rises): the trimmed fit, interpolated linearly across the cells it leaves out, passes
          above a convex bend of a monotone decay, and "0 anomalies" is a monotone decay however curved (without this
          rule the anomaly-free trend of Liuyuan showed five dips of 0.005-0.017 decades, its 8-station decimation
          one of 0.31).  In the inventory a dip lobe and the hump beside it stay one anomaly (split_signs off: a
          dipolar response).
    known: the measured time samples (bool, T).  Returns (trend, info with the runs, chord departure, threshold,
    footprint)."""
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    B_shape = np.asarray(B_shape, dtype=np.float64)
    kn = np.asarray(known, dtype=bool)
    D, T = L.shape
    noise = measured_station_noise(L[:, kn], x)
    thr_k = np.maximum(4.0 * noise, float(tol))
    thr = np.full(T, float(np.median(thr_k)))
    thr[kn] = thr_k
    if np.any(~kn):                                       # a late-time extension: the last measured threshold
        thr[~kn] = thr_k[-1]
    D0 = L - trend_minorant(L, x, 0.0)
    above = D0 > thr[None, :]
    dip = np.zeros_like(above)
    B_dip = B_shape
    n_bend = 0
    if two_sided:
        # the dips against the decay trend fitted without the rises (trimmed_isotonic_trend), where the profile
        # undulates: beside a rise against the decay (a bend of a monotone decay is trend)
        Mtr, _trim = trimmed_isotonic_trend(L, x, thr, start=B_start, start_thr=start_thr)
        dip = ((L - Mtr) < -thr[None, :]) & ~above
        keep_d = dips_beside_rises(dip, above, kn)
        n_bend = int(((dip & kn[None, :]).any(axis=1) & ~keep_d).sum())
        dip &= keep_d[:, None]
        D0 = np.where(dip, L - Mtr, D0)
        above = above | dip
        if sigma_m is not None and bool(dip.any()):
            B_dip = purify_background(Mtr, x, float(sigma_m), float(eps if eps is not None else 0.25 * float(tol)),
                                      decay_along=True)[0]
    fp = (above & kn[None, :]).any(axis=1)
    runs = _true_runs(fp)
    lt = np.log10(np.asarray(t_ms, dtype=np.float64))
    idx_k = np.nonzero(kn)[0]
    bridges = []
    for a, b in runs:
        # a run with a dip: the trend's shape from the decay trend fitted without rises and dips (the robust background
        # passed under the rises follows a dip)
        Bs_ = B_dip if bool(dip[a:b + 1].any()) else B_shape
        o1, o2 = a - 1, b + 1
        e1 = L[o1] - Bs_[o1] if o1 >= 0 else None
        e2 = L[o2] - Bs_[o2] if o2 < D else None
        if e1 is None and e2 is None:
            Bf = Bs_[a:b + 1]
        else:
            e1 = e2 if e1 is None else e1
            e2 = e1 if e2 is None else e2
            x1 = x[o1] if o1 >= 0 else x[a]
            x2 = x[o2] if o2 < D else x[b]
            s = ((x[a:b + 1] - x1) / max(x2 - x1, 1e-12))[:, None]
            Bf = Bs_[a:b + 1] + (1.0 - s) * e1[None, :] + s * e2[None, :]
        bridges.append(Bf)

    def _finish(wz: np.ndarray) -> np.ndarray:
        out_ = L.copy()
        for (a_, b_), Bf_ in zip(runs, bridges):
            out_[a_:b_ + 1] = wz[a_:b_ + 1] * Bf_ + (1.0 - wz[a_:b_ + 1]) * L[a_:b_ + 1]
        return _tmono_np(trend_minorant(out_, x, 0.0), axis=1)

    # pass 1: one weight per run (its own significant rises)
    w1 = np.zeros((D, T))
    for a, b in runs:
        w1[a:b + 1] = _significance_weight(above[a:b + 1][:, idx_k].any(axis=0), lt, idx_k, kn)[None, :]
    out = _finish(w1)
    parts = undulation_inventory(L - out, x, t_ms, kn, fp, thr, split_signs=not two_sided)["anomalies"] if runs else []
    if parts:
        # pass 2: one weight per anomaly part, blended across the cuts with the templates' partition
        Wp = ramp_partition(x, parts)                                               # (n, D)
        wp = np.stack([_significance_weight(above[(x >= p_["from_m"] - 1e-9) & (x <= p_["to_m"] + 1e-9)][:, idx_k]
                                            .any(axis=0), lt, idx_k, kn) for p_ in parts])   # (n, T)
        out = _finish(Wp.T @ wp)
    # the final minorant may cut a chord at a station just outside a run: that residual belongs to the run's template
    nz = (np.abs(L - out)[:, kn] > 1e-9).any(axis=1)
    fp2 = fp.copy()
    for a, b in runs:
        while a > 0 and nz[a - 1]:
            a -= 1
        while b < D - 1 and nz[b + 1]:
            b += 1
        fp2[a:b + 1] = True
    windows = []
    for p_ in parts:
        sig = above[(x >= p_["from_m"] - 1e-9) & (x <= p_["to_m"] + 1e-9)][:, idx_k].any(axis=0)
        on = idx_k[sig]
        windows.append({"from_m": float(p_["from_m"]), "to_m": float(p_["to_m"]),
                        "rise_significant_from_ms": float(t_ms[on[0]]) if on.size else float("nan"),
                        "rise_significant_to_ms": float(t_ms[on[-1]]) if on.size else float("nan")})
    info = {"rule": "measured", "sides": "two" if two_sided else "one", "dip_cells": int(np.sum(dip & kn[None, :])),
            "dip_stations_left_as_trend_bends": n_bend,
            "noise_decades_median": float(np.median(noise)),
            "threshold_decades_median": float(np.median(thr_k)),
            "runs": [{"from_m": float(x[a]), "to_m": float(x[b])} for a, b in _true_runs(fp2)], "parts": windows,
            "stations_in_undulations": int(fp2.sum())}
    return out, {"info": info, "threshold": thr, "chord_departure": D0, "footprint": fp2}


def undulation_inventory(R: np.ndarray, x: np.ndarray, t_ms: np.ndarray, known: np.ndarray, footprint: np.ndarray,
                         thr: np.ndarray, split_saddle: float = UNDULATION_SPLIT_SADDLE,
                         split_signs: bool = True) -> Dict[str, Any]:
    """The measured anomalies = the undulations of the profile (residual R = measured - trend, (D, T)).  Each run of
    the footprint is one anomaly, unless its envelope e(x) = max over the measured times of |R| has several maxima:
    two maxima are two anomalies when the saddle between them is lower than split_saddle x the lower one and the
    lower one stands at least 2 x the median threshold above the saddle (a shoulder of the size of the threshold is
    no anomaly of its own); the run is split at the saddle station and the two share it by a cosine ramp whose half
    width is half the distance from the saddle to the nearer maximum, so their templates add up to the residual
    (ramp_partition). Liuyuan: one run 155-245 m, maxima 0.23 (184 m) and 0.26 (220 m), saddle 0.07-0.09 at 196 m ->
    two anomalies."""
    R = np.asarray(R, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    kn = np.asarray(known, dtype=bool)
    tk = np.asarray(t_ms, dtype=np.float64)[kn]
    Rk = R[:, kn]
    thr_k = np.asarray(thr, dtype=np.float64)[kn]
    prom = 2.0 * float(np.median(thr_k)) if thr_k.size else 0.0
    env = np.abs(Rk).max(axis=1)
    sgn = (np.sign(Rk[np.arange(Rk.shape[0]), np.abs(Rk).argmax(axis=1)]) if Rk.shape[1]
           else np.zeros(Rk.shape[0]))                          # sign of the residual at each station's envelope
    anomalies: List[Dict[str, Any]] = []
    for a, b in _true_runs(footprint):
        seg = env[a:b + 1]
        n = seg.size
        pk = [k for k in range(n) if (k == 0 or seg[k] > seg[k - 1]) and (k == n - 1 or seg[k] >= seg[k + 1])]
        keep: List[int] = []
        for k in sorted(pk, key=lambda k_: -seg[k_]):
            ok = True
            for k2 in keep:
                if not split_signs and sgn[a + k] != sgn[a + k2]:
                    ok = False
                    break
                sad = seg[min(k, k2):max(k, k2) + 1].min()
                if sad > split_saddle * min(seg[k], seg[k2]) or min(seg[k], seg[k2]) - sad < prom:
                    ok = False
                    break
            if ok:
                keep.append(k)
        keep.sort()
        if not keep:
            keep = [int(np.argmax(seg))]
        cuts = [lo_ + int(np.argmin(seg[lo_:hi_ + 1])) for lo_, hi_ in zip(keep[:-1], keep[1:])]
        bounds = [0] + cuts + [n - 1]
        for j, k in enumerate(keep):
            i0, i1 = a + bounds[j], a + bounds[j + 1]
            sub = Rk[i0:i1 + 1]
            ii, gg = np.unravel_index(int(np.argmax(np.abs(sub))), sub.shape)
            pk_v = float(sub[ii, gg])
            # Gaussian-equivalent width at the peak's time sample: FWHM of the same-signed part around the peak / 2.355
            prof = np.clip(R[a:b + 1, np.nonzero(kn)[0][gg]] * np.sign(pk_v), 0.0, None)
            c = (i0 - a) + ii
            half = 0.5 * abs(pk_v)
            lo_i, hi_i = c, c
            while lo_i > 0 and prof[lo_i - 1] >= half:
                lo_i -= 1
            while hi_i < prof.size - 1 and prof[hi_i + 1] >= half:
                hi_i += 1
            xr = x[a:b + 1]

            def _cross(p: int, q: int) -> float:
                yp, yq = prof[p] - half, prof[q] - half
                return float(xr[p] + (xr[q] - xr[p]) * yp / (yp - yq)) if yp != yq else float(xr[p])

            left = _cross(lo_i, lo_i - 1) if lo_i > 0 else float(xr[0])
            right = _cross(hi_i, hi_i + 1) if hi_i < prof.size - 1 else float(xr[-1])
            cut_l = float(x[i0]) if j > 0 else None
            cut_r = float(x[i1]) if j < len(keep) - 1 else None
            ramp_l = 0.5 * min(float(x[i0]) - float(x[a + keep[j - 1]]), float(x[a + k]) - float(x[i0])) if j > 0 else 0.0
            ramp_r = (0.5 * min(float(x[i1]) - float(x[a + k]), float(x[a + keep[j + 1]]) - float(x[i1]))
                      if j < len(keep) - 1 else 0.0)
            anomalies.append({"from_m": float(x[i0]), "to_m": float(x[i1]), "peak_decades": pk_v,
                              "peak_time_ms": float(tk[gg]), "peak_at_m": float(x[i0 + ii]),
                              "envelope_max_at_m": float(x[a + k]),
                              "time_samples_above": int((np.abs(sub) > thr_k[None, :]).any(axis=0).sum()),
                              "width_sigma_m": float(max(right - left, 1e-6) / 2.355),
                              "partition": "ramp", "run_from_m": float(x[a]), "run_to_m": float(x[b]),
                              "cut_left_m": cut_l, "cut_right_m": cut_r, "ramp_left_m": float(ramp_l),
                              "ramp_right_m": float(ramp_r)})
    return {"rule": "undulation", "split_saddle": float(split_saddle), "split_min_prominence_decades": float(prom),
            "split_opposite_signs": bool(split_signs),
            "threshold_decades_median": float(np.median(thr_k)) if thr_k.size else float("nan"),
            "cells_above_fraction": float((np.abs(Rk) > thr_k[None, :]).mean()) if Rk.size else 0.0,
            "anomalies": anomalies}
