"""Measured and simulated data objects: local depth structure, the dense depth axis, the measured profile on the
anchored time grid (preprocess_measured), the simulated library on the measured grid and depth axis
(align_simulation) and a synthetic profile for the self-check."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common import array_digest
from .extension import calibrate_late_extension, extend_late_time
from .readers import ProfileMatrix, SurveySpec
from .time_grid import build_anchored_time_grid, depth_interp_profile, resample_loglog, robust_depth_trend


def local_structure_residual(L: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Departure of each interior station from the depth-weighted line through its two neighbours
    (decades).  Insensitive to smooth depth trends, so it measures LOCAL profile structure."""
    L = np.asarray(L, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    if d.size < 3:
        return np.zeros((0,) + L.shape[1:])
    w = (d[1:-1] - d[:-2]) / np.maximum(d[2:] - d[:-2], 1e-12)
    return L[1:-1] - ((1.0 - w)[:, None] * L[:-2] + w[:, None] * L[2:])


def build_dense_depth_axis(depth: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """n-point depth axis over the measured span with every measured station as an exact node (the measured
    depth axis stays the standard).  The n - D extra nodes are shared out over the station intervals in
    proportion to their length (largest remainder) and placed uniformly inside each interval.
    Returns (dense depth (n,), index of each measured station on the dense axis (D,))."""
    d = np.asarray(depth, dtype=np.float64)
    D = int(d.size)
    n = int(n)
    if n <= D or D < 2:
        return d.copy(), np.arange(D, dtype=np.int64)
    seg = np.diff(d)
    extra = n - D
    quota = extra * seg / max(seg.sum(), 1e-30)
    base = np.floor(quota).astype(np.int64)
    rem = int(extra - base.sum())
    if rem > 0:
        order = np.argsort(-(quota - base), kind="mergesort")
        base[order[:rem]] += 1
    pts: List[float] = [d[0]]
    st = [0]
    for k in range(D - 1):
        m = int(base[k])
        if m > 0:
            pts.extend(list(d[k] + seg[k] * (np.arange(1, m + 1) / (m + 1.0))))
        pts.append(d[k + 1])
        st.append(len(pts) - 1)
    out = np.asarray(pts, dtype=np.float64)
    st_arr = np.asarray(st, dtype=np.int64)
    out[st_arr] = d
    return out, st_arr


@dataclass
class MeasuredData:
    pm: ProfileMatrix
    grid_ms: np.ndarray          # (T,)
    gate_index: np.ndarray       # (n,) grid index of each measured gate (-1 = not embedded)
    depth: np.ndarray            # (D,)
    L: np.ndarray                # (D, T) log10 amplitude on the grid (positive mode) or symlog
    positive: bool
    report: Dict[str, Any]
    # survey mode of the profile axis, the last time whose values are measured (beyond it the grid is the late-time
    # extension, whose values follow the calibrated rule `late_ext`), and the generation axis when the profile is
    # extended beyond the measured stations (virtual stations; axis_known marks the measured ones)
    survey: Optional[SurveySpec] = None
    t_known_ms: float = 0.0
    late_ext: Optional[Dict[str, Any]] = None
    axis_depth: Optional[np.ndarray] = None
    axis_known: Optional[np.ndarray] = None
    # background_mode pure: the smooth geological background of the measured profile (D, T) on the grid and its
    # description (scale, rule, purity tolerance, residual statistics, the measured anomalies it leaves out)
    background: Optional[np.ndarray] = None
    background_info: Optional[Dict[str, Any]] = None
    # sim_role morphology: the simulated profile's anomalies at the measured stations (sim_morphology_library)
    sim_morph: Optional[Dict[str, Any]] = None

    @property
    def D(self) -> int:
        return int(self.depth.size)

    @property
    def T(self) -> int:
        return int(self.grid_ms.size)

    @property
    def gen_depth(self) -> np.ndarray:
        """Stations the network generates on: the measured stations, plus the virtual ones of a profile extension."""
        return self.depth if self.axis_depth is None else self.axis_depth

    @property
    def gen_known(self) -> np.ndarray:
        return np.ones(self.D, dtype=bool) if self.axis_known is None else self.axis_known

    def known_time_mask(self) -> np.ndarray:
        """Grid nodes whose values are measured (False on the late-time extension)."""
        if self.t_known_ms <= 0:
            return np.ones(self.T, dtype=bool)
        return self.grid_ms <= self.t_known_ms * (1.0 + 1e-9)

    def L_at(self, depth_q: np.ndarray) -> np.ndarray:
        return depth_interp_profile(self.depth, self.L, depth_q)

    def subset(self, keep: np.ndarray) -> "MeasuredData":
        """Measured profile restricted to the stations in `keep` (cross-validation folds)."""
        keep = np.asarray(keep, dtype=bool)
        pm2 = copy.copy(self.pm)
        pm2.depth = self.pm.depth[keep]
        pm2.values = self.pm.values[keep]
        return MeasuredData(pm=pm2, grid_ms=self.grid_ms, gate_index=self.gate_index, depth=self.depth[keep],
                            L=self.L[keep], positive=self.positive, report=dict(self.report), survey=self.survey,
                            t_known_ms=self.t_known_ms, late_ext=self.late_ext, sim_morph=self.sim_morph)

    def time_holdout(self, decades: float) -> Tuple["MeasuredData", np.ndarray]:
        """Time-direction cross-validation: the measured gates of the last `decades` of the window are hidden; the
        grid stays, the values after the cut follow the late-time extension rule chosen and calibrated on the
        remaining gates only.  Returns (fold data, hidden grid columns)."""
        if not self.positive:
            raise ValueError("[CV-TIME] the time-direction fold continues decays in log10: it needs positive measured "
                             "decays (this profile is in the signed-log domain)")
        if self.late_ext is not None:
            raise ValueError("[CV-TIME] the time-direction fold runs on the measured window (no --extend_time_ms)")
        t = np.asarray(self.pm.time_ms, dtype=np.float64)
        t_cut = float(10.0 ** (math.log10(t[-1]) - float(decades)))
        keep_g = t <= t_cut * (1.0 + 1e-9)
        if int(keep_g.sum()) < 4 or bool(np.all(keep_g)):
            raise ValueError(f"[CV-TIME] --cv_time_holdout {decades:g}: hiding the last {decades:g} decades leaves "
                             f"{int(keep_g.sum())} of {t.size} gates (needs >= 4 kept and >= 1 hidden)")
        t_cut = float(t[keep_g][-1])
        gi = self.gate_index
        if bool(np.all(gi >= 0)):
            Ln = self.L[:, gi]                     # the measured log values (non-positive samples already repaired)
        else:
            Ln = measured_log_values(self.pm)[0]   # the same log10 values the grid was built from
        cal = calibrate_late_extension(t[keep_g], Ln[:, keep_g], math.log10(t[-1] / t_cut))
        L = self.L.copy()
        hid = self.grid_ms > t_cut * (1.0 + 1e-9)
        # the known window is re-interpolated from the kept gates only (the PCHIP slope at the cut gate would otherwise
        # carry the first hidden gate), the hidden window follows the rule
        Lk, _, _ = resample_loglog(t[keep_g], Ln[:, keep_g], self.grid_ms[~hid], extrapolation="edge")
        L[:, ~hid] = Lk
        if bool(np.all(gi >= 0)):
            L[:, gi[keep_g]] = Ln[:, keep_g]
        L[:, hid] = extend_late_time(np.log10(t[keep_g]), Ln[:, keep_g], np.log10(self.grid_ms[hid]), cal["rule"])
        rep = dict(self.report)
        rep["time_holdout"] = {"hidden_decades": float(decades), "t_cut_ms": t_cut, "hidden_gates": int((~keep_g).sum()),
                               "rule": cal["rule"], "rule_rms_decades": cal["rms_decades"]}
        md = MeasuredData(pm=self.pm, grid_ms=self.grid_ms, gate_index=self.gate_index, depth=self.depth, L=L,
                          positive=self.positive, report=rep, survey=self.survey, t_known_ms=t_cut, late_ext=cal,
                          sim_morph=self.sim_morph)
        return md, np.where(hid)[0]


@dataclass
class SimPrior:
    pm: Optional[ProfileMatrix]
    depth: np.ndarray            # (N,) native sim depths (ascending)
    L_native_grid: np.ndarray    # (N, T) sim log10 amplitude on the measured grid (time-aligned)
    anomaly: np.ndarray          # (N, T) sim structure relative to its own background (decades)
    coverage: np.ndarray         # (T,) 1 = inside the sim's native time span (soft taper at the edges)
    level_shift: np.ndarray      # (T,) per-gate shift used by gate_median / global_median modes
    depth_mode: str              # "physical" | "relative" | "none"
    level_mode: str              # "structure" | "gate_median" | "global_median" | "none"
    gain: float                  # structure gain applied to `anomaly` (auto = measured-anchored)
    report: Dict[str, Any]

    @property
    def available(self) -> bool:
        return self.depth.size >= 2

    def rows_for(self, sim_depth_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear row interpolation weights on the native sim depth axis (any query shape)."""
        q = np.clip(np.asarray(sim_depth_q, dtype=np.float64), self.depth[0], self.depth[-1])
        j = np.clip(np.searchsorted(self.depth, q, side="right") - 1, 0, self.depth.size - 2)
        w = (q - self.depth[j]) / np.maximum(self.depth[j + 1] - self.depth[j], 1e-12)
        return j, j + 1, w

    def canonical_sim_depth(self, meas_depth: np.ndarray, m0: float, m1: float) -> np.ndarray:
        if self.depth_mode == "physical":
            return np.clip(np.asarray(meas_depth, dtype=np.float64), self.depth[0], self.depth[-1])
        s0, s1 = float(self.depth[0]), float(self.depth[-1])
        return s0 + (np.asarray(meas_depth, dtype=np.float64) - m0) / max(m1 - m0, 1e-9) * (s1 - s0)


def measured_log_values(pm: ProfileMatrix) -> Tuple[np.ndarray, bool, int, int]:
    """Measured values at the gates in the model's log domain: log10 when at most 1 % are non-positive, else symlog.
    Returns (L_native, positive, non-positive count, repaired count)."""
    V = np.asarray(pm.values, dtype=np.float64)
    nonpos = int(np.sum(V <= 0))
    frac = nonpos / max(V.size, 1)
    positive = frac <= 0.01
    repaired = 0
    if positive and nonpos > 0:
        lt = np.log10(pm.time_ms)
        for i in np.where((V <= 0).any(axis=1))[0]:
            ok = V[i] > 0
            if ok.sum() >= 2:
                V[i, ~ok] = 10.0 ** np.interp(lt[~ok], lt[ok], np.log10(V[i, ok]))
                repaired += int((~ok).sum())
    if positive:
        L_native = np.log10(V)
    else:
        scale = float(np.percentile(np.abs(V), 50)) + 1e-12
        L_native = np.sign(V) * np.log1p(np.abs(V) / scale)
    return L_native, bool(positive), nonpos, repaired


def preprocess_measured(pm: ProfileMatrix, target_len: int, verbose: bool = True, t_ext_ms: float = 0.0) -> MeasuredData:
    """Measured profile on the anchored time grid.  t_ext_ms > t_last extends the grid beyond the last gate: the
    decays continue there by the rule chosen and calibrated on the measured gates (calibrate_late_extension); the
    measured gates and their values are unchanged."""
    grid, gate_index, ginfo = build_anchored_time_grid(pm.time_ms, target_len, t_ext_ms)
    L_native, positive, nonpos, repaired = measured_log_values(pm)
    L, _, _ = resample_loglog(pm.time_ms, L_native, grid, extrapolation="edge")
    if ginfo["all_gates_embedded"]:
        L[:, gate_index] = L_native  # bit-exact at the measured gates
    late = None
    beyond = grid > float(pm.time_ms[-1]) * (1.0 + 1e-9)
    if np.any(beyond):
        if not positive:
            raise ValueError("[DATA-ANCHOR] the late-time extension needs positive decays (log10 domain)")
        late = calibrate_late_extension(pm.time_ms, L_native, math.log10(float(grid[-1]) / float(pm.time_ms[-1])))
        L[:, beyond] = extend_late_time(np.log10(pm.time_ms), L_native, np.log10(grid[beyond]), late["rule"])
    lsr = np.abs(local_structure_residual(L[:, ~beyond], pm.depth))      # the measured window
    dl = np.diff(L, axis=1)
    emb = gate_index >= 0
    if bool(np.all(emb)):
        readback = float(np.max(np.abs(L[:, gate_index] - L_native)))
    else:
        # gates off the grid: what the grid keeps of them, read back by the same log-log interpolation
        Lb, _, _ = resample_loglog(grid, L, pm.time_ms, extrapolation="edge")
        readback = float(np.max(np.abs(Lb - L_native)))
    rep = {
        "time_grid": ginfo,
        "positive_decays": bool(positive), "nonpositive_values": nonpos, "nonpositive_repaired": repaired,
        "time_monotone_violations": int(np.sum(dl > 1e-12)),
        "local_structure_decades": {"p50": float(np.percentile(lsr, 50)) if lsr.size else 0.0,
                                    "p95": float(np.percentile(lsr, 95)) if lsr.size else 0.0,
                                    "p99": float(np.percentile(lsr, 99)) if lsr.size else 0.0},
        "max_readback_error_decades": readback,
        "depth_axis_sha16": array_digest(pm.depth),
    }
    if verbose:
        print(f"[DATA-ANCHOR] measured time grid: T={ginfo['target_len']} over {ginfo['t_first_ms']:.6g}.."
              f"{ginfo['t_last_ms']:.6g} ms | all {ginfo['n_measured_gates']} gates embedded="
              f"{ginfo['all_gates_embedded']} | log10 step {ginfo['log10_step_min']:.4f}..{ginfo['log10_step_max']:.4f}")
        print(f"[DATA-ANCHOR] measured depth axis: {pm.depth.size} stations {pm.depth[0]:.6g}..{pm.depth[-1]:.6g} m "
              f"(unique standard) | positive decays={positive} (non-positive {nonpos}, repaired {repaired}) "
              f"| time-monotone violations={rep['time_monotone_violations']}")
        ls = rep["local_structure_decades"]
        print(f"[DATA-ANCHOR] measured local depth structure |p50/p95/p99| = {ls['p50']:.4f}/{ls['p95']:.4f}/"
              f"{ls['p99']:.4f} decades | gate read-back max error={rep['max_readback_error_decades']:.2e}")
    t_known = 0.0
    if late is not None:
        t_known = float(pm.time_ms[-1])
        rep["late_time_extension"] = {k: v for k, v in late.items() if k != "bins"}
        rep["late_time_extension"]["pseudo_holdout_bins"] = late.get("bins", [])
        if verbose:
            print(f"[DATA-ANCHOR] late-time extension {pm.time_ms[-1]:.6g} -> {grid[-1]:.6g} ms "
                  f"({int(beyond.sum())} grid nodes): rule {late['rule']} chosen on the measured gates (pseudo hold-out RMS "
                  f"{late['rms_decades']:.4f} decades; rules {', '.join(f'{k} {v:.4f}' for k, v in late['rules'].items())}) | "
                  f"per-profile spread a dlog t^g, a={late['a']:.4f} g={late['gamma']:.2f}")
            if late.get("fallback"):
                print(f"[DATA-ANCHOR] WARNING late-time extension: {late['fallback']}")
            if not ginfo["all_gates_embedded"]:
                print(f"[DATA-ANCHOR] WARNING {ginfo['late_time_extension'].get('warning', '')}: read-back max error "
                      f"{readback:.2e} decades")
    return MeasuredData(pm=pm, grid_ms=grid, gate_index=gate_index, depth=pm.depth.copy(), L=L,
                        positive=positive, report=rep, t_known_ms=t_known, late_ext=late)


def _soft_time_coverage(grid_ms: np.ndarray, t_lo: float, t_hi: float, taper_decades: float = 0.15) -> np.ndarray:
    """1 inside the native time span of a source, 0 outside, with a linear taper (in log10 t) that
    rises from 0 at the first / last native gate to 1 one taper-width inside the span."""
    lx = np.log10(np.asarray(grid_ms, dtype=np.float64))
    lo, hi = math.log10(t_lo), math.log10(t_hi)
    inside = (lx >= lo - 1e-12) & (lx <= hi + 1e-12)
    if taper_decades <= 0:
        return inside.astype(np.float64)
    edge_in = np.clip((lx - lo) / taper_decades, 0.0, 1.0)
    edge_out = np.clip((hi - lx) / taper_decades, 0.0, 1.0)
    return np.where(inside, np.minimum(edge_in, edge_out), 0.0)


def align_simulation(pm: Optional[ProfileMatrix], meas: MeasuredData, depth_match: str = "auto",
                     level_mode: str = "structure", extrapolation: str = "loglinear",
                     trend_sigma_frac: float = 0.12, structure_gain: Any = "auto",
                     verbose: bool = True) -> SimPrior:
    """Put the simulated library on the measured time grid and depth axis."""
    T = meas.T
    if pm is None:
        empty = np.zeros((0, T))
        return SimPrior(pm=None, depth=np.zeros(0), L_native_grid=empty, anomaly=empty,
                        coverage=np.zeros(T), level_shift=np.zeros(T), depth_mode="none",
                        level_mode="none", gain=0.0, report={"available": False})
    V = np.asarray(pm.values, dtype=np.float64)
    frac_np = float(np.mean(V <= 0))
    if meas.positive:
        if frac_np > 0:
            floor = float(np.percentile(V[V > 0], 0.1)) if np.any(V > 0) else 1e-12
            V = np.maximum(V, floor)
        L_native = np.log10(V)
    else:
        scale = float(np.percentile(np.abs(V), 50)) + 1e-12
        L_native = np.sign(V) * np.log1p(np.abs(V) / scale)
    L_grid, inside, tinfo = resample_loglog(pm.time_ms, L_native, meas.grid_ms, extrapolation=extrapolation)
    cov = _soft_time_coverage(meas.grid_ms, float(pm.time_ms[0]), float(pm.time_ms[-1]))
    # ---- depth registration ----
    m0, m1 = float(meas.depth[0]), float(meas.depth[-1])
    s0, s1 = float(pm.depth[0]), float(pm.depth[-1])
    overlap = max(0.0, min(m1, s1) - max(m0, s0)) / max(m1 - m0, 1e-9)
    mode = (depth_match or "auto").lower()
    if mode == "auto":
        mode = "physical" if overlap >= 0.95 else "relative"
    # ---- background / structure of the sim profile along its own depth ----
    span_s = s1 - s0
    sig_s = max(trend_sigma_frac * span_s, 2.0 * float(np.median(np.diff(pm.depth))))
    n_tr = int(min(pm.depth.size, 2048))
    idx_tr = np.unique(np.linspace(0, pm.depth.size - 1, n_tr).round().astype(np.int64))
    trend_s_sub = robust_depth_trend(L_grid[idx_tr], pm.depth[idx_tr], sig_s)
    trend_s = np.stack([np.interp(pm.depth, pm.depth[idx_tr], trend_s_sub[:, j]) for j in range(T)], axis=1)
    anomaly = L_grid - trend_s
    # ---- canonical registration onto the measured stations ----
    lm = (level_mode or "structure").lower()
    if mode == "physical":
        dq = np.clip(meas.depth, s0, s1)
    else:
        dq = s0 + (meas.depth - m0) / max(m1 - m0, 1e-9) * span_s
    j = np.clip(np.searchsorted(pm.depth, dq, side="right") - 1, 0, pm.depth.size - 2)
    w = (dq - pm.depth[j]) / np.maximum(pm.depth[j + 1] - pm.depth[j], 1e-12)
    L_sim_at_meas = (1 - w)[:, None] * L_grid[j] + w[:, None] * L_grid[j + 1]
    gate_shift = np.median(meas.L, axis=0) - np.median(L_sim_at_meas, axis=0)
    # statistics on the measured window only (a late-time extension / the hidden gates of the time-direction CV fold
    # carry rule values, not measurements)
    known = meas.known_time_mask()
    if lm == "gate_median":
        level_shift = gate_shift
    elif lm == "global_median":
        gk = (cov > 0.5) & known
        level_shift = np.full(T, float(np.median(gate_shift[gk] if np.any(gk) else gate_shift[known])))
    else:
        level_shift = np.zeros(T)
    # ---- measured-anchored structure gain ----
    sel = (cov > 0.5) & known
    lsr_m = np.abs(local_structure_residual(meas.L, meas.depth))
    lsr_s = np.abs(local_structure_residual(L_sim_at_meas, meas.depth))
    p99_m = float(np.percentile(lsr_m[:, sel], 99)) if (lsr_m.size and np.any(sel)) else 0.0
    p99_s = float(np.percentile(lsr_s[:, sel], 99)) if (lsr_s.size and np.any(sel)) else 0.0
    if isinstance(structure_gain, str) and structure_gain.lower() == "auto":
        gain = float(np.clip(p99_m / max(p99_s, 1e-9), 0.0, 1.0)) if p99_s > 0 else 0.0
        gain_rule = "auto: p99 local structure of the sim matched to the measured p99"
    else:
        gain = float(structure_gain)
        gain_rule = "fixed by flag"
    an_p95 = float(np.percentile(np.abs(anomaly[:, sel]), 95)) if np.any(sel) else 0.0
    rep = {"available": True, "summary": pm.summary(), "time_alignment": tinfo,
           "time_native_span_ms": [float(pm.time_ms[0]), float(pm.time_ms[-1])],
           "time_coverage_fraction_of_grid": float(np.mean(inside)),
           "time_gates_without_native_support": int(np.sum(inside < 0.5)),
           "depth_match": mode, "depth_overlap_fraction": float(overlap),
           "depth_registration": ("identity (physical depth)" if mode == "physical"
                                  else f"affine: sim [{s0:.6g},{s1:.6g}] m -> measured [{m0:.6g},{m1:.6g}] m"),
           "level_mode": lm, "per_gate_level_difference_decades": {
               "min": float(gate_shift.min()), "median": float(np.median(gate_shift)), "max": float(gate_shift.max())},
           "structure_background_sigma_m": float(sig_s), "structure_abs_p95_decades_unscaled": an_p95,
           "local_structure_p99_decades": {"measured": p99_m, "simulated": p99_s},
           "structure_gain": gain, "structure_gain_rule": gain_rule}
    if verbose:
        print(f"[SIM-ALIGN] time: native {pm.time_ms[0]:.6g}..{pm.time_ms[-1]:.6g} ms -> measured grid "
              f"{meas.grid_ms[0]:.6g}..{meas.grid_ms[-1]:.6g} ms | gates without native sim support: "
              f"{rep['time_gates_without_native_support']}/{T} (left extrap {tinfo['left_extrapolation_decades']:.2f} dec, "
              f"right {tinfo['right_extrapolation_decades']:.2f} dec; sim structure tapered to 0 there)")
        print(f"[SIM-ALIGN] depth: sim {s0:.6g}..{s1:.6g} m vs measured {m0:.6g}..{m1:.6g} m | overlap "
              f"{overlap:.1%} -> mode={mode} ({rep['depth_registration']})")
        print(f"[SIM-ALIGN] level: mode={lm} | sim-vs-measured per-gate level difference "
              f"{gate_shift.min():+.2f}..{gate_shift.max():+.2f} decades | local structure p99 measured="
              f"{p99_m:.4f} sim={p99_s:.4f} decades -> structure gain={gain:.4f} ({gain_rule})")
    return SimPrior(pm=pm, depth=pm.depth.copy(), L_native_grid=L_grid, anomaly=anomaly, coverage=cov,
                    level_shift=level_shift, depth_mode=mode, level_mode=lm, gain=gain, report=rep)


def synthetic_tem_profile(n_depth: int = 40, n_gates: int = 30, seed: int = 0,
                          depth_range: Tuple[float, float] = (40.0, 400.0),
                          time_range_ms: Tuple[float, float] = (0.02, 8.0),
                          kind: str = "measured") -> ProfileMatrix:
    """Physically shaped synthetic borehole-TEM profile (smoke tests / --data_name tem_simulate)."""
    rng = np.random.default_rng(seed)
    d = np.linspace(depth_range[0], depth_range[1], n_depth)
    t = np.geomspace(time_range_ms[0], time_range_ms[1], n_gates)
    u = (d - d[0]) / (d[-1] - d[0])
    lt = np.log10(t)[None, :]
    level = 1.4 - 1.6 * u[:, None]
    p_early = 0.75 - 0.35 * u[:, None]
    p_late = 2.0 + 0.45 * np.sin(np.pi * u[:, None])
    x = (lt - lt.min()) / (lt.max() - lt.min())
    L = level - (p_early * (1 - x) + p_late * x) * (lt - lt.min())
    if kind == "simulated":
        for c in rng.uniform(0.2, 0.8, size=2):
            L += 0.4 * np.exp(-0.5 * ((u[:, None] - c) / 0.05) ** 2) * x
        L += 3.0
    else:
        c = rng.uniform(0.3, 0.8)
        L += -0.07 * np.exp(-0.5 * ((u[:, None] - c) / 0.02) ** 2) * (1 - x)
    V = 10.0 ** L
    return ProfileMatrix(path=f"<synthetic:{kind}>", role=kind, depth=d, time_ms=t, values=V,
                         orientation="rows=depth", time_unit="ms", time_unit_reason="synthetic",
                         orientation_reason="synthetic", axis_name="depth (m)", encoding="n/a",
                         sha16=array_digest(V), notes=["synthetic profile"])
