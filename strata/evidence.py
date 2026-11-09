"""Evidence for the decomposition and the generated set: residual dependence along the profile (cross-validation
buffer, effective stations), decomposition stability, the injection-recovery identifiability bench and the diversity
of the generated profiles."""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import array_digest, save_json
from .config import AppConfig
from .data.contract import MeasuredData
from .data.decomposition import measured_station_noise
from .data.readers import _strict_float
from .data.time_grid import robust_depth_trend
from .morphology import MORPH_U
from .pipeline import measured_background


def _pct3(v: Any) -> Dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    q = np.percentile(a, [10, 50, 90])
    return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def residual_dependence(R: np.ndarray, x: np.ndarray, max_cols: int = 64) -> Dict[str, Any]:
    """Along-profile dependence of a residual R (D, T) at the stations x (D,), per time sample (<= max_cols) on a
    uniform axis at the median station spacing: the autocorrelation rho_k; correlation length = the first lag with
    rho < e^-1/2; range = the first lag with rho <= 0.05, the practical range beyond which two stations carry
    independent information (the block / buffer size of spatial cross-validation: Roberts et al. 2017, Ecography 40;
    Valavi et al. 2019, Methods Ecol. Evol. 10); effective number of independent stations n_eff = n / (1 + 2 sum_k (1
    - k / n) rho_k), summed over the initial positive sequence of rho (Geyer 1992).  Liuyuan (two-sided undulations):
    range 15 m (3 station spacings), n_eff ~ 17 of 50 (one-sided: 20 m, 15)."""
    R = np.asarray(R, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    xu = np.arange(x[0], x[-1] + 0.5 * dx, dx)
    n = int(xu.size)
    cols = np.unique(np.linspace(0, R.shape[1] - 1, min(int(max_cols), R.shape[1])).round().astype(np.int64))
    ell, rng_, neff = [], [], []
    for j in cols:
        r = np.interp(xu, x, R[:, j])
        r = r - r.mean()
        den = float(r @ r)
        if den <= 1e-20 or n < 4:
            continue
        ac = np.correlate(r, r, "full")[n - 1:] / den
        b = np.nonzero(ac < math.exp(-0.5))[0]
        ell.append(float(b[0] if b.size else n - 1) * dx)
        b = np.nonzero(ac <= 0.05)[0]
        rng_.append(float(b[0] if b.size else n - 1) * dx)
        acc_ = 0.0
        for k in range(1, n):
            if ac[k] <= 0.0:
                break
            acc_ += (1.0 - k / n) * float(ac[k])
        neff.append(n / (1.0 + 2.0 * acc_))
    ne = _pct3(neff)
    return {"spacing_m": dx, "uniform_samples": n, "stations": int(x.size), "time_samples": len(ell),
            "correlation_length_m": _pct3(ell), "range_m": _pct3(rng_), "n_eff": ne,
            "n_eff_share_of_samples": float(ne["p50"] / n) if n else float("nan"),
            "rule": "per time sample on a uniform axis: correlation length = first lag with rho < e^-1/2, range = first "
                    "lag with rho <= 0.05, n_eff = n / (1 + 2 sum (1 - k/n) rho_k) over the initial positive sequence"}


def cv_buffer_distance(meas: MeasuredData, cfg: AppConfig) -> Tuple[float, str, Dict[str, Any]]:
    """Buffer of the spatial block cross-validation (--cv_buffer_m): auto = the practical range of the measured
    residual's autocorrelation along the profile (residual from the measured background, or from a robust trend in
    the measured-anchor mode), median over the measured time samples."""
    spec = str(cfg.cv.buffer_m or "auto").strip().lower()
    kn = meas.known_time_mask()
    if meas.background is not None:
        base = np.asarray(meas.background, dtype=np.float64)
    else:
        base = robust_depth_trend(meas.L, meas.depth, 0.25 * float(meas.depth[-1] - meas.depth[0]))
    dep = residual_dependence((meas.L - base)[:, kn], meas.depth)
    if spec in ("0", "0.0", "0m", "off", "none", "no"):
        return 0.0, "off (v17 folds: the stations next to a held-out block are trained on)", dep
    if spec == "auto":
        return (float(dep["range_m"]["p50"]),
                f"auto: the practical range of the measured residual's autocorrelation along the profile (rho <= 0.05, "
                f"median over {dep['time_samples']} measured time samples)", dep)
    v = _strict_float(spec.replace("m", ""))
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"--cv_buffer_m expects auto, 0 or a distance in m >= 0, got {cfg.cv.buffer_m!r}")
    return float(v), f"--cv_buffer_m {v:g}", dep


def filter_sim_morph(sm: Optional[Dict[str, Any]], intervals: List[Tuple[float, float]],
                     why: str) -> Optional[Dict[str, Any]]:
    """The simulated-conductor library without the conductors overlapping the given depth intervals (a CV fold's
    held-out block and its buffer: the fold's model sees no simulated conductor there either)."""
    if sm is None or not intervals:
        return sm
    rep_ = copy.deepcopy(sm["report"])
    used = list(rep_.get("used", []))
    keep = [i for i, r in enumerate(used) if not any(r["from_m"] <= hi and r["to_m"] >= lo for lo, hi in intervals)]
    if len(keep) == len(used):
        return sm
    rep_["withheld"] = list(rep_.get("withheld", [])) + [dict(used[i], withheld=why) for i in range(len(used))
                                                          if i not in keep]
    rep_["used"] = [used[i] for i in keep]
    rep_["n_used"] = len(keep)
    A = np.asarray(sm["shapes"], dtype=np.float64)[keep] if keep else np.zeros((0, MORPH_U.size))
    return {"shapes": A, "sigma_m": np.asarray(sm["sigma_m"])[keep], "center_m": np.asarray(sm["center_m"])[keep],
            "peak_time_ms": np.asarray(sm["peak_time_ms"])[keep], "report": rep_,
            "digest": array_digest(A) if A.size else "empty"}


def decomposition_evidence(meas: MeasuredData, cfg: AppConfig, factors: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
                           verbose: bool = True) -> Dict[str, Any]:
    """Data-side evidence for the decomposition (decomposition_evidence.json): (1) the dependence of the measured
    residual along the profile (correlation length, range, effective number of independent stations); (2) the scale
    stability of the measured anomaly inventory - the same rule at sigma_B x 0.5 .. 2: are the same anomalies found,
    how far do their peaks move, how much do their amplitudes and the background change; (3) the simulated profile's
    own decay trend as independent evidence for the '0 anomalies = monotone decay trend' rule; (4) the
    decay-constraint test."""
    kn = meas.known_time_mask()
    if meas.background is not None and meas.background_info is not None:
        B0, i0 = np.asarray(meas.background, dtype=np.float64), meas.background_info
    else:   # the measured-anchor mode: the rule's background, for the record
        B0, i0 = measured_background(meas, cfg, verbose=False)
    out: Dict[str, Any] = {"decay_constraint": meas.report.get("decay_constraint"),
                           "residual_dependence": residual_dependence((meas.L - B0)[:, kn], meas.depth)}
    sig0 = float(i0["sigma_m"])
    base = list(i0["measured_anomalies"]["anomalies"])
    sp = float(np.median(np.diff(meas.depth))) if meas.D > 1 else 1.0
    rows = []
    for f in factors:
        if abs(f - 1.0) < 1e-12:
            B, inf = B0, i0
        else:
            c = copy.deepcopy(cfg)
            c.profile.background_sigma = f"{sig0 * f:.6g}"
            B, inf = measured_background(meas, c, verbose=False)
        an = list(inf["measured_anomalies"]["anomalies"])
        matches = []
        for a0 in base:
            near = [a_ for a_ in an if abs(a_["peak_at_m"] - a0["peak_at_m"]) <= 2.0 * sp]
            b_ = min(near, key=lambda a_: abs(a_["peak_at_m"] - a0["peak_at_m"])) if near else None
            matches.append({"base_peak_at_m": float(a0["peak_at_m"]), "found": b_ is not None,
                            "peak_at_m": float(b_["peak_at_m"]) if b_ else None,
                            "from_m": float(b_["from_m"]) if b_ else None, "to_m": float(b_["to_m"]) if b_ else None,
                            "peak_ratio": (float(b_["peak_decades"] / a0["peak_decades"])
                                           if (b_ and abs(a0["peak_decades"]) > 0) else None)})
        rows.append({"sigma_factor": float(f), "sigma_m": float(sig0 * f), "n_anomalies": len(an),
                     "anomalies": [{"from_m": float(a_["from_m"]), "to_m": float(a_["to_m"]),
                                    "peak_at_m": float(a_["peak_at_m"]), "peak_decades": float(a_["peak_decades"]),
                                    "peak_time_ms": float(a_["peak_time_ms"])} for a_ in an],
                     "base_anomalies_found": int(sum(m_["found"] for m_ in matches)), "matches": matches,
                     "background_change_max_decades": float(np.abs(B - B0)[:, kn].max()),
                     "trend_decays_along_profile_at_all_times": (inf.get("trend") or {}).get(
                         "decays_along_profile_at_all_times")})
    ratios = [m_["peak_ratio"] for r_ in rows for m_ in r_["matches"] if m_["peak_ratio"] is not None]
    shifts = [abs(m_["peak_at_m"] - m_["base_peak_at_m"]) for r_ in rows for m_ in r_["matches"] if m_["found"]]
    out["scale_stability"] = {
        "rule": "the measured decomposition repeated with the background scale sigma_B x factor; a measured anomaly is "
                "found again when an inventory anomaly peaks within 2 station spacings of it",
        "base_sigma_m": sig0, "factors": [float(f) for f in factors], "rows": rows,
        "all_measured_anomalies_found_at_every_scale": bool(all(r_["base_anomalies_found"] == len(base) for r_ in rows)),
        "extra_anomalies_max": int(max(r_["n_anomalies"] for r_ in rows) - len(base)) if rows else 0,
        "peak_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
        "peak_shift_max_m": float(max(shifts)) if shifts else None,
        "background_change_max_decades": float(max(r_["background_change_max_decades"] for r_ in rows)) if rows else 0.0}
    sm = meas.sim_morph
    out["simulated_background_trend"] = (sm["report"].get("background_trend") if sm else None)
    out["measured_background_trend"] = {k: v for k, v in (i0.get("trend") or {}).items() if k != "undulations"}
    if verbose:
        dep = out["residual_dependence"]
        ss = out["scale_stability"]
        st = out["simulated_background_trend"] or {}
        print(f"[EVIDENCE] measured residual along the profile: correlation length {dep['correlation_length_m']['p50']:.1f} m, "
              f"range {dep['range_m']['p50']:.1f} m, n_eff {dep['n_eff']['p50']:.1f} of {dep['uniform_samples']} samples "
              f"({dep['stations']} stations) | inventory at sigma_B x {', '.join(f'{f:g}' for f in factors)}: measured "
              f"anomalies found at every scale = {ss['all_measured_anomalies_found_at_every_scale']}, peak shift <= "
              f"{(ss['peak_shift_max_m'] or 0.0):.1f} m, amplitude ratio "
              + (f"{ss['peak_ratio_range'][0]:.2f}-{ss['peak_ratio_range'][1]:.2f}" if ss["peak_ratio_range"] else "-")
              + f", background change <= {ss['background_change_max_decades']:.3f} decades"
              + (f" | simulated background decays along the profile at {st.get('share_of_times_decaying_along_profile', 0.0):.0%} "
                 f"of the times" if st else ""))
    return out


# conductive-sphere injection-recovery bench (identifiability of the decomposition)
SPHERE_BENCH_CLASSES = (("intersected", 0.0, 0.8), ("near", 1.2, 2.5), ("far", 2.5, 6.0))   # hole offset / radius
SPHERE_BENCH_AMP = (0.02, 0.6)          # largest |departure| of an injected response (decades), log-uniform
SPHERE_BENCH_FOOTPRINT = (1.0, 6.0)     # radius (intersected) or hole offset, x the station spacing, log-uniform
SPHERE_BENCH_NULL_SHARE = 1.0 / 6.0     # noise-only trials (false alarms)
SPHERE_BENCH_AMP_BINS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6001)


def sphere_dipole_response(z: np.ndarray, zc: float, d: float, phi: float, alpha: float, a: float, t_ms: np.ndarray,
                           tau_ms: float, quantity: str = "dbdt") -> np.ndarray:
    """Axial (z) response along a vertical hole of a conductive sphere in a uniform primary field: radius a, centre
    depth zc, hole at horizontal offset d and azimuth phi, primary field along (sin alpha, 0, cos alpha).  Space: the
    late-time induced dipole outside the sphere, B_z ~ (3 (m.r) r_z / r^5 - m_z / r^3), and the uniform interior
    field 2 m_z / a^3 where the hole passes through it.  Time: the sphere's eigenmode series of the step-off response
    (Wait 1951; Nabighian 1970), dB/dt ~ sum_n exp(-n^2 t / tau), B ~ sum_n n^-2 exp(-n^2 t / tau), tau = mu0 sigma
    a^2 / pi^2.  Arbitrary units (scaled by the caller)."""
    z = np.asarray(z, dtype=np.float64)
    rx, ry, rz = d * math.cos(phi), d * math.sin(phi), z - zc
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    mx, mz = math.sin(alpha), math.cos(alpha)
    rr = np.maximum(r, 1e-9)
    g = np.where(r > a, 3.0 * (mx * rx + mz * rz) * rz / rr ** 5 - mz / rr ** 3, 2.0 * mz / a ** 3)
    n = np.arange(1, 65, dtype=np.float64)[:, None]
    tt = np.asarray(t_ms, dtype=np.float64)[None, :] / max(float(tau_ms), 1e-12)
    w = 1.0 / (n * n) if str(quantity) == "b" else np.ones_like(n)
    h = np.sum(w * np.exp(-np.minimum(n * n * tt, 700.0)), axis=0)
    return g[:, None] * h[None, :]


def identifiability_bench(meas: MeasuredData, cfg: AppConfig, n_trials: int, seed: int = 0,
                          reports_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Injection-recovery test of the measured-profile decomposition against an independent forward model (review
    point: the background purification is circular - does it keep a true conductor response?).  Each trial adds the
    response of a conductive sphere (sphere_dipole_response; footprint, offset / radius class, primary field
    direction, azimuth and time constant at random) to the MEASURED profile, away from its own anomalies, scaled to a
    largest departure of 0.02-0.6 decades, and decomposes the result by the same rule (measured_background).  The
    injected departure D_true = L_obs - L_meas is known exactly; the decomposition splits it into the change of the
    anomaly residual D_rec = (L_obs - B_obs) - (L_meas - B_meas) and the change of the background B_obs - B_meas (the
    absorbed part)."""
    kn = meas.known_time_mask()
    tk = np.asarray(meas.grid_ms, dtype=np.float64)[kn]
    x = np.asarray(meas.depth, dtype=np.float64)
    sp = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    if meas.background is not None and meas.background_info is not None:
        B0, i0 = np.asarray(meas.background, dtype=np.float64), meas.background_info
    else:
        B0, i0 = measured_background(meas, cfg, verbose=False)
    base = list(i0["measured_anomalies"]["anomalies"])
    # the injections stay away from the anomalies of both undulation rules (one- and two-sided): with the same seed both
    # rules decompose the same trials (a paired comparison, trial by trial)
    zones = list(base)
    if (i0.get("trend") or {}).get("rule") == "measured":
        cfg_o = copy.copy(cfg)
        cfg_o.profile = copy.copy(cfg.profile)
        two = str(getattr(cfg.profile, "undulation_sides", "two") or "two").strip().lower() == "two"
        cfg_o.profile.undulation_sides = "one" if two else "two"
        mo = copy.copy(meas)
        mo.sim_morph, mo.background, mo.background_info = None, None, None
        zones += list(measured_background(mo, cfg_o, verbose=False)[1]["measured_anomalies"]["anomalies"])
    busy = [(float(a_["from_m"]) - 2.0 * sp, float(a_["to_m"]) + 2.0 * sp) for a_ in zones]
    R0 = meas.L - B0
    sig_b = float(i0.get("sigma_m", 2.0 * sp))
    noise = measured_station_noise(meas.L[:, kn], x)
    qty = str(cfg.generation.signal_quantity or "auto").strip().lower()
    law = "b" if qty == "b" else "dbdt"
    rng = np.random.default_rng(int(seed))
    n_trials = int(n_trials)
    n_null = int(round(n_trials * SPHERE_BENCH_NULL_SHARE))
    kinds = ["null"] * n_null + [SPHERE_BENCH_CLASSES[i % len(SPHERE_BENCH_CLASSES)][0] for i in range(n_trials - n_null)]
    kinds = [kinds[i] for i in rng.permutation(len(kinds))]
    Vlin = np.power(10.0, np.clip(meas.L, -30.0, 30.0))
    ms = copy.copy(meas)
    ms.sim_morph = None
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    if verbose:
        print(f"[IDENT] sphere injection-recovery bench: {n_trials} trials ({n_null} noise-only nulls) on the measured "
              f"profile, away from its anomalies " + ", ".join(f"{a_['from_m']:.0f}-{a_['to_m']:.0f} m" for a_ in base)
              + f" | {law} eigenmode series, largest departure {SPHERE_BENCH_AMP[0]:g}-{SPHERE_BENCH_AMP[1]:g} decades")
    for i, kind in enumerate(kinds):
        row: Dict[str, Any] = {"trial": i, "class": kind}
        dtrue = None
        if kind == "null":
            Lo = meas.L.copy()
            Lo[:, kn] = Lo[:, kn] + rng.normal(size=(meas.D, int(kn.sum()))) * noise[None, :]
        else:
            lo_c, hi_c = next((l_, h_) for n_, l_, h_ in SPHERE_BENCH_CLASSES if n_ == kind)
            ell = sp * math.exp(rng.uniform(math.log(SPHERE_BENCH_FOOTPRINT[0]), math.log(SPHERE_BENCH_FOOTPRINT[1])))
            if kind == "intersected":
                a_r = ell
                d = float(rng.uniform(lo_c, hi_c)) * a_r
            else:
                d = ell
                a_r = d / float(rng.uniform(lo_c, hi_c))
            ext = 3.0 * ell
            zc = None
            for _ in range(400):
                z_ = float(rng.uniform(x[0], x[-1]))
                if not any(z_ + ext >= l_ and z_ - ext <= h_ for l_, h_ in busy):
                    zc = z_
                    break
            if zc is None:
                row["skipped"] = "no stretch of the profile away from its anomalies"
                rows.append(row)
                continue
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            alpha = float(rng.uniform(0.0, 0.5 * math.pi))
            tau = math.exp(rng.uniform(math.log(2.0 * tk[0]), math.log(0.5 * tk[-1])))
            amp = math.exp(rng.uniform(math.log(SPHERE_BENCH_AMP[0]), math.log(SPHERE_BENCH_AMP[1])))
            rel = sphere_dipole_response(x, zc, d, phi, alpha, a_r, meas.grid_ms, tau, law) / Vlin
            m_ = float(np.abs(rel[:, kn]).max())
            row.update({"centre_m": zc, "offset_m": d, "radius_m": a_r, "footprint_scale_m": ell,
                        "footprint_over_sigma_b": ell / max(sig_b, 1e-9),
                        "phi_deg": math.degrees(phi), "alpha_deg": math.degrees(alpha), "tau_ms": tau, "target_amp": amp})
            if not np.isfinite(m_) or m_ <= 0.0:
                row["skipped"] = "no response in the measured window"
                rows.append(row)
                continue

            def _dep(lg: float) -> np.ndarray:
                return np.log10(np.maximum(1.0 + (10.0 ** lg) * rel, 10.0 ** -1.5))

            la, lb = math.log10(1e-6 / m_), math.log10(1e3 / m_)
            for _ in range(60):
                lm = 0.5 * (la + lb)
                if float(np.abs(_dep(lm)[:, kn]).max()) < amp:
                    la = lm
                else:
                    lb = lm
            dtrue = _dep(lb)
            dtrue[:, ~kn] = 0.0
            Lo = meas.L + dtrue
        ms.L, ms.background, ms.background_info = Lo, None, None
        Bo, io_ = measured_background(ms, cfg, verbose=False)
        an = list(io_["measured_anomalies"]["anomalies"])
        is_base = [any(abs(a_["peak_at_m"] - b_["peak_at_m"]) <= 2.0 * sp for b_ in base) for a_ in an]
        row["base_anomalies_found"] = int(sum(any(abs(a_["peak_at_m"] - b_["peak_at_m"]) <= 2.0 * sp for a_ in an)
                                              for b_ in base))
        if dtrue is None:
            row["false_anomalies"] = int(sum(1 for b_ in is_base if not b_))
        else:
            amp = float(row["target_amp"])
            F = (np.abs(dtrue) >= max(0.1 * amp, 0.005)) & kn[None, :]
            st = np.nonzero(F.any(axis=1))[0]
            z_lo, z_hi = float(x[st[0]]), float(x[st[-1]])
            drec = (Lo - Bo) - R0
            den = float(np.sum(dtrue[F] ** 2))
            ii, jj = np.unravel_index(int(np.argmax(np.where(F, np.abs(dtrue), -1.0))), F.shape)
            hit = [a_ for a_, b_ in zip(an, is_base) if (not b_) and a_["to_m"] >= z_lo - 1e-9 and a_["from_m"] <= z_hi + 1e-9]
            out_ = ~F.any(axis=1)
            row.update({"peak_departure_true": float(dtrue[ii, jj]), "peak_at_m_true": float(x[ii]),
                        "peak_time_ms_true": float(meas.grid_ms[jj]), "footprint_from_m": z_lo, "footprint_to_m": z_hi,
                        "footprint_stations": int(st.size),
                        "has_rise": bool(np.any(dtrue[F] > 0.1 * amp)), "has_dip": bool(np.any(dtrue[F] < -0.1 * amp)),
                        "detected": bool(hit),
                        "localisation_error_m": (float(min(abs(a_["peak_at_m"] - x[ii]) for a_ in hit)) if hit
                                                 else float("nan")),
                        "energy_retention": float(np.sum((drec * dtrue)[F]) / max(den, 1e-12)),
                        "peak_retention": float(drec[ii, jj] / dtrue[ii, jj]),
                        "rel_pattern_error": float(math.sqrt(float(np.sum((drec - dtrue)[F] ** 2)) / max(den, 1e-12))),
                        "background_leakage_outside_max": (float(np.abs(Bo - B0)[out_][:, kn].max()) if out_.any()
                                                           else 0.0),
                        "false_anomalies": int(sum(1 for a_, b_ in zip(an, is_base) if (not b_) and a_ not in hit))})
        rows.append(row)
        if verbose and (i + 1) % max(1, n_trials // 10) == 0:
            print(f"[IDENT]   {i + 1}/{n_trials} trials ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    ok = df[(df["class"] != "null") & (df["skipped"].isna() if "skipped" in df.columns else True)].copy()
    for c_ in ("detected", "has_dip", "has_rise"):
        if c_ in ok.columns:
            ok[c_] = ok[c_].astype(bool)
    if "detected" not in ok.columns:
        ok["detected"], ok["has_dip"], ok["has_rise"] = False, False, False
    nul = df[df["class"] == "null"]
    bins = list(zip(SPHERE_BENCH_AMP_BINS[:-1], SPHERE_BENCH_AMP_BINS[1:]))
    table = []
    for cls in [c_[0] for c_ in SPHERE_BENCH_CLASSES] + ["all"]:
        sc = ok if cls == "all" else ok[ok["class"] == cls]
        for lo_, hi_ in bins:
            s_ = sc[(sc["target_amp"] >= lo_) & (sc["target_amp"] < hi_)] if len(sc) else sc
            if not len(s_):
                continue
            table.append({"class": cls, "amp_from": lo_, "amp_to": min(hi_, SPHERE_BENCH_AMP[1]), "n": int(len(s_)),
                          "detection_rate": float(s_["detected"].mean()),
                          "energy_retention": _pct3(s_["energy_retention"]),
                          "peak_retention_p50": float(s_["peak_retention"].median()),
                          "localisation_error_m_p50": float(s_["localisation_error_m"].median()),
                          "background_leakage_outside_p50": float(s_["background_leakage_outside_max"].median()),
                          "false_anomalies_mean": float(s_["false_anomalies"].mean())})
    meas_level = float(min(abs(a_["peak_decades"]) for a_ in base)) if base else float("nan")
    head: Dict[str, Any] = {"measured_anomaly_level_decades": meas_level}
    for cls in [c_[0] for c_ in SPHERE_BENCH_CLASSES] + ["all"]:
        sc = ok if cls == "all" else ok[ok["class"] == cls]
        if not len(sc):
            continue
        strong = sc[sc["target_amp"] >= meas_level] if np.isfinite(meas_level) else sc
        c50 = None
        for lo_, hi_ in bins:
            above_ = sc[sc["target_amp"] >= lo_]
            sub_ = [float(above_[(above_["target_amp"] >= l2) & (above_["target_amp"] < h2)]["detected"].mean())
                    for l2, h2 in bins if l2 >= lo_ and len(above_[(above_["target_amp"] >= l2) & (above_["target_amp"] < h2)])]
            if sub_ and min(sub_) >= 0.5:
                c50 = float(lo_)
                break
        nar = sc[sc["footprint_over_sigma_b"] <= 1.0]
        head[cls] = {"n": int(len(sc)), "detection_rate": float(sc["detected"].mean()),
                     "footprint_within_sigma_b": {"n": int(len(nar)),
                                                  "detection_rate": float(nar["detected"].mean()) if len(nar) else None,
                                                  "energy_retention": _pct3(nar["energy_retention"]) if len(nar) else None},
                     "at_measured_level": {"n": int(len(strong)),
                                           "detection_rate": float(strong["detected"].mean()) if len(strong) else None,
                                           "energy_retention": _pct3(strong["energy_retention"]) if len(strong) else None},
                     "completeness_50_from_decades": c50,
                     "energy_retention": _pct3(sc["energy_retention"]),
                     "with_dip": {"n": int(sc["has_dip"].sum()),
                                  "energy_retention": _pct3(sc[sc["has_dip"]]["energy_retention"]) if sc["has_dip"].any()
                                  else None}}
    res = {"rule": ("conductive-sphere responses (eigenmode series in time, induced dipole / uniform interior field along "
                    "the hole) injected into the measured profile away from its anomalies and decomposed by the measured "
                    "rule; D_true known exactly; retention = projection of the recovered anomaly change on D_true over "
                    "the footprint (|D_true| >= 10 % of the target); detection = a new inventory anomaly over the "
                    "footprint; nulls = the measured profile + white noise at its station noise"),
           "quantity": law, "trials": n_trials, "null_trials": int(len(nul)),
           "skipped": int(df["skipped"].notna().sum()) if "skipped" in df.columns else 0,
           "classes": {n_: {"offset_over_radius": [l_, h_]} for n_, l_, h_ in SPHERE_BENCH_CLASSES},
           "amplitude_range_decades": list(SPHERE_BENCH_AMP), "footprint_range_spacings": list(SPHERE_BENCH_FOOTPRINT),
           "background_sigma_m": sig_b, "undulation_sides": (i0.get("trend") or {}).get("undulations", {}).get("sides"),
           "injections_avoid_m": [[round(l_, 3), round(h_, 3)] for l_, h_ in busy],
           "injections_avoid_rule": ("the anomalies of both undulation rules (one- and two-sided) +- 2 station spacings, "
                                     "and 3 footprint scales around each injection: the same seed gives the same trials "
                                     "under either rule (paired)"),
           "headline": head, "table": table,
           "null": {"n": int(len(nul)), "false_anomalies_mean": float(nul["false_anomalies"].mean()) if len(nul) else None,
                    "share_with_false_anomaly": float((nul["false_anomalies"] > 0).mean()) if len(nul) else None,
                    "measured_anomalies_found_share": (float((nul["base_anomalies_found"] == len(base)).mean())
                                                       if len(nul) else None)},
           "seconds": float(time.time() - t0)}
    if reports_dir:
        df.to_csv(os.path.join(reports_dir, "identifiability_trials.csv"), index=False)
        save_json(res, os.path.join(reports_dir, "identifiability_bench.json"))
    if verbose:
        ha = head.get("all", {})
        hm = ha.get("at_measured_level") or {}
        dr_m = hm.get("detection_rate")
        print(f"[IDENT] {len(ok)} injections: detected {ha.get('detection_rate', float('nan')):.1%}; at the measured "
              f"anomaly level (>= {meas_level:.3f} decades) "
              f"{float('nan') if dr_m is None else dr_m:.1%}, energy retention p50 "
              f"{(hm.get('energy_retention') or {}).get('p50', float('nan')):.2f} | "
              + " | ".join(f"{c_}: det {head[c_]['detection_rate']:.0%}, 50 % from "
                           f"{head[c_]['completeness_50_from_decades'] if head[c_]['completeness_50_from_decades'] is not None else '-'} dec, "
                           f"retention p50 {head[c_]['energy_retention']['p50']:.2f}"
                           for c_ in [c[0] for c in SPHERE_BENCH_CLASSES] if c_ in head)
              + f" | nulls: {res['null']['false_anomalies_mean'] if res['null']['n'] else 0:.2f} false anomalies per "
              f"profile ({time.time() - t0:.0f} s)")
    return res


def _vendi_sqdist(D2: np.ndarray, h: float) -> float:
    n = int(D2.shape[0])
    if n < 2 or h <= 0:
        return 1.0
    lam = np.linalg.eigvalsh(np.exp(-D2 / (2.0 * h * h)) / n)
    lam = lam[lam > 1e-12]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def diversity_metrics(X: np.ndarray, sizes: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)) -> Dict[str, Any]:
    """Effective diversity of a set of samples X (n, p) beyond count, variance or mean distance: the Vendi score (exp
    of the entropy of the eigenvalues of K / n, RBF kernel at the median pairwise distance; Friedman & Dieng 2023) =
    the effective number of distinct samples, its saturation curve over nested subsets (the bandwidth fixed), the
    effective rank of the centred set (Roy & Vetterli 2007), the participation ratio of its covariance spectrum and
    the share of near duplicates (nearest neighbour < 1 % of the median distance)."""
    X = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
    n, p = X.shape
    sq = np.einsum("ij,ij->i", X, X)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    iu = np.triu_indices(n, 1)
    dist = np.sqrt(D2[iu]) if n > 1 else np.zeros(1)
    h = float(np.median(dist)) if dist.size else 0.0
    sat = [{"n": int(m), "vendi": _vendi_sqdist(D2[:m, :m], h)} for m in sorted(set([m for m in sizes if m < n] + [n]))]
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(Xc @ Xc.T), 0.0, None)
    sv = np.sqrt(lam)
    pr_ = sv / max(float(sv.sum()), 1e-300)
    pr_ = pr_[pr_ > 0]
    nn = np.sqrt(np.min(D2 + np.diag(np.full(n, np.inf)), axis=1)) if n > 1 else np.zeros(1)
    return {"n": int(n), "dims": int(p), "bandwidth": h, "vendi": sat[-1]["vendi"], "saturation": sat,
            "vendi_growth_last_doubling": (float(sat[-1]["vendi"] / sat[-2]["vendi"]) if len(sat) > 1 and
                                           sat[-2]["n"] * 2 >= sat[-1]["n"] else None),
            "effective_rank": float(np.exp(-np.sum(pr_ * np.log(pr_)))) if pr_.size else 0.0,
            "participation_ratio": float(lam.sum() ** 2 / max(float(np.sum(lam * lam)), 1e-300)),
            "pairwise_rms_decades_p50": float(np.median(dist) / math.sqrt(max(p, 1))),
            "nn_rms_decades_p50": float(np.median(nn) / math.sqrt(max(p, 1))),
            "near_duplicate_share": float(np.mean(nn < 0.01 * h)) if h > 0 else 1.0}


def diversity_report(L_gen: np.ndarray, anchors: Optional[np.ndarray], gate_cols: np.ndarray, trace: Dict[str, Any],
                     trainer: Any, n_max: int = 2048, seed: int = 0, reports_dir: Optional[str] = None,
                     verbose: bool = True) -> Dict[str, Any]:
    """Generated volume is not information (diversity_report.json): on a random subset (<= n_max) of the accepted
    profiles, at the measured stations and gates, diversity_metrics of the whole profiles, of their anomaly part
    (profile minus its anchor) and of their backgrounds (the anchors); the composition of the labelled anomalies
    (kinds, and for the morphology kind measured shapes / simulated shapes / learned morphologies, the measured
    anomalies kept in place); the morphology bank's own effective size by source (Vendi of the examples alone vs
    examples + learned)."""
    N = int(L_gen.shape[0])
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(N)[:min(N, max(int(n_max), 2))]
    gc = np.asarray(gate_cols, dtype=np.int64)
    P3 = np.asarray(L_gen, dtype=np.float64)[perm][:, :, gc]
    out: Dict[str, Any] = {"generated_profiles": N, "subset": int(perm.size), "stations": int(P3.shape[1]),
                           "gates": int(gc.size), "sets": {"profile": diversity_metrics(P3)}}
    if anchors is not None:
        A3 = np.asarray(anchors, dtype=np.float64)[perm][:, :, gc]
        out["sets"]["anomaly_part"] = diversity_metrics(P3 - A3)
        out["sets"]["background_part"] = diversity_metrics(A3)
    prm = trace.get("params") or {}
    comp: Dict[str, Any] = {}
    if "an_K" in prm and "an_kind" in prm:
        K_ = np.asarray(prm["an_K"]).astype(np.int64).reshape(-1)
        kd = np.asarray(prm["an_kind"]).astype(np.int64)
        kd = kd.reshape(K_.size, -1)
        on = np.arange(kd.shape[1])[None, :] < K_[:, None]
        names = ("intersected", "off_hole", "zone", "morphology")
        comp["synthetic_anomalies"] = int(on.sum())
        comp["by_kind"] = {names[k]: int((on & (kd == k)).sum()) for k in range(len(names))}
        if "an_src" in prm:
            sv = np.asarray(prm["an_src"]).astype(np.int64).reshape(kd.shape)
            m3 = on & (kd == 3)
            comp["morphology_by_source"] = {nm: int((m3 & (sv == j)).sum())
                                            for j, nm in enumerate(("measured_shape", "simulated_shape", "learned"))}
    if "ma_on" in prm:
        comp["measured_anomalies_kept_in_place"] = int((np.asarray(prm["ma_on"]) > 0.5).sum())
    out["composition"] = comp
    bank = getattr(trainer, "morph_bank", None)
    if bank is not None and len(bank.get("patches", [])):
        Pb = np.asarray(bank["patches"], dtype=np.float64)
        kt_ = int(np.sum(trainer.meas.known_time_mask()))
        Pb = Pb[:, ::4, :kt_][:, :, ::4]
        srcb = np.asarray(bank["src"])
        ex = srcb < 2
        out["bank"] = {"examples": int(ex.sum()), "learned": int((~ex).sum()),
                       "vendi_examples": diversity_metrics(Pb[ex])["vendi"] if ex.sum() > 1 else float(ex.sum()),
                       "vendi_examples_and_learned": diversity_metrics(Pb)["vendi"] if len(Pb) > 1 else 1.0,
                       "learned_filter": (getattr(trainer, "morph_report", {}) or {}).get("learned")}
    if reports_dir:
        save_json(out, os.path.join(reports_dir, "diversity_report.json"))
    if verbose:
        s_ = out["sets"]
        pr = s_["profile"]
        msg = (f"[DIVERSITY] {N} profiles (subset {perm.size}): Vendi effective count {pr['vendi']:.1f} (whole profile)"
               + (f", {s_['anomaly_part']['vendi']:.1f} (anomaly part), {s_['background_part']['vendi']:.1f} (background)"
                  if "anomaly_part" in s_ else "")
               + f" | last doubling x{(pr['vendi_growth_last_doubling'] or float('nan')):.3f} | effective rank "
               f"{pr['effective_rank']:.1f} | near duplicates {pr['near_duplicate_share']:.2%}")
        if "bank" in out:
            msg += (f" | morphology bank: Vendi {out['bank']['vendi_examples']:.1f} over {out['bank']['examples']} "
                    f"measured / simulated examples -> {out['bank']['vendi_examples_and_learned']:.1f} with "
                    f"{out['bank']['learned']} learned")
        print(msg)
    return out
