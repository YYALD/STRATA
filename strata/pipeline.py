"""Data contract of a run (load_data_contract): survey mode and extension plan, the measured background and anomaly
inventory, the simulated conductors as shape examples, the envelopes of unmeasured stations and times, the
normaliser, and the audit of what the acceptance selects."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import RFMG_PROGRAM_NAME, RFMG_VERSION
from .common import array_digest, stable_json_dumps
from .config import AppConfig
from .data.anomaly_laws import (anomaly_amp_range, borehole_physical_anomalies, measured_anomaly_mode,
                                measured_anomaly_scale_ranges, template_width_sigma)
from .data.contract import MeasuredData, SimPrior, align_simulation, preprocess_measured, synthetic_tem_profile
from .data.decomposition import (_DECAY, UNDULATION_SPLIT_SADDLE, _tmono_np, background_decomposition, background_scale,
                                 decay_constraint_evidence, highpass_profile_np, local_poly_smoother_np,
                                 measured_anomaly_inventory, measured_anomaly_templates, measured_trend_undulations,
                                 purify_background, undulation_inventory)
from .data.extension import anchor_profile_on, calibrate_unmeasured, extension_stations
from .data.normalization import NormalizationState, fit_normalizer_log
from .data.readers import (SURVEY_MODES, ProfileMatrix, SurveySpec, _strict_float, build_survey_spec,
                           detect_survey_mode, is_survey_line_file, read_measured_file, read_profile_matrix,
                           read_survey_lines)
from .data.time_grid import depth_interp_profile
from .morphology import MORPH_U
from .quality import PURITY_ANCHOR_SHARE


def _parse_gain(v: str) -> Any:
    try:
        return float(v)
    except Exception:
        return "auto"


def resolve_survey(cfg: AppConfig, pm: ProfileMatrix) -> SurveySpec:
    """Survey mode of the run: the requested mode (menu / --survey_mode), or the one detected from the measured file
    (auto); the mode written back into the config is the one in force."""
    req = str(cfg.profile.survey_mode or "auto").lower()
    det, why = detect_survey_mode(pm)
    if req in SURVEY_MODES:
        reason = f"--survey_mode {req}" + ("" if req == det else f" (the measured file suggests {det}: {why})")
        mode = req
    else:
        mode, reason = det, f"auto-detected: {why}"
    cfg.profile.survey_mode = mode
    return build_survey_spec(mode, pm, cfg.data.sensor_height, reason)


def resolve_extension_plan(cfg: AppConfig, pm: ProfileMatrix, survey: SurveySpec) -> Dict[str, Any]:
    """Profile-axis and late-time extensions of the run (both off by default = the measured coverage only)."""
    P = cfg.profile
    frac = max(0.0, float(P.extend_profile_frac))
    spec = str(P.extend_time_ms or "0").strip().lower()
    t_off = spec in ("", "0", "0.0", "off", "none")
    side = str(P.extend_side or "auto").strip().lower()
    if side == "auto":
        side = "high" if survey.mode == "borehole" else "both"
    if side not in ("low", "high", "both"):
        raise ValueError(f"--extend_side expects auto|low|high|both, got {P.extend_side!r}")
    if (frac > 0 or not t_off) and (int(cfg.cv.folds) > 0 or float(cfg.cv.time_holdout_decades) > 0):
        # the folds score held-out MEASURED stations / gates; the edge folds and the time fold test the extrapolation
        print("[EXTENSION] cross-validation run: no extension is built (the edge blocks and the time-direction fold test "
              "the extrapolation rules on measured data)")
        frac, spec, t_off = 0.0, "0", True
    if spec == "depth" and not (survey.mode == "borehole" and frac > 0 and side in ("high", "both")):
        raise ValueError("--extend_time_ms depth pairs the late times with a deeper borehole extension: it needs "
                         "--survey_mode borehole and --extend_profile_frac > 0 toward the deep end (--extend_side high)")
    if spec == "depth" and not 0.0 < float(pm.depth[0]) < float(pm.depth[-1]):
        raise ValueError(f"--extend_time_ms depth needs a positive depth axis increasing downhole (read "
                         f"{pm.depth[0]:g} .. {pm.depth[-1]:g}); give the target time in ms instead")
    axis, known = extension_stations(pm.depth, frac, side, nonnegative=(survey.mode == "borehole"))
    t_last = float(pm.time_ms[-1])
    if t_off:
        t_ext = 0.0
    elif spec == "depth":
        d_last, d_end = float(pm.depth[-1]), float(axis[-1])
        t_ext = t_last * (d_end / d_last) ** 2
    else:
        t_ext = _strict_float(spec.replace("ms", ""))
        if not np.isfinite(t_ext):
            raise ValueError(f"--extend_time_ms expects 0 / off, a time in ms or depth, got {P.extend_time_ms!r}")
        if t_ext <= t_last:
            raise ValueError(f"--extend_time_ms {t_ext:g} must exceed the last measured gate {t_last:g} ms")
    return {"profile_frac": frac, "side": side if frac > 0 else "", "axis": axis, "known": known,
            "n_virtual_stations": int((~known).sum()), "t_ext_ms": float(t_ext), "time_spec": spec}


def measured_background(meas: MeasuredData, cfg: AppConfig, verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Smooth geological background of the measured profile (background_decomposition at the survey's scale rule),
    the purity tolerance (half the smallest labelled anomaly amplitude, or --purity_tol) and the inventory of the
    measured anomalies the background leaves out (they are not copied into any generated profile)."""
    kn = meas.known_time_mask()
    sv = meas.survey
    mode = sv.mode if sv is not None else "borehole"
    family = sv.anomaly_family if sv is not None else "borehole"
    sig, why = background_scale(meas.L[:, kn], meas.depth, meas.grid_ms[kn], mode, cfg.profile.background_sigma)
    amp_lo = anomaly_amp_range(meas.L[:, kn], meas.depth, cfg.profile.anomaly_amp_range, family=family,
                               t_ms=meas.grid_ms[kn])[0]
    spec = str(cfg.profile.purity_tol or "auto").strip().lower()
    if spec in ("", "auto"):
        tol, tol_rule = 0.5 * float(amp_lo), f"auto: half the smallest labelled anomaly amplitude ({amp_lo:.4f} decades)"
    else:
        tol = _strict_float(spec)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError(f"--purity_tol expects auto or a positive number of decades, got {cfg.profile.purity_tol!r}")
        tol_rule = f"--purity_tol {tol:g}"
    # the background's shape rule ("0 anomalies" = the overall decay trend) and the inventory's noise rule; "measured" =
    # the measured profile itself off its undulations, the inventory = the undulations
    trend_rule = str(cfg.profile.background_trend or "auto").strip().lower()
    if trend_rule == "auto":
        trend_rule = "measured" if mode == "borehole" else "smooth"
    noise_rule = str(cfg.profile.inventory_noise or "auto").strip().lower()
    if noise_rule == "auto":
        noise_rule = "quiet" if mode == "borehole" else "mad"
    B0, _ = background_decomposition(meas.L, meas.depth, meas.grid_ms, sig)
    eps = PURITY_ANCHOR_SHARE * float(tol)
    B, n_pass, hp_b = purify_background(B0, meas.depth, sig, eps, decay_along=(trend_rule in ("monotone", "measured")))
    und = None
    if trend_rule == "measured":
        B_shape = B
        two = str(getattr(cfg.profile, "undulation_sides", "two") or "two").strip().lower() == "two"
        B, und = measured_trend_undulations(meas.L, meas.depth, meas.grid_ms, kn, B_shape, tol, two_sided=two,
                                            sigma_m=sig, eps=eps, B_start=B0, start_thr=0.5 * float(amp_lo))
        hp_b = float(np.abs(B - local_poly_smoother_np(meas.depth, sig) @ B).max())
    down = (B[-1] <= B[0])[None, :]
    trend_info: Dict[str, Any] = {"rule": trend_rule, "decays_along_profile_at_all_times": bool(np.all(down)),
                                  "largest_step_against_trend_decades": float(
                                      np.where(down, np.diff(B, axis=0), -np.diff(B, axis=0)).max()) if B.shape[0] > 1 else 0.0}
    if trend_rule in ("monotone", "measured"):
        Bs, _, _ = purify_background(B0, meas.depth, sig, eps)
        dts = Bs - B
        trend_info.update({"change_vs_smooth_background_max_decades": float(np.abs(dts).max()),
                           "cells_changed_more_than_0.001_decades": float(np.mean(np.abs(dts) > 1e-3)),
                           "smooth_background_largest_step_against_trend_decades": float(
                               np.where(down, np.diff(Bs, axis=0), -np.diff(Bs, axis=0)).max())})
    if und is not None:
        dmt = B - B_shape
        off = ~und["footprint"]
        trend_info.update({"undulations": und["info"],
                           "change_vs_v20_1_trend_max_decades": float(np.abs(dmt).max()),
                           "change_vs_v20_1_trend_off_undulations_max_decades": float(
                               np.abs(dmt[off][:, kn]).max()) if off.any() else 0.0,
                           "residual_off_undulations_max_decades": float(
                               np.abs((meas.L - B)[off][:, kn]).max()) if off.any() else 0.0,
                           "trend_highpass_max_decades": float(hp_b),
                           "measured_highpass_off_undulations_max_decades": float(np.abs(highpass_profile_np(
                               meas.L[:, kn], meas.depth, sig))[off].max()) if off.any() else 0.0})
    R = meas.L - B
    if und is not None:
        inv = undulation_inventory(R, meas.depth, meas.grid_ms, kn, und["footprint"], und["threshold"],
                                   split_signs=und["info"].get("sides") != "two")
        inv["noise_rule"] = "undulation"
        inv["noise_decades_median"] = und["info"]["noise_decades_median"]
    else:
        inv = measured_anomaly_inventory(R[:, kn], meas.depth, meas.grid_ms[kn], floor=tol, noise_rule=noise_rule)
        inv["noise_rule"] = noise_rule
    ar = np.abs(R[:, kn])
    ch = B - B0
    info = {"mode": "pure", "sigma_m": float(sig), "scale_rule": why, "method": "robust local quadratic along the profile "
            "(Gaussian kernel), Tukey biweight per time sample with the weights smoothed over 0.1 decades of log time, "
            "8 iterations, non-increasing decays; then smoothed again with the local-quadratic smoother until its own "
            "anomaly-scale content is <= a quarter of the purity tolerance" + (
                "; [v20.1] every pass followed by the greatest monotone minorant along the profile (the overall decay "
                "trend: no rise, no dip)" if trend_rule in ("monotone", "measured") else "") + (
                "; [v20.2] the trend is the measured profile itself off its undulations (rises above the lowest "
                "supporting chord by more than max(4 x station noise, tolerance)); under an undulation that smooth trend, "
                "tied to the measured values beside it, at the times the undulation is there" if trend_rule == "measured"
                else "") + ("; [v22] undulations two-sided: departures above or below the trimmed least-squares decay "
                            "trend (rises and dips), the trend's shape under them from that trimmed trend"
                            if (und is not None and und["info"].get("sides") == "two") else ""),
            "purity_tol_decades": float(tol),
            "purity_tol_rule": tol_rule, "trend": trend_info,
            "purification": {"eps_decades": float(eps), "passes": int(n_pass), "highpass_max_decades": float(hp_b),
                             "change_vs_robust_fit_rms_decades": float(np.sqrt(np.mean(ch ** 2))),
                             "change_vs_robust_fit_max_decades": float(np.abs(ch).max())},
            "residual_abs_decades": {"p50": float(np.percentile(ar, 50)), "p95": float(np.percentile(ar, 95)),
                                     "max": float(ar.max())},
            "measured_anomalies": inv, "background_digest": array_digest(B),
            "measured_anomaly_policy": {"prob": float(cfg.profile.measured_anomaly_prob),
                                        "scale": str(cfg.profile.measured_anomaly_scale),
                                        "mode": measured_anomaly_mode(cfg, family),
                                        "max_anomalies": int(cfg.profile.max_anomalies),
                                        "note": "each measured anomaly = the residual over its run (measured shape), "
                                                "scaled per profile; profiles with 0 anomalies carry none" + (
                                                    "; mode count: each one of the profile's anomalies (the count "
                                                    "includes them)" if measured_anomaly_mode(cfg, family) == "count"
                                                    else "")}}
    if verbose:
        runs = inv["anomalies"]
        top = sorted(runs, key=lambda r_: -abs(r_["peak_decades"]))[:6]
        print(f"[BACKGROUND] pure: anchors on the smooth geological background of the measured profile (robust local "
              f"quadratic, sigma {sig:.4g} m: {why}) | measured minus background |p50/p95/max| = {info['residual_abs_decades']['p50']:.4f}/"
              f"{info['residual_abs_decades']['p95']:.4f}/{info['residual_abs_decades']['max']:.4f} decades")
        if und is None:
            print(f"[BACKGROUND] purification: {n_pass} local-quadratic passes until the background's own anomaly-scale "
                  f"content <= {eps:.4f} decades (now {hp_b:.4f}); change vs the robust fit RMS "
                  f"{info['purification']['change_vs_robust_fit_rms_decades']:.4f}, max "
                  f"{info['purification']['change_vs_robust_fit_max_decades']:.4f} decades")
        else:
            print(f"[BACKGROUND] the trend's shape under the undulations: the robust fit with {n_pass} local-quadratic "
                  f"passes and the chord minorant (v20.1); the trend's own anomaly-scale content {hp_b:.4f} decades is the "
                  f"measured profile's bends where it is kept (the measured profile off the undulations: "
                  f"{trend_info['measured_highpass_off_undulations_max_decades']:.4f})")
        if und is not None:
            ui = und["info"]
            print(f"[BACKGROUND] [v20.2] '0 anomalies' = the measured profile's own decay trend: the measured profile off "
                  f"its undulations, bridged under {len(ui['runs'])} undulation run(s) "
                  + ", ".join(f"{w_['from_m']:.0f}-{w_['to_m']:.0f} m" for w_ in ui["runs"])
                  + " (parts: " + ", ".join(f"{w_['from_m']:.0f}-{w_['to_m']:.0f} m rising {w_['rise_significant_from_ms']:.3g}-"
                                            f"{w_['rise_significant_to_ms']:.3g} ms" for w_ in ui.get("parts", [])) + ")"
                  + f" (rise above the lowest supporting chord > max(4 x station noise {ui['noise_decades_median']:.4f}, "
                  f"tolerance), median threshold {ui['threshold_decades_median']:.4f} decades) | residual off the "
                  f"undulations <= {trend_info['residual_off_undulations_max_decades']:.4f}, change vs the v20.1 trend "
                  f"there up to {trend_info['change_vs_v20_1_trend_off_undulations_max_decades']:.4f} decades | largest "
                  f"step against the trend {trend_info['largest_step_against_trend_decades']:+.1e}"
                  + ("" if trend_info["decays_along_profile_at_all_times"] else " (some times rise along the profile)"))
        elif trend_rule == "monotone":
            print(f"[BACKGROUND] overall decay trend ('0 anomalies' profiles: no rise, no dip along the profile): the smooth "
                  f"background rose against its trend by up to "
                  f"{trend_info['smooth_background_largest_step_against_trend_decades']:.4f} decades per station step; the "
                  f"trend passes under it (change up to {trend_info['change_vs_smooth_background_max_decades']:.4f} "
                  f"decades, {trend_info['cells_changed_more_than_0.001_decades']:.1%} of the cells by > 0.001); now "
                  f"{trend_info['largest_step_against_trend_decades']:+.1e}"
                  + ("" if trend_info["decays_along_profile_at_all_times"] else " (some times rise along the profile)"))
            if trend_info["change_vs_smooth_background_max_decades"] > 5.0 * float(tol):
                i_, j_ = np.unravel_index(int(np.argmax(np.abs(dts))), dts.shape)
                print(f"[BACKGROUND] note: the smooth background departs from the overall decay trend by "
                      f"{trend_info['change_vs_smooth_background_max_decades']:.3f} decades (at {meas.depth[i_]:.4g} m, "
                      f"{meas.grid_ms[j_]:.4g} ms): that rise is labelled as part of a measured anomaly ('0 anomalies' "
                      f"decay along the profile); --background_trend smooth keeps it in the background")
        keep = float(cfg.profile.measured_anomaly_prob) > 0
        cnt_mode = measured_anomaly_mode(cfg, family) == "count"
        print(f"[BACKGROUND] {len(runs)} anomalies in the measured profile ("
              + ((f"kept as labelled anchor anomalies, measured shape, amplitude x {cfg.profile.measured_anomaly_scale} "
                  f"(log-uniform), " + ("each one of a profile's anomalies, drawn with the count" if cnt_mode else
                                        "in every profile with anomalies")) if keep else
                 "left out of every generated profile")
              + (f"; undulations, split at saddles below {inv.get('split_saddle', UNDULATION_SPLIT_SADDLE):g} x the lower "
                 f"maximum" if inv.get("rule") == "undulation" else
                 f"; threshold max(4 x station-scale noise ({noise_rule}), tolerance)")
              + f", median threshold {inv['threshold_decades_median']:.4f} decades): " + "; ".join(
                  f"{r_['from_m']:.0f}-{r_['to_m']:.0f} m peak {r_['peak_decades']:+.3f} at "
                  f"{r_.get('peak_at_m', float('nan')):.0f} m, {r_['peak_time_ms']:.3g} ms" for r_ in top)
              + (" ..." if len(runs) > len(top) else ""))
        print(f"[BACKGROUND] purity tolerance {tol:.4f} decades ({tol_rule}): off its labelled anomalies a generated "
              f"profile carries no bump or dip with a larger peak (up to the background scale in width; "
              + ("the anchor is the measured profile's own trend)" if und is not None else
                 f"the anchor itself <= {PURITY_ANCHOR_SHARE:g} x it at sigma_B)"))
    return B, info


def sim_role_of(cfg: AppConfig) -> str:
    """--sim_role auto: morphology with a pure background, texture otherwise."""
    role = str(cfg.data.sim_role or "auto").strip().lower()
    pure = str(cfg.profile.background_mode).strip().lower() == "pure"
    if role == "auto":
        return "morphology" if pure else "texture"
    if role == "texture" and pure:
        raise ValueError("--sim_role texture adds the simulated structure to the anchors: it needs --background_mode "
                         "measured (a pure background would carry it as an unlabelled anomaly)")
    if role == "morphology" and not pure:
        raise ValueError("--sim_role morphology labels the simulated anomalies: it needs --background_mode pure")
    return role


def _parse_depth_intervals(spec: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for tok in str(spec or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, sep, b = tok.partition("-")
        lo, hi = _strict_float(a), (_strict_float(b) if sep else float("nan"))
        if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo):
            raise ValueError(f"--sim_exclude_m expects 'lo-hi' depth intervals in m (comma list), got {tok!r}")
        out.append((float(lo), float(hi)))
    return out


def sim_morphology_library(sim: SimPrior, meas: MeasuredData, cfg: AppConfig,
                           verbose: bool = True) -> Optional[Dict[str, Any]]:
    """The simulated profile's conductors as labelled along-hole morphology examples (sim_role morphology). No
    simulated value enters a profile: the simulated decays sit 1-3 decades off the measured ones and floor at a
    constant level, so neither their level nor their time behaviour relative to that floor transfers.  What transfers
    is how a simulated conductor's response spreads along the hole."""
    if not sim.available:
        return None
    d0, d1 = float(meas.depth[0]), float(meas.depth[-1])
    kn = meas.known_time_mask()
    zq = sim.canonical_sim_depth(meas.depth, d0, d1)
    j0, j1, w = sim.rows_for(zq)
    Ls = (1.0 - w)[:, None] * sim.L_native_grid[j0] + w[:, None] * sim.L_native_grid[j1]       # (D, T) at the stations
    ms = copy.copy(meas)
    ms.L, ms.background, ms.background_info, ms.sim_morph = np.asarray(Ls, dtype=np.float64), None, None, None
    B, info = measured_background(ms, cfg, verbose=False)
    sp = float(np.median(np.diff(meas.depth))) if meas.depth.size > 1 else 1.0
    zf = np.arange(d0, d1 + 1e-9, max(sp / 5.0, 1e-3))
    j0f, j1f, wf = sim.rows_for(sim.canonical_sim_depth(zf, d0, d1))
    Lf = (1.0 - wf)[:, None] * sim.L_native_grid[j0f] + wf[:, None] * sim.L_native_grid[j1f]
    Ff = np.power(10.0, np.clip(Lf - depth_interp_profile(meas.depth, B, zf), -12.0, 12.0)) - 1.0
    a = Ff[:, kn].max(axis=1)
    floor = 10.0 ** (2.0 * float(info.get("purity_tol_decades", 0.01))) - 1.0
    pk = [i for i in range(a.size) if a[i] > floor and (i == 0 or a[i] >= a[i - 1]) and (i == a.size - 1 or a[i] > a[i + 1])]
    merged = True
    while merged and len(pk) > 1:
        merged = False
        for q in range(len(pk) - 1):
            i_, k_ = pk[q], pk[q + 1]
            if float(a[i_:k_ + 1].min()) >= 0.5 * min(a[i_], a[k_]):
                pk.pop(q + 1 if a[k_] <= a[i_] else q)
                merged = True
                break
    bounds = []
    for q, i_ in enumerate(pk):
        if q > 0:
            lo = pk[q - 1] + int(np.argmin(a[pk[q - 1]:i_ + 1]))
        else:
            below = np.nonzero(a[:i_] <= floor)[0]
            lo = int(below[-1]) if below.size else 0
        if q < len(pk) - 1:
            hi = i_ + int(np.argmin(a[i_:pk[q + 1] + 1]))
        else:
            below = np.nonzero(a[i_ + 1:] <= floor)[0]
            hi = i_ + 1 + int(below[0]) if below.size else a.size - 1
        bounds.append((lo, hi))
    excl = _parse_depth_intervals(cfg.data.sim_exclude_m)
    tg, kidx = meas.grid_ms, np.nonzero(kn)[0]
    shapes, sig, cen, tpk, used, withheld = [], [], [], [], [], []
    for i_, (lo, hi) in zip(pk, bounds):
        rec = {"from_m": float(zf[lo]), "to_m": float(zf[hi]), "maximum_at_m": float(zf[i_])}
        if any(rec["from_m"] <= hi_ and rec["to_m"] >= lo_ for lo_, hi_ in excl):
            rec["withheld"] = "--sim_exclude_m"
            withheld.append(rec)
            continue
        it = int(kidx[int(np.argmax(Ff[i_, kn]))])
        zz = zf[lo:hi + 1]
        g = Ff[lo:hi + 1, it] / max(float(Ff[i_, it]), 1e-12)
        if zz.size < 3:
            continue
        g = np.clip(g - (g[0] + (g[-1] - g[0]) * (zz - zz[0]) / max(zz[-1] - zz[0], 1e-12)), 0.0, None)
        if float(g.max()) <= 0:
            continue
        g = g / float(g.max())
        c = float(zz[int(np.argmax(g))])
        s_ = template_width_sigma(g[:, None], zz)
        if not np.isfinite(s_) or s_ < 0.5 * sp:
            rec["withheld"] = "narrower than half the station spacing (the survey cannot sample its shape)"
            withheld.append(rec)
            continue
        shapes.append(np.interp(c + s_ * MORPH_U, zz, g, left=0.0, right=0.0))
        sig.append(float(s_))
        cen.append(c)
        tpk.append(float(tg[it]))
        rec.update({"centre_m": c, "sigma_m": float(s_), "peak_time_ms": float(tg[it]),
                    "peak_decades_vs_simulated_trend": float(np.log10(1.0 + Ff[i_, it]))})
        used.append(rec)
    A = np.asarray(shapes, dtype=np.float64).reshape(-1, MORPH_U.size)
    dn = (B[-1] <= B[0])
    steps = (np.where(dn[None, :], np.diff(B, axis=0), -np.diff(B, axis=0)) if B.shape[0] > 1
             else np.zeros((1, B.shape[1])))
    rep = {"rule": "simulated conductors: maxima of the relative response change F = 10^(L - trend) - 1 along the hole "
                   "(fine axis, measured window), split at saddles below half the lower maximum; shape = F at the "
                   "conductor's peak time between its saddles minus the line through its ends, normalised, in "
                   "u = (z - centre) / sigma; no simulated level or time behaviour is used",
           "depth_registration": sim.report.get("depth_registration", sim.depth_mode),
           "fine_axis_step_m": float(zf[1] - zf[0]) if zf.size > 1 else 0.0,
           "trend_sigma_m": float(info["sigma_m"]), "n_conductors": len(pk), "n_used": len(used),
           "excluded_intervals_m": [list(e) for e in excl], "used": used, "withheld": withheld,
           "background_trend": {"rule": (info.get("trend") or {}).get("rule"),
                                "decays_along_profile_at_all_times": bool(np.all(dn[kn])),
                                "share_of_times_decaying_along_profile": float(np.mean(dn[kn])),
                                "largest_step_against_trend_decades": float(steps[:, kn].max()) if steps.size else 0.0},
           "level_offset_vs_measured_decades": sim.report.get("per_gate_level_difference_decades")}
    if verbose:
        print(f"[SIM-MORPH] simulated profile -> {len(used)} conductor shapes along the hole (of {len(pk)} maxima of its "
              f"relative response change"
              + (f"; {len(withheld)} withheld" if withheld else "") + "): "
              + "; ".join(f"{r_['centre_m']:.0f} m sigma {r_['sigma_m']:.1f} m" for r_ in used[:10])
              + (" ..." if len(used) > 10 else "")
              + " | no simulated value or time behaviour enters a profile")
    return {"shapes": A, "sigma_m": np.asarray(sig, dtype=np.float64), "center_m": np.asarray(cen, dtype=np.float64),
            "peak_time_ms": np.asarray(tpk, dtype=np.float64), "report": rep,
            "digest": array_digest(A) if A.size else "empty"}


def _ks_w1(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and Wasserstein-1 distance (b against a; NaN-free inputs)."""
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    x = np.concatenate([a, b])
    ks = float(np.max(np.abs(np.searchsorted(a, x, side="right") / a.size - np.searchsorted(b, x, side="right") / b.size)))
    q = np.linspace(0.0, 1.0, 201)
    return ks, float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance of two categorical samples."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    cats = np.union1d(a, b)
    pa = np.array([np.mean(a == c) for c in cats])
    pb = np.array([np.mean(b == c) for c in cats])
    return float(0.5 * np.abs(pa - pb).sum())


def qc_selection_audit(params: Dict[str, np.ndarray], status: np.ndarray) -> Dict[str, Any]:
    """What the acceptance selects: the accepted profiles' parameters against every drawn slot (the configured law).
    Counts / kinds / morphology sources: total-variation distance; amplitudes, widths, centres, time constants, peak
    times and background draws: KS statistic and Wasserstein-1 / the drawn law's interquartile range."""
    st = np.asarray(status).astype(int)
    acc = st == 0
    out: Dict[str, Any] = {"slots": int(st.size), "accepted": int(acc.sum()), "review": int((st == 1).sum()),
                           "rejected": int((st == 2).sum()), "categorical_tvd": {}, "continuous": {}}
    if st.size == 0 or not acc.any():
        return out
    if "an_K" in params:
        K = np.asarray(params["an_K"]).astype(int)
        if "ma_on" in params:
            K = K + (np.asarray(params["ma_on"]) > 0.5).sum(axis=1)
        out["categorical_tvd"]["anomaly_count"] = _tvd(K, K[acc])
        act = np.arange(np.asarray(params["an_A"]).shape[1])[None, :] < np.asarray(params["an_K"]).astype(int)[:, None]
        acc2 = act & acc[:, None]

        def _pair(key: str, f=None) -> Tuple[np.ndarray, np.ndarray]:
            v = np.asarray(params[key], dtype=np.float64)
            v = f(v) if f is not None else v
            return v[act & np.isfinite(v)], v[acc2 & np.isfinite(v)]

        if "an_kind" in params:
            out["categorical_tvd"]["anomaly_kind"] = _tvd(*_pair("an_kind"))
        if "an_src" in params:
            sv = np.asarray(params["an_src"])
            k3 = act & (np.asarray(params.get("an_kind", np.zeros_like(sv))) == 3)
            out["categorical_tvd"]["morphology_source"] = _tvd(sv[k3], sv[k3 & acc[:, None]])
        for key, f, name in (("an_A", np.abs, "abs_peak_decades"), ("an_w0", None, "width_m"),
                             ("an_z0", None, "centre_m"), ("an_ltau", None, "log10_tau_ms"),
                             ("lab_peak_ms", None, "peak_time_ms"), ("an_s", None, "size_factor")):
            if key in params:
                a_, b_ = _pair(key, f)
                ks, w1 = _ks_w1(a_, b_)
                iqr = float(np.subtract(*np.percentile(a_, [75, 25]))) if a_.size else float("nan")
                out["continuous"][name] = {"ks": ks, "w1_over_iqr": (w1 / iqr if iqr > 0 else float("nan")),
                                           "median_drawn": float(np.median(a_)) if a_.size else float("nan"),
                                           "median_accepted": float(np.median(b_)) if b_.size else float("nan")}
    for key in ("bg_contrast", "bg_contrast_t", "bg_level", "bg_tilt", "bg_lslope", "bg_tslope"):
        if key in params:
            v = np.asarray(params[key], dtype=np.float64).ravel()
            ks, w1 = _ks_w1(v, v[acc])
            iqr = float(np.subtract(*np.percentile(v, [75, 25])))
            out["continuous"][key] = {"ks": ks, "w1_over_iqr": (w1 / iqr if iqr > 0 else float("nan"))}
    if "ma_scale" in params and "ma_on" in params:
        on = np.asarray(params["ma_on"]) > 0.5
        sc = np.asarray(params["ma_scale"], dtype=np.float64)
        ks, w1 = _ks_w1(sc[on], sc[on & acc[:, None]])
        out["continuous"]["measured_anomaly_scale"] = {"ks": ks, "w1": w1}
        out["categorical_tvd"]["measured_anomaly_on"] = _tvd(on.astype(int), on[acc].astype(int))
    tv = [v for v in out["categorical_tvd"].values() if np.isfinite(v)]
    ksv = [v["ks"] for v in out["continuous"].values() if np.isfinite(v.get("ks", np.nan))]
    out["max_tvd"] = float(max(tv)) if tv else 0.0
    out["max_ks"] = float(max(ksv)) if ksv else 0.0
    out["note"] = ("accepted vs drawn: a TVD / KS near 0 means the acceptance did not reshape the labelled law; the "
                   "configured law is the drawn one, the data set's is the accepted one")
    return out


def load_data_contract(cfg: AppConfig, verbose: bool = True) -> Tuple[MeasuredData, SimPrior, NormalizationState, Dict[str, Any]]:
    dn = cfg.data.data_name
    if dn == "tem_simulate":
        pm = synthetic_tem_profile(n_depth=48, n_gates=36, seed=cfg.runtime.seed, kind="measured")
        pms = synthetic_tem_profile(n_depth=800, n_gates=120, seed=cfg.runtime.seed + 1, depth_range=(10.0, 260.0),
                                    time_range_ms=(0.05, 16.0), kind="simulated")
        pm.meta.setdefault("format", "synthetic")
        print("[DATA] data_name=tem_simulate: synthetic measured + synthetic simulated profiles (self-test data)")
    elif dn == "simulata":
        pm = read_measured_file(cfg.data.simulata_data_file, "measured-standard (simulata)", cfg.data,
                                cfg.data.sim_time_unit, verbose=verbose)
        pms = None
    else:
        pm = read_measured_file(cfg.data.real_data_file, "measured", cfg.data, cfg.data.real_time_unit, verbose=verbose)
        pms = None
        if dn in ("combined", "combined+sine"):
            if cfg.data.simulata_data_file and os.path.isfile(cfg.data.simulata_data_file):
                if is_survey_line_file(cfg.data.simulata_data_file):
                    pms = read_survey_lines(cfg.data.simulata_data_file, "simulated", cfg.data.gate_time_file,
                                            "auto", cfg.data.component_block, "auto", cfg.data.min_positive_frac,
                                            cfg.data.sim_time_unit, verbose=verbose)
                else:
                    pms = read_profile_matrix(cfg.data.simulata_data_file, "simulated", "auto", cfg.data.sim_time_unit,
                                              verbose=verbose)
            else:
                print("[DATA] WARNING: simulated file not found -> measured-only run (no simulated structure prior)")
        if dn == "combined+sine":
            print("[DATA] combined+sine: the sine bank of v15 is not used in v16 (treated as combined)")
    # ---- survey mode: profile axis, anomaly family, extension directions ----
    survey = resolve_survey(cfg, pm)
    if pms is not None and str(cfg.data.sim_survey_check).lower() != "off":
        s_mode, s_why = detect_survey_mode(pms)
        if (s_mode == "borehole") != (survey.mode == "borehole"):
            print(f"[SIM-ALIGN] the simulated file looks like a {s_mode} survey ({s_why}) while the measured survey is "
                  f"{survey.mode}: it is not used as structure prior (--sim_survey_check off uses it anyway)")
            pms = None
    plan = resolve_extension_plan(cfg, pm, survey)
    print(f"[SURVEY] mode={survey.mode} ({survey.reason}) | profile axis = {survey.axis_label} | anomaly family = "
          f"{survey.anomaly_family} | sensor height {survey.sensor_height_m if np.isfinite(survey.sensor_height_m) else '-'}"
          f" m | profile extension {plan['profile_frac']:.2f} x span "
          f"({plan['side'] or 'off'}, {plan['n_virtual_stations']} virtual stations) | late-time extension "
          f"{(str(round(plan['t_ext_ms'], 4)) + ' ms') if plan['t_ext_ms'] > 0 else 'off'}")
    meas = preprocess_measured(pm, cfg.data.target_len, verbose=verbose, t_ext_ms=plan["t_ext_ms"])
    meas.survey = survey
    # the time-monotone constraint only where the measured quantity supports it (before any data-level fit)
    meas.report["decay_constraint"] = decay_constraint_evidence(meas, cfg, verbose=verbose)
    _DECAY["data_monotone"] = meas.report["decay_constraint"]["mode"] == "hard"
    pure = str(cfg.profile.background_mode).strip().lower() == "pure"
    if pure:
        meas.background, meas.background_info = measured_background(meas, cfg, verbose=verbose)
    L_bg = meas.background if pure else meas.L          # the profile the anchors are built on
    unmeasured = None
    if plan["n_virtual_stations"] > 0:
        meas.axis_depth, meas.axis_known = plan["axis"], plan["known"]
        unmeasured = calibrate_unmeasured(meas.depth, L_bg)
        meas.report["profile_extension"] = {
            "fraction_of_span": plan["profile_frac"], "side": plan["side"], "n_virtual_stations": plan["n_virtual_stations"],
            "axis_from": float(plan["axis"][0]), "axis_to": float(plan["axis"][-1]),
            "rules": {s: {"rule": unmeasured[s]["rule"], "pseudo_holdout_rms_decades": unmeasured[s]["rms_decades"],
                          "candidates": unmeasured[s].get("rules", {}), "gamma": unmeasured[s]["gamma"],
                          "a_median": float(np.median(unmeasured[s]["a"]))} for s in ("low", "high", "gap")}}
        print(f"[EXTENSION] profile axis {plan['axis'][0]:.6g} .. {plan['axis'][-1]:.6g} ({plan['n_virtual_stations']} "
              f"virtual stations, measured {meas.depth[0]:.6g} .. {meas.depth[-1]:.6g}) | rules chosen on the measured "
              f"stations: low end {unmeasured['low']['rule']} (pseudo hold-out RMS {unmeasured['low']['rms_decades']:.4f}), "
              f"high end {unmeasured['high']['rule']} ({unmeasured['high']['rms_decades']:.4f}) | spread a dist^g, g "
              f"{unmeasured['low']['gamma']:.2f}/{unmeasured['high']['gamma']:.2f}")
    sim = align_simulation(pms, meas, cfg.data.sim_depth_match, cfg.data.sim_level_anchor, cfg.data.sim_time_extrapolation,
                           structure_gain=_parse_gain(cfg.data.sim_structure_gain), verbose=verbose)
    role = sim_role_of(cfg)
    if sim.available and role == "off":
        rep_sim = sim.report
        sim = align_simulation(None, meas, verbose=False)
        sim.report = {"available": False, "role": "off", "read_but_not_used": "--sim_role off",
                      "alignment_report": rep_sim}
        print("[SIM-ALIGN] --sim_role off: the simulated profile is read but not used")
    elif pure and sim.available:
        # the simulated profile's local structure (features of tens of metres) would be an unlabelled anomaly in every
        # anchor of a pure background, so its values never enter a profile.  sim_role morphology: its anomalies,
        # relative to its own background and labelled, are examples of the anomaly-morphology prior instead
        rep_sim = sim.report
        if role == "morphology":
            meas.sim_morph = sim_morphology_library(sim, meas, cfg, verbose=verbose)
        sim = align_simulation(None, meas, verbose=False)
        sim.report = {"available": False, "role": role, "alignment_report": rep_sim,
                      "values_used": "none: the simulated decays are not part of any anchor (background_mode pure)",
                      "morphology": (meas.sim_morph or {}).get("report")}
        if role != "morphology":
            print("[SIM-ALIGN] background_mode pure: the simulated structure is not used (its local features would be "
                  "unlabelled anomalies in every profile)")
    if cfg.normalization.mode in ("symlog_minmax", "signed_log1p") and meas.positive:
        print(f"[NORM] --normalization_mode {cfg.normalization.mode} requested, but the measured decays are positive: "
              f"the log10 min-max normaliser of the data contract is used (v15's symlog knee compressed the late gates)")
    extra = extension_envelope(meas, unmeasured, float(cfg.profile.unmeasured_spread), L=L_bg)
    if pure:
        extra = [meas.background] + extra
        runs_m = (meas.background_info.get("measured_anomalies") or {}).get("anomalies", [])
        if runs_m and float(cfg.profile.measured_anomaly_prob) > 0:
            # each measured anomaly at the top of its own factor range
            fam_ = survey.anomaly_family
            a_vis = anomaly_amp_range(meas.L[:, meas.known_time_mask()], meas.depth, cfg.profile.anomaly_amp_range,
                                      family=fam_, t_ms=meas.grid_ms[meas.known_time_mask()],
                                      visible=borehole_physical_anomalies(cfg, fam_))[0]
            s_hi = np.array([h for _, h in measured_anomaly_scale_ranges(runs_m, cfg.profile.measured_anomaly_scale, a_vis)])
            Am = measured_anomaly_templates(meas.L - meas.background, meas.depth, runs_m,
                                            float(meas.background_info["sigma_m"]))
            extra = extra + [meas.background + (s_hi[:, None, None] * Am).sum(axis=0)]
    norm = normalizer_for(meas.L, meas.positive, sim, cfg, depth=meas.depth, extra_arrays=extra, t_ms=meas.grid_ms,
                          family=survey.anomaly_family, t_known_ms=meas.t_known_ms)
    span = norm.max_val - norm.min_val
    if extra:
        n0 = normalizer_for(meas.L, meas.positive, sim, cfg, depth=meas.depth, t_ms=meas.grid_ms,
                            family=survey.anomaly_family, t_known_ms=meas.t_known_ms)
        grow = span / max(n0.max_val - n0.min_val, 1e-9)
        meas.report["normaliser_span_growth_from_extension"] = float(grow)
        if grow > 1.3:
            print(f"[NORM] WARNING the extension envelope widens the normalised range {grow:.2f}x ({n0.max_val - n0.min_val:.2f} "
                  f"-> {span:.2f} decades): the measured part gets a coarser share of it; a shorter extension keeps it finer")
    late = meas.L[:, int(0.75 * meas.T):]
    print(f"[NORM] mode={norm.mode} | log10 range [{norm.min_val:.3f}, {norm.max_val:.3f}] ({span:.2f} decades incl. "
          f"{cfg.normalization.margin:.0%} margins) | measured late-25% gates occupy "
          f"{(late.max() - late.min()) / span:.1%} of the normalised range")
    key_src = {"grid": array_digest(meas.grid_ms), "depth": array_digest(meas.depth), "T": meas.T,
               "norm": [round(norm.min_val, 6), round(norm.max_val, 6), norm.mode], "positive": meas.positive,
               "sim": [sim.level_mode, sim.depth_mode, round(float(sim.gain), 6), bool(sim.available)],
               "data_name": dn}
    if survey.mode != "borehole" or meas.axis_depth is not None or meas.late_ext is not None:
        key_src["survey"] = {"mode": survey.mode, "family": survey.anomaly_family,
                             "axis": array_digest(meas.gen_depth), "t_known_ms": round(float(meas.t_known_ms), 9)}
    if pure:
        key_src["background"] = {"mode": "pure", "sigma_m": round(float(meas.background_info["sigma_m"]), 6),
                                 "purity_tol_decades": round(float(meas.background_info["purity_tol_decades"]), 9),
                                 "digest": array_digest(meas.background)}
        inv_ = meas.background_info.get("measured_anomalies") or {}
        if inv_.get("noise_rule", "mad") != "mad":
            # the inventory's rule decides the labelled measured anomalies, which no array digest above sees
            key_src["background"]["inventory"] = {"noise_rule": inv_["noise_rule"], "runs": [
                [round(float(r_["from_m"]), 6), round(float(r_["to_m"]), 6)] for r_ in inv_.get("anomalies", [])]}
    if meas.report["decay_constraint"]["mode"] != "hard":
        key_src["decay"] = meas.report["decay_constraint"]["mode"]
    if meas.sim_morph is not None:
        key_src["sim_morphology"] = meas.sim_morph["digest"]
    contract = {
        "program": RFMG_PROGRAM_NAME, "version": RFMG_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_key": hashlib.sha256(stable_json_dumps(key_src).encode("utf-8")).hexdigest()[:16],
        "rule": "measured time and depth axes are the only standard; simulated data are re-sampled, registered and "
                "level-anchored onto them",
        "measured": meas.pm.summary(), "measured_preprocess": meas.report,
        "time_grid_ms": [float(x) for x in meas.grid_ms], "measured_gate_index_on_grid": [int(x) for x in meas.gate_index],
        "depth_axis_m": [float(x) for x in meas.depth], "simulated": sim.report, "normalizer": norm.to_dict(),
        "target_len": int(meas.T), "key_source": key_src, "survey": survey.to_dict(),
        "background": (meas.background_info if pure else {"mode": "measured", "note": "anchors on the measured profile "
                                                                                       "itself (v16 / v17)"}),
        "decay_constraint": meas.report["decay_constraint"],
    }
    if meas.axis_depth is not None:
        contract["generation_axis"] = [float(x) for x in meas.axis_depth]
    return meas, sim, norm, contract


def unknown_envelope(kd: np.ndarray, kL: np.ndarray, qd: np.ndarray, um: Optional[Dict[str, Any]], spread: float,
                     positive: bool = True) -> List[np.ndarray]:
    """Rows A -/+ 3 sigma at positions qd without a measurement: A = the anchor rule from the known stations (kd, kL)
    as the trainer builds it (time-monotone in log10), sigma = the calibrated spread (ends: a dist^gamma from the end
    station; gaps: a deff^gamma with the bridge distance d1 d2 / (d1 + d2))."""
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    if um is None or qd.size == 0:
        return []
    kd = np.asarray(kd, dtype=np.float64)
    A = anchor_profile_on(kd, kL, qd, (um["low"]["rule"], um["high"]["rule"]))
    if positive:
        A = _tmono_np(A, axis=1)
    sig = np.zeros_like(A)
    lo, hi = qd < kd[0], qd > kd[-1]
    if np.any(lo):
        sig[lo] = um["low"]["a"][None, :] * np.power(kd[0] - qd[lo], um["low"]["gamma"])[:, None]
    if np.any(hi):
        sig[hi] = um["high"]["a"][None, :] * np.power(qd[hi] - kd[-1], um["high"]["gamma"])[:, None]
    mid = ~(lo | hi)
    if np.any(mid):
        j = np.clip(np.searchsorted(kd, qd[mid]), 1, kd.size - 1)
        d1, d2 = np.clip(qd[mid] - kd[j - 1], 0.0, None), np.clip(kd[j] - qd[mid], 0.0, None)
        deff = d1 * d2 / np.maximum(d1 + d2, 1e-12)
        sig[mid] = um["gap"]["a"][None, :] * np.power(deff, um["gap"]["gamma"])[:, None]
    s = 3.0 * float(spread) * sig
    return [A - s, A + s]


def extension_envelope(meas: MeasuredData, unmeasured: Optional[Dict[str, Any]], spread: float,
                       L: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Extra rows for the normaliser: the anchor at the virtual stations and beyond the last gate +- 3 sigma of the
    calibrated spread, so that every extended anchor stays inside the normalised range.  L = the profile the anchors
    are built on (the measured background in background_mode pure; default the measured profile)."""
    Lb = meas.L if L is None else np.asarray(L, dtype=np.float64)
    out: List[np.ndarray] = []
    if meas.late_ext is not None:
        lt = np.log10(meas.grid_ms)
        dt = np.clip(lt - math.log10(meas.t_known_ms), 0.0, None)
        s = 3.0 * spread * float(meas.late_ext["a"]) * np.power(dt, float(meas.late_ext["gamma"]))
        out += [Lb.min(0, keepdims=True) - s[None, :], Lb.max(0, keepdims=True) + s[None, :]]
    if unmeasured is not None and meas.axis_depth is not None:
        out += unknown_envelope(meas.depth, Lb, meas.axis_depth[~meas.axis_known], unmeasured, spread, meas.positive)
    return out


def normalizer_for(L_meas: np.ndarray, positive: bool, sim: SimPrior, cfg: AppConfig,
                   depth: Optional[np.ndarray] = None, extra_arrays: Optional[List[np.ndarray]] = None,
                   t_ms: Optional[np.ndarray] = None, family: str = "borehole", t_known_ms: float = 0.0) -> NormalizationState:
    """log10 min-max range covering the measured profile and every value the anchors can take: sim structure, random
    anomalies (total capped at 1.5 x the upper |peak|) and the background level / tilt modulation (4 sigma);
    extra_arrays = the extended anchors +- 3 sigma of their calibrated spread; the anomaly amplitude rule of the
    survey's anomaly family is read on the measured window (t <= t_known_ms when the grid extends beyond it)."""
    arrays = [L_meas] + list(extra_arrays or [])
    # the anomaly / background / sim margins apply to the extended anchors as well
    L_all = np.concatenate([L_meas] + list(extra_arrays), axis=0) if extra_arrays else L_meas
    ext = 0.0
    if cfg.profile.anomaly_enabled and depth is not None:
        kt = L_meas.shape[1]
        if t_ms is not None and float(t_known_ms) > 0:
            kt = int(np.sum(np.asarray(t_ms) <= float(t_known_ms) * (1.0 + 1e-9)))
        ext += 1.5 * anomaly_amp_range(L_meas[:, :kt], depth, cfg.profile.anomaly_amp_range, family=family,
                                       t_ms=None if t_ms is None else np.asarray(t_ms)[:kt],
                                       visible=borehole_physical_anomalies(cfg, family))[1]
    ext += 4.0 * max(float(cfg.profile.bg_level_sigma), 0.0) + 2.0 * max(float(cfg.profile.bg_tilt_sigma), 0.0)
    if ext > 0:
        arrays += [L_all.min(0, keepdims=True) - ext, L_all.max(0, keepdims=True) + ext]
    if sim.available:
        spread = max(float(cfg.profile.structure_gain_spread), 1.0)
        if sim.level_mode == "structure":
            g = sim.gain * spread * sim.coverage
            arrays += [L_all.min(0, keepdims=True) + g[None, :] * np.minimum(sim.anomaly.min(0, keepdims=True), 0.0),
                       L_all.max(0, keepdims=True) + g[None, :] * np.maximum(sim.anomaly.max(0, keepdims=True), 0.0)]
        else:
            Ls = sim.L_native_grid + sim.level_shift[None, :]
            arrays += [np.where(sim.coverage[None, :] > 0, Ls.min(0, keepdims=True), L_all.min(0, keepdims=True)),
                       np.where(sim.coverage[None, :] > 0, Ls.max(0, keepdims=True), L_all.max(0, keepdims=True))]
    return fit_normalizer_log(arrays, positive, margin=cfg.normalization.margin, eps=cfg.normalization.eps)
