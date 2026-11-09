"""Cross-validation: leave-block-out folds along the profile and a time-direction fold, a universal-kriging
reference, CRPS and coverage scores."""

from __future__ import annotations

import copy
import math
import os
import shutil
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .ablation import _rfmg_apply_ablation
from .checkpoints import select_best_checkpoint
from .common import save_json, set_seed
from .config import AppConfig
from .data.contract import MeasuredData, SimPrior, align_simulation, local_structure_residual
from .data.extension import calibrate_unmeasured
from .data.normalization import NormalizationState
from .data.time_grid import depth_interp_profile
from .evidence import cv_buffer_distance, filter_sim_morph
from .figures import _axis_names, _plt
from .pipeline import _parse_gain, extension_envelope, measured_background, normalizer_for, unknown_envelope
from .quality import QC_GATES, acceptance_summary, profile_quality_metrics, structure_fidelity_metrics
from .trainer import RFMGGenTrainer


def fold_anchors(meas: MeasuredData, sim: SimPrior, keep: np.ndarray, cfg: AppConfig) -> Tuple[SimPrior, NormalizationState]:
    """Cross-validation fold: structure gain (auto rule) and normaliser recomputed from the TRAINING stations only,
    so no statistic of the held-out block reaches the fold's model.  The normaliser also covers the held-out
    positions' anchor (the rule from the training stations) +- 3 sigma of its calibrated spread, as production covers
    its extension."""
    mk = meas.subset(keep)
    sim_k = sim
    if sim.available and str(cfg.data.sim_structure_gain).lower() == "auto" and sim.level_mode == "structure":
        sel = sim.coverage > 0.5
        lsr_m = np.abs(local_structure_residual(mk.L, mk.depth))
        p99_m = float(np.percentile(lsr_m[:, sel], 99)) if (lsr_m.size and np.any(sel)) else 0.0
        p99_s = float(sim.report.get("local_structure_p99_decades", {}).get("simulated", 0.0))
        gain = float(np.clip(p99_m / max(p99_s, 1e-9), 0.0, 1.0)) if p99_s > 0 else 0.0
        sim_k = copy.copy(sim)
        sim_k.gain = gain
    # background_mode pure: the fold's anchors are the background of its training stations (scale and tolerance from
    # them too; the trainer derives the same background)
    pure = str(cfg.profile.background_mode).strip().lower() == "pure"
    Lb = measured_background(mk, cfg, verbose=False)[0] if pure else mk.L
    um = calibrate_unmeasured(mk.depth, Lb)
    extra = unknown_envelope(mk.depth, Lb, meas.depth[~np.asarray(keep, dtype=bool)], um,
                             float(cfg.profile.unmeasured_spread), meas.positive) + ([Lb] if pure else [])
    fam = meas.survey.anomaly_family if meas.survey is not None else "borehole"
    return sim_k, normalizer_for(mk.L, meas.positive, sim_k, cfg, depth=mk.depth, extra_arrays=extra, t_ms=meas.grid_ms,
                                 family=fam)


def _train_all_stages(trainer: RFMGGenTrainer, cfg: AppConfig) -> None:
    pool = trainer.build_curve_pool(cfg.data.measured_pool_size, cfg.data.sim_reservoir_rows // 2,
                                    cfg.data.sim_reservoir_rows // 2, seed=cfg.runtime.seed + 5)
    print(f"[POOL] curve pool for Stage-0/0B/1A: {pool['y'].shape[0]} curves "
          f"(measured={int((pool['domain'] == 0).sum())}, sim-structured={int((pool['domain'] == 1).sum())}, "
          f"bridge={int((pool['domain'] == 2).sum())})")
    trainer.stage0_autoencoder(pool)
    trainer.stage0b_condition(pool)
    trainer.stage1a_supervisor(pool)
    trainer.train_generator_stage("stage1B")
    trainer.train_generator_stage("stage2")


def ensemble_crps(ens: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble (M, ...) for observations y (...), energy form, in the data units."""
    e = np.asarray(ens, dtype=np.float64)
    t1 = np.mean(np.abs(e - y[None]), axis=0)
    es = np.sort(e, axis=0)
    M = e.shape[0]
    w = (2.0 * np.arange(1, M + 1) - M - 1).reshape((M,) + (1,) * (e.ndim - 1))
    t2 = np.sum(w * es, axis=0) / (M * M)       # = 0.5 * E|X - X'|
    return t1 - t2


def gaussian_crps(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of N(mu, sd^2) for observations y (closed form, Gneiting & Raftery 2007), in the data units."""
    sd = np.maximum(np.asarray(sd, dtype=np.float64), 1e-12)
    w = (np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / sd
    wt = torch.from_numpy(np.ascontiguousarray(w))
    cdf = torch.special.ndtr(wt).numpy()
    pdf = np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
    return sd * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def kriging_predict(x: np.ndarray, Y: np.ndarray, xq: np.ndarray, n_len: int = 24,
                    nuggets: Sequence[float] = (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)) -> Optional[Dict[str, Any]]:
    """Probabilistic reference of the cross-validation: universal kriging along one axis for C series that share the
    covariance shape - Y (n, C) observed at x (n,), linear drift in x (generalised least squares), Matern-3/2
    correlation of length ell plus a nugget eta, both as fractions of each series' own variance sigma_c^2 (profiled
    out, restricted likelihood); ell and eta maximise the restricted likelihood pooled over the series on a grid (ell
    geometric from half the median spacing to twice the span, eta 1e-4 .. 1).  Returns the predictive mean and
    standard deviation (m, C) of a new observation at xq (nugget and drift uncertainty included), ell, eta; None with
    fewer than 5 points."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(x.size, -1)
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    n, C = Y.shape
    if n < 5 or not np.all(np.isfinite(Y)):
        return None
    c0, sc = float(np.mean(x)), max(float(np.ptp(x)), 1e-9)
    F = np.stack([np.ones(n), (x - c0) / sc], axis=1)
    Fq = np.stack([np.ones(xq.size), (xq - c0) / sc], axis=1)
    p_ = F.shape[1]
    dx = np.abs(x[:, None] - x[None, :])
    dq = np.abs(x[:, None] - xq[None, :])
    sp = float(np.median(np.diff(np.sort(x))))
    sp = sp if sp > 0 else sc / max(n - 1, 1)

    def corr(d: np.ndarray, ell: float) -> np.ndarray:
        a = math.sqrt(3.0) * d / ell
        return (1.0 + a) * np.exp(-a)

    best = None
    for ell in np.geomspace(0.5 * sp, 2.0 * sc, int(n_len)):
        R0 = corr(dx, float(ell))
        for eta in nuggets:
            R = R0 + float(eta) * np.eye(n)
            sg, ldR = np.linalg.slogdet(R)
            if sg <= 0:
                continue
            RiF = np.linalg.solve(R, F)
            A = F.T @ RiF
            sgA, ldA = np.linalg.slogdet(A)
            if sgA <= 0:
                continue
            B = np.linalg.solve(A, RiF.T @ Y)                                   # (p, C) drift
            Res = Y - F @ B
            RiRes = np.linalg.solve(R, Res)
            s2 = np.maximum(np.sum(Res * RiRes, axis=0) / (n - p_), 1e-30)       # (C,) restricted ML variances
            ll = -0.5 * ((n - p_) * float(np.sum(np.log(s2))) + C * (ldR + ldA))
            if best is None or ll > best[0]:
                best = (ll, float(ell), float(eta), R, A, B, RiRes, s2)
    if best is None:
        return None
    ll, ell, eta, R, A, B, RiRes, s2 = best
    rq = corr(dq, ell)                                                           # (n, m)
    Rirq = np.linalg.solve(R, rq)
    mean = Fq @ B + rq.T @ RiRes                                                 # (m, C)
    u = Fq.T - F.T @ Rirq                                                        # (p, m)
    v = (1.0 + eta) - np.sum(rq * Rirq, axis=0) + np.sum(u * np.linalg.solve(A, u), axis=0)
    sd = np.sqrt(np.maximum(v, 1e-12))[:, None] * np.sqrt(s2)[None, :]
    return {"mean": mean, "sd": sd, "length": ell, "nugget": eta, "restricted_loglik": float(ll)}


def cv_fold_blocks(D: int, K: int) -> List[np.ndarray]:
    """Contiguous depth blocks of near-equal size (rule-based, no hand-picked hold-out region)."""
    return [b for b in np.array_split(np.arange(D), K) if b.size]


def _cv_fold_cfg(cfg: AppConfig, var: str, subdir: str) -> AppConfig:
    """Config of one cross-validation fold: the ablation variant, the quick / medium epoch scale, its own model folder."""
    c = copy.deepcopy(cfg)
    _rfmg_apply_ablation(c, var)
    if c.cv.scale in ("quick", "medium"):
        t = c.train
        div = 4 if c.cv.scale == "quick" else 2
        t.stage0_epochs, t.stage0b_epochs = max(3, t.stage0_epochs // div), max(1, t.stage0b_epochs // 2)
        t.stage1_supervisor_epochs = max(2, t.stage1_supervisor_epochs // div)
        t.stage1_epochs, t.stage2_epochs = max(4, t.stage1_epochs // div), max(6, t.stage2_epochs // div)
        t.steps_per_epoch = max(30, t.steps_per_epoch // 2)
        c.data.measured_pool_size = min(c.data.measured_pool_size, 2048)
        c.data.sim_reservoir_rows = min(c.data.sim_reservoir_rows, 4096)
    c.export.model_dir = os.path.join(cfg.export.model_dir, "cv", var, subdir)
    c.export.save_checkpoint = True
    c.checkpoint.save_best_per_stage = True
    return c


CV_COVER_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _cv_scores(ens: np.ndarray, truth: np.ndarray, base: np.ndarray,
               gp: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, float]:
    """Held-out scores of an ensemble (M, ...) against the measured truth (...): coverage, CRPS, the deterministic
    rule's MAE (= its CRPS) and the skill against it; the observed coverage of the ensemble's central 10-90 %
    intervals and, with gp = (mean, sd), the kriging reference's CRPS, coverage and the ensemble's skill against it."""
    q = np.percentile(ens, [2.5, 25, 75, 97.5], axis=0)
    crps = ensemble_crps(ens, truth)
    mae_base = np.abs(base - truth)
    out = {"heldout_coverage95": float(np.mean((truth >= q[0]) & (truth <= q[3]))),
           "heldout_coverage50": float(np.mean((truth >= q[1]) & (truth <= q[2]))),
           "heldout_crps_decades": float(np.mean(crps)),
           "heldout_interp_mae_decades": float(np.mean(mae_base)),
           "heldout_crpss_vs_interp": float(1.0 - np.mean(crps) / max(np.mean(mae_base), 1e-12)),
           "heldout_median_abs_err_decades": float(np.mean(np.abs(np.median(ens, axis=0) - truth))),
           "pit_mean": float(np.mean(np.mean(ens <= truth[None], axis=0)))}
    for lv in CV_COVER_LEVELS:
        lo_, hi_ = np.percentile(ens, [50.0 * (1.0 - lv), 50.0 * (1.0 + lv)], axis=0)
        out[f"cover_ens_p{int(round(100 * lv)):02d}"] = float(np.mean((truth >= lo_) & (truth <= hi_)))
    if gp is not None:
        mu, sd = np.asarray(gp[0], dtype=np.float64), np.asarray(gp[1], dtype=np.float64)
        cg = gaussian_crps(mu, sd, truth)
        w = np.abs(truth - mu) / np.maximum(sd, 1e-12)
        nd = NormalDist()
        out.update({"heldout_gp_crps_decades": float(np.mean(cg)),
                    "heldout_gp_mae_decades": float(np.mean(np.abs(mu - truth))),
                    "heldout_gp_coverage95": float(np.mean(w <= nd.inv_cdf(0.975))),
                    "heldout_gp_coverage50": float(np.mean(w <= nd.inv_cdf(0.75))),
                    "heldout_crpss_vs_gp": float(1.0 - np.mean(crps) / max(float(np.mean(cg)), 1e-12))})
        for lv in CV_COVER_LEVELS:
            out[f"cover_gp_p{int(round(100 * lv)):02d}"] = float(np.mean(w <= nd.inv_cdf(0.5 + 0.5 * lv)))
    return out


def run_cross_validation(cfg: AppConfig, device: torch.device, meas: MeasuredData, sim: SimPrior,
                         norm: NormalizationState, contract: Dict[str, Any], reports_dir: str, plots_dir: str) -> Dict[str, Any]:
    """Leave-block-out cross-validation along the profile axis (contiguous blocks of stations, rule-based) and the
    time-direction fold (the last N decades of the measured gates hidden).  Held-out stations / times are built like
    unmeasured positions in production: the rule chosen on the remaining measured data plus the calibrated
    per-profile spread."""
    K = int(cfg.cv.folds)
    if K == 1:
        raise ValueError("--cv_folds 1 would hold out every station: use K >= 2 (or 0 with --cv_time_holdout only)")
    variants = ["full"] + [v.strip() for v in cfg.cv.ablations.split(",") if v.strip() and v.strip() != "full"]
    try:
        sel = {int(x) for x in str(cfg.cv.folds_select).replace(";", ",").split(",") if x.strip()}
    except ValueError:
        raise ValueError(f"--cv_folds_select expects fold numbers like 1,5, got {cfg.cv.folds_select!r}")
    if sel and (K < 2 or not sel <= set(range(1, K + 1))):
        raise ValueError(f"--cv_folds_select {sorted(sel)}: folds are numbered 1..{K} (--cv_folds {K})")
    if float(cfg.cv.time_holdout_decades) > 0:
        meas.time_holdout(float(cfg.cv.time_holdout_decades))      # validated before any fold trains
    blocks = cv_fold_blocks(meas.D, K) if K > 0 else []
    ax_name = _axis_names(meas)["name"]
    # spatial buffer: stations within one range of the residual's autocorrelation of a held-out block are not trained on
    # (their values carry the block's information); capped at half a block, reported
    buf_m, buf_rule, buf_dep = cv_buffer_distance(meas, cfg) if K > 0 else (0.0, "no blocks", {})
    if blocks:
        cap = 0.5 * min(float(meas.depth[b[-1]] - meas.depth[b[0]]) + float(np.median(np.diff(meas.depth))) for b in blocks)
        if buf_m > cap:
            print(f"[CV] WARNING: buffer {buf_m:.1f} m capped at half a block ({cap:.1f} m): the folds are not independent "
                  f"beyond it")
            buf_rule += f"; capped at half a block ({cap:.4g} m from {buf_m:.4g} m)"
            buf_m = cap
    results: Dict[str, Any] = {"K": K, "blocks_depth_m": [[float(meas.depth[b[0]]), float(meas.depth[b[-1]])] for b in blocks],
                               "survey": meas.survey.to_dict() if meas.survey is not None else {}, "variants": {},
                               "buffer": {"m": float(buf_m), "rule": buf_rule, "residual_dependence": buf_dep}}
    print("=" * 88)
    if K > 0:
        print(f"[CV] leave-block-out cross-validation along the {ax_name} axis: K={K} contiguous blocks of "
              f"{[int(b.size) for b in blocks]} stations{f' (folds {sorted(sel)})' if sel else ''} | variants={variants} | "
              f"scale={cfg.cv.scale} | buffer {buf_m:.1f} m ({buf_rule})")
    if float(cfg.cv.time_holdout_decades) > 0:
        print(f"[CV] time-direction fold: the last {cfg.cv.time_holdout_decades:g} decades of the measured gates are hidden")
    print("=" * 88)
    for var in variants:
        rows = []
        for k, blk in enumerate(blocks):
            if sel and (k + 1) not in sel:
                continue
            c = _cv_fold_cfg(cfg, var, f"fold_{k + 1}")
            keep = np.ones(meas.D, dtype=bool)
            keep[blk] = False
            z_lo, z_hi = float(meas.depth[blk[0]]) - buf_m, float(meas.depth[blk[-1]]) + buf_m
            bufz = keep & (meas.depth >= z_lo - 1e-9) & (meas.depth <= z_hi + 1e-9)
            keep[bufz] = False
            # the fold's model sees no simulated conductor over the held-out block and its buffer either
            meas_k = copy.copy(meas)
            meas_k.sim_morph = filter_sim_morph(meas.sim_morph, [(z_lo, z_hi)],
                                                f"CV fold {k + 1}: held-out block + buffer {z_lo:.1f}-{z_hi:.1f} m")
            n_sw = (len(((meas_k.sim_morph or {}).get("report") or {}).get("withheld", []))
                    - len(((meas.sim_morph or {}).get("report") or {}).get("withheld", [])))
            edge = bool(blk[0] == 0 or blk[-1] == meas.D - 1)
            set_seed(cfg.runtime.seed + 101 * (k + 1))
            print(f"[CV] variant={var} fold {k + 1}/{K}: hold out stations {blk[0]}..{blk[-1]} "
                  f"({meas.depth[blk[0]]:.1f}-{meas.depth[blk[-1]]:.1f}; {'edge' if edge else 'interior'} block); the anchor "
                  f"there is built from the remaining stations | buffer {buf_m:.1f} m: {int(bufz.sum())} more stations not "
                  f"trained on, {int(keep.sum())} training stations"
                  + (f", {n_sw} simulated conductor(s) there withheld" if n_sw else ""))
            sim_k, norm_k = fold_anchors(meas_k, sim, keep, c)
            print(f"[CV]   fold anchors from training stations only: structure gain {sim_k.gain:.4f} "
                  f"(all stations {sim.gain:.4f}), log10 range [{norm_k.min_val:.3f}, {norm_k.max_val:.3f}]")
            ck = dict(contract)
            ck["contract_key"] = f"{contract['contract_key']}-cv{K}f{k + 1}"
            tr = RFMGGenTrainer(c, device, meas_k, sim_k, norm_k, ck, keep_mask=keep, tag=f"cv_{var}_{k + 1}")
            _train_all_stages(tr, c)
            best = select_best_checkpoint(tr)
            if best:
                tr.load_checkpoint(best)
            Lg, trace = tr.generate_profiles(int(cfg.cv.generate_profiles), seed=cfg.runtime.seed + 900 + k,
                                             mode="gen", return_anchor=True)
            truth = meas.L[blk]
            ens = Lg[:, blk, :]
            row: Dict[str, Any] = {"variant": var, "fold": k + 1, "fold_type": "edge" if edge else "interior",
                                   "depth_from_m": float(meas.depth[blk[0]]), "depth_to_m": float(meas.depth[blk[-1]]),
                                   "buffer_m": float(buf_m), "buffer_stations": int(bufz.sum()),
                                   "train_stations": int(keep.sum()), "sim_conductors_withheld": int(n_sw)}
            # probabilistic reference: kriging along the axis from the fold's training stations only
            kg = kriging_predict(meas.depth[keep], meas.L[keep], meas.depth[blk])
            row.update(_cv_scores(ens, truth, tr.L_anchor_np[blk], gp=(kg["mean"], kg["sd"]) if kg else None))
            if kg:
                row["gp_length_m"], row["gp_nugget"] = kg["length"], kg["nugget"]
            row["heldout_v16_clamp_mae_decades"] = float(np.mean(np.abs(
                depth_interp_profile(meas.depth[keep], meas.L[keep], meas.depth[blk]) - truth)))
            if tr.unmeasured is not None:
                row["rule_low_end"], row["rule_high_end"] = tr.unmeasured["low"]["rule"], tr.unmeasured["high"]["rule"]
            kept = np.where(keep)[0]
            qi = np.percentile(Lg[:, kept, :], [2.5, 97.5], axis=0)
            row["insample_coverage95"] = float(np.mean((meas.L[kept] >= qi[0]) & (meas.L[kept] <= qi[1])))
            pmq = profile_quality_metrics(Lg, tr.L_anchor_np, meas.depth, P_anchor=trace["anchor_L"])
            row["rough_ratio_vs_anchor"] = pmq.get("rough_ratio_vs_anchor_median", float("nan"))
            row["diversity_decades"] = pmq.get("pairwise_profile_distance_decades_median", float("nan"))
            row["dist_to_measured_decades"] = pmq.get("distance_to_measured_rms_decades_median", float("nan"))
            acc = acceptance_summary(trace.get("qc", {}))
            row["accept_rate_first_pass"] = acc.get("accept_rate", float("nan"))
            row["anomaly_retention"] = acc.get("anomaly_retention_median", float("nan"))
            row["anomaly_corr"] = acc.get("anomaly_corr_median", float("nan"))
            # structure fidelity on the fold's training stations (held-out stations carry no measured feature cell)
            smq, _ = structure_fidelity_metrics(Lg, trace["anchor_L"], tr.L_anchor_np, meas.depth, tr.gate_cols,
                                                tr.struct_thr, keep=keep)
            topq = smq.get("strongest_feature", {})
            row["struct_retention"] = smq.get("struct_retention", float("nan"))
            row["measured_structure_retention_departure"] = smq.get("pooled_measured_structure_retention", {}).get(
                "departure", float("nan"))
            row["strongest_feature_depth_m"] = topq.get("depth_m", float("nan"))
            row["strongest_feature_retention_departure_min"] = topq.get("retention_departure_min", float("nan"))
            rows.append(row)
            print(f"[CV] {var} fold {k + 1}: held-out coverage95={row['heldout_coverage95']:.3f} "
                  f"coverage50={row['heldout_coverage50']:.3f} CRPS={row['heldout_crps_decades']:.4f} dec "
                  f"(anchor-rule MAE {row['heldout_interp_mae_decades']:.4f}, v16 clamp MAE "
                  f"{row['heldout_v16_clamp_mae_decades']:.4f}, CRPSS={row['heldout_crpss_vs_interp']:+.3f}; kriging CRPS "
                  f"{row.get('heldout_gp_crps_decades', float('nan')):.4f} coverage95 "
                  f"{row.get('heldout_gp_coverage95', float('nan')):.3f}, CRPSS vs kriging "
                  f"{row.get('heldout_crpss_vs_gp', float('nan')):+.3f}) | "
                  f"rough/anchor={row['rough_ratio_vs_anchor']:.3f} div={row['diversity_decades']:.4f} "
                  f"dist-to-measured={row['dist_to_measured_decades']:.4f} | first-pass accept={row['accept_rate_first_pass']:.2f} "
                  f"anomaly retention={row['anomaly_retention']:.3f} corr={row['anomaly_corr']:.3f} | structure retention="
                  f"{row['struct_retention']:.3f} measured features={row['measured_structure_retention_departure']:.3f} "
                  f"(strongest {row['strongest_feature_depth_m']:.0f} min {row['strongest_feature_retention_departure_min']:.3f})")
            del tr
            if str(cfg.checkpoint.keep) == "final":       # a fold's checkpoints are not needed once it is scored
                shutil.rmtree(c.export.model_dir, ignore_errors=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if rows:
            df = pd.DataFrame(rows)
            skip = ("variant", "fold", "fold_type", "depth_from_m", "depth_to_m", "strongest_feature_depth_m",
                    "rule_low_end", "rule_high_end", "buffer_m", "buffer_stations", "train_stations",
                    "sim_conductors_withheld")
            summ = {c_: {"mean": float(df[c_].mean()), "std": float(df[c_].std(ddof=0))} for c_ in df.columns if c_ not in skip}
            results["variants"][var] = {"folds": rows, "summary": summ}
            df.to_csv(os.path.join(reports_dir, f"cv_{var}.csv"), index=False)
            print(f"[CV] {var} summary: " + " | ".join(f"{k_}={v_['mean']:.4f}±{v_['std']:.4f}" for k_, v_ in summ.items()
                                                       if not k_.startswith("cover_")))
        # ---- time-direction fold ----
        dec = float(cfg.cv.time_holdout_decades)
        if dec > 0:
            c = _cv_fold_cfg(cfg, var, f"time_{dec:g}dec")
            md, hid = meas.time_holdout(dec)
            gi = meas.gate_index
            emb = bool(np.all(gi >= 0))
            # scored columns: the hidden measured gates (grid nodes when the gates are embedded), else the hidden grid nodes
            hid_g = gi[np.isin(gi, hid)] if emb else hid
            n_hid_gates = int(np.sum(meas.pm.time_ms > md.t_known_ms * (1.0 + 1e-9)))
            k_last = int(np.nonzero(md.known_time_mask())[0][-1])          # last known grid node (the last kept gate)
            print(f"[CV] variant={var} time fold: gates after {md.t_known_ms:.6g} ms hidden ({n_hid_gates} measured gates, "
                  f"{hid.size} grid nodes up to {meas.grid_ms[-1]:.6g} ms; scored on {hid_g.size} "
                  f"{'gate' if emb else 'grid'} columns); rule {md.late_ext['rule']} chosen on the rest")
            set_seed(cfg.runtime.seed + 707)
            # the sim prior's level shift and auto gain re-derived from the known window (they used every gate)
            sim_t = (align_simulation(sim.pm, md, c.data.sim_depth_match, c.data.sim_level_anchor,
                                      c.data.sim_time_extrapolation, structure_gain=_parse_gain(c.data.sim_structure_gain),
                                      verbose=False) if sim.available else sim)
            pure_t = str(c.profile.background_mode).strip().lower() == "pure"
            if pure_t:   # the fold's background from the known window only (scale and tolerance too)
                md.background, md.background_info = measured_background(md, c, verbose=False)
            extra = (extension_envelope(md, None, float(c.profile.unmeasured_spread), L=(md.background if pure_t else None))
                     + ([md.background] if pure_t else []))
            fam = md.survey.anomaly_family if md.survey is not None else "borehole"
            norm_t = normalizer_for(md.L, md.positive, sim_t, c, depth=md.depth, extra_arrays=extra, t_ms=md.grid_ms,
                                    family=fam, t_known_ms=md.t_known_ms)
            print(f"[CV]   time-fold anchors from the known window only: structure gain {sim_t.gain:.4f} (all gates "
                  f"{sim.gain:.4f}), log10 range [{norm_t.min_val:.3f}, {norm_t.max_val:.3f}]")
            ck = dict(contract)
            ck["contract_key"] = f"{contract['contract_key']}-cvt{dec:g}"
            tr = RFMGGenTrainer(c, device, md, sim_t, norm_t, ck, tag=f"cv_{var}_time")
            _train_all_stages(tr, c)
            best = select_best_checkpoint(tr)
            if best:
                tr.load_checkpoint(best)
            Lg, trace = tr.generate_profiles(int(cfg.cv.generate_profiles), seed=cfg.runtime.seed + 950, mode="gen",
                                             return_anchor=True)
            truth = meas.L[:, hid_g]
            row = {"variant": var, "hidden_decades": dec, "t_known_ms": float(md.t_known_ms),
                   "hidden_measured_gates": n_hid_gates, "scored_columns": int(hid_g.size),
                   "scored_on": "measured gates" if emb else "grid nodes (gates not embedded)", "rule": md.late_ext["rule"],
                   "structure_gain_fold": float(sim_t.gain)}
            # probabilistic reference: kriging along log time, per station, on the late half of the known gates
            kt_cols = gi[meas.pm.time_ms <= md.t_known_ms * (1.0 + 1e-9)] if emb else np.zeros(0, dtype=np.int64)
            gpt = None
            if kt_cols.size >= 5:
                xl = np.log10(meas.grid_ms[kt_cols])
                late = xl >= xl.min() + 0.5 * (xl.max() - xl.min())
                late = late if int(late.sum()) >= 5 else (np.arange(xl.size) >= xl.size - 5)
                kg = kriging_predict(xl[late], md.L[:, kt_cols[late]].T, np.log10(meas.grid_ms[hid_g]))
                if kg:
                    gpt = (kg["mean"].T, kg["sd"].T)
                    row["gp_length_log10_t"], row["gp_nugget"] = kg["length"], kg["nugget"]
                    row["gp_known_gates"] = int(late.sum())
            row.update(_cv_scores(Lg[:, :, hid_g], truth, md.L[:, hid_g], gp=gpt))
            row["heldout_v16_edge_mae_decades"] = float(np.mean(np.abs(md.L[:, [k_last]] - truth)))
            dl = np.log10(meas.grid_ms[hid_g]) - math.log10(md.t_known_ms)
            for lab, lo_, hi_ in (("first_third", 0.0, 1 / 3), ("last_third", 2 / 3, 1.01)):
                s_ = (dl >= lo_ * dl.max()) & (dl <= hi_ * dl.max())
                if np.any(s_):
                    sc = _cv_scores(Lg[:, :, hid_g[s_]], truth[:, s_], md.L[:, hid_g[s_]],
                                    gp=(gpt[0][:, s_], gpt[1][:, s_]) if gpt is not None else None)
                    row[f"coverage95_{lab}"] = sc["heldout_coverage95"]
                    row[f"crps_{lab}"] = sc["heldout_crps_decades"]
                    if gpt is not None:
                        row[f"gp_crps_{lab}"] = sc["heldout_gp_crps_decades"]
            results.setdefault("time_fold", {})[var] = row
            pd.DataFrame([row]).to_csv(os.path.join(reports_dir, f"cv_time_{var}.csv"), index=False)
            print(f"[CV] {var} time fold: hidden-gate coverage95={row['heldout_coverage95']:.3f} coverage50="
                  f"{row['heldout_coverage50']:.3f} CRPS={row['heldout_crps_decades']:.4f} dec (rule MAE "
                  f"{row['heldout_interp_mae_decades']:.4f}, v16 edge-constant MAE {row['heldout_v16_edge_mae_decades']:.4f}, "
                  f"CRPSS={row['heldout_crpss_vs_interp']:+.3f}; kriging CRPS "
                  f"{row.get('heldout_gp_crps_decades', float('nan')):.4f}, CRPSS vs kriging "
                  f"{row.get('heldout_crpss_vs_gp', float('nan')):+.3f}) | first / last third coverage95 "
                  f"{row.get('coverage95_first_third', float('nan')):.3f} / {row.get('coverage95_last_third', float('nan')):.3f}")
            del tr
            if str(cfg.checkpoint.keep) == "final":       # a fold's checkpoints are not needed once it is scored
                shutil.rmtree(c.export.model_dir, ignore_errors=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    save_json(results, os.path.join(reports_dir, "cv_report.json"))
    if not any(v.get("folds") for v in results["variants"].values()):
        return results
    try:
        plt = _plt()
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
        vv = [v for v in variants if v in results["variants"]]
        width = 0.8 / max(len(vv), 1)
        for i_, var in enumerate(vv):
            f = results["variants"][var]["folds"]
            x = np.arange(len(f)) + i_ * width
            axs[0].bar(x, [r["heldout_coverage95"] for r in f], width=width, label=var)
            axs[1].bar(x, [r["heldout_crpss_vs_interp"] for r in f], width=width, label=var)
            axs[2].bar(x, [r["rough_ratio_vs_anchor"] for r in f], width=width, label=var)
        for ax, t in zip(axs, ("held-out coverage (95% band)", "CRPS skill vs the anchor rule", "profile roughness / anchor")):
            ax.set_title(t)
            ax.set_xlabel(f"fold ({ax_name} block)")
            ax.grid(alpha=0.3)
        axs[0].axhline(0.95, color="k", ls="--", lw=1)
        axs[1].axhline(0.0, color="k", lw=1)
        axs[2].axhline(QC_GATES["rough_ratio_vs_anchor_max"], color="r", ls="--", lw=1)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "cross_validation.png"), dpi=int(cfg.visualization.dpi))
        plt.close(fig)
    except Exception as e:
        print(f"[CV] figure skipped: {e}")
    return results
