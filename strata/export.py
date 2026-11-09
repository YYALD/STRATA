"""Output files: CSV layouts of the generated profiles and decays, the streamed profile store, the profile index, the
QC split, the manifold report, disk checks and the training history."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .common import ensure_dir, save_json
from .config import AppConfig
from .data.contract import MeasuredData
from .data.normalization import NormalizationState, raw_from_L
from .data.readers import SurveySpec


def _write_csv_chunks(path: str, header: Optional[List[str]], rows_fn, n_rows: int, chunk: int = 20000,
                      float_format: str = "%.6e") -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        if header is not None:
            f.write(",".join(header) + "\n")
        for s in range(0, n_rows, chunk):
            e = min(s + chunk, n_rows)
            pd.DataFrame(rows_fn(s, e)).to_csv(f, index=False, header=False, float_format=float_format)


_DIGITS = np.frombuffer(b"0123456789", dtype=np.uint8)


def format_sci7(X: np.ndarray) -> Optional[np.ndarray]:
    """Vectorised CSV formatting of a (n, k) block of positive finite values as 'd.dddddde+XX'.  Returns (n, 13 k)
    uint8 ASCII with ',' separators and '\n' ending every row, or None when the block holds non-positive / non-finite
    values or |exponent| > 99 (the caller then uses pandas)."""
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.size == 0 or not bool(np.all(np.isfinite(x))) or not bool(np.all(x > 0)):
        return None
    e = np.floor(np.log10(x)).astype(np.int64)
    m = x / np.power(10.0, e)
    lo, hi = m < 1.0, m >= 10.0
    if lo.any() or hi.any():
        e = e - lo + hi
        m = x / np.power(10.0, e)
    q = np.rint(m * 1e6).astype(np.int64)
    over = q >= 10_000_000
    if over.any():
        q[over] = 1_000_000
        e[over] += 1
    if bool(np.any(np.abs(e) > 99)):
        return None
    n, k = x.shape
    out = np.empty((n, k, 13), dtype=np.uint8)
    out[..., 0] = _DIGITS[q // 1_000_000]
    out[..., 1] = 46
    for j in range(6):
        out[..., 2 + j] = _DIGITS[(q // 10 ** (5 - j)) % 10]
    out[..., 8] = 101
    out[..., 9] = np.where(e < 0, 45, 43)
    ae = np.abs(e)
    out[..., 10] = _DIGITS[ae // 10]
    out[..., 11] = _DIGITS[ae % 10]
    out[..., 12] = 44
    out[:, -1, 12] = 10
    return out.reshape(n, k * 13)


def _write_csv_rows(f, prefixes: List[bytes], X: np.ndarray) -> None:
    """Write rows 'prefix + values' (prefix already ends with ',')."""
    body = format_sci7(X)
    if body is None:
        import io
        buf = io.StringIO()
        pd.DataFrame(np.asarray(X, dtype=np.float64)).to_csv(buf, index=False, header=False, float_format="%.6e",
                                                             lineterminator="\n")
        lines = buf.getvalue().split("\n")
        f.write(b"".join(pre + ln.encode("ascii") + b"\n" for pre, ln in zip(prefixes, lines)))
        return
    W = body.shape[1]
    mv = memoryview(body.reshape(-1))
    parts: List[Any] = []
    for i, pre in enumerate(prefixes):
        parts.append(pre)
        parts.append(mv[i * W:(i + 1) * W])
    f.write(b"".join(parts))


def _depth_label(d: float) -> str:
    return f"{d:.4f}"


class GeneratedProfileStore:
    """Dense generated profiles in measured units (float32), streamed to an .npy memmap during generation;
    the CSV files are written from it afterwards.  Keeps the first `n_keep` profiles (log10) for the figures."""

    def __init__(self, out_dir: str, n_profiles: int, depth_dense: np.ndarray, grid_ms: np.ndarray,
                 norm: NormalizationState, keep_npy: bool = True, n_keep: int = 64) -> None:
        ensure_dir(out_dir)
        self.keep_npy = bool(keep_npy)
        self.path = os.path.join(out_dir, "generated_profiles_float32.npy" if keep_npy else ".generated_profiles_tmp.npy")
        self.norm = norm
        self.shape = (int(n_profiles), int(len(depth_dense)), int(len(grid_ms)))
        self.arr = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float32, shape=self.shape)
        self.n_keep = int(n_keep)
        self.log_keep: List[np.ndarray] = []
        self.n_written = 0

    def put(self, i0: int, L_dense: np.ndarray) -> None:
        V = raw_from_L(np.asarray(L_dense, dtype=np.float64), self.norm).astype(np.float32)
        self.arr[i0:i0 + V.shape[0]] = V
        self.n_written = max(self.n_written, i0 + V.shape[0])
        have = sum(x.shape[0] for x in self.log_keep)
        if have < self.n_keep:
            self.log_keep.append(np.asarray(L_dense[: self.n_keep - have], dtype=np.float32).copy())

    def log_sample(self) -> np.ndarray:
        return np.concatenate(self.log_keep) if self.log_keep else np.zeros((0,) + self.shape[1:], np.float32)

    def truncate(self, n: int) -> None:
        """Keep the first n profiles (qc_export split: fewer accepted than requested within the slot budget)."""
        n = int(n)
        if n >= self.shape[0] or self.arr is None:
            return
        tmp = self.path + ".part.npy"
        new = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n,) + self.shape[1:])
        for i0 in range(0, n, 256):
            new[i0:min(i0 + 256, n)] = self.arr[i0:min(i0 + 256, n)]
        new.flush()
        for a_ in (new, self.arr):
            mm = getattr(a_, "_mmap", None)
            if mm is not None:
                mm.close()
        del new
        self.arr = None
        os.replace(tmp, self.path)
        self.arr = np.load(self.path, mmap_mode="r+")
        self.shape = (n,) + self.shape[1:]
        self.n_written = min(self.n_written, n)

    def close(self) -> None:
        try:
            self.arr.flush()
            mm = getattr(self.arr, "_mmap", None)
            del self.arr
            if mm is not None:
                mm.close()
        except Exception:
            pass
        self.arr = None
        if not self.keep_npy:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def reader(self) -> np.ndarray:
        return self.arr if self.arr is not None else np.load(self.path, mmap_mode="r")


def export_qc_split(qs: Dict[str, Any], trace: Dict[str, Any], meas: MeasuredData, cfg: AppConfig,
                    reports_dir: str) -> Dict[str, Any]:
    """qc_export split: qc_split_report.json (rule, counts, rates, gates failed, selection audit) and the review /
    rejected profiles apart from the data set (<generated data>/qc_review, qc_rejected: station-level log10 profiles
    + their index with every parameter and acceptance metric)."""
    rep = {k: v for k, v in qs.items() if k not in ("review", "rejected")}
    base = {k: v for k, v in trace.items() if k not in ("params", "qc", "qc_split", "anchor_L", "anchor_L0")}
    for name, folder in (("review", "qc_review"), ("rejected", "qc_rejected")):
        box = qs.get(name) or {}
        L = box.get("L")
        if L is None or not len(L):
            rep[f"{name}_export"] = None
            continue
        d = ensure_dir(os.path.join(cfg.export.generated_data_path, folder))
        np.savez_compressed(os.path.join(d, "profiles_station_level_log10.npz"), L_log10=np.asarray(L, dtype=np.float32),
                            depth_m=np.asarray(meas.gen_depth, dtype=np.float64),
                            time_ms=np.asarray(meas.grid_ms, dtype=np.float64))
        idx = build_profile_index({**base, "params": box.get("params", {}), "qc": box.get("qc", {})}, int(len(L)),
                                  meas.survey)
        idx.insert(1, "qc_status", name)
        idx.to_csv(os.path.join(d, "index.csv"), index=False)
        rep[f"{name}_export"] = {"folder": d, "profiles": int(len(L)),
                                 "note": "not part of the generated data set; station-level profiles (log10) + index"}
    save_json(rep, os.path.join(reports_dir, "qc_split_report.json"))
    return rep


def build_profile_index(trace: Dict[str, Any], n: int, survey: Optional[SurveySpec] = None) -> pd.DataFrame:
    """One row per generated profile: sample_id, acceptance, every random parameter (incl. each anomaly). Line
    surveys label the anomaly centre as a position along the line and the time-flat type as a host zone; extended
    runs add the profile's departure draws from the extension rule (in units of the calibrated spread)."""
    prm, qc = trace.get("params", {}), trace.get("qc", {})
    buried = survey is not None and survey.anomaly_family == "buried"
    pos_col, flat_name = ("position_m", "host-zone") if buried else ("depth_m", "near-hole")
    nan = np.full(n, np.nan)
    cols: Dict[str, Any] = {"sample_id": np.arange(1, n + 1)}
    if "pass" in qc:
        cols["qc_pass"] = np.asarray(qc["pass"]).astype(int)
        cols["tries"] = np.asarray(qc.get("tries", np.ones(n))).astype(int)
        if "qc_margin" in qc:     # smallest normalised slack of the acceptance gates
            cols["qc_margin"] = np.asarray(qc["qc_margin"], dtype=np.float64)
    cols["alpha"] = prm.get("alpha", nan)
    cols["sim_registration_zoom"] = prm.get("zoom", nan)
    cols["sim_registration_offset"] = prm.get("offset", nan)
    cols["sim_structure_gain"] = prm.get("gain", nan)
    cols["bg_level_decades"] = prm.get("bg_level", np.zeros(n))
    cols["bg_tilt_decades"] = prm.get("bg_tilt", np.zeros(n))
    if "bg_lslope" in prm:   # pure: the profile's regional trend (decades across the measured span)
        cols["bg_level_slope_decades"] = prm["bg_lslope"]
        cols["bg_level_curvature_decades"] = prm["bg_lcurv"]
        cols["bg_tilt_slope_decades"] = prm["bg_tslope"]
    if "bg_contrast" in prm:  # trend: log contrast factor c0 and its slope c1 over the normalised log time
        cols["bg_contrast_log_factor"] = prm["bg_contrast"]
        cols["bg_contrast_time_slope"] = prm["bg_contrast_t"]
    if "an_A" in prm:
        K = np.asarray(prm["an_K"]).astype(int)
        cols["n_anomalies"] = K
        if "ma_on" in prm:
            # the measured anomalies are labelled anomalies too: total = synthetic + measured
            on = np.asarray(prm["ma_on"]) > 0.5
            cols["n_anomalies"] = K + on.sum(axis=1)
            cols["n_synthetic_anomalies"] = K
            cols["n_measured_anomalies"] = on.sum(axis=1)
            scl = np.asarray(prm["ma_scale"])
            for j, r_ in enumerate(trace.get("measured_anomalies", [])):
                cols[f"measured{j + 1}_from_m"] = np.where(on[:, j], float(r_["from_m"]), np.nan)
                cols[f"measured{j + 1}_to_m"] = np.where(on[:, j], float(r_["to_m"]), np.nan)
                cols[f"measured{j + 1}_peak_at_m"] = np.where(on[:, j], float(r_["peak_at_m"]), np.nan)
                cols[f"measured{j + 1}_peak_time_ms"] = np.where(on[:, j], float(r_["peak_time_ms"]), np.nan)
                cols[f"measured{j + 1}_scale"] = np.where(on[:, j], scl[:, j], np.nan)
                cols[f"measured{j + 1}_peak_decades"] = np.where(on[:, j], scl[:, j] * float(r_["peak_decades"]), np.nan)
                if "measured_anomaly_time_profile" in trace:
                    # the times at which the scaled measured anomaly reaches half the visibility floor somewhere along
                    # the profile (the synthetic anomalies' visible-window rule); empty when it never does
                    tp = np.asarray(trace["measured_anomaly_time_profile"])[j]
                    tg = np.asarray(trace["time_ms"])
                    vis = scl[:, j, None] * tp[None, :] >= 0.5 * float(trace["visibility_floor_decades"])
                    anyv = on[:, j] & vis.any(axis=1)
                    first = np.argmax(vis, axis=1)
                    last = tp.size - 1 - np.argmax(vis[:, ::-1], axis=1)
                    cols[f"measured{j + 1}_visible_from_ms"] = np.where(anyv, tg[first], np.nan)
                    cols[f"measured{j + 1}_visible_to_ms"] = np.where(anyv, tg[last], np.nan)
        Kmax = prm["an_A"].shape[1]
        phys = "an_kind" in prm
        bank_names = trace.get("morphology_bank_names")          # kind 3 from the morphology bank
        ml_name = "morphology" if bank_names else "measured-like"
        kind_names = np.array(["conductor, hole through", "conductor beside the hole", "near-hole zone", ml_name])
        mlk = "an_tpl" in prm                     # measured model: measured shapes at other depths
        for k in range(Kmax):
            act = K > k
            cols[f"anomaly{k + 1}_{pos_col}"] = np.where(act, prm["an_z0"][:, k], np.nan)
            # the signed peak is the anomaly's largest |change| as labelled from its own field (= the drawn amplitude;
            # for a conductor beside the hole the lobe that reaches it)
            pk = prm["lab_peak"][:, k] if "lab_peak" in prm else prm["an_A"][:, k]
            cols[f"anomaly{k + 1}_peak_decades"] = np.where(act, pk, np.nan)
            cols[f"anomaly{k + 1}_sigma0_m"] = np.where(act, prm["an_w0"][:, k], np.nan)
            kd_ = (np.clip(np.asarray(prm["an_kind"][:, k]).round().astype(int), 0, 3) if phys
                   else np.zeros(n, dtype=int))
            is_ml = phys & (kd_ == 3)
            cols[f"anomaly{k + 1}_tau_ms"] = np.where(act & ~is_ml, 10.0 ** prm["an_ltau"][:, k], np.nan)
            cols[f"anomaly{k + 1}_type"] = np.where(act, np.where(prm["an_type"][:, k] > 0.5, flat_name,
                                                                  np.where(is_ml, ml_name, "conductor")), "")
            if phys:
                names_k = kind_names[kd_]
                if mlk and bank_names:
                    bn = np.array([f"morphology ({n_})" for n_ in bank_names] + [""])
                    tp_ = np.clip(np.asarray(prm["an_tpl"][:, k]).round().astype(int), 0, len(bank_names) - 1)
                    names_k = np.where(is_ml, bn[tp_], names_k)
                    if "an_src" in prm:
                        sn = np.array(["measured", "simulated", "learned", ""])
                        sv = np.asarray(prm["an_src"][:, k]).round().astype(int)
                        cols[f"anomaly{k + 1}_morphology_source"] = np.where(act & is_ml, sn[np.clip(sv, 0, 2)], "")
                elif mlk:
                    tp_ = np.asarray(prm["an_tpl"][:, k]).round().astype(int) + 1
                    names_k = np.where(is_ml, np.char.add(np.char.add("measured-like (shape of measured ", tp_.astype(str)),
                                                          ")"), names_k)
                cols[f"anomaly{k + 1}_kind"] = np.where(act, names_k, "")
                cols[f"anomaly{k + 1}_distance_m"] = np.where(act & ~is_ml, prm["an_w0"][:, k], np.nan)
                if mlk:
                    # which measured anomaly's shape and its along-hole size factor (with the depth and the peak they
                    # rebuild the field exactly: _measured_like_fields)
                    cols[f"anomaly{k + 1}_template"] = np.where(act & is_ml, np.asarray(prm["an_tpl"][:, k]) + 1, np.nan)
                    cols[f"anomaly{k + 1}_size_factor"] = np.where(act & is_ml, prm["an_s"][:, k], np.nan)
                    # the |peak| the field is scaled to on the moved template's own stations (the dense peak above can
                    # be a little larger between them)
                    cols[f"anomaly{k + 1}_drawn_peak_decades"] = np.where(act & is_ml, np.abs(prm["an_A"][:, k]), np.nan)
                cols[f"anomaly{k + 1}_dip_deg"] = np.where(act & (kd_ == 1), np.degrees(prm["an_dip"][:, k]), np.nan)
                # orientation of the dipole kernel (+1: at small dips the negative-centred three-peak anomaly); with the
                # peak, distance, dip, tau and depth it reproduces the anomaly field exactly
                cols[f"anomaly{k + 1}_polarity"] = np.where(act & (kd_ == 1), np.sign(prm["an_A"][:, k]), np.nan)
            if "lab_peak" in prm:
                cols[f"anomaly{k + 1}_peak_at_m"] = np.where(act, prm["lab_peak_at"][:, k], np.nan)
                cols[f"anomaly{k + 1}_peak_time_ms"] = np.where(act, prm["lab_peak_ms"][:, k], np.nan)
                cols[f"anomaly{k + 1}_visible_from_ms"] = np.where(act, prm["lab_vis_from_ms"][:, k], np.nan)
                cols[f"anomaly{k + 1}_visible_to_ms"] = np.where(act, prm["lab_vis_to_ms"][:, k], np.nan)
            if "lab_rise" in prm:
                # the undulation the anomaly alone makes on the trend: largest rise against the decay along the profile
                # (decades, dense axis, measured window)
                cols[f"anomaly{k + 1}_rise_decades"] = np.where(act, prm["lab_rise"][:, k], np.nan)
    else:
        cols["n_anomalies"] = np.zeros(n, dtype=int)
    if "unk_eps" in prm:
        ue = np.asarray(prm["unk_eps"])
        sides = trace.get("extension_sides", ("low", "high"))
        for j, side in enumerate(("low", "high")):
            if side in sides:                     # only the ends that are extended carry a draw that acts
                cols[f"extension_draw_{side}_end"] = ue[:, j]
        for g in range(2, ue.shape[1]):
            cols[f"gap{g - 1}_draw"] = ue[:, g]
    if "text_eps" in prm:
        cols["late_time_extension_draw"] = np.asarray(prm["text_eps"])
    if "warp_b" in prm:   # background warp: mode coefficients (|dz'/dz - 1| <= sum |b_j|) and the largest shift (m)
        wb = np.asarray(prm["warp_b"])
        for j in range(wb.shape[1]):
            cols[f"warp_b{j + 1}"] = wb[:, j]
        cols["warp_max_shift_m"] = np.asarray(prm["warp_umax_m"])
    for k in ("rough_ratio", "envelope_frac", "anomaly_retention", "anomaly_contrast", "anomaly_corr",
              "max_abs_anomaly_decades", "dist_to_measured_decades", "purity_max_decades", "field_admissible_frac",
              "cmg_gate_mean"):
        if k in qc:
            cols[k] = qc[k]
    if "purity_ok" in qc:   # 1 = no unlabelled anomaly-scale feature (within the purity tolerance)
        cols["purity_ok"] = np.asarray(qc["purity_ok"]).astype(int)
    if "trend_step_decades" in qc:   # "0 anomalies": largest step against the decay trend before export (else NaN)
        cols["trend_step_decades"] = qc["trend_step_decades"]
    return pd.DataFrame(cols)


def manifold_report(trainer: "RFMGGenTrainer", trace: Dict[str, Any], acc: Dict[str, float]) -> Dict[str, Any]:
    """The manifold constraints of a run: the FAM (anchors, radii, leave-one-out occupancy of the measured curves,
    occupancy of the generated background curves), the CMG gate (form, contraction statistics) and the warp
    background (modes, strain bound, largest shifts)."""
    cfg = trainer.cfg
    rep_: Dict[str, Any] = {"manifold_constraint": str(cfg.profile.manifold_constraint),
                            "background_manifold": ("warp" if getattr(trainer, "bg_warp", False) else
                                                    ("affine" if getattr(trainer, "bg_trend", False) else "quadratic"))}
    fam = getattr(trainer, "fam", None)
    if fam is not None:
        rep_["fam"] = {**fam.summary(), "enabled": bool(trainer.fam_on), "min_frac": float(cfg.profile.fam_min_frac),
                       "lambda_field_shell": float(cfg.losses.lambda_field_shell),
                       "background_tolerance_decades": float(trainer.fam_bg_tol),
                       "generated_occupancy": acc.get("field_admissible_occupancy", float("nan")),
                       "generated_frac_median": acc.get("field_admissible_frac_median", float("nan")),
                       "generated_frac_p10": acc.get("field_admissible_frac_p10", float("nan"))}
    if getattr(trainer, "cmg", None) is not None:
        rep_["cmg"] = {"form": str(getattr(trainer.cmg, "form", "v20")),
                       "gate_median": acc.get("cmg_gate_median", float("nan")),
                       "gate_p90": acc.get("cmg_gate_p90", float("nan"))}
        if getattr(trainer.cmg, "form", "v20") == "eaai":
            rep_["cmg"]["tau_U"] = float(torch.exp(trainer.cmg.log_tau_U).clamp(0.005, 0.5).item())
    prm = trace.get("params", {})
    if "warp_umax_m" in prm:
        u = np.asarray(prm["warp_umax_m"], dtype=np.float64)
        b = np.asarray(prm["warp_b"], dtype=np.float64)
        rep_["warp"] = {"modes": int(b.shape[1]), "max_strain": float(trainer.warp_strain),
                        "shift_m_median": float(np.median(u)), "shift_m_p90": float(np.percentile(u, 90)),
                        "shift_m_max": float(u.max()), "strain_bound_median": float(np.median(np.abs(b).sum(axis=1))),
                        "coef_sd": np.std(b, axis=0).tolist(),
                        # purity refers to the warped anchor; the warp's own content (ratio to the unwarped background,
                        # bound (1 + s)^4)
                        "purity_reference": "warped anchor (v22.2)",
                        "content_bound_factor": float((1.0 + float(trainer.warp_strain)) ** 4),
                        "content_ratio_median": acc.get("warp_content_ratio_median", float("nan")),
                        "content_ratio_p90": acc.get("warp_content_ratio_p90", float("nan")),
                        "content_ratio_max": acc.get("warp_content_ratio_max", float("nan")),
                        "warp_ok_rate": acc.get("warp_ok_rate", float("nan")),
                        "v21_charge_decades_median": acc.get("warp_content_decades_median", float("nan")),
                        "v21_charge_decades_p90": acc.get("warp_content_decades_p90", float("nan"))}
    return rep_


def export_size_estimate(n_profiles: int, n_points: int, T: int, n_gates: int, cfg: "AppConfig") -> Dict[str, float]:
    """Bytes the export will write (CSV cells ~14 bytes at 7 significant digits, row prefixes ~36 bytes, float32
    npy), for the free-space preflight."""
    cell, pre = 14.0, 36.0
    est = {"decay_curves_csv": n_profiles * n_points * (T * cell + pre),
           "profiles_csv": n_profiles * T * (n_points * cell + pre)}
    if cfg.export.export_measured_gates:
        est["measured_gates_csv"] = n_profiles * n_points * (n_gates * cell + pre)
    est["npy"] = n_profiles * n_points * T * 4.0
    est["total"] = float(sum(est.values()))
    return est


def disk_preflight(path: str, need_bytes: float, stage: str, hard: bool) -> None:
    """Free space of the output folder against the export size: a warning before training, an error before the
    generation (the trained checkpoint is saved by then; free the space and rerun with --train_action generate_only)."""
    try:
        free = float(shutil.disk_usage(ensure_dir(path)).free)
    except Exception as e:
        print(f"[DISK] free-space check skipped ({e})")
        return
    msg = (f"[DISK] {stage}: the export needs about {need_bytes / 2 ** 30:.1f} GB, {free / 2 ** 30:.1f} GB free on the drive "
           f"of {path}")
    if free >= 1.05 * need_bytes:
        print(msg + " -> ok")
        return
    if hard:
        raise RuntimeError(msg + " -> not enough space; free it (or lower --generated_profile_count / --profile_depth_points)"
                           " and rerun with --train_action generate_only")
    print(msg + " -> WARNING: not enough space for the export at the moment")


def save_generated_outputs(store: GeneratedProfileStore, depth_dense: np.ndarray, station_index: np.ndarray,
                           meas: MeasuredData, trace: Dict[str, Any], cfg: AppConfig, out_dir: str,
                           norm: NormalizationState, L_stations: Optional[np.ndarray] = None,
                           chunk_profiles: int = 16) -> Dict[str, str]:
    """Single-file exports (measured units, 7 significant digits), numbered by sample_id = 1..N:
      generated_decay_curves.csv   one decay curve per row: sample_id, depth_index, depth (m), measured_station,
                                   then the T grid times (ms) as columns          -> N x Dd rows
      generated_profiles.csv       one depth profile per row: sample_id, gate_index, time (ms), measured_gate,
                                   then the Dd depths (m) as columns               -> N x T rows
      generated_decay_curves_measured_gates.csv  decay curves on the measured gate times only (measured layout)
      generated_profile_index.csv  per sample_id: acceptance, every random parameter, each anomaly
      generated_profiles_float32.npy + generated_axes.npz  the same values (N, Dd, T) for fast reading
    Indices (depth_index, gate_index) are 1-based; measured_station / measured_gate flag the measured nodes. Line
    surveys (airborne / ground) name the profile axis position_index, distance (m).  A profile extension adds the
    column measured_span (1 inside the measured stations' span, 0 on the extension) to the decay-curve files, a
    late-time extension the column measured_window (1 up to the last measured gate) to the profile file."""
    ensure_dir(out_dir)
    t0 = time.time()
    N, Dd, T = store.shape
    arr = store.reader()
    grid = meas.grid_ms
    paths: Dict[str, str] = {}
    sid = [str(i).encode("ascii") for i in range(1, N + 1)]
    st_flag = np.zeros(Dd, dtype=int)
    st_flag[np.asarray(station_index, dtype=int)] = 1
    dlab = [_depth_label(d) for d in depth_dense]
    gate_flag = np.zeros(T, dtype=int)
    gi = meas.gate_index
    if np.all(gi >= 0):
        gate_flag[gi] = 1
    tlab = [f"{t:.9g}" for t in grid]
    sv = meas.survey
    ax_lab = sv.axis_label if sv is not None else "depth (m)"
    ix_lab = sv.index_label if sv is not None else "depth_index"
    ext_prof = meas.axis_depth is not None
    dd_ = np.asarray(depth_dense, dtype=np.float64)
    span_flag = ((dd_ >= meas.depth[0] - 1e-9) & (dd_ <= meas.depth[-1] + 1e-9)).astype(int)
    ext_time = meas.late_ext is not None and float(meas.t_known_ms) > 0
    win_flag = (grid <= float(meas.t_known_ms) * (1.0 + 1e-9)).astype(int) if ext_time else np.ones(T, dtype=int)
    span_hdr = ",measured_span" if ext_prof else ""
    # --no_export_csv: experiment runs keep the array, the index and the measured-gate CSV (1000 x 1000 x 256 values are
    # two CSVs of ~3.3 GB each)
    dpre = [(f",{j + 1},{dlab[j]},{st_flag[j]}" + (f",{span_flag[j]}" if ext_prof else "") + ",").encode("ascii")
            for j in range(Dd)]
    if cfg.export.export_csv:
        # 1. decay curves (rows = sample x depth)
        p1 = os.path.join(out_dir, "generated_decay_curves.csv")
        with open(p1, "wb") as f:
            f.write((f"sample_id,{ix_lab},{ax_lab},measured_station{span_hdr}," + ",".join(tlab) + "\n").encode("utf-8"))
            for s0 in range(0, N, chunk_profiles):
                blk = np.asarray(arr[s0:s0 + chunk_profiles])
                pre = [sid[s0 + i] + dpre[j] for i in range(blk.shape[0]) for j in range(Dd)]
                _write_csv_rows(f, pre, blk.reshape(-1, T))
        paths["decay_curves_csv"] = p1
        # 2. profiles (rows = sample x time gate)
        p2 = os.path.join(out_dir, "generated_profiles.csv")
        gpre = [(f",{g + 1},{tlab[g]},{gate_flag[g]}" + (f",{win_flag[g]}" if ext_time else "") + ",").encode("ascii")
                for g in range(T)]
        with open(p2, "wb") as f:
            f.write(("sample_id,gate_index,time (ms),measured_gate" + (",measured_window" if ext_time else "") + ","
                     + ",".join(dlab) + "\n").encode("utf-8"))
            for s0 in range(0, N, chunk_profiles):
                blk = np.asarray(arr[s0:s0 + chunk_profiles]).transpose(0, 2, 1)
                pre = [sid[s0 + i] + gpre[g] for i in range(blk.shape[0]) for g in range(T)]
                _write_csv_rows(f, pre, np.ascontiguousarray(blk).reshape(-1, Dd))
        paths["profiles_csv"] = p2
    else:
        print("[EXPORT] decay-curve / profile CSVs skipped (--no_export_csv)")
    # 3. decay curves on the measured gates
    if cfg.export.export_measured_gates and not np.all(gi >= 0):
        print("[EXPORT] generated_decay_curves_measured_gates.csv not written: the measured gates are not grid nodes "
              "(target_len below the gate count, or a late-time extension at target_len = gate count)")
    if cfg.export.export_measured_gates and np.all(gi >= 0):
        p3 = os.path.join(out_dir, "generated_decay_curves_measured_gates.csv")
        with open(p3, "wb") as f:
            f.write((f"sample_id,{ix_lab},{ax_lab},measured_station{span_hdr},"
                     + ",".join(f"{t:.9g}" for t in meas.pm.time_ms) + "\n").encode("utf-8"))
            for s0 in range(0, N, chunk_profiles):
                blk = np.asarray(arr[s0:s0 + chunk_profiles])[:, :, gi]
                pre = [sid[s0 + i] + dpre[j] for i in range(blk.shape[0]) for j in range(Dd)]
                _write_csv_rows(f, pre, np.ascontiguousarray(blk).reshape(-1, gi.size))
        paths["decay_curves_measured_gates_csv"] = p3
    # 4. index
    p4 = os.path.join(out_dir, "generated_profile_index.csv")
    build_profile_index(trace, N, meas.survey).to_csv(p4, index=False, float_format="%.6g")
    paths["profile_index_csv"] = p4
    # 5. arrays
    if store.keep_npy:
        p5 = os.path.join(out_dir, "generated_axes.npz")
        np.savez(p5, time_ms=grid, depth_m=np.asarray(depth_dense), station_index_dense=np.asarray(station_index),
                 measured_gate_index=gi, sample_id=np.arange(1, N + 1), measured_depth_m=meas.depth,
                 measured_time_ms=meas.pm.time_ms, profile_axis_label=np.array(ax_lab),
                 survey_mode=np.array(sv.mode if sv is not None else "borehole"), measured_span=span_flag,
                 measured_window=win_flag)
        paths["profiles_npy"] = store.path
        paths["axes_npz"] = p5
    # 6. optional per-profile files (first K)
    k = int(max(0, min(cfg.export.export_profile_samples, N)))
    if k:
        pdir = ensure_dir(os.path.join(out_dir, "profiles"))
        for i in range(k):
            with open(os.path.join(pdir, f"profile_{i + 1:05d}.csv"), "wb") as f:
                f.write((f"{ax_lab}," + ",".join(tlab) + "\n").encode("utf-8"))
                _write_csv_rows(f, [(dlab[j] + ",").encode("ascii") for j in range(Dd)], np.asarray(arr[i]))
        paths["profile_samples_dir"] = pdir
    if cfg.export.export_legacy_csv and L_stations is not None:
        paths.update(save_legacy_outputs(L_stations, meas, trace, out_dir, norm))
    sizes = {k_: os.path.getsize(v_) for k_, v_ in paths.items() if os.path.isfile(v_)}
    print(f"[EXPORT] {N} profiles x {Dd} {'depths' if ix_lab == 'depth_index' else 'positions'} x {T} gates -> {out_dir} "
          f"({time.time() - t0:.0f}s)")
    for k_, v_ in paths.items():
        sz = f" ({sizes[k_] / 2 ** 20:.1f} MB)" if k_ in sizes else ""
        print(f"[EXPORT]   {k_:<34}: {v_}{sz}")
    return paths


def save_legacy_outputs(L_gen: np.ndarray, meas: MeasuredData, trace: Dict[str, Any], out_dir: str,
                        norm: NormalizationState) -> Dict[str, str]:
    """Legacy layouts on the measured stations: generated_data_flat.csv (row id + T values) and
    generated_data_matrix.csv (header [axis name, grid times]; rows [depth, T values]; profiles stacked)."""
    P, D, T = L_gen.shape
    V = raw_from_L(L_gen, norm).astype(np.float64)
    flat = V.reshape(P * D, T)
    paths: Dict[str, str] = {}
    p_flat = os.path.join(out_dir, "generated_data_flat.csv")
    _write_csv_chunks(p_flat, None, lambda s, e: np.concatenate([np.arange(s, e, dtype=np.float64)[:, None], flat[s:e]], 1),
                      P * D)
    paths["legacy_flat_csv"] = p_flat
    dep_col = np.tile(meas.depth, P)
    p_mat = os.path.join(out_dir, "generated_data_matrix.csv")
    _write_csv_chunks(p_mat, [meas.pm.axis_name or "depth (m)"] + [f"{t:.9g}" for t in meas.grid_ms],
                      lambda s, e: np.concatenate([dep_col[s:e, None], flat[s:e]], 1), P * D)
    paths["legacy_matrix_csv"] = p_mat
    return paths


def export_history(history: List[Dict[str, Any]], reports_dir: str) -> Dict[str, str]:
    ensure_dir(reports_dir)
    p_csv = os.path.join(reports_dir, "training_history.csv")
    p_js = os.path.join(reports_dir, "training_history.json")
    if history:
        pd.DataFrame(history).to_csv(p_csv, index=False)
    save_json(history, p_js)
    return {"csv": p_csv, "json": p_js}


def print_qc_table(qc: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"[QC GATE] status = {qc['status']}")
    for k, v in qc["checks"].items():
        val = "n/a" if v["value"] is None else f"{v['value']:.4f}"
        print(f"[QC GATE]   {'PASS' if v['pass'] else 'FAIL'}  {k:<30} {val}")
    print("=" * 88)
