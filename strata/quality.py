"""Quality metrics of decays and profiles, structure fidelity, the QC gates and the composite score that selects
checkpoints."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .data.contract import local_structure_residual
from .losses import local_structure_torch, structure_projection
from .physics import PHYS_NAMES, PhysicsDescriptor


QC_GATES = {
    "monotone_fraction_min": 0.999,       # non-increasing decays (non-crossing profile families)
    "finite_positive_fraction_min": 1.0,
    "rough_ratio_vs_anchor_max": 1.5,     # profile local structure vs the union-bridge anchors
    "measured_coverage_min": 0.80,        # measured cells inside the generated 2.5-97.5 % band
    "copy_fraction_max": 0.05,            # profiles within 0.01 decades RMS of the measured profile
    "descriptor_precision_min": 0.90,
    "descriptor_recall_min": 0.90,
    "profile_diversity_min_decades": 0.02,
    "anomaly_retention_tol": 0.30,        # |1 - median slope of generated vs intended anomaly field|
    "anomaly_corr_min": 0.70,             # median correlation of generated vs intended anomaly field
    "accept_rate_min": 0.95,              # profiles passing the per-profile acceptance after redraws
    # signals, no added noise: station-scale depth texture and high-frequency time texture of the network's departure
    # from the anchors, relative to the anchors' own (median over profiles, departure_texture)
    "departure_depth_texture_max": 0.25,
    "departure_time_texture_max": 0.10,
    # background_mode pure: profiles whose anomaly-scale content off their labelled anomalies is within the purity
    # tolerance (a profile labelled with k anomalies shows exactly those k; "0 anomalies" = pure smooth background)
    "background_purity_rate_min": 0.99,
    # background_trend monotone: "0 anomalies" profiles with no step against the overall decay trend along the profile
    # larger than a quarter of the purity tolerance (no rise, no dip)
    "zero_anomaly_trend_rate_min": 0.99,
    # field-admissible manifold: background curves of the generated profiles inside the union of the measured background
    # curves' descriptor balls (FieldAdmissibleManifold), pooled over the profiles
    "field_admissible_occupancy_min": 0.95,
}
# validation-score tolerance of the anchor-structure retention (a selection target, not a QC gate): within 0.9-1.1 the
# checkpoint score is indifferent, outside it the loss of (or added) anchor structure counts like the anomaly terms
STRUCTURE_RETENTION_TOL = 0.10
# background purity.  The tolerance (purity_tol, auto = half the smallest labelled anomaly amplitude) bounds the
# anomaly-scale content (high-pass at the background scale) of a generated profile off its labelled anomalies.  The
# background anchor may use a quarter of it (purify_background), the network's departure the rest.
PURITY_ANCHOR_SHARE = 0.25
PURITY_LOSS_MARGIN = 0.80
# the regional departure head's label weight switches between the network's departure (on a synthetic label) and the
# regional trend; it is filled and ramped along log time over +-this many decades (synthetic_label_weight), so a switch
# never happens within one or two time samples
LABEL_WEIGHT_SMOOTH_DECADES = 0.05
# validation-score tolerance of the roughness ratio vs the anchors: up to 10 % above the anchors is the designed latent
# texture; beyond it log(ratio / 1.1) counts
ROUGH_SELECT_TOL = 0.10


def _nn_dist(a: np.ndarray, b: np.ndarray, block: int = 2048, exclude_self: bool = False) -> np.ndarray:
    out = np.empty(a.shape[0], dtype=np.float64)
    for s in range(0, a.shape[0], block):
        e = min(s + block, a.shape[0])
        d = np.sqrt(np.maximum(((a[s:e, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
        if exclude_self:
            for i in range(s, e):
                if i < b.shape[0]:
                    d[i - s, i] = np.inf
        out[s:e] = d.min(axis=1)
    return out


def curve_quality_metrics(L_gen: np.ndarray, L_meas: np.ndarray, L_union: np.ndarray, phys: PhysicsDescriptor,
                          max_gen: int = 6000, seed: int = 0) -> Dict[str, float]:
    """L_*: (N, T) log10 curves.  Precision = generated curves near the measured+sim-structured union,
    recall = measured curves with a generated neighbour; radii from the reference's own NN spacing."""
    rng = np.random.default_rng(seed)
    G = np.asarray(L_gen, dtype=np.float64).reshape(-1, phys.T)
    if G.shape[0] > max_gen:
        G = G[rng.choice(G.shape[0], max_gen, replace=False)]
    out: Dict[str, float] = {}
    out["finite_positive_fraction"] = float(np.mean(np.all(np.isfinite(G), axis=1)))
    out["monotone_fraction"] = float(np.mean(np.all(np.diff(G, axis=1) <= 1e-9, axis=1)))
    fm, fu, fg = phys.numpy(L_meas), phys.numpy(L_union), phys.numpy(G)
    mu = np.median(fu, axis=0)
    sd = 1.4826 * np.median(np.abs(fu - mu), axis=0)
    sd = np.maximum(sd, 0.05 * (np.percentile(fu, 95, axis=0) - np.percentile(fu, 5, axis=0)) + 1e-6)
    zm, zu, zg = (fm - mu) / sd, (fu - mu) / sd, (fg - mu) / sd
    # precision = plausibility: generated descriptors inside the union envelope (+25 % of its range, the same envelope
    # the physics-guidance term uses); recall = the measured descriptors inside the generated envelope (+5 %).
    lo, hi = np.percentile(zu, 0.5, axis=0), np.percentile(zu, 99.5, axis=0)
    mg = 0.25 * (hi - lo) + 0.25
    out["descriptor_precision"] = float(np.mean(np.all((zg >= lo - mg) & (zg <= hi + mg), axis=1)))
    glo, ghi = np.percentile(zg, 0.5, axis=0), np.percentile(zg, 99.5, axis=0)
    gmg = 0.05 * (ghi - glo) + 1e-6
    out["descriptor_recall"] = float(np.mean(np.all((zm >= glo - gmg) & (zm <= ghi + gmg), axis=1)))
    r_m = 2.0 * float(np.median(_nn_dist(zm, zm, exclude_self=True))) if zm.shape[0] > 1 else 1.0
    r_u = 2.0 * float(np.median(_nn_dist(zu, zu, exclude_self=True))) if zu.shape[0] > 1 else 1.0
    r_m, r_u = max(r_m, 0.25), max(r_u, 0.25)
    out["descriptor_recall_knn"] = float(np.mean(_nn_dist(zm, zg) <= r_m))
    out["descriptor_precision_knn"] = float(np.mean(_nn_dist(zg, zu) <= r_u))
    out["descriptor_radius_measured"] = r_m
    out["descriptor_radius_union"] = r_u
    for k, name in enumerate(PHYS_NAMES):
        out[f"phys_{name}_gen_median"] = float(np.median(fg[:, k]))
        out[f"phys_{name}_meas_median"] = float(np.median(fm[:, k]))
    dg = np.sqrt(np.mean(np.diff(G, n=2, axis=1) ** 2, axis=1))
    dm = np.sqrt(np.mean(np.diff(np.asarray(L_meas), n=2, axis=1) ** 2, axis=1))
    out["time_roughness_ratio"] = float(np.median(dg) / max(np.median(dm), 1e-12))
    return out


def pairwise_profile_distance(P_gen: np.ndarray, max_profiles: int = 2000) -> Tuple[float, float, int]:
    """Profile diversity: median and p10 of the RMS distance (decades) over all pairs of up to max_profiles profiles
    (P, D, T)."""
    P = np.asarray(P_gen, dtype=np.float64)
    nP = int(P.shape[0])
    idx = np.arange(nP) if nP <= max_profiles else np.sort(np.random.default_rng(0).choice(nP, max_profiles, replace=False))
    if len(idx) < 2:
        return 0.0, 0.0, int(len(idx))
    Q = P[idx].reshape(len(idx), -1)
    Q = Q - Q.mean(axis=0, keepdims=True)
    sq = (Q * Q).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Q @ Q.T), 0.0) / Q.shape[1]
    dq = np.sqrt(d2[np.triu_indices(len(idx), 1)])
    return float(np.median(dq)), float(np.percentile(dq, 10)), int(len(idx))


def departure_texture(P_gen: np.ndarray, P_anchor: np.ndarray, depth: np.ndarray,
                      A_floor: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Noise-like texture the network adds to its anchors, per profile (log10 profiles (P, D, T)). depth_texture: RMS
    of the local structure (station minus the line through its neighbours) of the departure P - A over the RMS of the
    anchor's own local structure; time_texture: RMS of the high-frequency curvature along the time grid (second
    difference minus its 5-point moving average, as the tsmooth term) of the departure over the anchor's own.  Smooth
    departures (levels, decay-rate changes over several stations) give ~0; white noise of the anchor's
    local-structure size gives ~1.2 in depth.  A_floor (D, T): the depth reference is at least its local structure
    (background_mode pure: the measured profile; the smooth background anchors have almost none)."""
    P = np.asarray(P_gen, dtype=np.float64)
    A = np.asarray(P_anchor, dtype=np.float64)
    dev = P - A
    out: Dict[str, np.ndarray] = {}
    d = np.asarray(depth, dtype=np.float64)
    if d.size >= 3:
        w = ((d[1:-1] - d[:-2]) / np.maximum(d[2:] - d[:-2], 1e-12))[None, :, None]

        def _ls_rms(X: np.ndarray) -> np.ndarray:
            r = X[:, 1:-1] - ((1.0 - w) * X[:, :-2] + w * X[:, 2:])
            return np.sqrt(np.mean(r * r, axis=(1, 2)))

        ref = _ls_rms(A)
        if A_floor is not None:
            ref = np.maximum(ref, _ls_rms(np.asarray(A_floor, dtype=np.float64)[None]))
        out["depth_texture"] = _ls_rms(dev) / np.maximum(ref, 1e-12)
    if P.shape[2] >= 7:
        def _hf_rms(X: np.ndarray) -> np.ndarray:
            d2 = X[..., 2:] - 2.0 * X[..., 1:-1] + X[..., :-2]
            pad = np.concatenate([d2[..., :1], d2[..., :1], d2, d2[..., -1:], d2[..., -1:]], axis=-1)
            sm = sum(pad[..., i:i + d2.shape[-1]] for i in range(5)) / 5.0
            return np.sqrt(np.mean((d2 - sm) ** 2, axis=(1, 2)))

        out["time_texture"] = _hf_rms(dev) / np.maximum(_hf_rms(A), 1e-30)
    return out


def profile_quality_metrics(P_gen: np.ndarray, L_meas: np.ndarray, depth: np.ndarray,
                            P_anchor: Optional[np.ndarray] = None, n_bands: int = 8,
                            copy_tol_decades: float = 0.01, L_floor: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """P_gen: (P, D, T) log10 profiles on the measured stations; L_meas: (D, T) the reference (the measured profile,
    or its background in background_mode pure).  L_floor (D, T): roughness / texture references are at least its
    local structure, and the roughness ratio "vs measured" is taken against it (pure: the raw measured profile)."""
    P = np.asarray(P_gen, dtype=np.float64)
    M = np.asarray(L_meas, dtype=np.float64)
    nP, D, T = P.shape
    out: Dict[str, Any] = {"n_profiles": int(nP), "n_stations": int(D)}
    out["noncrossing_fraction"] = float(np.mean(np.all(np.diff(P, axis=2) <= 1e-9, axis=2)))
    bands = np.array_split(np.arange(T), n_bands)

    def _band_rms(X: np.ndarray) -> np.ndarray:
        r = local_structure_residual(X, depth)
        return np.array([np.sqrt(np.mean(r[:, b] ** 2)) for b in bands])

    rm = _band_rms(M)
    if L_floor is not None:
        rm = np.maximum(rm, _band_rms(np.asarray(L_floor, dtype=np.float64)))
    rg = np.stack([_band_rms(P[i]) for i in range(nP)])
    out["rough_rms_measured_by_band"] = rm.tolist()
    out["rough_rms_generated_median_by_band"] = np.median(rg, axis=0).tolist()
    out["rough_ratio_vs_measured_median"] = float(np.median(np.median(rg, axis=0) / np.maximum(rm, 1e-6)))
    if P_anchor is not None:
        A = np.asarray(P_anchor, dtype=np.float64)
        ra = np.stack([_band_rms(A[i]) for i in range(nP)])
        ratio = rg / np.maximum(ra, np.maximum(rm[None, :], 1e-6))
        out["rough_ratio_vs_anchor_median"] = float(np.median(ratio))
        out["rough_ratio_vs_anchor_p90"] = float(np.percentile(ratio, 90))
        dev = P - A
        out["anchor_deviation_rms_decades_median"] = float(np.median(np.sqrt(np.mean(dev ** 2, axis=(1, 2)))))
        tex = departure_texture(P, A, depth, A_floor=L_floor)
        for k, v in tex.items():
            out[f"departure_{k}_ratio_median"] = float(np.median(v))
            out[f"departure_{k}_ratio_p90"] = float(np.percentile(v, 90))
    dist_meas = np.sqrt(np.mean((P - M[None]) ** 2, axis=(1, 2)))
    out["distance_to_measured_rms_decades_median"] = float(np.median(dist_meas))
    out["distance_to_measured_rms_decades_p90"] = float(np.percentile(dist_meas, 90))
    out["copy_fraction"] = float(np.mean(dist_meas < copy_tol_decades))
    med, p10, n_div = pairwise_profile_distance(P)
    out["pairwise_profile_distance_n_profiles"] = n_div
    out["pairwise_profile_distance_decades_median"] = med
    out["pairwise_profile_distance_decades_p10"] = p10
    out["ensemble_spread_decades_median"] = float(np.median(np.std(P, axis=0)))
    if nP >= 8:
        lo, hi = np.percentile(P, 2.5, axis=0), np.percentile(P, 97.5, axis=0)
        inside = (M >= lo) & (M <= hi)
        out["measured_coverage"] = float(np.mean(inside))
        out["measured_coverage_by_station"] = np.mean(inside, axis=1).tolist()
        pit = np.mean(P <= M[None], axis=0)
        out["measured_pit_mean"] = float(np.mean(pit))
    else:
        out["measured_coverage"] = float("nan")
    return out


def structure_fidelity_metrics(L_gen: np.ndarray, L_anchor: Optional[np.ndarray], L_meas: np.ndarray,
                               depth: np.ndarray, gate_cols: np.ndarray, thr: float, keep: Optional[np.ndarray] = None,
                               gate_ms: Optional[np.ndarray] = None, n_bands: int = 8,
                               chunk: int = 64) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Does the generated ensemble keep the measured profile's own narrow depth features?
    L_gen / L_anchor: (P, D, T) log10 on the measured stations, L_meas (D, T), gate_cols = grid index of each measured
    gate, thr = the measured local-structure p95 (decades) of the training term, keep = known stations (CV).
    * per station: median departure of the generated profiles from the measured profile, of the anchors from the
      measured profile and of the generated profiles from their anchors (the network's own departure);
    * measured narrow features: cells (station, gate) with |LS_measured| >= thr, LS = departure from the neighbour
      line (local_structure_residual).  Three retention estimates per cell: median_P LS(generated) / LS(measured),
      the same for the anchors, and the network's own effect 1 + median_P LS(generated - anchor) / LS(measured),
      which is free of the anchors' random anomalies.  The strongest feature (largest |LS_measured|) is reported at
      every gate; every local maximum of max-over-gates |LS_measured| >= 2 thr is listed with its retention over its
      significant gates;
    * structure retention: 1 + median of the per-band projection coefficient of LS(generated - anchor) on LS(anchor)
      over whole profiles (structure_projection: the quantity the training term drives to 0).
    Returns (metrics, per-station table)."""
    G = np.asarray(L_gen, dtype=np.float64)
    Mx = np.asarray(L_meas, dtype=np.float64)
    nP, D, T = G.shape
    d = np.asarray(depth, dtype=np.float64)
    gi = np.asarray(gate_cols, dtype=np.int64)
    nG = int(gi.size)
    cols = [0, (nG - 1) // 2, nG - 1]
    known = np.ones(D, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    A = None if L_anchor is None else np.asarray(L_anchor, dtype=np.float64)
    out: Dict[str, Any] = {"thr_decades": float(thr), "n_profiles": int(nP), "report_gate_index": cols}
    if gate_ms is not None:
        out["report_gate_ms"] = [float(gate_ms[c]) for c in cols]
    tab: Dict[str, Any] = {"depth_m": d, "known": known.astype(int)}
    Gg = G[:, :, gi]
    dev_gm = np.median(Gg - Mx[None][:, :, gi], axis=0)                         # (D, nG)
    tab["med_dev_gen_meas_mean"] = dev_gm.mean(axis=1)
    if A is not None:
        Ag = A[:, :, gi]
        dev_am = np.median(Ag - Mx[None][:, :, gi], axis=0)
        dev_ga = np.median(Gg - Ag, axis=0)
        tab["med_dev_anchor_meas_mean"] = dev_am.mean(axis=1)
        tab["med_dev_gen_anchor_mean"] = dev_ga.mean(axis=1)
    for nm, c in zip(("first", "mid", "last"), cols):
        tab[f"med_dev_gen_meas_{nm}"] = dev_gm[:, c]
        if A is not None:
            tab[f"med_dev_anchor_meas_{nm}"] = dev_am[:, c]
            tab[f"med_dev_gen_anchor_{nm}"] = dev_ga[:, c]
    if D < 3:
        return out, pd.DataFrame(tab)

    def _ls(X: np.ndarray) -> np.ndarray:            # (P, D, nG) -> (D-2, P, nG)
        return local_structure_residual(X.transpose(1, 0, 2).reshape(D, -1), d).reshape(D - 2, X.shape[0], nG)

    ls_m = local_structure_residual(Mx[:, gi], d)                                  # (D-2, nG)
    ls_g = _ls(Gg)
    med_g = np.median(ls_g, axis=1)
    med_a = med_d = None
    if A is not None:
        ls_a = _ls(Ag)
        med_a = np.median(ls_a, axis=1)
        med_d = np.median(ls_g - ls_a, axis=1)
    ok_st = known[:-2] & known[1:-1] & known[2:]
    sig = (np.abs(ls_m) >= float(thr)) & ok_st[:, None]
    den = np.where(sig, ls_m, np.nan)
    r_gen = med_g / den
    r_anc = med_a / den if med_a is not None else None
    r_dep = 1.0 + med_d / den if med_d is not None else None

    def _pad(X: np.ndarray) -> np.ndarray:           # interior (D-2) -> all stations (NaN at the ends)
        return np.concatenate([np.full((1,) + X.shape[1:], np.nan), X, np.full((1,) + X.shape[1:], np.nan)])

    for nm, c in zip(("first", "mid", "last"), cols):
        tab[f"ls_meas_{nm}"] = _pad(ls_m[:, c])
        tab[f"ls_gen_median_{nm}"] = _pad(med_g[:, c])
        tab[f"retention_gen_{nm}"] = _pad(r_gen[:, c])
        if A is not None:
            tab[f"ls_anchor_median_{nm}"] = _pad(med_a[:, c])
            tab[f"retention_anchor_{nm}"] = _pad(r_anc[:, c])
            tab[f"retention_departure_{nm}"] = _pad(r_dep[:, c])
    w = ls_m * ls_m * sig
    if float(w.sum()) > 0:
        pooled = {"generated_median": float((med_g * ls_m * sig).sum() / w.sum())}
        if A is not None:
            pooled["anchor_median"] = float((med_a * ls_m * sig).sum() / w.sum())
            pooled["departure"] = float(1.0 + (med_d * ls_m * sig).sum() / w.sum())
        out["pooled_measured_structure_retention"] = pooled
        out["n_significant_cells"] = int(sig.sum())
    peak = np.where(ok_st, np.abs(ls_m).max(axis=1), 0.0)

    def _feature(j: int) -> Dict[str, Any]:
        s = sig[j]
        f: Dict[str, Any] = {"depth_m": float(d[j + 1]), "station_index": int(j + 1), "peak_abs_ls_decades": float(peak[j]),
                             "n_significant_gates": int(s.sum())}
        for nm, r in (("generated", r_gen), ("anchor", r_anc), ("departure", r_dep)):
            if r is None or not s.any():
                continue
            f[f"retention_{nm}_first_mid_last"] = [float(r[j, c]) for c in cols]
            v_ = np.where(s, r[j], np.nan)
            if np.isfinite(v_).any():
                f[f"retention_{nm}_min"] = float(np.nanmin(v_))
                f[f"retention_{nm}_median"] = float(np.nanmedian(v_))
                f[f"retention_{nm}_min_gate"] = int(np.nanargmin(v_))
            else:       # no finite retention at the feature's gates (non-finite profiles): NaN, not an exception
                f[f"retention_{nm}_min"] = f[f"retention_{nm}_median"] = float("nan")
                f[f"retention_{nm}_min_gate"] = -1
        return f

    if peak.max() > 0:
        j0 = int(np.argmax(peak))
        top = _feature(j0)
        if gate_ms is not None:
            top["gate_ms"] = [float(v) for v in gate_ms]
        top["measured_ls_by_gate"] = [float(v) for v in ls_m[j0]]
        top["generated_median_ls_by_gate"] = [float(v) for v in med_g[j0]]
        for nm, r in (("generated", r_gen), ("anchor", r_anc), ("departure", r_dep)):
            if r is not None:
                top[f"retention_{nm}_by_gate"] = [float(v) for v in r[j0]]
        out["strongest_feature"] = top
        lm = [j for j in range(D - 2) if peak[j] >= 2.0 * float(thr)
              and (j == 0 or peak[j] >= peak[j - 1]) and (j == D - 3 or peak[j] >= peak[j + 1])]
        out["features"] = [_feature(j) for j in lm]
    if A is not None and float(thr) > 0:
        bs, os_ = [], []
        dep_t = torch.as_tensor(d, dtype=torch.float32)
        for s0 in range(0, nP, chunk):
            g_ = torch.as_tensor(G[s0:s0 + chunk] - A[s0:s0 + chunk], dtype=torch.float32)
            a_ = torch.as_tensor(A[s0:s0 + chunk], dtype=torch.float32)
            dd = dep_t.unsqueeze(0).expand(g_.shape[0], -1)
            b_, o_ = structure_projection(local_structure_torch(g_, dd), local_structure_torch(a_, dd), float(thr), n_bands)
            bs.append(b_.numpy())
            os_.append(o_.numpy())
        beta, okb = np.concatenate(bs), np.concatenate(os_)
        if okb.any():
            out["struct_retention"] = float(1.0 + np.median(beta[okb]))
            out["struct_retention_p10_p90"] = [float(1.0 + np.percentile(beta[okb], 10)),
                                               float(1.0 + np.percentile(beta[okb], 90))]
            out["struct_retention_by_band"] = [float(1.0 + np.median(beta[okb[:, b], b])) if okb[:, b].any() else float("nan")
                                               for b in range(beta.shape[1])]
    return out, pd.DataFrame(tab)


def qc_gate_status(cm: Dict[str, float], pm: Dict[str, Any]) -> Dict[str, Any]:
    g = QC_GATES
    checks = {
        "monotone_decay": (cm.get("monotone_fraction", 0.0) >= g["monotone_fraction_min"], cm.get("monotone_fraction")),
        "finite_positive": (cm.get("finite_positive_fraction", 0.0) >= g["finite_positive_fraction_min"],
                            cm.get("finite_positive_fraction")),
        "profile_noncrossing": (pm.get("noncrossing_fraction", 0.0) >= g["monotone_fraction_min"],
                                pm.get("noncrossing_fraction")),
        "profile_roughness_vs_anchor": (pm.get("rough_ratio_vs_anchor_median", float("inf")) <= g["rough_ratio_vs_anchor_max"],
                                        pm.get("rough_ratio_vs_anchor_median")),
        "measured_coverage": (pm.get("measured_coverage", 0.0) >= g["measured_coverage_min"], pm.get("measured_coverage")),
        "copy_fraction": (pm.get("copy_fraction", 1.0) <= g["copy_fraction_max"], pm.get("copy_fraction")),
        "descriptor_precision": (cm.get("descriptor_precision", 0.0) >= g["descriptor_precision_min"],
                                 cm.get("descriptor_precision")),
        "descriptor_recall": (cm.get("descriptor_recall", 0.0) >= g["descriptor_recall_min"], cm.get("descriptor_recall")),
        "profile_diversity": (pm.get("pairwise_profile_distance_decades_median", 0.0) >= g["profile_diversity_min_decades"],
                              pm.get("pairwise_profile_distance_decades_median")),
    }
    ret = pm.get("anomaly_retention_median", float("nan"))
    if ret is not None and math.isfinite(float(ret)):
        checks["anomaly_retention"] = (abs(1.0 - float(ret)) <= g["anomaly_retention_tol"], ret)
    cor = pm.get("anomaly_corr_median", float("nan"))
    if cor is not None and math.isfinite(float(cor)):
        checks["anomaly_shape_corr"] = (float(cor) >= g["anomaly_corr_min"], cor)
    if "accept_rate" in pm:
        checks["per_profile_acceptance"] = (float(pm["accept_rate"]) >= g["accept_rate_min"], pm["accept_rate"])
    for key, gate in (("departure_depth_texture", "departure_depth_texture_max"),
                      ("departure_time_texture", "departure_time_texture_max")):
        v = pm.get(key + "_ratio_median", float("nan"))
        if v is not None and math.isfinite(float(v)):
            checks["no_added_" + key.split("_")[1] + "_texture"] = (float(v) <= g[gate], v)
    pr = pm.get("purity_rate", None)
    if pr is not None and math.isfinite(float(pr)):
        checks["background_purity"] = (float(pr) >= g["background_purity_rate_min"], pr)
    tr_ = pm.get("trend_rate", None)
    if tr_ is not None and math.isfinite(float(tr_)):
        checks["zero_anomaly_trend"] = (float(tr_) >= g["zero_anomaly_trend_rate_min"], tr_)
    if "anomaly_count_ok" in pm:   # count 0 .. max_anomalies, every configured count present (value: largest)
        checks["anomaly_count"] = (float(pm["anomaly_count_ok"]) >= 1.0, pm.get("anomaly_count_max"))
    fo = pm.get("field_admissible_occupancy", None)
    if fo is not None and math.isfinite(float(fo)) and float(pm.get("field_admissible_gate", 0.0)) > 0:
        checks["field_admissible_occupancy"] = (float(fo) >= g["field_admissible_occupancy_min"], fo)
    failed = [k for k, (ok, _) in checks.items() if not ok]
    return {"status": "PASS" if not failed else "FAIL:" + ",".join(failed),
            "checks": {k: {"pass": bool(ok), "value": (None if v is None else float(v))} for k, (ok, v) in checks.items()},
            "thresholds": dict(g)}


def qc_fail_reasons(q: Dict[str, np.ndarray], status: np.ndarray, trainer: Any) -> Dict[str, int]:
    """For the rejected slots: how many failed each acceptance gate (a slot can fail several)."""
    g = trainer.cfg.generation
    st = np.asarray(status)
    rj = st == 2
    out: Dict[str, int] = {"rejected": int(rj.sum())}
    if not rj.any():
        return out

    def _a(k: str) -> np.ndarray:
        return np.asarray(q.get(k, np.full(st.size, np.nan)), dtype=np.float64)

    out["not_finite"] = int((rj & (_a("finite") < 0.5)).sum())
    out["roughness"] = int((rj & (_a("rough_ratio") > float(g.qc_rough_max))).sum())
    out["physics_envelope"] = int((rj & (_a("envelope_frac") < float(g.qc_envelope_min))).sum())
    lo_r, hi_r = (float(v) for v in str(g.qc_retention_range).split(","))
    lo_k, hi_k = (float(v) for v in str(g.qc_contrast_range).split(","))
    ret, con, cor = _a("anomaly_retention"), _a("anomaly_contrast"), _a("anomaly_corr")
    an_bad = np.isfinite(ret) & ((ret < lo_r) | (ret > hi_r) | (con < lo_k) | (con > hi_k) | (cor < float(g.qc_anomaly_corr_min)))
    out["anomaly_retention_shape"] = int((rj & an_bad).sum())
    if "purity_ok" in q:
        out["background_purity"] = int((rj & (_a("purity_ok") < 0.5)).sum())
    if "warp_ok" in q:      # the warp's own anomaly-scale content beyond the strain bound
        out["warp_content"] = int((rj & (_a("warp_ok") < 0.5)).sum())
    if "trend_ok" in q:
        out["zero_anomaly_trend"] = int((rj & (_a("trend_ok") < 0.5)).sum())
    if "field_admissible_frac" in q and getattr(trainer, "fam_on", False):
        fa = _a("field_admissible_frac")
        out["field_admissible"] = int((rj & np.isfinite(fa) & (fa < float(trainer.cfg.profile.fam_min_frac))).sum())
    return out


def acceptance_summary(qc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Medians / rates of the per-profile acceptance metrics (trainer.profile_acceptance)."""
    out: Dict[str, float] = {}
    if not qc or "pass" not in qc:
        return out
    out["accept_rate"] = float(np.mean(qc["pass"]))
    for k in ("rough_ratio", "envelope_frac", "anomaly_retention", "anomaly_contrast", "anomaly_corr",
              "dist_to_measured_decades", "purity_max_decades"):
        if k == "purity_max_decades" and k not in qc:
            continue
        v = np.asarray(qc.get(k, []), dtype=np.float64)
        v = v[np.isfinite(v)]
        out[f"{k}_median"] = float(np.median(v)) if v.size else float("nan")
        if k in ("anomaly_retention", "anomaly_contrast", "anomaly_corr", "dist_to_measured_decades",
                 "purity_max_decades") and v.size:
            out[f"{k}_p10"] = float(np.percentile(v, 10))
            out[f"{k}_p90"] = float(np.percentile(v, 90))
        if k == "purity_max_decades" and v.size:
            out["purity_max_decades_max"] = float(v.max())
    if "purity_ok" in qc:   # background purity of the profiles (background_mode pure)
        out["purity_rate"] = float(np.mean(np.asarray(qc["purity_ok"], dtype=np.float64)))
    if "warp_ok" in qc:
        out["warp_ok_rate"] = float(np.mean(np.asarray(qc["warp_ok"], dtype=np.float64)))
        for k in ("warp_content_ratio", "warp_content_decades"):
            v = np.asarray(qc.get(k, []), dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                out[f"{k}_median"], out[f"{k}_p90"] = float(np.median(v)), float(np.percentile(v, 90))
                out[f"{k}_max"] = float(v.max())
    if "trend_ok" in qc:    # "0 anomalies" profiles = the overall decay trend (background_trend monotone)
        v = np.asarray(qc["trend_ok"], dtype=np.float64)
        s = np.asarray(qc.get("trend_step_decades", []), dtype=np.float64)
        if np.isfinite(v).any():
            out["trend_rate"] = float(np.nanmean(v))
            out["trend_profiles"] = int(np.isfinite(v).sum())
            out["trend_step_decades_max"] = float(np.nanmax(s)) if np.isfinite(s).any() else float("nan")
    if "field_admissible_frac" in qc:   # FAM: per profile over its background stations; pooled = mean of profiles
        v = np.asarray(qc["field_admissible_frac"], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            out["field_admissible_occupancy"] = float(v.mean())
            out["field_admissible_frac_median"] = float(np.median(v))
            out["field_admissible_frac_p10"] = float(np.percentile(v, 10))
    if "cmg_gate_mean" in qc:          # mean CMG contraction strength per profile
        v = np.asarray(qc["cmg_gate_mean"], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            out["cmg_gate_median"], out["cmg_gate_p90"] = float(np.median(v)), float(np.percentile(v, 90))
    if "tries" in qc:
        out["mean_tries"] = float(np.mean(qc["tries"]))
    return out


def composite_quality_score(cm: Dict[str, float], pm: Dict[str, Any]) -> float:
    """Lower is better; with the anchor structure kept, that picked the smoothest epochs, late in Stage-2 where the
    anchor terms are weakest."""
    s = 0.0
    s += 10.0 * max(0.0, 1.0 - float(cm.get("monotone_fraction", 0.0)))
    s += max(0.0, math.log(max(float(pm.get("rough_ratio_vs_anchor_median", 1.0)), 1e-6) / (1.0 + ROUGH_SELECT_TOL)))
    s += max(0.0, 1.0 - float(pm.get("measured_coverage", 0.0) if math.isfinite(pm.get("measured_coverage", 0.0)) else 0.0))
    s += max(0.0, 1.0 - float(cm.get("descriptor_recall", 0.0)))
    s += max(0.0, 1.0 - float(cm.get("descriptor_precision", 0.0)))
    s += float(pm.get("copy_fraction", 0.0))
    div = float(pm.get("pairwise_profile_distance_decades_median", 0.0))
    tgt = QC_GATES["profile_diversity_min_decades"]
    s += max(0.0, (tgt - div) / tgt)
    div_a = float(pm.get("anchor_pairwise_distance_decades_median", float("nan")))
    if math.isfinite(div_a) and div_a > 0 and div > 0:
        s += math.log(div_a / div)            # < 0 when the network adds variety to its anchors
    # anomalies must survive generation: retention slope ~ 1 and shape correlation high
    ret = float(pm.get("anomaly_retention_median", float("nan")))
    if math.isfinite(ret):
        s += max(0.0, abs(1.0 - ret) - QC_GATES["anomaly_retention_tol"])
    cor = float(pm.get("anomaly_corr_median", float("nan")))
    if math.isfinite(cor):
        s += -math.log(max(cor, 1e-6))
    # the anchor's local depth structure must survive as well (retention = 1 + median band projection of the departure,
    # structure_fidelity_metrics): the roughness term above alone rewards checkpoints that smooth it away
    sr = float(pm.get("struct_retention", float("nan")))
    if math.isfinite(sr):
        s += max(0.0, abs(1.0 - sr) - STRUCTURE_RETENTION_TOL)
    # a checkpoint whose departure adds depth texture beyond the QC limit is not a signal generator
    tx = float(pm.get("departure_depth_texture_ratio_median", float("nan")))
    if math.isfinite(tx):
        s += max(0.0, tx - QC_GATES["departure_depth_texture_max"])
    # a checkpoint whose profiles carry unlabelled anomaly-scale features (background_mode pure) is not selected:
    # weighted like the monotone decays (10 x the share of failing profiles)
    pr = float(pm.get("purity_rate", float("nan")))
    if math.isfinite(pr):
        s += 10.0 * max(0.0, 1.0 - pr)
    # likewise "0 anomalies" profiles that leave the overall decay trend along the profile (background_trend)
    tr_ = float(pm.get("trend_rate", float("nan")))
    if math.isfinite(tr_):
        s += 10.0 * max(0.0, 1.0 - tr_)
    return float(s)
