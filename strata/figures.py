"""Figures and printed reports of a run: profiles, anomalies, structure, background purity and extensions."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import ensure_dir
from .data.contract import MeasuredData, SimPrior, local_structure_residual
from .data.time_grid import depth_interp_profile
from .manifold.joint import SEPARATION_CORE_SIGMAS, highpass_profile_np, local_poly_smoother_np
from .physics import PhysicsDescriptor
from .quality import PURITY_ANCHOR_SHARE


def _axis_names(meas: "MeasuredData") -> Dict[str, str]:
    """Profile-axis label / name of the survey for figures: depth (m) along the hole, distance (m) along a line."""
    lab = meas.survey.axis_label if getattr(meas, "survey", None) is not None else "depth (m)"
    name = lab.split(" ")[0]
    key = {"depth": "depth_m", "distance": "distance_m"}.get(name, "station")      # column / key of the station axis
    return {"label": lab, "name": name, "key": key}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _gate_subset(T: int, gate_index: np.ndarray, n: int = 10) -> np.ndarray:
    gi = gate_index[gate_index >= 0] if np.any(gate_index >= 0) else np.arange(T)
    return gi[np.unique(np.linspace(0, gi.size - 1, min(n, gi.size)).round().astype(int))]


def save_figures(L_gen: np.ndarray, anchor_L: Optional[np.ndarray], meas: MeasuredData, L_union: np.ndarray,
                 phys: PhysicsDescriptor, history: List[Dict[str, Any]], plots_dir: str, dpi: int = 200,
                 sim: Optional[SimPrior] = None, title_tag: str = "") -> Dict[str, str]:
    _AX = _axis_names(meas)
    plt = _plt()
    ensure_dir(plots_dir)
    out: Dict[str, str] = {}
    P, D, T = L_gen.shape
    depth, grid = meas.depth, meas.grid_ms
    gsel = _gate_subset(T, meas.gate_index, 10)
    cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng(0)
    show = rng.choice(P, size=min(5, P), replace=False)
    # 1. profile families (V vs depth, one line per gate) — measured + generated examples
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    panels = [("measured", meas.L)] + [(f"generated #{i}", L_gen[i]) for i in show]
    for ax, (name, Lp) in zip(axs.ravel(), panels):
        for j, g in enumerate(gsel):
            ax.plot(depth, 10 ** Lp[:, g], color=cmap(j / max(len(gsel) - 1, 1)), lw=1.1)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel(_AX["label"])
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("|dB/dt| (measured units)")
    axs[1, 0].set_ylabel("|dB/dt| (measured units)")
    fig.suptitle(f"Profile families (one line per gate, {grid[gsel[0]]:.3g}-{grid[gsel[-1]]:.3g} ms) {title_tag}")
    fig.tight_layout()
    p = os.path.join(plots_dir, "profile_families.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["profile_families"] = p
    # 2. ensemble band per gate vs depth
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, g in zip(axs.ravel(), _gate_subset(T, meas.gate_index, 6)):
        q = np.percentile(L_gen[:, :, g], [2.5, 25, 50, 75, 97.5], axis=0)
        ax.fill_between(depth, 10 ** q[0], 10 ** q[4], color="tab:blue", alpha=0.18, label="generated 2.5-97.5%")
        ax.fill_between(depth, 10 ** q[1], 10 ** q[3], color="tab:blue", alpha=0.35, label="generated 25-75%")
        ax.plot(depth, 10 ** q[2], color="tab:blue", lw=1.2, label="generated median")
        ax.plot(depth, 10 ** meas.L[:, g], color="k", lw=1.5, label="measured")
        ax.set_yscale("log")
        ax.set_title(f"gate t = {grid[g]:.4g} ms")
        ax.set_xlabel(_AX["label"])
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Generated ensemble vs measured along the profile ({_AX['name']}, {P} profiles) {title_tag}")
    fig.tight_layout()
    p = os.path.join(plots_dir, "ensemble_band_vs_depth.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["ensemble_band"] = p
    # 3. decay curves at four depth positions (quantiles of the measured span)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)
    for ax, q in zip(axs, (0.05, 0.35, 0.65, 0.95)):
        i = int(round(q * (D - 1)))
        for k in rng.choice(P, size=min(25, P), replace=False):
            ax.plot(grid, 10 ** L_gen[k, i], color="tab:orange", lw=0.6, alpha=0.6)
        ktm = int(np.sum(meas.known_time_mask()))
        ax.plot(grid[:ktm], 10 ** meas.L[i, :ktm], color="k", lw=1.6, label="measured")
        if ktm < grid.size:
            ax.plot(grid[ktm - 1:], 10 ** meas.L[i, ktm - 1:], color="k", lw=1.0, ls="--", label="extension rule")
        ax.plot(meas.pm.time_ms, meas.pm.values[i], "k.", ms=4, label="measured gates")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{_AX['name']} {depth[i]:.1f} m")
        ax.set_xlabel("t (ms)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("|dB/dt|")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Decay curves: measured (black) and 25 generated profiles (orange) {title_tag}")
    fig.tight_layout()
    p = os.path.join(plots_dir, "decay_curves.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["decay_curves"] = p
    # 4. depth roughness by gate band + diversity histograms
    bands = np.array_split(np.arange(T), 8)

    def _rms(X):
        r = local_structure_residual(X, depth)
        return np.array([np.sqrt(np.mean(r[:, b] ** 2)) for b in bands])

    rm = _rms(meas.L)
    rg = np.stack([_rms(L_gen[i]) for i in range(min(P, 128))])
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    xb = np.arange(len(bands))
    axs[0].bar(xb - 0.2, rm, width=0.4, label="measured", color="k")
    axs[0].bar(xb + 0.2, np.median(rg, 0), width=0.4, label="generated (median)", color="tab:blue")
    if anchor_L is not None:
        ra = np.stack([_rms(anchor_L[i]) for i in range(min(P, 128))])
        axs[0].plot(xb, np.median(ra, 0), "o-", color="tab:green", label="anchors (median)")
    axs[0].set_xticks(xb)
    axs[0].set_xticklabels([f"{grid[b[0]]:.2g}-{grid[b[-1]]:.2g}" for b in bands], rotation=45, fontsize=7)
    axs[0].set_ylabel("RMS local depth structure (decades)")
    axs[0].set_title("Depth roughness by gate band")
    axs[0].legend(fontsize=8)
    Q = L_gen[rng.choice(P, size=min(P, 96), replace=False)].reshape(min(P, 96), -1)
    dq = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).mean(-1))
    iu = np.triu_indices(Q.shape[0], 1)
    axs[1].hist(dq[iu], bins=40, color="tab:blue", alpha=0.8)
    axs[1].set_xlabel("RMS distance between generated profiles (decades)")
    axs[1].set_title("Inter-profile diversity")
    dmeas = np.sqrt(((L_gen - meas.L[None]) ** 2).mean(axis=(1, 2)))
    axs[2].hist(dmeas, bins=40, color="tab:purple", alpha=0.8)
    axs[2].axvline(0.01, color="r", ls="--", label="copy threshold 0.01")
    axs[2].set_xlabel("RMS distance to the measured profile (decades)")
    axs[2].set_title("Distance to measured (anti-copy)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(plots_dir, "roughness_and_diversity.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["roughness_and_diversity"] = p
    # 5. physics descriptor PCA (measured / union reference / generated)
    try:
        # the columns the descriptor was built on (the trainer's gate_cols: measured gates, else the measured window)
        kn = meas.known_time_mask()
        gc = (meas.gate_index[kn[meas.gate_index]] if bool(np.all(meas.gate_index >= 0)) else np.nonzero(kn)[0])
        fm, fu = phys.numpy(meas.L[:, gc]), phys.numpy(L_union[:, gc])
        Gs = L_gen.reshape(-1, T)[:, gc]
        fg = phys.numpy(Gs[rng.choice(Gs.shape[0], size=min(Gs.shape[0], 4000), replace=False)])
        mu, sd = fu.mean(0), fu.std(0) + 1e-9
        U, S, Vt = np.linalg.svd((fu - mu) / sd, full_matrices=False)
        pr = lambda f: ((f - mu) / sd) @ Vt[:2].T
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        a = pr(fg)
        ax.scatter(a[:, 0], a[:, 1], s=3, alpha=0.25, color="tab:orange", label="generated")
        b = pr(fu)
        ax.scatter(b[:, 0], b[:, 1], s=6, alpha=0.5, color="tab:green", label="union reference (measured + sim-structured)")
        c = pr(fm)
        ax.scatter(c[:, 0], c[:, 1], s=18, color="k", label="measured stations")
        ax.set_xlabel("PC1 (physics descriptor)")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.set_title("TEM physics descriptor space")
        fig.tight_layout()
        p = os.path.join(plots_dir, "descriptor_pca.png")
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        out["descriptor_pca"] = p
    except Exception as e:
        print(f"[PLOT] descriptor PCA skipped: {e}")
    # 6. anchored time grid
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(grid, np.zeros_like(grid), "|", ms=10, color="tab:blue", label=f"generation grid (T={T})")
    ax.plot(meas.pm.time_ms, np.ones_like(meas.pm.time_ms), "|", ms=14, color="k", label=f"measured gates ({meas.pm.time_ms.size})")
    if sim is not None and sim.available:
        ax.plot(grid, -1 + 0 * grid, "|", ms=6, color="0.7")
        ax.plot(grid[sim.coverage > 0], -np.ones(int((sim.coverage > 0).sum())), "|", ms=10, color="tab:green",
                label="sim native support on the grid")
    ax.set_xscale("log")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["sim support", "grid", "measured"])
    ax.set_xlabel("t (ms)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Measured-anchored time grid (measured gates embedded exactly)")
    fig.tight_layout()
    p = os.path.join(plots_dir, "time_grid.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["time_grid"] = p
    # 7. training curves
    if history:
        try:
            df = pd.DataFrame(history)
            keys = [k for k in ("recon", "sup", "total", "wave", "rough", "trend", "ms", "anom", "struct", "purity", "adv",
                                "prof_adv", "D", "val_score", "val_rough_ratio_vs_anchor", "val_measured_coverage",
                                "val_diversity_decades", "val_anomaly_retention", "val_anomaly_corr",
                                "val_struct_retention", "val_feature_retention_min", "val_accept_rate",
                                "val_purity_rate") if k in df.columns]
            n = len(keys)
            if n:
                cols = 4
                rows = int(math.ceil(n / cols))
                fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows))
                axs = np.atleast_1d(axs).ravel()
                # one colour per stage in every panel (the per-axes colour cycle mislabelled the stages)
                stage_colors = {"stage0": "tab:blue", "stage0B": "tab:purple", "stage1A": "tab:gray",
                                "stage1B": "tab:orange", "stage2": "tab:green"}
                handles = {}
                for ax, k in zip(axs, keys):
                    for stg, sub in df.groupby("stage", sort=False):
                        s = sub[["global_epoch", k]].dropna()
                        if len(s):
                            ln, = ax.plot(s["global_epoch"], s[k], ".-", lw=1, label=stg,
                                          color=stage_colors.get(str(stg), "black"))
                            handles.setdefault(str(stg), ln)
                    ax.set_title(k)
                    ax.grid(alpha=0.3)
                for ax in axs[n:]:
                    ax.axis("off")
                fig.legend(list(handles.values()), list(handles.keys()), loc="upper right", fontsize=8,
                           ncol=len(handles))
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                p = os.path.join(plots_dir, "training_curves.png")
                fig.savefig(p, dpi=dpi)
                plt.close(fig)
                out["training_curves"] = p
        except Exception as e:
            print(f"[PLOT] training curves skipped: {e}")
    return out


def save_anomaly_figures(L_dense: np.ndarray, depth_dense: np.ndarray, meas: MeasuredData, index: pd.DataFrame,
                         plots_dir: str, dpi: int = 200) -> Dict[str, str]:
    """L_dense: (n, Dd, T) log10 of the first generated profiles (dense depth).  Figures:
    anomaly_gallery.png        profiles with 0 anomalies / the measured anomalies only / + 1 / + 2 or more synthetic ones;
                               every measured gate (colour = time, a multi-channel profile plot), each synthetic
                               anomaly shaded over its footprint with its peak and visible time window
    anomaly_gallery_deviation.png  the same profiles as log10(generated / background) along depth, every gate
    anomaly_kinds.png          examples of each kind of synthetic anomaly and of late-time-only ones: departure from
                               the background per gate around the anomaly
    anomaly_time_windows.png   peak time vs tau, tau vs distance r, visible window vs peak, onset vs r (all profiles)
    anomaly_statistics.png     count / signed peak / centre depth / width / tau / retention distributions
    anomaly_decay_examples.png decay curves at anomaly centres vs the background curve at the same depth"""
    _AX = _axis_names(meas)
    _pc = "position_m" if any(str(c).endswith("_position_m") for c in index.columns) else "depth_m"
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    ensure_dir(plots_dir)
    out: Dict[str, str] = {}
    n = L_dense.shape[0]
    if n == 0 or "n_anomalies" not in index.columns:
        return out
    T = L_dense.shape[2]
    grid = meas.grid_ms
    kt = int(np.sum(meas.known_time_mask()))
    gi = np.asarray(meas.gate_index)
    # every measured gate of the measured window (at most 48 lines; a dense grid file: an even subset)
    gall = gi[(gi >= 0) & (gi < kt)] if np.any(gi >= 0) else np.arange(kt)
    if gall.size > 48:
        gall = gall[np.unique(np.linspace(0, gall.size - 1, 48).round().astype(int))]
    tcm = LinearSegmentedColormap.from_list("gates", plt.get_cmap("turbo")(np.linspace(0.04, 0.96, 256)))
    tnorm = LogNorm(vmin=float(grid[gall[0]]), vmax=float(grid[gall[-1]]))
    cmap = plt.get_cmap("viridis")
    idx = index.iloc[:n]
    # background_mode pure: the reference line is the measured profile's background, which the profiles are anchored on
    # (the measured profile's own anomalies are in none of them: background_decomposition.png)
    pure = meas.background is not None
    Lref = meas.background if pure else meas.L
    ref_name = "measured background" if pure else "measured"
    Lm_d = depth_interp_profile(meas.depth, Lref, np.asarray(depth_dense))
    Kc = idx["n_anomalies"].to_numpy()
    Kmax = int(sum(1 for c in idx.columns if re.fullmatch(r"anomaly\d+_peak_decades", str(c))))
    Mmax = int(sum(1 for c in idx.columns if re.fullmatch(r"measured\d+_scale", str(c))))
    pol = ((meas.background_info or {}).get("measured_anomaly_policy") or {}) if pure else {}
    nm_ = idx["n_measured_anomalies"].to_numpy() if "n_measured_anomalies" in idx.columns else None
    if nm_ is not None and pol.get("mode") == "count":
        # one row per anomaly count 0 .. max (the count includes the measured anomalies); per row up to three profiles
        # of different composition (the most measured anomalies first, then fewer), so the measured set alone, mixed and
        # synthetic-only profiles and their positions are seen side by side
        def _pick(c: int) -> List[int]:
            ids = [int(i) for i in np.nonzero(Kc == c)[0]]
            if c == 0:
                return ids[:3]
            got: List[int] = []
            for want in sorted({int(nm_[i]) for i in ids}, reverse=True):
                got += [i for i in ids if int(nm_[i]) == want][:1]
            got += [i for i in ids if i not in got]
            return got[:3]

        groups = [(f"{c} anomal{'y' if c == 1 else 'ies'}" + ("\n(pure background)" if c == 0 else ""), _pick(c))
                  for c in range(int(Kc.max()) + 1 if Kc.size else 1)]
    elif nm_ is not None:
        # rows: pure background / the measured anomalies only / + 1 synthetic / + 2 or more synthetic
        ns_ = idx["n_synthetic_anomalies"].to_numpy()
        groups = [("0 anomalies\n(pure background)", Kc == 0), ("measured\nanomalies only", (nm_ > 0) & (ns_ == 0)),
                  ("measured\n+ 1 synthetic", (nm_ > 0) & (ns_ == 1)), ("measured\n+ 2+ synthetic", (nm_ > 0) & (ns_ >= 2))]
    else:
        groups = [(f"{kk}{'+' if kk == 3 else ''} anomalies", sel) for kk, sel in
                  ((0, Kc == 0), (1, Kc == 1), (2, Kc == 2), (3, Kc >= 3))]
    _span_pad = float(np.median(np.diff(meas.depth))) if meas.depth.size > 1 else 0.0

    def _syn(i: int, k: int) -> Optional[Tuple[float, float, float, float, float, str]]:
        """Synthetic anomaly k of profile i: (centre, footprint half width, signed peak, visible from / to (ms), kind)."""
        row = idx.iloc[i]
        z = row.get(f"anomaly{k + 1}_{_pc}", np.nan)
        if not np.isfinite(z):
            return None
        kind = str(row.get(f"anomaly{k + 1}_kind", "") or row.get(f"anomaly{k + 1}_type", ""))
        half = (2.0 if "beside" in kind else SEPARATION_CORE_SIGMAS if "measured-like" in kind else 1.0) * float(
            row.get(f"anomaly{k + 1}_sigma0_m", 0.0))
        return (float(z), half, float(row.get(f"anomaly{k + 1}_peak_decades", np.nan)),
                float(row.get(f"anomaly{k + 1}_visible_from_ms", np.nan)),
                float(row.get(f"anomaly{k + 1}_visible_to_ms", np.nan)), kind)

    span_x = float(np.ptp(depth_dense)) if np.size(depth_dense) > 1 else 1.0

    def _mark(ax, i: int) -> None:
        syn = sorted([(k, s_) for k, s_ in ((k, _syn(i, k)) for k in range(Kmax)) if s_ is not None], key=lambda e: e[1][0])
        z_prev, level = -np.inf, 0
        for k, (z, half, a, vf, vt, knd) in syn:
            colr = "tab:red" if a > 0 else "tab:blue"
            ax.axvspan(z - half, z + half, color=colr, alpha=0.10, lw=0)
            ax.axvline(z, color=colr, ls=":", lw=1.0)
            # labels of neighbouring anomalies step through three levels: above the panel, inside its top, lower inside
            # (up to five anomalies per profile)
            level = (level + 1) % 3 if z - z_prev < 0.12 * span_x else 0
            z_prev = z
            inside = level > 0
            mt = re.search(r"measured (\d+)", knd)
            tag = (f" M{mt.group(1)} shape" if mt else " beside" if "beside" in knd else " zone" if "zone" in knd
                   else " cond." if "hole through" in knd else "")
            ax.text(z, (1.0, 0.985, 0.86)[level], f"S{k + 1} {a:+.2f}{tag}" + (f"\n{vf:.2g}-{vt:.2g} ms" if np.isfinite(vf) else ""),
                    color=colr, fontsize=7, transform=ax.get_xaxis_transform(), ha="center",
                    va="top" if inside else "bottom",
                    bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=0.5) if inside else None)
        for j in range(Mmax):
            a0, a1 = idx.iloc[i].get(f"measured{j + 1}_from_m", np.nan), idx.iloc[i].get(f"measured{j + 1}_to_m", np.nan)
            if np.isfinite(a0):
                sc_ = idx.iloc[i].get(f"measured{j + 1}_scale", np.nan)
                ax.axvspan(a0 - 0.5 * _span_pad, a1 + 0.5 * _span_pad, color="tab:orange", alpha=0.12, lw=0)
                ax.text(0.5 * (a0 + a1), 0.0, f"M{j + 1} x{sc_:.2f}", color="tab:orange", fontsize=7,
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom")

    def _time_colorbar(fig) -> None:
        cax = fig.add_axes([0.945, 0.3, 0.011, 0.4])
        sm = plt.cm.ScalarMappable(norm=tnorm, cmap=tcm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="time (ms)")

    for fname, dev_ in (("anomaly_gallery.png", False), ("anomaly_gallery_deviation.png", True)):
        fig, axs = plt.subplots(len(groups), 3, figsize=(17, 4.3 * len(groups)), sharex=True, squeeze=False)
        for r, (kk, sel) in enumerate(groups):
            ids = sel if isinstance(sel, list) else np.nonzero(sel)[0][:3]
            for c in range(3):
                ax = axs[r, c]
                if c >= len(ids):
                    ax.axis("off")
                    continue
                i = int(ids[c])
                for g in gall:
                    ax.plot(depth_dense, (L_dense[i, :, g] - Lm_d[:, g]) if dev_ else 10 ** L_dense[i, :, g],
                            color=tcm(tnorm(float(grid[g]))), lw=0.8)
                if dev_:
                    ax.axhline(0.0, color="k", lw=0.7)
                else:
                    ax.set_yscale("log")
                _mark(ax, i)
                ax.grid(alpha=0.3)
                comp = ""
                if nm_ is not None and Kc[i] > 0:
                    nmi, nsi = int(nm_[i]), int(Kc[i]) - int(nm_[i])
                    comp = " (" + " + ".join(p_ for p_ in ((f"{nmi} measured" if nmi else ""),
                                                           (f"{nsi} synthetic" if nsi else "")) if p_) + ")"
                ax.set_title(f"sample_id {int(idx.iloc[i]['sample_id'])}: {int(Kc[i])} anomal{'y' if Kc[i] == 1 else 'ies'}"
                             + comp, fontsize=10, pad=24)
                if c == 0:
                    ax.set_ylabel(f"{kk}\n" + (f"log10(gen / {'background' if pure else 'measured'})" if dev_ else "|dB/dt|"))
                if r == len(groups) - 1:
                    ax.set_xlabel(_AX["label"])
        if dev_:
            fig.suptitle(f"Same profiles as departures from the {ref_name} (decades), every measured gate (colour = time): "
                         "background modulation, the network's regional departure and the anomalies\nshaded = synthetic "
                         "anomaly (label = peak, visible time window), orange = measured anomaly (label = amplitude factor)")
        else:
            fig.suptitle(f"Generated profiles, every measured gate (colour = time); shaded = synthetic anomaly footprint, "
                         f"label = peak (decades) and the time window in which it is visible\norange band = measured "
                         f"anomaly (measured shape), label = amplitude factor")
        fig.tight_layout(rect=(0, 0, 0.93, 0.97))
        _time_colorbar(fig)
        p = os.path.join(plots_dir, fname)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        out[fname[:-4]] = p
    # examples of each kind of synthetic anomaly and of late-time-only anomalies (concealed conductors)
    if "anomaly1_kind" in idx.columns:
        rows_k = [("conductor, hole through", lambda s_: s_[5] == "conductor, hole through"),
                  ("conductor beside the hole", lambda s_: s_[5] == "conductor beside the hole"),
                  ("near-hole zone", lambda s_: s_[5] == "near-hole zone"),
                  ("late-time only\n(visible from >= 1 ms)", lambda s_: np.isfinite(s_[3]) and s_[3] >= 1.0)]
        kinds_all = {str(v_) for k_ in range(Kmax) for v_ in idx[f"anomaly{k_ + 1}_kind"].to_numpy()}
        if any(v_.startswith("measured-like") for v_ in kinds_all):
            # measured anomaly shapes at other depths first; zones only when the run has them
            rows_k = ([("measured-like\n(a measured shape\nat another depth)",
                        lambda s_: str(s_[5]).startswith("measured-like"))]
                      + [r_ for r_ in rows_k if r_[0] != "near-hole zone" or "near-hole zone" in kinds_all])

        def _isolated(i: int, k: int) -> bool:
            z, half = _syn(i, k)[:2]
            for k2 in range(Kmax):
                s2 = _syn(i, k2)
                if k2 != k and s2 is not None and abs(s2[0] - z) < 2.0 * (half + s2[1]):
                    return False
            for j in range(Mmax):
                a0 = idx.iloc[i].get(f"measured{j + 1}_from_m", np.nan)
                a1 = idx.iloc[i].get(f"measured{j + 1}_to_m", np.nan)
                if np.isfinite(a0) and z + 2.0 * half > a0 and z - 2.0 * half < a1:
                    return False
            return True

        picks: List[List[Tuple[int, int]]] = []
        for _, cond in rows_k:
            got: List[Tuple[int, int]] = []
            shown = {e for row_ in picks for e in row_}
            for want_iso in (True, False):
                for i in range(n):
                    for k in range(Kmax):
                        s_ = _syn(i, k)
                        if (s_ is not None and cond(s_) and (i, k) not in got and (i, k) not in shown and len(got) < 3
                                and (not want_iso or _isolated(i, k))):
                            got.append((i, k))
            picks.append(got)
        if any(picks):
            fig, axs = plt.subplots(len(rows_k), 3, figsize=(17, 3.9 * len(rows_k)), squeeze=False)
            for r, ((lab_, _), got) in enumerate(zip(rows_k, picks)):
                for c in range(3):
                    ax = axs[r, c]
                    if c >= len(got):
                        ax.axis("off")
                        continue
                    i, k = got[c]
                    z, half, a, vf, vt, kind = _syn(i, k)
                    row = idx.iloc[i]
                    w = max(3.0 * half, 4.0 * _span_pad)
                    m = (depth_dense >= z - w) & (depth_dense <= z + w)
                    for g in gall:
                        ax.plot(depth_dense[m], L_dense[i, m, g] - Lm_d[m, g], color=tcm(tnorm(float(grid[g]))), lw=0.9)
                    ax.axhline(0.0, color="k", lw=0.7)
                    ax.axvline(z, color="tab:red" if a > 0 else "tab:blue", ls=":", lw=1.0)
                    dip = row.get(f"anomaly{k + 1}_dip_deg", np.nan)
                    if str(kind).startswith("measured-like"):
                        head = (f"{kind} at {z:.0f} m, size x{row.get(f'anomaly{k + 1}_size_factor', np.nan):.2f}"
                                f"\nsigma {row[f'anomaly{k + 1}_sigma0_m']:.1f} m")
                    else:
                        head = (f"{kind}, r {row[f'anomaly{k + 1}_sigma0_m']:.0f} m"
                                + (f", dip {dip:+.0f} deg" if np.isfinite(dip) else "")
                                + f"\ntau {row[f'anomaly{k + 1}_tau_ms']:.2g} ms")
                    ax.set_title(f"sample {int(row['sample_id'])} S{k + 1}: {head}, peak {a:+.2f} at "
                                 f"{row.get(f'anomaly{k + 1}_peak_time_ms', np.nan):.2g} ms, visible {vf:.2g}-{vt:.2g} ms",
                                 fontsize=9)
                    ax.grid(alpha=0.3)
                    if c == 0:
                        ax.set_ylabel(f"{lab_}\nlog10(gen / background)")
                    ax.set_xlabel(_AX["label"])
            fig.suptitle("Synthetic anomaly kinds in generated profiles: departure from the background per measured gate "
                         "(colour = time)\n" + ", ".join(r_[0].split("\n")[0] for r_ in rows_k))
            fig.tight_layout(rect=(0, 0, 0.93, 0.97))
            _time_colorbar(fig)
            p = os.path.join(plots_dir, "anomaly_kinds.png")
            fig.savefig(p, dpi=dpi)
            plt.close(fig)
            out["anomaly_kinds"] = p
    # statistics over ALL generated profiles (index)
    full = index
    peaks = np.concatenate([full[f"anomaly{k + 1}_peak_decades"].to_numpy() for k in range(Kmax)]) if Kmax else np.zeros(0)
    deps = np.concatenate([full[f"anomaly{k + 1}_{_pc}"].to_numpy() for k in range(Kmax)]) if Kmax else np.zeros(0)
    sig = np.concatenate([full[f"anomaly{k + 1}_sigma0_m"].to_numpy() for k in range(Kmax)]) if Kmax else np.zeros(0)
    tau = np.concatenate([full[f"anomaly{k + 1}_tau_ms"].to_numpy() for k in range(Kmax)]) if Kmax else np.zeros(0)
    typ = np.concatenate([full[f"anomaly{k + 1}_type"].to_numpy() for k in range(Kmax)]) if Kmax else np.zeros(0)
    ok = np.isfinite(peaks)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    kc = full["n_anomalies"].to_numpy()
    vals, cnts = np.unique(kc, return_counts=True)
    axs[0, 0].bar(vals, cnts / max(len(kc), 1), color="tab:blue")
    axs[0, 0].set_xlabel("anomalies per profile")
    axs[0, 0].set_ylabel("share of profiles")
    axs[0, 0].set_title(f"anomaly count ({len(kc)} profiles)")
    axs[0, 1].hist(peaks[ok], bins=50, color="tab:purple")
    axs[0, 1].set_xlabel("signed peak (decades, + raises / - lowers the response)")
    axs[0, 1].set_title("anomaly amplitude")
    phys = "anomaly1_kind" in full.columns
    kinds = (np.concatenate([full[f"anomaly{k + 1}_kind"].astype(str).to_numpy() for k in range(Kmax)])
             if (phys and Kmax) else typ)
    if phys and Kmax:
        # where the anomalies sit, by origin: measured anomalies at their own place, measured shapes moved, conductors
        # through / beside the hole (stacked)
        ks_ = kinds.astype(str)
        grp = [("measured (own place)", np.concatenate([np.where(full[f"measured{j + 1}_scale"].notna(),
                                                                 full[f"measured{j + 1}_peak_at_m"], np.nan)
                                                        for j in range(Mmax)]) if Mmax else np.zeros(0), "tab:orange"),
               ("measured-like", deps[ok & np.char.startswith(ks_, "measured-like")], "tab:brown"),
               ("hole through", deps[ok & (ks_ == "conductor, hole through")], "tab:red"),
               ("beside the hole", deps[ok & (ks_ == "conductor beside the hole")], "tab:purple"),
               ("near-hole zone", deps[ok & (ks_ == "near-hole zone")], "tab:gray")]
        grp = [(n_, np.asarray(v_, dtype=float)[np.isfinite(np.asarray(v_, dtype=float))], c_) for n_, v_, c_ in grp]
        grp = [g_ for g_ in grp if g_[1].size]
        if grp:
            axs[0, 2].hist([v_ for _, v_, _ in grp], bins=40, stacked=True, color=[c_ for _, _, c_ in grp],
                           label=[f"{n_} ({v_.size})" for n_, v_, _ in grp])
            axs[0, 2].legend(fontsize=7)
    else:
        axs[0, 2].hist(deps[ok], bins=40, color="tab:green")
    axs[0, 2].set_xlabel(f"centre {_AX['label']}")
    axs[0, 2].set_title("anomaly position\n(" + ("synthetic: from max(first station, min depth + core), cores apart"
                                                 if phys else "uniform over the " + ("generation axis incl. the extension"
                                                                                     if meas.axis_depth is not None
                                                                                     else "measured span")) + ")")
    if ok.any():
        axs[1, 0].hist(np.log10(sig[ok]), bins=40, color="tab:orange")
    axs[1, 0].set_xlabel("log10 distance r from the hole (m)" if phys else "log10 footprint sigma at onset (m)")
    axs[1, 0].set_title("anomaly width")
    kc_names = (("conductor, hole through", "tab:red"), ("conductor beside the hole", "tab:purple"),
                ("near-hole zone", "tab:gray")) if phys else (("conductor", "tab:red"), ("near-hole", "tab:gray"),
                                                               ("host-zone", "tab:gray"))
    for name, colr in kc_names:
        sel = ok & (kinds == name)
        if sel.any():
            axs[1, 1].hist(np.log10(tau[sel]), bins=40, alpha=0.6, color=colr, label=f"{name} ({int(sel.sum())})")
    axs[1, 1].set_xlabel("log10 tau (ms)")
    axs[1, 1].set_title("eddy-current time constant / " + ("kind" if phys else "type"))
    axs[1, 1].legend(fontsize=8)
    if "anomaly_retention" in full.columns:
        rr = full["anomaly_retention"].to_numpy(dtype=float)
        cc = full["anomaly_corr"].to_numpy(dtype=float)
        m = np.isfinite(rr) & np.isfinite(cc)
        axs[1, 2].scatter(rr[m], cc[m], s=6, alpha=0.5)
        axs[1, 2].axvline(1.0, color="k", lw=0.8)
        axs[1, 2].set_xlabel("retention slope (generated / intended anomaly)")
        axs[1, 2].set_ylabel("correlation generated vs intended")
        axs[1, 2].set_title("anomalies kept by the generator")
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plots_dir, "anomaly_statistics.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    out["anomaly_statistics"] = p
    # time windows of the synthetic anomalies over all profiles: when they peak (vs tau), the tau-distance law, how long
    # they stay visible (vs their peak) and when they appear (vs their distance from the hole)
    if Kmax and "anomaly1_visible_from_ms" in full.columns:
        def _cat(suffix: str) -> np.ndarray:
            return np.concatenate([full[f"anomaly{k + 1}_{suffix}"].to_numpy(dtype=float) for k in range(Kmax)])
        tpk, vf, vt = _cat("peak_time_ms"), _cat("visible_from_ms"), _cat("visible_to_ms")
        okw = ok & np.isfinite(tpk)
        fig, axs = plt.subplots(1, 4, figsize=(22, 5.0))
        for name, colr in kc_names:
            sel = okw & (kinds == name)
            if sel.any():
                axs[0].scatter(tau[sel], tpk[sel], s=7, alpha=0.6, color=colr, label=name)
                axs[1].scatter(sig[sel], tau[sel], s=7, alpha=0.6, color=colr, label=name)
        tl = np.array([float(grid[0]), float(grid[kt - 1])])
        axs[0].plot(tl, tl, "k--", lw=0.8, label="peak time = tau")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("tau (ms)")
        axs[0].set_ylabel("time of the largest change (ms)")
        axs[0].set_title("peak time ~ tau x the background's log-slope")
        axs[0].legend(fontsize=8)
        if phys and meas.depth.size > 1:
            rr_ = np.geomspace(max(float(np.nanmin(sig[okw])), 1e-3), float(np.nanmax(sig[okw])), 50)
            sp_ = float(np.median(np.diff(meas.depth)))
            axs[1].plot(rr_, float(grid[0]) * np.maximum(rr_ / sp_, 1.0) ** 2, "k--", lw=1.0,
                        label=f"tau >= t1 (r / {sp_:.3g} m)^2")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("distance r from the hole / footprint (m)")
        axs[1].set_ylabel("tau (ms)")
        axs[1].set_title("farther = larger body = longer time constant")
        axs[1].legend(fontsize=8)
        sv = okw & np.isfinite(vf)
        sc = axs[2].scatter(np.abs(peaks[sv]), np.log10(vt[sv] / vf[sv]), c=np.log10(tau[sv]), s=7, cmap="viridis")
        axs[2].set_xscale("log")
        axs[2].set_xlabel("|peak| (decades)")
        axs[2].set_ylabel("visible window (decades of time)")
        axs[2].set_title("a stronger anomaly is visible over more gates")
        fig.colorbar(sc, ax=axs[2], label="log10 tau (ms)")
        # conductors only: a near-hole zone is visible from the first gate whatever its width
        sc_ = sv & ~np.isin(kinds, ("near-hole zone", "near-hole", "host-zone"))
        sc = axs[3].scatter(sig[sc_], vf[sc_], c=np.log10(tau[sc_]), s=7, cmap="viridis")
        axs[3].set_xscale("log")
        axs[3].set_yscale("log")
        axs[3].set_xlabel("distance r from the hole / footprint (m)")
        axs[3].set_ylabel("first visible time (ms)")
        axs[3].set_title("conductors: farther from the hole = later onset")
        fig.colorbar(sc, ax=axs[3], label="log10 tau (ms)")
        for ax in axs:
            ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = os.path.join(plots_dir, "anomaly_time_windows.png")
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        out["anomaly_time_windows"] = p
    # decay curves at anomaly centres vs the measured background at the same depth
    ex = []
    for i in range(n):
        for k in range(Kmax):
            a = idx.iloc[i].get(f"anomaly{k + 1}_peak_decades", np.nan)
            if np.isfinite(a) and abs(a) >= 0.08:
                ex.append((i, k))
                break
        if len(ex) >= 4:
            break
    if ex:
        fig, axs = plt.subplots(2, len(ex), figsize=(4.6 * len(ex), 8), squeeze=False)
        for c, (i, k) in enumerate(ex):
            z = float(idx.iloc[i][f"anomaly{k + 1}_{_pc}"])
            j = int(np.argmin(np.abs(depth_dense - z)))
            Lm = depth_interp_profile(meas.depth, Lref, np.array([depth_dense[j]]))[0]
            axs[0, c].plot(grid, 10 ** L_dense[i, j], color="tab:red", lw=1.4, label="generated at centre")
            axs[0, c].plot(grid, 10 ** Lm, color="k", lw=1.0, ls="--", label=f"{ref_name} (same {_AX['name']})")
            axs[0, c].set_xscale("log")
            axs[0, c].set_yscale("log")
            axs[0, c].set_title(f"sample {int(idx.iloc[i]['sample_id'])}, {z:.0f} m, "
                                f"{idx.iloc[i][f'anomaly{k + 1}_type']}, tau "
                                f"{idx.iloc[i][f'anomaly{k + 1}_tau_ms']:.2g} ms", fontsize=9)
            axs[1, c].plot(grid, L_dense[i, j] - Lm, color="tab:red")
            axs[1, c].axhline(0.0, color="k", lw=0.8)
            axs[1, c].set_xscale("log")
            axs[1, c].set_xlabel("t (ms)")
            axs[1, c].set_title(f"log10(generated / {'background' if pure else 'measured'}), intended peak "
                                f"{idx.iloc[i][f'anomaly{k + 1}_peak_decades']:+.2f}", fontsize=9)
            for ax in axs[:, c]:
                ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(plots_dir, "anomaly_decay_examples.png")
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        out["anomaly_decay_examples"] = p
    return out


def save_structure_figure(sm: Dict[str, Any], tab: pd.DataFrame, plots_dir: str, dpi: int = 200,
                          title_tag: str = "", axis_label: str = "depth (m)") -> Dict[str, str]:
    """structure_fidelity.png: local depth structure of the measured profile vs the median generated / anchor profiles,
    per-station median departures (first / middle / last measured gate), and the strongest measured feature's
    retention by gate + the anchor-structure retention by time band (structure_fidelity_metrics)."""
    _AX = {"label": axis_label, "name": axis_label.split(" ")[0]}
    plt = _plt()
    ensure_dir(plots_dir)
    d = tab["depth_m"].to_numpy()
    gms = sm.get("report_gate_ms", [None, None, None])
    has_a = "ls_anchor_median_first" in tab.columns
    fig, axs = plt.subplots(2, 4, figsize=(21, 8.5))
    for c, nm in enumerate(("first", "mid", "last")):
        lab = f"{nm} measured gate" + (f" ({gms[c]:.3g} ms)" if gms[c] is not None else "")
        ax = axs[0, c]
        ax.plot(d, tab[f"ls_meas_{nm}"], "k.-", lw=1.4, label="measured")
        ax.plot(d, tab[f"ls_gen_median_{nm}"], color="tab:blue", lw=1.2, label="generated (median)")
        if has_a:
            ax.plot(d, tab[f"ls_anchor_median_{nm}"], color="tab:green", ls="--", lw=1.0, label="anchors (median)")
        thr = float(sm.get("thr_decades", 0.0))
        ax.axhspan(-thr, thr, color="0.85", alpha=0.6, lw=0, label="|LS| < threshold (not a feature)")
        ax.set_title(f"local depth structure, {lab}", fontsize=10)
        ax.set_xlabel(_AX["label"])
        ax.grid(alpha=0.3)
        ax = axs[1, c]
        ax.plot(d, tab[f"med_dev_gen_meas_{nm}"], color="tab:blue", lw=1.2, label="generated - measured")
        if has_a:
            ax.plot(d, tab[f"med_dev_anchor_meas_{nm}"], color="tab:green", ls="--", lw=1.0, label="anchors - measured")
            ax.plot(d, tab[f"med_dev_gen_anchor_{nm}"], color="tab:red", lw=1.2, label="generated - anchors (network)")
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_title(f"median departure by station, {lab}", fontsize=10)
        ax.set_xlabel(_AX["label"])
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("LS = departure from the neighbour line (decades)")
    axs[1, 0].set_ylabel("median departure (decades)")
    axs[0, 0].legend(fontsize=7)
    axs[1, 0].legend(fontsize=7)
    top = sm.get("strongest_feature", {})
    ax = axs[0, 3]
    keys_t = ("retention_generated_by_gate", "retention_anchor_by_gate", "retention_departure_by_gate")
    has_top = bool(top.get("gate_ms")) and any(np.any(np.isfinite(np.asarray(top[k], dtype=float)))
                                               for k in keys_t if k in top)
    if not has_top:   # background_mode pure: the reference (the measured background) has no narrow feature
        ax.text(0.5, 0.5, "no narrow feature in the reference profile\n(background_mode pure: the measured background)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    if has_top:
        t = np.asarray(top["gate_ms"])
        for key, col, lab in (("retention_generated_by_gate", "tab:blue", "generated median / measured"),
                              ("retention_anchor_by_gate", "tab:green", "anchor median / measured"),
                              ("retention_departure_by_gate", "tab:red", "1 + median(generated - anchor) / measured")):
            if key in top:
                ax.plot(t, top[key], ".-", color=col, lw=1.1, label=lab)
        ax.axhline(1.0, color="k", lw=0.7)
        ax.axhline(0.9, color="0.5", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("t (ms)")
        ax.legend(fontsize=7)
    ax.set_title(f"strongest measured feature ({top.get('depth_m', float('nan')):.0f} m): retention by gate", fontsize=10)
    ax.grid(alpha=0.3)
    ax = axs[1, 3]
    rb = sm.get("struct_retention_by_band")
    if rb:
        ax.bar(np.arange(len(rb)), rb, color="tab:purple")
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_xlabel("time band (early -> late)")
    ax.set_title(f"anchor-structure retention by time band (median {sm.get('struct_retention', float('nan')):.3f})",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Structure fidelity: measured narrow features and the network's departure ({sm.get('n_profiles', 0)} "
                 f"profiles) {title_tag}")
    fig.tight_layout()
    p = os.path.join(plots_dir, "structure_fidelity.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return {"structure_fidelity": p}


def print_structure_summary(sm: Dict[str, Any]) -> None:
    top = sm.get("strongest_feature")
    if not top:
        print("[STRUCTURE] no measured local feature above the threshold")
        return
    gms = sm.get("report_gate_ms")
    gtxt = (" / ".join(f"{g:.3g}" for g in gms) + " ms") if gms else "first / middle / last gate"
    print(f"[STRUCTURE] measured narrow features: |LS| >= {sm['thr_decades']:.4f} decades (measured p95) | strongest at "
          f"{top['depth_m']:.1f} m (peak |LS| {top['peak_abs_ls_decades']:.4f}, {top['n_significant_gates']} significant gates)")
    print(f"[STRUCTURE]   retention at {gtxt} | min over the significant gates:")
    for nm, lab in (("generated", "generated median / measured"), ("anchor", "anchor median / measured"),
                    ("departure", "1 + median(generated - anchor) / measured")):
        if f"retention_{nm}_first_mid_last" in top:
            v = top[f"retention_{nm}_first_mid_last"]
            g = top.get(f"retention_{nm}_min_gate")
            gtx = f" at {top['gate_ms'][g]:.3g} ms" if (g is not None and top.get("gate_ms")) else ""
            print(f"[STRUCTURE]     {lab:42s} {v[0]:.3f} {v[1]:.3f} {v[2]:.3f} | min {top[f'retention_{nm}_min']:.3f}{gtx}")
    feats = sm.get("features", [])
    if feats:
        print("[STRUCTURE]   features (local maxima >= 2 x threshold), min retention generated / departure: " + ", ".join(
            f"{f['depth_m']:.0f} m {f.get('retention_generated_min', float('nan')):.2f}/{f.get('retention_departure_min', float('nan')):.2f}"
            for f in feats))
    pr = sm.get("pooled_measured_structure_retention", {})
    if pr:
        print(f"[STRUCTURE]   pooled over {sm.get('n_significant_cells', 0)} measured feature cells: "
              + ", ".join(f"{k} {v:.3f}" for k, v in pr.items()))
    if "struct_retention" in sm:
        lo, hi = sm["struct_retention_p10_p90"]
        print(f"[STRUCTURE]   anchor-structure retention (band projection, whole profiles): median {sm['struct_retention']:.3f} "
              f"[p10 {lo:.3f}, p90 {hi:.3f}] | by band " + " ".join(f"{v:.2f}" for v in sm["struct_retention_by_band"]))


def background_purity_report(L_gen: np.ndarray, trace: Dict[str, Any], meas: MeasuredData, trainer: "RFMGGenTrainer",
                             kt: int, anchors: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Purity of the delivered profiles (background_mode pure), recomputed in numpy on the measured stations and
    window.  "0 anomalies" profiles: the network's departure from the anchor at the departure scale sigma_R against
    the calibrated threshold, and the whole profile's anomaly-scale content at the background scale sigma_B against
    the tolerance; all profiles: the acceptance's off-label departure purity (trace, all stations of the axis); the
    background anchor's and the measured profile's own content; the raw measured profile as a diagnostic."""
    sig, tol = float(trainer.bg_sigma_m), float(trainer.purity_tol)
    sig_r, thr = float(trainer.purity_sigma_m), float(trainer.purity_thr)
    S = local_poly_smoother_np(meas.depth, sig)
    Sr = local_poly_smoother_np(meas.depth, sig_r)

    def _hp(X: np.ndarray, Sm: np.ndarray = S) -> np.ndarray:
        return np.abs(highpass_profile_np(X, meas.depth, sig, S=Sm))

    Bq, Lq = np.asarray(meas.background[:, :kt]), np.asarray(meas.L[:, :kt])
    hm = _hp(Lq)
    trend_meas = ((meas.background_info or {}).get("trend") or {}).get("rule") == "measured"
    out: Dict[str, Any] = {"sigma_B_m": sig, "sigma_R_m": sig_r, "purity_tol_decades": tol,
                           "departure_threshold_decades": thr, "response_to_sigma_B_bump": float(trainer.purity_resp),
                           "anchor_share": PURITY_ANCHOR_SHARE,
                           "background_anchor_highpass_max_decades": float(_hp(Bq).max()),
                           "measured_highpass_max_decades": float(hm.max()),
                           "measured_cells_above_tol_fraction": float(np.mean(hm > tol))}

    def _stats(v: np.ndarray, lim: float) -> Dict[str, Any]:
        return {"n": int(v.size), "limit_decades": float(lim), "within_fraction": float(np.mean(v <= lim)),
                "max_decades_median": float(np.median(v)), "max_decades_p99": float(np.percentile(v, 99)),
                "max_decades_max": float(v.max())}

    qc = trace.get("qc", {})
    if "purity_max_decades" in qc:
        out["all_profiles_departure_off_labels"] = _stats(np.asarray(qc["purity_max_decades"], dtype=np.float64), thr)
    prm_ = trace.get("params", {})
    K = np.asarray(prm_.get("an_K", np.zeros(L_gen.shape[0])), dtype=int)
    if "ma_on" in prm_:                     # the measured anomalies count
        K = K + (np.asarray(prm_["ma_on"]) > 0.5).sum(axis=1)
    z = np.nonzero(K == 0)[0]
    if z.size:
        if anchors is not None:
            out["zero_anomaly_departure"] = _stats(np.array([_hp(np.asarray(L_gen[i] - anchors[i], dtype=np.float64),
                                                                 Sr).max() for i in z]), thr)
        # background_trend measured: the anchor keeps the measured profile's own bends (its content at sigma_B,
        # out["background_anchor_highpass_max_decades"]); a "0 anomalies" profile may carry that plus the tolerance
        lim_w = tol + (out["background_anchor_highpass_max_decades"] if trend_meas else 0.0)
        out["zero_anomaly_whole_profile"] = _stats(np.array([_hp(np.asarray(L_gen[i], dtype=np.float64)).max()
                                                             for i in z]), lim_w)
        if getattr(trainer, "bg_trend", False):
            # the overall decay trend: largest step against it along the profile, every time sample (decades)
            dn = trainer.t_bg_down.cpu().numpy()[:kt][None, :]
            st = [float(np.where(dn, np.diff(L_gen[i], axis=0), -np.diff(L_gen[i], axis=0)).max()) for i in z]
            out["zero_anomaly_trend_steps"] = _stats(np.asarray(st), PURITY_ANCHOR_SHARE * tol)
    lo, hi = np.percentile(L_gen, 2.5, axis=0), np.percentile(L_gen, 97.5, axis=0)
    out["background_coverage"] = float(np.mean((Bq >= lo) & (Bq <= hi)))
    out["raw_measured_coverage"] = float(np.mean((Lq >= lo) & (Lq <= hi)))
    out["distance_to_raw_measured_rms_decades_median"] = float(np.median(np.sqrt(np.mean((L_gen - Lq[None]) ** 2,
                                                                                          axis=(1, 2)))))
    print(f"[PURITY] tolerance {tol:.4f} decades (peak) | background anchor: anomaly-scale content at sigma_B {sig:.4g} m "
          f"{out['background_anchor_highpass_max_decades']:.4f} "
          + ("(the measured profile's own bends: the v20.2 trend is the measured profile off its undulations)"
             if trend_meas else f"(<= {PURITY_ANCHOR_SHARE:g} x tolerance)") + f" | measured profile "
          f"{out['measured_highpass_max_decades']:.4f} ({out['measured_cells_above_tol_fraction']:.1%} of its cells above the "
          f"tolerance: its own anomalies{', kept as labelled anomalies' if getattr(trainer, 'n_ma', 0) else ', in no generated profile'}) | departure test at sigma_R {sig_r:.4g} m, threshold "
          f"{thr:.4f}")
    for key, name in (("zero_anomaly_departure", "'0 anomalies' profiles, network departure (numpy recheck)"),
                      ("zero_anomaly_whole_profile", "'0 anomalies' profiles, whole profile at sigma_B"),
                      ("zero_anomaly_trend_steps", "'0 anomalies' profiles, step against the overall decay trend"),
                      ("all_profiles_departure_off_labels", "all profiles off their labelled anomalies (acceptance)")):
        if key in out:
            v = out[key]
            print(f"[PURITY] {name}: {v['within_fraction']:.1%} of {v['n']} within {v['limit_decades']:.4f} | max per "
                  f"profile median {v['max_decades_median']:.4f}, p99 {v['max_decades_p99']:.4f}, largest "
                  f"{v['max_decades_max']:.4f}")
    print(f"[PURITY] raw measured profile (diagnostic): inside the generated 2.5-97.5 % band {out['raw_measured_coverage']:.1%} "
          f"of its cells (background {out['background_coverage']:.1%}); median RMS distance "
          f"{out['distance_to_raw_measured_rms_decades_median']:.4f} decades")
    return out


def save_background_figure(meas: MeasuredData, plots_dir: str, dpi: int = 200) -> Dict[str, str]:
    """background_decomposition.png: the measured profile (points), its smooth geological background (the anchors of
    every generated profile) and the residual = the measured profile's own anomalies with the purity tolerance band,
    at six gates; the inventory's anomalies are shaded (one colour each, the measured trend described as such)."""
    bi = meas.background_info or {}
    if meas.background is None:
        return {}
    plt = _plt()
    ensure_dir(plots_dir)
    _AX = _axis_names(meas)
    kn = meas.known_time_mask()
    gsel = [g for g in _gate_subset(meas.T, meas.gate_index, 6) if kn[g]]
    tol = float(bi.get("purity_tol_decades", float("nan")))
    fig, axs = plt.subplots(2, len(gsel), figsize=(3.3 * len(gsel), 7.2), sharex=True, squeeze=False)
    runs = (bi.get("measured_anomalies") or {}).get("anomalies", [])
    for c, g in enumerate(gsel):
        a0, a1 = axs[0, c], axs[1, c]
        a0.plot(meas.depth, 10 ** meas.L[:, g], ".", ms=3, color="0.45", label="measured")
        a0.plot(meas.depth, 10 ** meas.background[:, g], "-", lw=1.8, color="tab:orange", label="background (anchor)")
        a0.set_yscale("log")
        a0.set_title(f"{meas.grid_ms[g]:.3g} ms", fontsize=9)
        a1.plot(meas.depth, meas.L[:, g] - meas.background[:, g], "-", lw=1.0, color="tab:blue")
        if math.isfinite(tol):
            a1.axhspan(-tol, tol, color="tab:green", alpha=0.15, lw=0)
        a1.axhline(0.0, color="k", lw=0.7)
        for q_, r_ in enumerate(runs):
            # each measured anomaly in its own colour (adjacent parts of one undulation run stay apart)
            colr = ("tab:red", "tab:purple", "tab:brown", "tab:pink")[q_ % 4]
            for ax in (a0, a1):
                ax.axvspan(r_["from_m"], max(r_["to_m"], r_["from_m"] + 1e-6), color=colr, alpha=0.08, lw=0)
            if c == len(gsel) - 1 or c == 0:
                a1.text(0.5 * (r_["from_m"] + r_["to_m"]), 0.98, f"M{q_ + 1}", color=colr, fontsize=8, ha="center",
                        va="top", transform=a1.get_xaxis_transform())
        a1.set_xlabel(_AX["label"])
        for ax in (a0, a1):
            ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("|dB/dt|")
    axs[1, 0].set_ylabel("measured - background\n(log10, decades)")
    axs[0, 0].legend(fontsize=7)
    pur = bi.get("purification", {})
    kept = float(((bi.get("measured_anomaly_policy") or {}).get("prob", 1.0))) > 0
    fate = "labelled anchor anomalies" if kept else "left out"
    if ((bi.get("trend") or {}).get("rule")) == "measured":
        head = (f"Measured profile = its '0 anomalies' decay trend (orange, anchor of every generated profile: the measured "
                f"profile off its undulations, passed under them) + its {len(runs)} anomalies (the undulations, M1..; "
                f"{fate})")
    else:
        head = (f"Measured profile = smooth geological background (orange, anchor of every generated profile; sigma "
                f"{bi.get('sigma_m', float('nan')):.4g} m, {pur.get('passes', 0)} purification passes) + its own anomalies "
                f"(shaded, {len(runs)} runs; {fate})")
    fig.suptitle(head + f". Green band = purity tolerance +-{tol:.4f} decades", fontsize=9)
    fig.tight_layout()
    p = os.path.join(plots_dir, "background_decomposition.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return {"background_decomposition": p}


def extension_diagnostics(L_all: np.ndarray, A_all: Optional[np.ndarray], trainer: "RFMGGenTrainer",
                          meas: MeasuredData) -> Dict[str, Any]:
    """Checks of the generated profiles where nothing is measured. Profile extension, per extended end: decay
    monotonicity of the generated curves, the step from the last measured station into the extension relative to the
    measured station-to-station steps, the departure's local structure relative to the anchor's, and the ensemble
    spread around the rule (std over profiles of generated - rule at the outermost station) against the calibrated
    spread a dist^gamma.  Late-time extension: monotonicity, the late log-log slope of the extension against the
    measured late slopes, the ensemble spread at the last node against a dlog t^gamma."""
    out: Dict[str, Any] = {}
    L = np.asarray(L_all, dtype=np.float64)
    known = np.asarray(trainer.keep_mask, dtype=bool)
    d = trainer.depth_np
    kt = int(np.sum(meas.known_time_mask()))
    um = trainer.unmeasured
    if um is not None and not bool(np.all(known)):
        rule_L = trainer.L_anchor_np                               # the rule (no per-profile departure)
        mono = np.diff(L, axis=2) <= 1e-9
        dm = np.diff(L[:, known][:, :, :kt], axis=1)
        step_ref = float(np.median(np.abs(dm))) + 1e-12
        prof: Dict[str, Any] = {}
        for side, sel, edge in (("low", d < d[known][0], int(np.nonzero(known)[0][0])),
                                ("high", d > d[known][-1], int(np.nonzero(known)[0][-1]))):
            if not np.any(sel):
                continue
            idx = np.nonzero(sel)[0]
            far = int(idx[0]) if side == "low" else int(idx[-1])
            near = int(idx[-1]) if side == "low" else int(idx[0])
            dist_far = abs(float(d[far] - d[edge]))
            sd = np.std(L[:, far, :kt] - rule_L[far, :kt], axis=0)
            cal = float(np.median(um[side]["a"][:kt])) * dist_far ** float(um[side]["gamma"]) * float(
                trainer.cfg.profile.unmeasured_spread)
            step = float(np.median(np.abs(L[:, near, :kt] - L[:, edge, :kt])))
            res: Dict[str, Any] = {"n_virtual_stations": int(idx.size), "rule": um[side]["rule"],
                                   "distance_to_far_end": dist_far,
                                   "monotone_fraction": float(np.mean(np.all(mono[:, idx], axis=2))),
                                   "edge_step_vs_measured_step": step / step_ref,
                                   "spread_far_end_decades": float(np.median(sd)),
                                   "calibrated_spread_far_end_decades": cal}
            if A_all is not None and idx.size >= 3:
                Ld, Ad = L[:, idx, :kt], np.asarray(A_all, dtype=np.float64)[:, idx, :kt]
                lsd = np.abs(local_structure_residual((Ld - Ad).transpose(1, 0, 2).reshape(idx.size, -1), d[idx]))
                lsa = np.abs(local_structure_residual(Ad.transpose(1, 0, 2).reshape(idx.size, -1), d[idx]))
                res["departure_local_structure_vs_anchor"] = float(np.sqrt(np.mean(lsd ** 2)) /
                                                                   max(np.sqrt(np.mean(lsa ** 2)), 1e-12))
            prof[side] = res
        out["profile"] = prof
    if trainer.t_time_spread is not None and kt < meas.T:
        lt = np.log10(meas.grid_ms)
        Lk = L[:, np.asarray(trainer.meas_cols)]
        mono = np.all(np.diff(Lk[:, :, kt - 1:], axis=2) <= 1e-9, axis=2)
        sl_ext = (Lk[:, :, -1] - Lk[:, :, kt - 1]) / max(lt[-1] - lt[kt - 1], 1e-12)
        k0 = max(0, kt - max(3, int(round(0.1 * kt))))
        sl_meas = (meas.L[:, kt - 1] - meas.L[:, k0]) / max(lt[kt - 1] - lt[k0], 1e-12)
        rule = trainer.L_anchor_np[np.asarray(trainer.meas_cols)]
        sd = np.std(Lk[:, :, -1] - rule[None, :, -1], axis=0)
        out["time"] = {"t_known_ms": float(meas.t_known_ms), "t_end_ms": float(meas.grid_ms[-1]),
                       "rule": meas.late_ext["rule"], "monotone_fraction": float(np.mean(mono)),
                       "extension_slope_median": float(np.median(sl_ext)),
                       "extension_slope_p5_p95": [float(np.percentile(sl_ext, 5)), float(np.percentile(sl_ext, 95))],
                       "measured_late_slope_p5_p95": [float(np.percentile(sl_meas, 5)), float(np.percentile(sl_meas, 95))],
                       "spread_last_node_decades": float(np.median(sd)),
                       "calibrated_spread_last_node_decades": float(trainer.t_time_spread[-1].item())}
    if out:
        print("[EXTENSION] " + json.dumps(out, ensure_ascii=False, default=float))
    return out


def save_extension_figure(L_all: np.ndarray, trainer: "RFMGGenTrainer", meas: MeasuredData, diag: Dict[str, Any],
                          plots_dir: str, dpi: int = 200, title_tag: str = "") -> Dict[str, str]:
    """extension.png: generated ensemble (median, 50 % / 95 % bands) along the generation axis at three measured
    gates with the measured stations and the extension bands shaded; with a late-time extension, decay curves at
    three measured stations beyond the last measured gate."""
    plt = _plt()
    ensure_dir(plots_dir)
    _AX = _axis_names(meas)
    d, grid = trainer.depth_np, meas.grid_ms
    known = np.asarray(trainer.keep_mask, dtype=bool)
    kt = int(np.sum(meas.known_time_mask()))
    has_t = "time" in diag
    ncol = 3
    fig, axs = plt.subplots(2 if has_t else 1, ncol, figsize=(17, 9 if has_t else 4.8), squeeze=False)
    gs = _gate_subset(kt, meas.gate_index[meas.gate_index < kt], 3)
    for ax, g in zip(axs[0], gs):
        q = np.percentile(L_all[:, :, g], [2.5, 25, 50, 75, 97.5], axis=0)
        ax.fill_between(d, 10 ** q[0], 10 ** q[4], color="tab:blue", alpha=0.18, label="generated 2.5-97.5%")
        ax.fill_between(d, 10 ** q[1], 10 ** q[3], color="tab:blue", alpha=0.35, label="generated 25-75%")
        ax.plot(d, 10 ** q[2], color="tab:blue", lw=1.2, label="generated median")
        ax.plot(meas.depth, 10 ** meas.L[:, g], "k.-", lw=1.0, ms=3, label="measured")
        for side_sel in (d < d[known][0], d > d[known][-1]):
            if np.any(side_sel):
                ax.axvspan(float(d[side_sel].min()), float(d[side_sel].max()), color="tab:orange", alpha=0.10)
        ax.set_yscale("log")
        ax.set_title(f"t = {grid[g]:.4g} ms (orange = extension)")
        ax.set_xlabel(_AX["label"])
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    if has_t:
        mcols = np.asarray(trainer.meas_cols)
        rng = np.random.default_rng(1)
        for ax, q_ in zip(axs[1], (0.1, 0.5, 0.9)):
            i = int(mcols[int(round(q_ * (mcols.size - 1)))])
            j = int(np.nonzero(mcols == i)[0][0])
            for k in rng.choice(L_all.shape[0], size=min(25, L_all.shape[0]), replace=False):
                ax.plot(grid, 10 ** L_all[k, i], color="tab:orange", lw=0.6, alpha=0.6)
            ax.plot(meas.pm.time_ms, meas.pm.values[j], "k.", ms=4, label="measured gates")
            ax.axvspan(float(grid[kt - 1]), float(grid[-1]), color="tab:orange", alpha=0.10)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{_AX['name']} {d[i]:.1f} m: late-time extension after {grid[kt - 1]:.4g} ms")
            ax.set_xlabel("t (ms)")
            ax.grid(alpha=0.3)
        axs[1, 0].legend(fontsize=8)
    fig.suptitle(f"Generation beyond the measured coverage {title_tag}")
    fig.tight_layout()
    p_ = os.path.join(plots_dir, "extension.png")
    fig.savefig(p_, dpi=dpi)
    plt.close(fig)
    return {"extension": p_}
