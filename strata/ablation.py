"""Ablations: single-factor variants applied to the configuration, their output folders and report, and the
standalone GAN / CGAN baselines."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cli import apply_manifold_constraint, apply_v21_generator
from .common import ensure_dir
from .config import _RFMG_ABLATION_FLAGS, AppConfig
from .data.normalization import torch_L_from_norm
from .figures import _plt
from .losses import ns_d_loss, ns_g_loss
from .models.baselines import _AblBaselineDiscriminator, _AblBaselineGenerator
from .trainer import RFMGGenTrainer


_RFMG_ABLATION_MODES = ("full", "no_film", "no_curriculum", "single_critic", "no_manifold_gating",
                        "no_mechanism_anchor", "no_physics_pull", "no_diversity", "no_real_endpoint", "no_bridge",
                        "cgan", "gan", "no_profile", "no_profile_critic", "no_anomalies",
                        # the field-admissible manifold constraints, one at a time, and all of them
                        "no_field_manifold", "affine_background", "cmg_credibility", "v20_manifold",
                        "no_sim_morphology", "no_learned_morphology", "soft_decay", "one_sided_undulations",
                        "v21_generator")


_RFMG_ABLATION_ACTIVE = "full"


def _rfmg_apply_ablation(cfg: AppConfig, mode: str) -> None:
    """Single-factor ablations: exactly one mechanism is removed; data, seed, epochs, normaliser and
    the evaluation frame stay identical ('full' = no override)."""
    global _RFMG_ABLATION_ACTIVE
    mode = str(mode or "full").lower()
    assert mode in _RFMG_ABLATION_MODES, f"unknown ablation mode {mode}"
    _RFMG_ABLATION_ACTIVE = mode
    _RFMG_ABLATION_FLAGS.update({"no_film": False, "no_condition": False, "plain_export": False})
    cfg.reproduction.ablation_mode = mode
    L, R, P = cfg.losses, cfg.reproduction, cfg.profile
    if mode == "no_film":
        _RFMG_ABLATION_FLAGS["no_film"] = True
        R.film_residual = 0.0
    elif mode == "no_curriculum":
        R.curriculum = False
    elif mode == "single_critic":
        R.use_dual_discriminator = False
    elif mode == "no_manifold_gating":
        R.cmg_enabled = False
    elif mode == "no_mechanism_anchor":
        L.lambda_bridge_feat = 0.0
        L.lambda_mechanism_anchor = 0.0
    elif mode == "no_physics_pull":
        L.lambda_guid = 0.0
        L.lambda_dist = 0.0
    elif mode == "no_diversity":
        L.lambda_div = 0.0
        L.lambda_prof_ms = 0.0
        L.lambda_copy_guard = 0.0
        R.mode_seek_two_z_enabled = False
    elif mode == "no_real_endpoint":
        cfg.generation.union_real_weight = 0.0
    elif mode == "no_bridge":
        cfg.generation.union_bridge_weight = 0.0
    elif mode in ("cgan", "gan"):
        _RFMG_ABLATION_FLAGS["plain_export"] = True
        if mode == "gan":
            _RFMG_ABLATION_FLAGS["no_condition"] = True
    elif mode == "no_profile":
        P.enabled = False
        P.critic = False
    elif mode == "no_profile_critic":
        P.critic = False
    elif mode == "no_anomalies":
        P.anomaly_enabled = False
    elif mode == "no_field_manifold":           # no FAM shell term, no FAM acceptance gate
        P.fam_enabled = False
        L.lambda_field_shell = 0.0
    elif mode == "affine_background":
        P.background_manifold = "affine"
    elif mode == "cmg_credibility":
        R.cmg_form = "v20"
    elif mode == "v20_manifold":
        P.manifold_constraint = "v20"
        apply_manifold_constraint(cfg)
    elif mode == "no_sim_morphology":           # the morphology bank and its prior without the simulated anomalies
        cfg.data.sim_role = "off"
    elif mode == "no_learned_morphology":       # the bank with the measured / simulated morphologies as they are
        P.morphology_learned = 0
    elif mode == "soft_decay":                  # hinge penalty on rises instead of the time-monotone projection
        cfg.generation.decay_constraint = "soft"
    elif mode == "one_sided_undulations":
        P.undulation_sides = "one"
    elif mode == "v21_generator":
        apply_v21_generator(cfg)
    print(f"[ABLATION] mode={mode} applied")


def _rfmg_ablation_redirect_dirs(cfg: AppConfig, mode: str) -> None:
    if mode == "full" or not cfg.export.ablation_subdir:      # --no_ablation_subdir: the folders as given
        return
    cfg.export.model_dir = os.path.join(cfg.export.model_dir, "ablation", mode)
    cfg.export.generated_data_path = os.path.join(cfg.export.generated_data_path, "ablation", mode)
    for attr in ("log_dir", "plot_dir", "report_dir"):        # the separate folders follow
        if getattr(cfg.export, attr):
            setattr(cfg.export, attr, os.path.join(getattr(cfg.export, attr), "ablation", mode))
    print(f"[ABLATION] outputs redirected -> {cfg.export.model_dir} | {cfg.export.generated_data_path}")


def _rfmg_ablation_report(cfg: AppConfig, modes: Optional[List[str]] = None) -> None:
    base = cfg.export.model_dir
    if os.sep + "ablation" + os.sep in base:
        base = base.split(os.sep + "ablation" + os.sep)[0]
    rows = []
    for m in (modes or list(_RFMG_ABLATION_MODES)):
        md = base if m == "full" else os.path.join(base, "ablation", m)
        p = os.path.join(md, "analysis", "reports", "generation_metrics.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            js = json.load(f)
        prof, cur = js.get("profile", {}), js.get("curve", {})
        rows.append({"mode": m, "status": js.get("qc", {}).get("status", ""),
                     "struct_retention": js.get("structure", {}).get("struct_retention"),
                     "rough_ratio_vs_anchor": prof.get("rough_ratio_vs_anchor_median"),
                     "rough_ratio_vs_measured": prof.get("rough_ratio_vs_measured_median"),
                     "measured_coverage": prof.get("measured_coverage"),
                     "diversity_decades": prof.get("pairwise_profile_distance_decades_median"),
                     "copy_fraction": prof.get("copy_fraction"), "noncrossing": prof.get("noncrossing_fraction"),
                     "precision": cur.get("descriptor_precision"), "recall": cur.get("descriptor_recall")})
    if not rows:
        print("[ABLATION] no completed variants found for the report")
        return
    rep_dir = ensure_dir(os.path.join(base, "analysis", "reports"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rep_dir, "ablation_report.csv"), index=False)
    print("[ABLATION] report:\n" + df.to_string(index=False))
    try:
        plt = _plt()
        keys = ["rough_ratio_vs_anchor", "measured_coverage", "diversity_decades", "copy_fraction", "precision", "recall"]
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for ax, k in zip(axs.ravel(), keys):
            ax.bar(df["mode"], df[k].astype(float))
            ax.set_title(k)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, "ablation_report.png"), dpi=int(cfg.visualization.dpi))
        plt.close(fig)
    except Exception as e:
        print(f"[ABLATION] figure skipped: {e}")


def _abl_downsample(y: torch.Tensor, k: int) -> torch.Tensor:
    s = y[..., 0] if y.dim() == 3 else y
    return F.adaptive_avg_pool1d(s.unsqueeze(1), k).squeeze(1)


def run_standalone_baseline(cfg: AppConfig, mode: str, trainer: RFMGGenTrainer, device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """cgan / gan baselines: textbook DCGAN-1D curves (no RFMG mechanism, no profile level); profiles are
    assembled from independently generated curves on the measured stations."""
    T = trainer.T
    pool = trainer.build_curve_pool(cfg.data.measured_pool_size, cfg.data.sim_reservoir_rows // 2,
                                    cfg.data.sim_reservoir_rows // 2, seed=cfg.runtime.seed + 5)
    kc = 16 if mode == "cgan" else 0
    G = _AblBaselineGenerator(32, T, cond_dim=kc).to(device)
    Dn = _AblBaselineDiscriminator(T, cond_dim=kc).to(device)
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(Dn.parameters(), lr=2e-4, betas=(0.5, 0.999))
    steps = int(max(1, (cfg.train.stage1_epochs + cfg.train.stage2_epochs) * cfg.train.steps_per_epoch))
    y_all = pool["y"]
    bs = int(cfg.runtime.batch_size)
    print(f"[BASELINE] {mode}: {steps} steps on {y_all.shape[0]} curves (condition={'16-bin downsample' if kc else 'none'})")
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, y_all.shape[0], (bs,), device=device)
        yr = y_all[idx]
        c = _abl_downsample(yr, kc) if kc else None
        z = torch.randn(bs, 32, device=device)
        yf = G(z, c)
        od.zero_grad(set_to_none=True)
        ld = ns_d_loss(Dn(yr, c), Dn(yf.detach(), c))
        ld.backward()
        od.step()
        og.zero_grad(set_to_none=True)
        lg = ns_g_loss(Dn(yf, c))
        lg.backward()
        og.step()
        if (it + 1) % max(1, steps // 10) == 0:
            print(f"[BASELINE] {mode} step {it + 1}/{steps} | D={ld.item():.4f} G={lg.item():.4f} | {time.time() - t0:.1f}s")
    G.eval()
    n_prof = int(cfg.generation.generated_profile_count or math.ceil(cfg.data.generated_data_count / trainer.D))
    outs, tr = [], {"alpha": [], "zoom": [], "offset": [], "gain": []}
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.runtime.seed + 2027)
    with torch.no_grad():
        done = 0
        while done < n_prof:
            nb = min(max(1, 512 // trainer.D), n_prof - done)
            b = trainer.build_profile_batch(nb, trainer.D, mode="gen", station_jitter=False, gen=gen,
                                            starts=np.zeros(nb, dtype=np.int64))
            ya = b["y_A"].reshape(-1, T, 1)
            c = _abl_downsample(ya, kc) if kc else None
            yf = G(torch.randn(ya.shape[0], 32, device=device, generator=gen), c).reshape(nb, trainer.D, T)
            outs.append(torch_L_from_norm(yf.float(), trainer.norm).cpu().numpy())
            tr["alpha"].append(b["alpha"].cpu().numpy())
            for k in ("zoom", "offset", "gain"):
                tr[k].append(b[f"reg_{k}"].cpu().numpy())
            done += nb
    return np.concatenate(outs), {k: np.concatenate(v) for k, v in tr.items()}
