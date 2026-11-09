"""Staged training and generation (RFMGGenTrainer): anchors with labelled anomalies, the autoencoder and condition
stages, adversarial training against the latent, decay and profile critics, the regional head (_warp_background), the
acceptance tests of the profiles and generation with an exponential moving average of the generator."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import RFMG_PROGRAM_NAME, RFMG_SCHEMA_VERSION, RFMG_VERSION
from .checkpoints import _format_stage_progress
from .common import (AMP_CALIBRATION_MIN_SCALE, AverageMeter, GradScaler, _parse_mix, array_digest, atomic_torch_save,
                     autocast, count_parameters, ensure_dir, generator_epoch_verdict, set_module_trainable,
                     stable_json_dumps)
from .config import AppConfig
from .data.anomaly_laws import (MEASURED_LIKE_SIZE_RANGE, SEPARATION_CANDIDATES, SEPARATION_CORE_SIGMAS,
                                UNDULATION_LOCAL_POINTS, UNDULATION_REDRAWS, UNION_REFERENCE_PROFILES,
                                UNION_REFERENCE_PROFILES_MEASURED, anomaly_amp_range, anomaly_count_law,
                                anomaly_tau_range_log10, borehole_physical_anomalies, dipole_axial_kernel,
                                lateral_correlation_law, measured_anomaly_mode, measured_anomaly_scale_ranges,
                                template_width_sigma)
from .data.contract import MeasuredData, SimPrior, local_structure_residual
from .data.decomposition import local_poly_smoother_np, measured_anomaly_templates
from .data.extension import anchor_profile_on, calibrate_unmeasured
from .data.normalization import L_from_norm, NormalizationState, norm_from_L, torch_L_from_norm
from .data.readers import build_survey_spec
from .data.time_grid import depth_interp_profile, pchip_slopes
from .losses import (ANOMALY_SHAPE_TOL, anomaly_shape_residual, centred_smooth_l1, depth_lowpass_torch,
                     local_structure_torch, masked_quadratic_residual_torch, masked_quadratic_trend_torch,
                     masked_trend_fit_torch, ns_d_loss, ns_g_loss, r1_penalty, segment_weights, soft_chamfer,
                     structure_projection)
from .models.critics import ConditionalDiscriminator, SequencePatchDiscriminator
from .models.decay_manifold import (ConditionSupervisionHead, Embedder, Recovery, TEMConditionExtractor, _FinCMG,
                                    _FinFiLMMixer)
from .models.generator import BridgeResidualGenerator, Supervisor
from .models.spatial import (ProfileDepthContext, ProfilePatchDiscriminator, depth_features, fourier_field,
                             fourier_params_t)
from .morphology import (MORPH_EXAMPLE_SHARE, MORPH_FILTER, MORPH_PHYS_KERNELS, MORPH_PHYS_SHARES, MORPH_SIM_TAUS,
                         MORPH_U, MORPH_U_HALF, MORPH_U_POINTS, MORPH_VAE_BETA, MORPH_VAE_LATENT, MORPH_VAE_NT,
                         MORPH_VAE_NU, MORPH_VAE_PIX_SD, MORPH_VAE_U_HALF, MorphologyVAE, _interp_matrix,
                         _normalise_patches, morphology_filters, morphology_mix)
from .physics import PHYS_NAMES, FieldAdmissibleManifold, PhysicsDescriptor
from .pipeline import measured_background, qc_selection_audit
from .quality import (LABEL_WEIGHT_SMOOTH_DECADES, PURITY_ANCHOR_SHARE, PURITY_LOSS_MARGIN, QC_GATES, _nn_dist,
                      acceptance_summary, composite_quality_score, curve_quality_metrics, pairwise_profile_distance,
                      profile_quality_metrics, qc_fail_reasons, structure_fidelity_metrics)


def torch_pchip_slopes(x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Fritsch-Carlson monotone slopes along the last axis of Y (..., n) for knots x (n,) — the torch twin of
    pchip_slopes (same edge rule), used to densify generated profiles along depth on the GPU."""
    n = Y.shape[-1]
    h = torch.diff(x)
    delta = torch.diff(Y, dim=-1) / h
    if n == 2:
        return delta[..., :1].expand_as(Y).clone()
    m = torch.zeros_like(Y)
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    d0, d1 = delta[..., :-1], delta[..., 1:]
    same = (d0 * d1) > 0
    hm = (w1 + w2) / (w1 / torch.where(same, d0, torch.ones_like(d0)) + w2 / torch.where(same, d1, torch.ones_like(d1)))
    m[..., 1:-1] = torch.where(same, hm, torch.zeros_like(hm))

    def _edge(h0, h1, m0, m1):
        d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        d = torch.where(torch.sign(d) != torch.sign(m0), torch.zeros_like(d), d)
        return torch.where((torch.sign(m0) != torch.sign(m1)) & (d.abs() > 3.0 * m0.abs()), 3.0 * m0, d)

    m[..., 0] = _edge(h[0], h[1], delta[..., 0], delta[..., 1])
    m[..., -1] = _edge(h[-1], h[-2], delta[..., -1], delta[..., -2])
    return torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)


def torch_hermite_dense(x: torch.Tensor, Y: torch.Tensor, m: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Evaluate the cubic Hermite interpolant (knots x (n,), values / slopes (..., n)) at xq (q,) -> (..., q).
    Exact at the knots."""
    n = x.numel()
    k = (torch.searchsorted(x, xq.clamp(x[0], x[-1]).contiguous(), right=True) - 1).clamp(0, n - 2)
    h = x[k + 1] - x[k]
    s = (xq.clamp(x[0], x[-1]) - x[k]) / h
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    return h00 * Y[..., k] + h10 * h * m[..., k] + h01 * Y[..., k + 1] + h11 * h * m[..., k + 1]


class _EMA:
    """Exponential moving average of the generator path with the usual warm-up
    (effective decay min(decay, (1+n)/(10+n))) so short runs are not dominated by stale weights."""

    def __init__(self, modules: Dict[str, nn.Module], decay: float) -> None:
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow = {n: copy.deepcopy(m).eval() for n, m in modules.items() if m is not None}
        for m in self.shadow.values():
            set_module_trainable(m, False)

    @torch.no_grad()
    def reset(self, modules: Dict[str, nn.Module]) -> None:
        for n, m in modules.items():
            if m is not None and n in self.shadow:
                self.shadow[n].load_state_dict(m.state_dict())
        self.n_updates = 0

    @torch.no_grad()
    def update(self, modules: Dict[str, nn.Module]) -> None:
        self.n_updates += 1
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        for n, m in modules.items():
            if m is None or n not in self.shadow:
                continue
            for ps, pm in zip(self.shadow[n].parameters(), m.parameters()):
                ps.mul_(d).add_(pm.detach(), alpha=1.0 - d)
            for bs, bm in zip(self.shadow[n].buffers(), m.buffers()):
                bs.copy_(bm)

    def state_dict(self) -> Dict[str, Any]:
        sd = {n: m.state_dict() for n, m in self.shadow.items()}
        sd["__n_updates__"] = self.n_updates
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        for n, s in (sd or {}).items():
            if n == "__n_updates__":
                self.n_updates = int(s)
            elif n in self.shadow:
                self.shadow[n].load_state_dict(s, strict=False)


class RFMGGenTrainer:
    """RFMG-Gen trainer: Stage-0 AE, Stage-0B condition frame, Stage-1A supervisor, Stage-1B profile union-bridge
    pretraining, Stage-2 adversarial refinement (3 critics)."""

    # trainable generator path (EMA-tracked).
    G_PATH = ("generator", "profile_ctx", "recovery", "cmg", "mixer")

    def __init__(self, cfg: AppConfig, device: torch.device, meas: MeasuredData, sim: SimPrior,
                 norm: NormalizationState, contract: Dict[str, Any], keep_mask: Optional[np.ndarray] = None,
                 tag: str = "") -> None:
        self.cfg, self.device, self.meas, self.sim, self.norm = cfg, device, meas, sim, norm
        self.contract = contract
        self.tag = tag
        self.T, self.D = meas.T, meas.D
        self.use_amp = bool(cfg.runtime.use_amp and device.type == "cuda")
        m, R, P = cfg.model, cfg.reproduction, cfg.profile
        T = self.T
        # time-monotone constraint of this run (decay_constraint_evidence): hard = running minimum over time in every
        # anchor / output + the monotone recovery head; soft = no projection, a hinge on rises beyond the measured
        # gate-to-gate noise; off = no time constraint (signed data)
        dc_ = meas.report.get("decay_constraint") or {"mode": "hard"}
        self.decay_mode = str(dc_.get("mode", "hard"))
        self.decay_hard, self.decay_soft = self.decay_mode == "hard", self.decay_mode == "soft"
        self.decay_tol = float(dc_.get("rise_tolerance_decades", 0.0) or 0.0)
        self.embedder = Embedder(1, m.hidden_dim, m.num_layer, m.module).to(device)
        self.recovery = Recovery(m.hidden_dim, 1, m.num_layer, m.module,
                                 monotone_head=bool(m.recovery_monotone_head and self.decay_hard)).to(device)
        self.supervisor = Supervisor(m.hidden_dim, max(1, m.num_layer - 1), m.module).to(device)
        self.condition_extractor = TEMConditionExtractor(T, m.cond_dim, m.inception_base_ch, m.condition_deep_dim,
                                                         m.condition_branch_dim).to(device)
        self.cond_head = ConditionSupervisionHead(m.cond_dim, stat_dim=len(PHYS_NAMES)).to(device)
        self.generator = BridgeResidualGenerator(T, m.cond_dim, m.z_dim, m.hidden_dim, m.dropout, m.gen_shared_dim,
                                                 m.gen_static_dim, m.gen_z_proj_dim, m.gen_temporal_hidden).to(device)
        self.profile_ctx = (ProfileDepthContext(m.cond_dim, m.z_dim, m.profile_code_dim, m.profile_ctx_hidden).to(device)
                            if P.enabled else None)
        self.discriminator = ConditionalDiscriminator(m.hidden_dim, m.cond_dim, T, m.disc_branch_ch).to(device)
        self.sequence_discriminator = (SequencePatchDiscriminator(T, m.cond_dim, m.disc_branch_ch).to(device)
                                       if R.use_dual_discriminator else None)
        self.profile_discriminator = ProfilePatchDiscriminator(T, m.profile_disc_ch).to(device) if P.critic else None
        self.cmg = _FinCMG(m.cond_dim, form=str(getattr(R, "cmg_form", "eaai"))).to(device) if R.cmg_enabled else None
        self.mixer = _FinFiLMMixer(m.hidden_dim, m.cond_dim).to(device) if R.cmg_enabled else None
        t = cfg.train
        self.opt_er = torch.optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=t.lr_er)
        self.opt_s = torch.optim.Adam(self.supervisor.parameters(), lr=t.lr_s)
        self.opt_c = torch.optim.Adam(list(self.condition_extractor.parameters()) + list(self.cond_head.parameters()),
                                      lr=1e-3, weight_decay=1e-5)
        self.opt_g = torch.optim.Adam(self._g_params(), lr=t.lr_g, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self._d_params(), lr=t.lr_d, betas=(0.0, 0.9)) if self._d_params() else None
        self.scaler_er, self.scaler_s, self.scaler_c = GradScaler(self.use_amp), GradScaler(self.use_amp), GradScaler(self.use_amp)
        self.scaler_g, self.scaler_d = GradScaler(self.use_amp), GradScaler(self.use_amp)
        self.ema = _EMA(self._g_modules(), t.ema_decay)
        self.skipped_steps = 0      # optimizer steps skipped for a non-finite gradient (all stages)
        self.history: List[Dict[str, Any]] = []
        self.global_epoch = 0
        self.stage_progress = {"stage0": 0, "stage0B": 0, "stage1A": 0, "stage1B": 0, "stage2": 0}
        self.best_stage_metric: Dict[str, float] = {}
        self.best_quality: Dict[str, Any] = {}
        self.checkpoint_dir = ensure_dir(os.path.join(cfg.export.model_dir, cfg.checkpoint.checkpoint_subdir))
        self.np_rng = np.random.default_rng(int(cfg.runtime.seed) + 17)
        self._prepare_data(keep_mask)

    # ------------------------------------------------------------------ parameters
    def _tcm(self, X: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Running minimum over time (axis dim) under the hard decay constraint, else X unchanged."""
        return torch.cummin(X, dim=dim).values if self.decay_hard else X

    def _g_modules(self) -> Dict[str, nn.Module]:
        return {n: getattr(self, n) for n in self.G_PATH if getattr(self, n) is not None}

    def _g_params(self) -> List[nn.Parameter]:
        ps: List[nn.Parameter] = []
        for mod in self._g_modules().values():
            ps += list(mod.parameters())
        return ps

    def _d_params(self) -> List[nn.Parameter]:
        ps = list(self.discriminator.parameters())
        for d in (self.sequence_discriminator, self.profile_discriminator):
            if d is not None:
                ps += list(d.parameters())
        return ps

    def parameter_report(self) -> Dict[str, Dict[str, int]]:
        return {n: count_parameters(getattr(self, n)) for n in (
            "embedder", "recovery", "supervisor", "condition_extractor", "generator", "profile_ctx", "discriminator",
            "sequence_discriminator", "profile_discriminator", "cmg", "mixer")}

    # ------------------------------------------------------------------ data on device
    def _prepare_data(self, keep_mask: Optional[np.ndarray]) -> None:
        cfg, dev, meas, sim = self.cfg, self.device, self.meas, self.sim
        # generation axis = the measured stations, plus the virtual stations of a profile extension
        self.survey = meas.survey or build_survey_spec("borehole", meas.pm, "auto", "default")
        self.depth_np = np.asarray(meas.gen_depth, dtype=np.float64)
        self.D = int(self.depth_np.size)
        self.meas_cols = np.nonzero(np.asarray(meas.gen_known, dtype=bool))[0]   # measured stations on the axis
        D, T = self.D, self.T
        if keep_mask is None:
            keep = np.asarray(meas.gen_known, dtype=bool).copy()
        else:
            keep = np.asarray(keep_mask, dtype=bool)        # cross-validation: on the measured axis (no extension)
        self.keep_mask = keep
        self.d0, self.d1 = float(self.depth_np[0]), float(self.depth_np[-1])
        # measured span: the sim registration, the anomaly width cap and the latent scales keep the measured stations'
        # statistics when a profile extension adds virtual stations (= d0, d1 without an extension)
        self.m0, self.m1 = float(meas.depth[0]), float(meas.depth[-1])
        self.spacing_med = float(np.median(np.diff(self.depth_np))) if D > 1 else 1.0
        mw = int(cfg.profile.max_window)
        self.max_window = mw if mw > 0 else (64 if self.survey.mode != "borehole" else 1 << 30)
        kd = self.depth_np[keep]
        kL = meas.L[keep[self.meas_cols]]
        kL_meas = kL
        # background_mode pure: the anchors are built on the smooth geological background of the known stations, so the
        # measured profile's own anomalies (a borehole's dip, the conductor peaks of a flight line) are in no anchor;
        # the measured profile only sets the anomaly statistics (amplitude range, footprint law) below
        self.bg_pure = str(cfg.profile.background_mode).strip().lower() == "pure"
        self.bg_sigma_m, self.purity_tol = float("nan"), float("nan")
        self.bg_info: Dict[str, Any] = {}
        if self.bg_pure:
            kmask = np.asarray(keep[self.meas_cols], dtype=bool)
            if meas.background is not None and meas.background_info is not None and bool(np.all(kmask)):
                kL, self.bg_info = np.asarray(meas.background, dtype=np.float64), meas.background_info
            else:   # cross-validation folds: background, scale and tolerance of the TRAINING stations / window only
                kL, self.bg_info = measured_background(meas if bool(np.all(kmask)) else meas.subset(kmask), cfg,
                                                       verbose=False)
            self.bg_sigma_m = float(self.bg_info["sigma_m"])
            self.purity_tol = float(self.bg_info["purity_tol_decades"])
        # anchor profile L_M(depth): PCHIP along depth through the KNOWN stations (all, or the CV training ones).
        # wherever a station is unknown every profile also draws its own departure from the rule, with the spread
        # calibrated by pseudo hold-out (calibrate_unmeasured; drawn in sample_profile_params).
        self.unmeasured = calibrate_unmeasured(kd, kL) if bool(np.any(~keep)) else None
        rules = ((self.unmeasured["low"]["rule"], self.unmeasured["high"]["rule"]) if self.unmeasured is not None
                 else ("clamp", "clamp"))
        Lfull = anchor_profile_on(kd, kL, self.depth_np, rules)        # (D, T) on all stations
        if meas.positive and self.decay_hard and bool(np.any(~keep)):
            # a rule continued per time sample can rise in time far from the known stations: a decay does not rise
            Lfull[~keep] = np.minimum.accumulate(Lfull[~keep], axis=1)
        self.L_anchor_np = Lfull
        if self.bg_pure:
            # the measured profile itself on the generation axis (QC diagnostics, the acceptance's roughness floor)
            Lfull_meas = anchor_profile_on(kd, kL_meas, self.depth_np, rules)
            if meas.positive and self.decay_hard and bool(np.any(~keep)):
                Lfull_meas[~keep] = np.minimum.accumulate(Lfull_meas[~keep], axis=1)
        else:
            Lfull_meas = Lfull
        self.L_meas_np = Lfull_meas
        # background_trend monotone: the per-profile regional modulation keeps the overall decay trend.  instead the
        # trend's contrast about its value at the middle of the measured span is scaled by exp(c0 + c1 (tau - 1/2)),
        # which keeps the direction of the decay along the profile.
        self.bg_trend = bool(self.bg_pure and (self.bg_info.get("trend") or {}).get("rule") in ("monotone", "measured"))
        self.bg_contrast_sigma = (0.0, 0.0)
        if self.bg_trend:
            zmid = 0.5 * (self.m0 + self.m1)
            Lmid = np.array([np.interp(zmid, self.depth_np, Lfull[:, j]) for j in range(Lfull.shape[1])])
            kn_ = meas.known_time_mask()
            half = float(np.median(np.maximum(np.abs(Lfull[0] - Lmid), np.abs(Lmid - Lfull[-1]))[kn_]))
            self.t_Lmid = torch.tensor(Lmid, dtype=torch.float32, device=dev)
            self.bg_contrast_sigma = (0.5 * float(cfg.profile.bg_level_sigma) / max(half, 1e-6),
                                      0.5 * float(cfg.profile.bg_tilt_sigma) / max(half, 1e-6))
            self.t_bg_down = torch.tensor(Lfull[-1] <= Lfull[0], dtype=torch.bool, device=dev)   # trend direction per time
        sp_unk = float(cfg.profile.unmeasured_spread)
        self.unk_gaps: List[Tuple[float, float]] = []
        self.n_unk_eps = 0
        self.ext_sides: Tuple[str, ...] = ()
        if self.unmeasured is not None and sp_unk > 0:
            um = self.unmeasured
            kidx = np.nonzero(keep)[0]
            self.unk_gaps = [(float(self.depth_np[a]), float(self.depth_np[b])) for a, b in zip(kidx[:-1], kidx[1:])
                             if b - a > 1]
            self.n_unk_eps = 2 + len(self.unk_gaps)
            self.unk_edges = (float(kd[0]), float(kd[-1]))
            self.ext_sides = tuple(sd for sd, on in (("low", bool(np.any(self.depth_np < kd[0]))),
                                                     ("high", bool(np.any(self.depth_np > kd[-1])))) if on)
            self.t_unk_a = torch.tensor(sp_unk * np.stack([um["low"]["a"], um["high"]["a"], um["gap"]["a"]]),
                                        dtype=torch.float32, device=dev)                    # (3, T) decades / m^gamma
            self.unk_gamma = (float(um["low"]["gamma"]), float(um["high"]["gamma"]), float(um["gap"]["gamma"]))
            print(f"[UNMEASURED] {int((~keep).sum())} stations without a measurement on the generation axis | rules "
                  f"low {rules[0]} / high {rules[1]} / gaps pchip (pseudo hold-out RMS {um['low']['rms_decades']:.4f} / "
                  f"{um['high']['rms_decades']:.4f} / {um['gap']['rms_decades']:.4f} decades) | spread x{sp_unk:g}, "
                  f"gamma {self.unk_gamma[0]:.2f}/{self.unk_gamma[1]:.2f}/{self.unk_gamma[2]:.2f}, {len(self.unk_gaps)} gap(s)")
        # late-time extension (production --extend_time_ms, or the time-direction CV fold): grid nodes after the last
        # known time follow the calibrated rule; each profile draws its own departure a dlog t^gamma
        self.time_known = meas.known_time_mask()
        self.kt = int(self.time_known.sum())        # known time window = grid[:kt] (the extension follows it)
        # soft decay: the measured gate-pair rise tolerance per grid step (the grid is finer than the gates)
        lg_ = np.log10(np.asarray(meas.grid_ms, dtype=np.float64))
        lgg_ = np.log10(np.asarray(meas.pm.time_ms, dtype=np.float64))
        st_ = float(np.median(np.diff(lg_))) / max(float(np.median(np.diff(lgg_))) if lgg_.size > 1 else 1.0, 1e-12)
        self.decay_tol_n = float(self.decay_tol * min(st_, 1.0) / max(float(self.norm.max_val - self.norm.min_val), 1e-9))
        kt = self.kt
        self.t_time_spread = None
        if meas.late_ext is not None and sp_unk > 0:
            dtl = np.clip(np.log10(meas.grid_ms) - math.log10(meas.t_known_ms), 0.0, None)
            spread_t = sp_unk * float(meas.late_ext["a"]) * np.power(dtl, float(meas.late_ext["gamma"]))
            self.t_time_spread = torch.tensor(spread_t, dtype=torch.float32, device=dev)
            print(f"[LATE-EXT] {int((~self.time_known).sum())} grid nodes after {meas.t_known_ms:.6g} ms follow rule "
                  f"{meas.late_ext['rule']} | per-profile spread a dlog t^g (a={meas.late_ext['a']:.4f}, "
                  f"g={meas.late_ext['gamma']:.2f}) x{sp_unk:g}: {spread_t[-1]:.4f} decades at {meas.grid_ms[-1]:.6g} ms")
        slopes = pchip_slopes(self.depth_np, Lfull.T).T                # (D, T) d L / d depth
        self.t_depth = torch.tensor(self.depth_np, dtype=torch.float32, device=dev)
        self.t_Lk = torch.tensor(Lfull, dtype=torch.float32, device=dev)
        self.t_mk = torch.tensor(slopes, dtype=torch.float32, device=dev)
        # the measured anomalies as labelled anchor anomalies (pure mode): templates from the residual of the KNOWN
        # stations (CV folds: their own background and inventory), on the generation axis by PCHIP inside the known
        # span, zero beyond it synthetic anomaly amplitude range (borehole physical: from the visibility floor) and the
        # smallest anomaly amplitude the purity tolerance and the acceptance refer to
        self.anom_family = self.survey.anomaly_family
        self.anom_phys = borehole_physical_anomalies(cfg, self.anom_family)
        L_amp = (kL_meas if self.bg_pure else Lfull[keep])[:, :kt]
        self.anom_test_amp = anomaly_amp_range(L_amp, kd, cfg.profile.anomaly_amp_range, family=self.anom_family,
                                               t_ms=meas.grid_ms[:kt])[0]
        self.anom_amp = anomaly_amp_range(L_amp, kd, cfg.profile.anomaly_amp_range, family=self.anom_family,
                                          t_ms=meas.grid_ms[:kt], visible=self.anom_phys)
        self.n_ma, self.ma_runs, self.ma_prob, self.ma_scale = 0, [], 0.0, [(1.0, 1.0)]
        self.t_MA = self.t_mMA = None
        if self.bg_pure and float(cfg.profile.measured_anomaly_prob) > 0:
            runs = list((self.bg_info.get("measured_anomalies") or {}).get("anomalies", []))
            if runs:
                Ak = measured_anomaly_templates(kL_meas - kL, kd, runs, self.bg_sigma_m)
                inside = (self.depth_np >= kd[0] - 1e-9) & (self.depth_np <= kd[-1] + 1e-9)
                Af = np.stack([anchor_profile_on(kd, Ak[i], self.depth_np, ("clamp", "clamp")) * inside[:, None]
                               for i in range(len(runs))])
                mA = np.stack([pchip_slopes(self.depth_np, Af[i].T).T for i in range(len(runs))])
                self.t_MA = torch.tensor(Af, dtype=torch.float32, device=dev)            # (n, D, T)
                self.t_mMA = torch.tensor(mA, dtype=torch.float32, device=dev)
                self.n_ma, self.ma_runs = len(runs), runs
                self.ma_prob = float(min(max(cfg.profile.measured_anomaly_prob, 0.0), 1.0))
                # one factor range per measured anomaly (auto: a slight one may reach twice the visibility floor)
                self.ma_scale = measured_anomaly_scale_ranges(runs, cfg.profile.measured_anomaly_scale, self.anom_amp[0])
                # geometry of each measured anomaly (known stations): centre = the station of its largest |value| over
                # the times (envelope maximum), support = the stations where its template is not 0, width = FWHM / 2.355
                # at its peak time; for the calibrated synthetic law and the measured_like kind
                env = np.abs(Ak).max(axis=2)                                               # (n, n_known)
                # support: the nonzero stations and the zero station on each side (the template falls to 0 there; at a
                # run's end the residual is still ~0.02 decades)
                sup = []
                for i in range(len(runs)):
                    nz = np.nonzero(env[i] > 0)[0]
                    sup.append(np.arange(max(int(nz[0]) - 1, 0), min(int(nz[-1]) + 2, kd.size)) if nz.size else nz)
                self.ma_center = np.array([float(kd[int(np.argmax(env[i]))]) for i in range(len(runs))])
                self.ma_support = [(float(kd[s_[0]]), float(kd[s_[-1]])) if s_.size else (c_, c_)
                                   for s_, c_ in zip(sup, self.ma_center)]
                self.ma_sigma = np.array([template_width_sigma(Ak[i], kd) for i in range(len(runs))])
                self.ma_peak = np.array([float(np.abs(Ak[i][:, :kt]).max()) for i in range(len(runs))])
                self.t_ma_c = torch.tensor(self.ma_center, dtype=torch.float32, device=dev)
                self.t_ma_lo = torch.tensor([s_[0] for s_ in self.ma_support], dtype=torch.float32, device=dev)
                self.t_ma_hi = torch.tensor([s_[1] for s_ in self.ma_support], dtype=torch.float32, device=dev)
                self.t_ma_sigma = torch.tensor(np.nan_to_num(self.ma_sigma, nan=float(self.spacing_med)),
                                               dtype=torch.float32, device=dev)
                self.t_ma_peak = torch.tensor(np.maximum(self.ma_peak, 1e-9), dtype=torch.float32, device=dev)
        self.t_logt = torch.tensor(np.log10(meas.grid_ms), dtype=torch.float32, device=dev)
        # normalised log time of the MEASURED window (0 .. 1; > 1 on a late-time extension): latent time basis and the
        # background tilt pivot keep their meaning on the measured gates
        tau = (np.log10(meas.grid_ms) - np.log10(meas.grid_ms[0])) / max(np.log10(meas.grid_ms[kt - 1] / meas.grid_ms[0]), 1e-12)
        J = int(cfg.profile.latent_basis)
        psi = np.stack([np.ones(T)] + [np.cos(np.pi * j * tau) for j in range(1, J)], axis=0)
        self.t_psi = torch.tensor(psi, dtype=torch.float32, device=dev)
        self.t_wseg = torch.tensor(segment_weights(meas.grid_ms, cfg.losses.lambda_late, meas.t_known_ms), device=dev)
        wsh = segment_weights(meas.grid_ms[1:], cfg.losses.lambda_late, meas.t_known_ms)
        self.t_wseg_d = torch.tensor(wsh, device=dev)
        # physics descriptors are evaluated on the measured gate times (the measured standard): the grid
        # points between gates are PCHIP interpolation for the measured curves and must not be compared as data
        gi = meas.gate_index
        # only the gates inside the known time window (all of them, unless the time-direction CV hides the last)
        self.gate_cols = gi[self.time_known[gi]] if bool(np.all(gi >= 0)) else np.nonzero(self.time_known)[0]
        self.t_gate_cols = torch.tensor(self.gate_cols, dtype=torch.long, device=dev)
        self.phys = PhysicsDescriptor(meas.grid_ms[self.gate_cols])
        # windows allowed: W consecutive stations that are all known (CV keeps held-out blocks out)
        self._window_starts: Dict[int, np.ndarray] = {}
        # sim on device background_mode pure: the simulated structure is not used (load_data_contract does not pass it
        # either; this keeps a directly constructed trainer on the same contract)
        self.has_sim = bool(sim.available) and cfg.data.data_name in ("combined", "combined+sine") and not self.bg_pure
        if self.has_sim:
            self.t_sim_depth = torch.tensor(sim.depth, dtype=torch.float32, device=dev)
            self.t_sim_anom = torch.tensor(sim.anomaly, dtype=torch.float32, device=dev)
            self.t_sim_L = (torch.tensor(sim.L_native_grid + sim.level_shift[None, :], dtype=torch.float32, device=dev)
                            if sim.level_mode != "structure" else None)
            self.t_cov = torch.tensor(sim.coverage, dtype=torch.float32, device=dev)
            self.sim_s0, self.sim_s1 = float(sim.depth[0]), float(sim.depth[-1])
        # latent depth scales anchored to the measured span / spacing
        span = max(self.m1 - self.m0, 1e-6)
        dmed = max(self.spacing_med, 1e-6)
        self.latent_bands = [(1.0 * span, 4.0 * span), (6.0 * dmed, max(0.5 * span, 8.0 * dmed)),
                             (2.5 * dmed, 6.0 * dmed)]
        self.lat_law: Optional[Dict[str, Any]] = None
        if self.survey.anomaly_family == "buried":
            # a line survey varies along the line on its own lateral correlation length (e.g. 409 m = 18 stations);
            # latent wavelengths down to 6 spacings gave the network's departure a station-scale texture the measured
            # line does not have: the mid band starts at the measured correlation length
            self.lat_law = lateral_correlation_law((kL_meas if self.bg_pure else Lfull[keep])[:, :kt], kd, meas.grid_ms[:kt])
            mid_lo = max(6.0 * dmed, float(self.lat_law["l1_m"]))
            self.latent_bands[1] = (mid_lo, max(0.5 * span, 8.0 * dmed, 1.5 * mid_lo))
            print(f"[LATENT] line survey: latent wavelengths long {self.latent_bands[0][0]:.0f}-{self.latent_bands[0][1]:.0f}, "
                  f"mid {self.latent_bands[1][0]:.0f}-{self.latent_bands[1][1]:.0f} (from the measured lateral correlation "
                  f"length {self.lat_law['l1_m']:.0f}; 6 spacings = {6.0 * dmed:.0f})")
        self.latent_probs = [float(x) for x in cfg.profile.latent_scale_probs.split(",")]
        self.gen_alpha_mix = []
        for k, w in _parse_mix(cfg.profile.gen_alpha_mix, "0.85-1.0:1.0"):
            lo, _, hi = k.partition("-")
            self.gen_alpha_mix.append((float(lo), float(hi or lo), w))
        dd = np.diff(self.depth_np) if D > 1 else np.ones(1)
        sp_st = np.concatenate([dd[:1], 0.5 * (dd[1:] + dd[:-1]), dd[-1:]]) if D > 2 else np.full(D, float(dd[0]))
        self.t_spacing = torch.tensor(sp_st, dtype=torch.float32, device=dev)
        self.t_tau = torch.tensor(tau, dtype=torch.float32, device=dev)
        # random anomaly composition (count / amplitude / depth / width / time window), anchored to the measured
        Pc = cfg.profile
        self.anom_enabled = bool(Pc.anomaly_enabled)
        # measured_anomaly_mode count: the measured anomalies are counted with the synthetic ones (a profile with K
        # anomalies carries m measured + K - m synthetic, K <= max_anomalies)
        self.ma_mode = measured_anomaly_mode(cfg, self.anom_family)
        self.ma_count = bool(self.n_ma > 0 and self.ma_mode == "count")
        cnt = anomaly_count_law(Pc.anomaly_count_probs, Pc.max_anomalies, self.ma_mode)
        self.anom_kmax = max(1, max(k for k, _ in cnt))
        pv = np.zeros(self.anom_kmax + 1)
        for k_, w_ in cnt:
            pv[k_] += w_
        self.t_anom_p = torch.tensor(pv / max(pv.sum(), 1e-12), dtype=torch.float32, device=dev)
        # borehole_anomaly_model measured: the physical law calibrated on the measured anomalies (when there are any)
        self.anom_meas = bool(self.anom_phys and str(Pc.borehole_anomaly_model).strip().lower() == "measured")
        self.anom_cal = bool(self.anom_meas and self.n_ma > 0)
        self._morph_bank_plan()          # the morphology kind draws from the morphology bank
        if self.anom_cal:
            # |peak| up to the strongest measured anomaly at the top of the measured factor range (x 2), at least 4 x
            # the visibility floor: synthetic anomalies as strong as the measured ones can be shown, not 3 x stronger
            self.anom_amp = (float(self.anom_amp[0]), float(max(2.0 * float(np.max(self.ma_peak)), 4.0 * self.anom_amp[0])))
        # anomaly family of the survey: "borehole" or "buried" (airborne / ground lines: conductors under the line,
        # footprint anchored to the measured line's lateral correlation length at the anomaly's own time scale, so later
        # = deeper = wider); self.anom_family / anom_amp / anom_test_amp are set with the measured anomalies above
        self.anom_cap = 1.5 * self.anom_amp[1]            # overlapping anomalies: total capped at 1.5 x the peak range
        if self.anom_phys:
            # physical borehole anomalies: kind shares, the shallowest source depth, the tau-distance law's reference; +
            # measured_like (measured model with measured anomalies)
            spec_k = str(Pc.borehole_anomaly_kinds or "auto").strip().lower()
            if spec_k in ("", "auto"):
                # with the morphology bank the morphology kind is drawn without measured anomalies too (blind)
                spec_k = ("measured_like:0.50,intersected:0.35,off_hole:0.15,zone:0" if self.anom_cal
                          else "measured_like:0.40,intersected:0.20,off_hole:0.30,zone:0.10" if self.anom_bank_on
                          else "intersected:0.35,off_hole:0.50,zone:0.15")
            spec_k = spec_k.replace("morphology:", "measured_like:")
            kp = {str(k_).strip().lower(): float(w_) for k_, w_ in _parse_mix(spec_k, "intersected:0.35,off_hole:0.50,zone:0.15")}
            names = ("intersected", "off_hole", "zone", "measured_like")
            unknown = sorted(set(kp) - set(names))
            pk = np.array([max(kp.get(n_, 0.0), 0.0) for n_ in names])
            if unknown or pk.sum() <= 0:
                raise ValueError(f"--borehole_anomaly_kinds expects shares of {', '.join(names)} (e.g. "
                                 f"intersected:0.35,off_hole:0.5,zone:0.15), got {Pc.borehole_anomaly_kinds!r}")
            if pk[3] > 0 and not (self.anom_cal or self.anom_bank_on):
                raise ValueError("--borehole_anomaly_kinds measured_like needs --borehole_anomaly_model measured and measured "
                                 "anomalies in the profile's inventory (background_mode pure, measured_anomaly_prob > 0)")
            self.anom_kind_p = pk / pk.sum()
            self.anom_zmin = max(float(Pc.anomaly_min_depth_m), 0.0)
            self.anom_r_ref = float(self.spacing_med)
        self.anom_neg_frac = float(Pc.anomaly_neg_frac)
        wr = [float(x) for x in str(Pc.anomaly_width_range).split(",")]
        self.anom_w_lo_sp = float(wr[0])
        self.anom_w_hi = float(wr[1]) * max(self.m1 - self.m0, 1e-6)
        if self.anom_cal:
            # sigma / distance up to twice the widest measured anomaly
            self.anom_w_hi = float(min(self.anom_w_hi, 2.0 * float(np.nanmax(self.ma_sigma))))
        self.anom_ltau = anomaly_tau_range_log10(meas.grid_ms[:kt], Pc.anomaly_tau_range)
        self.anom_bb_frac = float(Pc.anomaly_broadband_frac)
        # calibrated model: no footprint widening in time - the measured anomalies keep their width along the decay
        # (Liuyuan 184 m: FWHM 11-13 m from 2.9 to 11 ms; 220 m: 20 m at 3.5 ms, 14 m at 16 ms)
        self.anom_diffusion = bool(Pc.anomaly_diffusion) and not self.anom_cal
        # measured model: every labelled anomaly keeps its designed shape (compose_anchor: L_baseM = the anchor)
        self.anom_fixed = bool(self.anom_meas and self.bg_pure and bool(cfg.profile.pure_departure_head))
        self.t_time_ms = torch.tensor(meas.grid_ms, dtype=torch.float32, device=dev)
        self.bg_enabled = bool(Pc.bg_level_sigma > 0 or Pc.bg_tilt_sigma > 0)
        if self.anom_family == "buried":
            fr = [float(x) for x in str(Pc.anomaly_footprint_range).split(",")]
            if len(fr) != 2 or not 0.0 < fr[0] <= fr[1]:
                raise ValueError(f"--anomaly_footprint_range expects 'lo,hi' with 0 < lo <= hi, got {Pc.anomaly_footprint_range!r}")
            self.anom_fp = (fr[0], fr[1])
        self.purity_sigma_m, self.purity_resp, self.purity_thr = float("nan"), float("nan"), float("nan")
        # half-width of the label weight's time smoothing, in samples of the (log-spaced) time grid
        lt_g = np.log10(np.asarray(meas.grid_ms, dtype=np.float64))
        dl_g = float(np.median(np.diff(lt_g))) if lt_g.size > 1 else 1.0
        self.label_k = int(max(1, round(LABEL_WEIGHT_SMOOTH_DECADES / max(dl_g, 1e-9))))
        if self.bg_pure:
            # the network's departure from its anchor is regional: smoother than the background scale and than the
            # widest labelled anomaly at onset, so no departure can mimic a label.  Its anomaly-scale content is tested
            # with a local-quadratic fit at sigma_R = max(2 sigma_B, widest footprint)
            if self.anom_family == "buried":
                lw = self.lat_law
                t_hi = float(min(10.0 ** self.anom_ltau[1], meas.grid_ms[kt - 1]))
                w_on = self.anom_fp[1] * float(lw["l1_m"]) * (t_hi / float(lw["t1_ms"])) ** float(lw["beta"])
                w_on = min(w_on, 0.25 * max(self.m1 - self.m0, 1e-6))
            else:
                w_on = float(self.anom_w_hi)
            self.purity_sigma_m = float(max(2.0 * self.bg_sigma_m, w_on))
            Sr = local_poly_smoother_np(self.depth_np, self.purity_sigma_m)
            ic = int(np.argmin(np.abs(self.depth_np - 0.5 * (self.depth_np[0] + self.depth_np[-1]))))
            g_b = np.exp(-0.5 * ((self.depth_np - self.depth_np[ic]) / self.bg_sigma_m) ** 2)
            self.purity_resp = float((g_b - Sr @ g_b)[ic])
            self.purity_thr = float(self.purity_tol * self.purity_resp)
            lo_b = 2.0 * math.pi * self.purity_sigma_m
            mlo, mhi = self.latent_bands[1]
            if lo_b > mlo:
                self.latent_bands[1] = (lo_b, max(mhi, 1.5 * lo_b))
            print(f"[BACKGROUND] anchors on the measured background (sigma_B {self.bg_sigma_m:.4g} m) | departure test at "
                  f"sigma_R {self.purity_sigma_m:.4g} m (2 sigma_B {2.0 * self.bg_sigma_m:.4g}, widest footprint at onset "
                  f"{w_on:.4g}) | tolerance {self.purity_tol:.4f} decades peak x response {self.purity_resp:.3f} to a "
                  f"sigma_B bump = threshold {self.purity_thr:.4f} | latent wavelengths long {self.latent_bands[0][0]:.0f}-"
                  f"{self.latent_bands[0][1]:.0f}, mid {self.latent_bands[1][0]:.0f}-{self.latent_bands[1][1]:.0f} m | "
                  f"simulated structure {'used' if self.has_sim else 'not used'}")
            if self.n_ma:
                print(f"[BACKGROUND] {self.n_ma} measured anomalies kept as labelled anchor anomalies (measured shape, "
                      f"amplitude factor log-uniform per anomaly ({cfg.profile.measured_anomaly_scale}), "
                      + ("[v20.2] each one of the profile's anomalies: m of them uniform over 0 .. min(count, "
                         f"{self.n_ma}), which ones at random" if getattr(self, "ma_count", False) else
                         f"probability {self.ma_prob:g} in every profile with anomalies; slot 1 of the anomaly count")
                      + "): " + "; ".join(
                          f"{r_['from_m']:.0f}-{r_['to_m']:.0f} m ({r_['peak_decades']:+.3f} at {r_['peak_time_ms']:.3g} ms,"
                          f" x {s_[0]:g}-{s_[1]:.3g})" for r_, s_ in zip(self.ma_runs[:6], self.ma_scale[:6]))
                      + (" ..." if self.n_ma > 6 else ""))
        if self.anom_enabled:
            if self.anom_family == "buried":
                lw = self.lat_law
                wtxt = (f"sigma0 {self.anom_fp[0]:g}-{self.anom_fp[1]:g} x the measured lateral correlation length "
                        f"l(tau) = {lw['l1_m']:.0f} m (t/{lw['t1_ms']:.3g} ms)^{lw['beta']:.2f}")
            else:
                wtxt = f"sigma {self.anom_w_lo_sp:g} x local spacing .. {self.anom_w_hi:.1f} m"
            if self.anom_cal:
                kp_ = self.anom_kind_p
                print(f"[ANOMALY] family borehole, physical law calibrated on the {self.n_ma} measured anomalies (v20.2) | "
                      f"count probs {np.round(pv / max(pv.sum(), 1e-12), 3).tolist()} (measured + synthetic"
                      + (f", m measured uniform over 0 .. min(count, {self.n_ma})" if self.ma_count else
                         ", every measured anomaly in every profile with anomalies") + f") | kinds: measured_like "
                      f"{kp_[3]:.2f} (a measured shape at another depth, size x {MEASURED_LIKE_SIZE_RANGE[0]:g}-"
                      f"{MEASURED_LIKE_SIZE_RANGE[1]:g}), hole through a conductor {kp_[0]:.2f}, beside the hole "
                      f"{kp_[1]:.2f}, near-hole zone {kp_[2]:.2f} | |peak| {self.anom_amp[0]:.3f}-{self.anom_amp[1]:.3f} "
                      f"decades log-uniform (floor = visibility, top = max(2 x the strongest measured "
                      f"{float(np.max(self.ma_peak)):.3f}, 4 x floor)) | sigma / distance {self.anom_w_lo_sp:g} x local "
                      f"spacing .. {self.anom_w_hi:.1f} m (2 x the widest measured, sigma "
                      + "/".join(f"{v_:.1f}" for v_ in self.ma_sigma) + " m), no widening in time | tau "
                      f"{10 ** self.anom_ltau[0]:.3g}-{10 ** self.anom_ltau[1]:.3g} ms, >= {10 ** self.anom_ltau[0]:.3g} "
                      f"ms x (r / {self.anom_r_ref:.3g} m)^2 | centres >= {self.anom_zmin:g} m + core, cores (+-"
                      f"{SEPARATION_CORE_SIGMAS:g} widths) apart from the profile's other anomalies")
            elif self.anom_phys:
                kp_ = self.anom_kind_p
                print(f"[ANOMALY] family borehole, physical (v20) | count probs "
                      f"{np.round(pv / max(pv.sum(), 1e-12), 3).tolist()} | kinds: hole through a conductor {kp_[0]:.2f} "
                      f"(single positive peak), conductor beside the hole {kp_[1]:.2f} (axial dipole field, dip "
                      f"-90..90 deg), near-hole zone {kp_[2]:.2f} (all times) | |peak| {self.anom_amp[0]:.3f}-"
                      f"{self.anom_amp[1]:.3f} decades log-uniform (visibility floor = the measured profile's strongest "
                      f"local feature; share {self.anom_neg_frac:.2f} negative zones / flipped dipole kernels) | distance r "
                      f"{self.anom_w_lo_sp:g} x local spacing .. {self.anom_w_hi:.1f} m log-uniform | tau "
                      f"{10 ** self.anom_ltau[0]:.3g}-{10 ** self.anom_ltau[1]:.3g} ms, >= {10 ** self.anom_ltau[0]:.3g} "
                      f"ms x (r / {self.anom_r_ref:.3g} m)^2 | centres >= {self.anom_zmin:g} m + r | response relative "
                      f"to the measured background decay, footprint widening sqrt(1 + t / tau) (hole through / zone)")
            else:
                print(f"[ANOMALY] family {self.anom_family} | count probs {np.round(pv / max(pv.sum(), 1e-12), 3).tolist()} | "
                      f"|peak| {self.anom_amp[0]:.3f}-{self.anom_amp[1]:.3f} decades (log-uniform, negative share "
                      f"{self.anom_neg_frac:.2f}) | {wtxt} | "
                      f"tau {10 ** self.anom_ltau[0]:.3g}-{10 ** self.anom_ltau[1]:.3g} ms | time-flat share "
                      f"{self.anom_bb_frac:.2f} | diffusion broadening {self.anom_diffusion}")
        if self.anom_bank_on:
            self._build_morphology_bank()
        # union reference for descriptors / envelopes: measured stations + canonical sim-structured curves +
        # composed anchors (with their random anomalies), so the physics envelope admits the intended diversity
        ref = [Lfull[keep]] + ([kL_meas] if self.bg_pure else [])
        with torch.no_grad():
            if self.has_sim:
                b = self.build_profile_batch(n_windows=8, W=self.D, mode="canonical", station_jitter=False)
                ref.append(L_from_norm(b["y_S"].reshape(-1, T).cpu().numpy(), self.norm))
            if self.anom_enabled or self.bg_enabled:
                g_ref = torch.Generator(device=dev)
                g_ref.manual_seed(int(cfg.runtime.seed) + 77)
                nref = int(UNION_REFERENCE_PROFILES_MEASURED if getattr(self, "anom_meas", False)
                           else UNION_REFERENCE_PROFILES)
                b = self.build_profile_batch(n_windows=nref, W=self.D, mode="gen", station_jitter=False, gen=g_ref,
                                             starts=np.zeros(nref, dtype=np.int64))
                ref.append(L_from_norm(b["y_A"].reshape(-1, T).cpu().numpy(), self.norm))
        self.L_union_np = np.concatenate(ref, axis=0)
        fu = self.phys.numpy(self.L_union_np[:, self.gate_cols])
        self.phys_mu = np.median(fu, axis=0)
        sd = 1.4826 * np.median(np.abs(fu - self.phys_mu), axis=0)
        self.phys_sd = np.maximum(sd, 0.05 * (np.percentile(fu, 95, axis=0) - np.percentile(fu, 5, axis=0)) + 1e-3)
        z = (fu - self.phys_mu) / self.phys_sd
        self.t_phys_mu = torch.tensor(self.phys_mu, dtype=torch.float32, device=dev)
        self.t_phys_sd = torch.tensor(self.phys_sd, dtype=torch.float32, device=dev)
        self.t_phys_ref = torch.tensor(z, dtype=torch.float32, device=dev)
        lo, hi = np.percentile(z, 0.5, axis=0), np.percentile(z, 99.5, axis=0)
        mg = 0.25 * (hi - lo) + 0.25
        self.t_phys_lo = torch.tensor(lo - mg, dtype=torch.float32, device=dev)
        self.t_phys_hi = torch.tensor(hi + mg, dtype=torch.float32, device=dev)
        if z.shape[0] > 4096:   # the spacing from a subsample (the reference grew with the anomaly law's variety)
            zs_ = z[np.random.default_rng(int(cfg.runtime.seed) + 78).choice(z.shape[0], 4096, replace=False)]
            r = _nn_dist(zs_, zs_, block=512, exclude_self=True)
        else:
            r = _nn_dist(z, z, exclude_self=True) if z.shape[0] > 1 else np.ones(1)
        self.phys_radius = float(max(2.0 * np.median(r), 1.0))   # >= 1 robust sd: tolerance, not the station spacing
        yu = norm_from_L(self.L_union_np, self.norm)
        span_y = yu.max(0) - yu.min(0)
        self.t_env_lo = torch.tensor(yu.min(0) - 0.25 * span_y - 0.02, dtype=torch.float32, device=dev)
        self.t_env_hi = torch.tensor(yu.max(0) + 0.25 * span_y + 0.02, dtype=torch.float32, device=dev)
        # profile local-structure floor (normalised units) from the measured profile pure: from the measured profile
        # itself (the background has almost no local structure: a floor from it would make the roughness term's
        # normaliser vanish; anomaly-scale content is bounded by the purity term)
        lsr = np.abs(local_structure_residual(norm_from_L((kL_meas if self.bg_pure else Lfull[keep])[:, :kt],
                                                          self.norm).astype(np.float64), kd))
        self.rough_floor = float(np.percentile(lsr, 50)) if lsr.size else 0.0
        # structure-fidelity threshold = the measured local-structure p95 (a barely visible narrow feature, the rule of the
        # lower anomaly amplitude): weaker anchor structure is not tested by the projection
        self.struct_thr_n = float(np.percentile(lsr, 95)) if lsr.size else 0.0
        self.struct_thr = self.struct_thr_n * float(self.norm.max_val - self.norm.min_val)
        self.div_target = float(QC_GATES["profile_diversity_min_decades"] / max(self.norm.max_val - self.norm.min_val, 1e-6))
        # per-profile acceptance: local depth-structure weights of the stations and the measured band floor
        if D >= 3:
            wls = (self.depth_np[1:-1] - self.depth_np[:-2]) / np.maximum(self.depth_np[2:] - self.depth_np[:-2], 1e-12)
        else:
            wls = np.zeros(0)
        self.t_ls_w = torch.tensor(wls, dtype=torch.float32, device=dev).view(1, -1, 1)
        self.t_rough_meas_band = self._band_rough(
            (torch.tensor(self.L_meas_np, dtype=torch.float32, device=dev) if self.bg_pure else self.t_Lk).unsqueeze(0))[0]
        self.copy_min = float(cfg.profile.copy_min_dist)
        self._prepare_manifold(Lfull, keep)

    def _prepare_manifold(self, Lfull: np.ndarray, keep: np.ndarray) -> None:
        """The manifold constraints on the background: FAM - the field-admissible manifold of the known stations'
        background anchor curves (FieldAdmissibleManifold); always built, so its occupancy is reported for every run;
        with fam_enabled the generator is trained against its shell (lambda_field_shell) and a profile passes only if
        >= fam_min_frac of its background stations (no labelled anomaly above a quarter of the purity tolerance,
        measured and inside the measured span) are admissible."""
        cfg, dev, Pc = self.cfg, self.device, self.cfg.profile
        self.fam: Optional[FieldAdmissibleManifold] = None
        self.fam_on = bool(Pc.fam_enabled)
        try:
            self.fam = FieldAdmissibleManifold(self.phys.numpy(np.asarray(Lfull)[keep][:, self.gate_cols]),
                                               k=int(Pc.fam_k), kappa=float(Pc.fam_kappa), device=dev)
        except ValueError as e:
            print(f"[FAM] not built ({e}); field-shell term and admissibility gate off")
            self.fam_on = False
        tol = PURITY_ANCHOR_SHARE * self.purity_tol if math.isfinite(self.purity_tol) else 0.25 * float(self.anom_test_amp)
        self.fam_bg_tol = float(max(tol, 1e-6))   # decades: a station off every labelled anomaly
        # the FAM is the manifold of the known stations' curves and judges those only: inside the measured span and
        # measured.
        self.t_fam_span = torch.tensor((self.depth_np >= self.m0 - 1e-9) & (self.depth_np <= self.m1 + 1e-9)
                                       & np.asarray(keep, dtype=bool), dtype=torch.bool, device=dev)
        self.fam_all_known = bool(self.t_fam_span.all())
        if self.fam is not None:
            s_ = self.fam.summary()
            on = self.fam_on and float(cfg.losses.lambda_field_shell) > 0
            print(f"[FAM] field-admissible manifold: {s_['anchors']} background anchor curves x {s_['descriptors']} "
                  f"descriptors | k={s_['k']} kappa={s_['kappa']:g} | radius median {s_['radius_median']:.3f} "
                  f"[{s_['radius_min']:.3f}, {s_['radius_max']:.3f}] robust sd (floor {s_['r_floor']:.3f}) | leave-one-out "
                  f"occupancy {s_['loo_occupancy']:.3f} | shell term {'on' if on else 'off'} (lambda "
                  f"{float(cfg.losses.lambda_field_shell):g}), acceptance >= {float(Pc.fam_min_frac):.2f} of the background "
                  f"stations {'on' if self.fam_on else 'off (reported only)'} | background = |labels| <= {self.fam_bg_tol:.4f} dec")
        bm = str(Pc.background_manifold).strip().lower()
        self.bg_warp = bool(self.bg_trend and bool(Pc.pure_departure_head) and bm in ("auto", "warp")
                            and int(Pc.warp_modes) > 0 and float(Pc.warp_max_strain) > 0)
        self.warp_J = int(Pc.warp_modes) if self.bg_warp else 0
        self.warp_strain = float(min(max(float(Pc.warp_max_strain), 0.0), 0.95))
        # MAP ridge of the mode fit: departures below the purity tolerance (normalised) explain no warp beyond the strain
        sig_n = (float(self.purity_tol) if math.isfinite(self.purity_tol) else float(self.anom_test_amp)) / max(
            float(self.norm.max_val - self.norm.min_val), 1e-9)
        self.warp_ridge = float((sig_n / max(self.warp_strain, 1e-6)) ** 2)
        if self.bg_warp:
            span = self.m1 - self.m0
            print(f"[WARP] background manifold = measured trend under monotone depth warps: {self.warp_J} modes, "
                  f"|du/dz| < {self.warp_strain:g} (largest shift {span * self.warp_strain / math.pi:.1f} m for mode 1 over "
                  f"the {span:.0f} m span), u = 0 at {self.m0:g} / {self.m1:g} m, + level / contrast per time (v20.1)")
        elif bm == "warp":
            print(f"[WARP] --background_manifold warp needs background_mode pure with a monotone trend and the pure "
                  f"departure head; the regional head keeps its v20 form (bg_trend={self.bg_trend})")

    # ------------------------------------------------------------------ anomaly-morphology prior
    def _morph_bank_plan(self) -> None:
        """Whether the morphology kind draws from the morphology bank, and the bank's source shares over the sources
        present (decided before the kind shares)."""
        Pc = self.cfg.profile
        spec = str(Pc.morphology_bank or "auto").strip().lower()
        sm = getattr(self.meas, "sim_morph", None)
        self.mb_n_sim = int(len(sm["sigma_m"])) if sm is not None else 0
        self.mb_mix = morphology_mix(Pc.morphology_mix, self.n_ma, self.mb_n_sim, max(int(Pc.morphology_learned), 0))
        ok = bool(self.anom_enabled and self.anom_meas and self.bg_pure and sum(self.mb_mix.values()) > 0)
        if spec == "bank" and not ok:
            raise ValueError("--morphology_bank bank needs background_mode pure, the borehole measured anomaly model and a "
                             "source (measured anomalies, a simulated profile or --morphology_learned > 0)")
        self.anom_bank_on = bool(spec in ("auto", "bank") and ok)
        self.morph_report: Dict[str, Any] = {}

    def _morph_bank_key(self) -> str:
        Pc = self.cfg.profile
        sm = getattr(self.meas, "sim_morph", None)
        src = {"bank": "v22.0.0", "seed": int(self.cfg.runtime.seed), "mix": self.mb_mix,
               "learned": int(Pc.morphology_learned), "novelty": float(Pc.morphology_novelty),
               "steps": int(Pc.morphology_vae_steps), "contract": self.contract.get("contract_key", ""),
               "keep": array_digest(np.asarray(self.keep_mask, dtype=np.float64)),
               "measured": [[float(r_["from_m"]), float(r_["to_m"])] for r_ in self.ma_runs],
               "measured_digest": array_digest(self.t_MA.cpu().numpy()) if self.t_MA is not None else "none",
               "sim": (sm or {}).get("digest", "none"),
               "law": [float(v) for v in (self.anom_amp[0], self.anom_amp[1], self.anom_w_lo_sp, self.anom_w_hi,
                                          self.anom_ltau[0], self.anom_ltau[1], self.anom_neg_frac, self.anom_r_ref)],
               "diffusion": bool(self.anom_diffusion), "kt": int(self.kt), "T": int(self.T), "D": int(self.D)}
        return hashlib.sha1(stable_json_dumps(src).encode("utf-8")).hexdigest()[:16]

    def _build_morphology_bank(self) -> None:
        """The run's morphology bank: <checkpoint dir>/morphology_bank.pt when its key matches (resumed and
        generation-only runs use the same bank), else built (MorphologyPrior) and saved there."""
        key = self._morph_bank_key()
        path = os.path.join(self.checkpoint_dir, "morphology_bank.pt")
        bank = None
        if os.path.isfile(path):
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if ck.get("key") == key:
                    bank = ck
                    print(f"[MORPH] morphology bank reused: {path} (key {key})")
                else:
                    print(f"[MORPH] {path} belongs to other data / settings (key {ck.get('key')} vs {key}): rebuilt")
            except Exception as e:
                print(f"[MORPH] {path} not readable ({e}): rebuilt")
        if bank is None:
            t0 = time.time()
            bank = self._make_morphology_bank()
            bank["key"] = key
            bank["report"]["build_seconds"] = float(time.time() - t0)
            atomic_torch_save(bank, path)
        self._set_morph_bank(bank)

    def _morph_physics_kernels(self, n: int, gen: torch.Generator, uv: np.ndarray, Wt: np.ndarray) -> np.ndarray:
        """n physics kernels on the prior's patch grid (n, NU, NT): this run's analytic anomaly law (anomaly_fields:
        hole through a conductor, conductor beside the hole, near-hole zone in MORPH_PHYS_SHARES; |peak|, distance,
        time constant and dip drawn as for the profiles' own synthetic anomalies), relative to the measured
        background at a random depth of the profile, along the hole in units of the distance r (u = (z - z0) / r)."""
        dev = self.device
        u = torch.rand(n, 1, 7, device=dev, generator=gen)
        p0, p1 = MORPH_PHYS_SHARES[0], MORPH_PHYS_SHARES[0] + MORPH_PHYS_SHARES[1]
        kind = (u[..., 0] >= p0).long() + (u[..., 0] >= p1).long()
        lo, hi = self.anom_amp
        amp = torch.exp(math.log(lo) + u[..., 1] * (math.log(hi) - math.log(lo)))
        sign = torch.where(u[..., 2] < self.anom_neg_frac, -torch.ones_like(amp), torch.ones_like(amp))
        sign = torch.where(kind == 0, torch.ones_like(sign), sign)
        rlo = float(self.anom_w_lo_sp * self.spacing_med)
        rhi = float(max(self.anom_w_hi, 1.5 * rlo))
        r = torch.exp(math.log(rlo) + u[..., 3] * (math.log(rhi) - math.log(rlo)))
        z0 = self.d0 + u[..., 4] * (self.d1 - self.d0)
        lt0, lt1 = self.anom_ltau
        ltlo = (lt0 + 2.0 * torch.log10((r / max(self.anom_r_ref, 1e-9)).clamp_min(1.0))).clamp(max=max(lt1 - 0.5, lt0))
        P = {"an_A": sign * amp, "an_z0": z0, "an_w0": r, "an_ltau": ltlo + u[..., 5] * (lt1 - ltlo),
             "an_type": (kind == 2).float(), "an_kind": kind.float(), "an_dip": (2.0 * u[..., 6] - 1.0) * (0.5 * math.pi)}
        tu = torch.tensor(uv, dtype=torch.float32, device=dev).view(1, -1)
        out = []
        for i0 in range(0, n, 256):
            Pi = {k_: v_[i0:i0 + 256] for k_, v_ in P.items()}
            F_ = self.anomaly_fields(Pi["an_z0"] + Pi["an_w0"] * tu, Pi)[:, 0]               # (b, NU, T)
            out.append(torch.log10((1.0 + F_).clamp_min(1e-12)).double().cpu().numpy())
        return _normalise_patches(np.einsum("nut,bt->nub", np.concatenate(out), Wt))

    def _train_morph_prior(self, Xk: np.ndarray, Xe: np.ndarray, steps: int,
                           seed: int) -> Tuple["MorphologyVAE", List[List[float]]]:
        """The beta-VAE on physics kernels + measured / simulated examples (MORPH_EXAMPLE_SHARE of every batch, with
        along-hole scale x exp(+-0.1), mirror u -> -u and +-1 log-time node shifts), on its own random stream."""
        dev = self.device
        devs = list(range(torch.cuda.device_count())) if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(int(seed))
            vae = MorphologyVAE().to(dev)
            opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(steps), 1), eta_min=1e-4)
            Xp = torch.tensor(Xk, dtype=torch.float32, device=dev).unsqueeze(1)
            Xq = torch.tensor(Xe, dtype=torch.float32, device=dev).unsqueeze(1) if len(Xe) else None
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed) + 1)
            bs = 256
            ne = int(round(MORPH_EXAMPLE_SHARE * bs)) if Xq is not None else 0
            hist: List[List[float]] = []
            vae.train()
            for it in range(int(steps)):
                xb = Xp[torch.randint(0, Xp.shape[0], (bs - ne,), device=dev, generator=g)]
                if ne:
                    xe = Xq[torch.randint(0, Xq.shape[0], (ne,), device=dev, generator=g)]
                    r3 = torch.rand(ne, 3, device=dev, generator=g)
                    th = torch.zeros(ne, 2, 3, device=dev)
                    th[:, 0, 0] = 1.0
                    th[:, 0, 2] = (2.0 * r3[:, 0] - 1.0) * (2.0 / (MORPH_VAE_NT - 1))
                    th[:, 1, 1] = torch.where(r3[:, 1] < 0.5, -1.0, 1.0) * torch.exp((2.0 * r3[:, 2] - 1.0) * 0.1)
                    xe = F.grid_sample(xe, F.affine_grid(th, list(xe.shape), align_corners=True), mode="bilinear",
                                       padding_mode="border", align_corners=True)
                    xb = torch.cat([xb, xe], dim=0)
                rec, mu, lv = vae(xb)
                rl = ((rec - xb) ** 2).sum(dim=(1, 2, 3)) / (2.0 * MORPH_VAE_PIX_SD ** 2)
                kl = -0.5 * (1.0 + lv - mu * mu - lv.exp()).sum(dim=1)
                loss = (rl + MORPH_VAE_BETA * kl).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
                opt.step()
                sch.step()
                if it % 250 == 0 or it == int(steps) - 1:
                    hist.append([float(it), float(rl.mean()), float(kl.mean())])
            vae.eval()
        return vae, hist

    def _sample_learned_morphologies(self, vae: "MorphologyVAE", Xk: np.ndarray, Xe: np.ndarray, n_want: int,
                                     novelty: float, seed: int, uv: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Learned morphologies: decoded latent draws z ~ N(0, I), kept when physically admissible
        (morphology_filters), on the training manifold (nearest training patch, RMS per pixel, <= the training set's
        own 99th-percentile spacing x 1.25) and not a copy (relative L2 >= novelty from every measured / simulated
        example and its mirror image).  The nearest physics kernel is reported (the kernels are a law, not data).
        Without any example (no measured anomaly, every simulated conductor narrower than half the station spacing:
        blind_dec4 / dec8) only the admissibility and manifold tests apply and the novelty statistics are NaN."""
        dev = self.device
        Xtr = np.concatenate([Xk, Xe]) if len(Xe) else Xk
        npx = int(np.prod(Xk.shape[1:]))
        ftr = torch.tensor(Xtr.reshape(Xtr.shape[0], -1), dtype=torch.float32, device=dev)
        dtr = torch.cdist(ftr, ftr) / math.sqrt(npx)
        dtr.fill_diagonal_(float("inf"))
        r_ok = float(np.quantile(dtr.min(dim=1).values.cpu().numpy(), MORPH_FILTER["plausibility_quantile"])
                     * MORPH_FILTER["plausibility_factor"])
        del dtr
        E = np.concatenate([Xe, Xe[:, ::-1, :]]) if len(Xe) else np.zeros((0,) + Xk.shape[1:])
        # reshape to the pixel count: an empty example set has no size to infer a -1 from
        fE = torch.tensor(E.reshape(E.shape[0], npx), dtype=torch.float32, device=dev)
        nE = fE.norm(dim=1).clamp_min(1e-9)
        fK = torch.tensor(Xk.reshape(Xk.shape[0], -1), dtype=torch.float32, device=dev)
        nK = fK.norm(dim=1).clamp_min(1e-9)
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed) + 2)
        keep: List[np.ndarray] = []
        nov: List[float] = []
        nker: List[float] = []
        st = {"drawn": 0, "admissible": 0, "on_manifold": 0, "novel": 0, "fail_outside_energy": 0, "fail_time_peaks": 0,
              "fail_sign_changes": 0}
        for _ in range(32):
            if len(keep) >= n_want:
                break
            with torch.no_grad():
                M = _normalise_patches(vae.decode(torch.randn(1024, MORPH_VAE_LATENT, device=dev, generator=g))[:, 0]
                                       .double().cpu().numpy())
            st["drawn"] += M.shape[0]
            f = morphology_filters(M, uv)
            st["fail_outside_energy"] += int((f["outside_energy"] > MORPH_FILTER["outside_energy_max"]).sum())
            st["fail_time_peaks"] += int((f["time_peaks"] != 1).sum())
            st["fail_sign_changes"] += int((f["sign_changes"] > MORPH_FILTER["sign_change_max"]).sum())
            ok = f["ok"]
            st["admissible"] += int(ok.sum())
            fm = torch.tensor(M.reshape(M.shape[0], -1), dtype=torch.float32, device=dev)
            dn = (torch.cdist(fm, ftr) / math.sqrt(npx)).min(dim=1).values.cpu().numpy()
            ok = ok & (dn <= r_ok)
            st["on_manifold"] += int(ok.sum())
            rel = ((torch.cdist(fm, fE) / nE.view(1, -1)).min(dim=1).values.cpu().numpy() if len(E)
                   else np.full(M.shape[0], np.inf))
            ok = ok & (rel >= float(novelty))
            st["novel"] += int(ok.sum())
            relk = (torch.cdist(fm, fK) / nK.view(1, -1)).min(dim=1).values.cpu().numpy()
            for i in np.nonzero(ok)[0]:
                if len(keep) >= n_want:
                    break
                keep.append(M[i])
                nov.append(float(rel[i]))
                nker.append(float(relk[i]))
        Lr = np.stack(keep) if keep else np.zeros((0,) + Xk.shape[1:])
        pw = float("nan")
        if Lr.shape[0] > 1:
            fl = torch.tensor(Lr.reshape(Lr.shape[0], -1), dtype=torch.float32, device=dev)
            dl = torch.cdist(fl, fl) / fl.norm(dim=1).clamp_min(1e-9).view(-1, 1)
            pw = float(dl[~torch.eye(fl.shape[0], dtype=torch.bool, device=dev)].median())
        nv = np.asarray(nov, dtype=np.float64)
        st.update({"kept": int(Lr.shape[0]), "wanted": int(n_want),
                   "admissible_rate": st["admissible"] / max(st["drawn"], 1),
                   "on_manifold_rate": st["on_manifold"] / max(st["drawn"], 1), "novel_rate": st["novel"] / max(st["drawn"], 1),
                   "manifold_radius_rms_per_pixel": r_ok, "novelty_threshold": float(novelty),
                   "novelty_vs_examples_min": float(nv[np.isfinite(nv)].min()) if np.isfinite(nv).any() else float("nan"),
                   "novelty_vs_examples_median": float(np.median(nv[np.isfinite(nv)])) if np.isfinite(nv).any() else float("nan"),
                   "nearest_physics_kernel_rel_median": float(np.median(nker)) if nker else float("nan"),
                   "pairwise_rel_distance_median": pw})
        return Lr, st

    def _make_morphology_bank(self) -> Dict[str, Any]:
        """MorphologyPrior: the bank's measured and simulated examples, the physics kernels, the beta-VAE and its
        learned morphologies (bank patches (n, MORPH_U_POINTS, T), max |.| = 1 over the measured window)."""
        Pc, dev = self.cfg.profile, self.device
        T, kt = self.T, self.kt
        seed = int(self.cfg.runtime.seed) + 4240
        uv = np.linspace(-MORPH_VAE_U_HALF, MORPH_VAE_U_HALF, MORPH_VAE_NU)
        Wu = _interp_matrix(uv, MORPH_U)                                                   # bank u -> prior u
        Wt = _interp_matrix(np.linspace(0.0, kt - 1.0, MORPH_VAE_NT), np.arange(T, dtype=np.float64))   # (NT, T)

        def _unit(Pm: np.ndarray) -> Optional[np.ndarray]:
            m_ = float(np.abs(Pm[:, :kt]).max()) if Pm.size else 0.0
            return Pm / m_ if m_ > 0 else None

        names: List[str] = []
        pats: List[np.ndarray] = []
        sig: List[float] = []
        src: List[int] = []
        tsh: List[float] = []
        dnp = self.depth_np
        if self.n_ma > 0 and self.mb_mix["measured"] > 0:
            Af = self.t_MA.double().cpu().numpy()                                          # (n, D, T) generation axis
            for j in range(self.n_ma):
                c = float(self.ma_center[j])
                sg = float(self.ma_sigma[j]) if np.isfinite(self.ma_sigma[j]) else float(self.spacing_med)
                lo_, hi_ = self.ma_support[j]
                zq = c + sg * MORPH_U
                Pm = _unit(depth_interp_profile(dnp, Af[j], np.clip(zq, dnp[0], dnp[-1]))
                           * ((zq >= lo_ - 1e-9) & (zq <= hi_ + 1e-9))[:, None])
                if Pm is not None:
                    pats.append(Pm)
                    sig.append(sg)
                    src.append(0)
                    tsh.append(1.0)
                    names.append(f"measured {j + 1} ({self.ma_runs[j]['from_m']:.0f}-{self.ma_runs[j]['to_m']:.0f} m)")
        n_meas = len(pats)
        sm = getattr(self.meas, "sim_morph", None)
        lt0, lt1 = self.anom_ltau
        a_ref = float(math.sqrt(self.anom_amp[0] * self.anom_amp[1]))
        tms = np.asarray(self.meas.grid_ms, dtype=np.float64)
        if sm is not None and len(sm["sigma_m"]) and self.mb_mix["sim"] > 0:
            for i in range(len(sm["sigma_m"])):
                sg, c = float(sm["sigma_m"][i]), float(sm["center_m"][i])
                Lc = self._measured_at(self._depth_to_pos(torch.tensor([[c]], dtype=torch.float32, device=dev)))[0]
                Lc = Lc[0, 0].double().cpu().numpy()
                ltlo = min(lt0 + 2.0 * math.log10(max(sg / max(self.anom_r_ref, 1e-9), 1.0)), max(lt1 - 0.5, lt0))
                shp = np.clip(np.asarray(sm["shapes"][i], dtype=np.float64), 0.0, None)
                for f_ in MORPH_SIM_TAUS:
                    tau = 10.0 ** (ltlo + f_ * (lt1 - ltlo))
                    lu = -tms / tau * math.log10(math.e) - Lc
                    un = 10.0 ** (lu - lu[:kt].max())
                    un[kt:] = np.minimum(un[kt:], 1.0)
                    Pm = _unit(np.log10(1.0 + (10.0 ** a_ref - 1.0) * shp[:, None] * un[None, :]))
                    if Pm is not None:
                        pats.append(Pm)
                        sig.append(sg)
                        src.append(1)
                        tsh.append(1.0)
                        names.append(f"simulated conductor {c:.0f} m, tau {tau:.3g} ms")
        n_simp = len(pats) - n_meas
        Xe = (_normalise_patches(np.einsum("au,nut,bt->nab", Wu, np.stack(pats), Wt)) if pats
              else np.zeros((0, MORPH_VAE_NU, MORPH_VAE_NT)))
        n_want = max(int(Pc.morphology_learned), 0) if self.mb_mix["learned"] > 0 else 0
        vae_state, hist, lst = None, [], {"kept": 0, "wanted": 0}
        Xk = np.zeros((0, MORPH_VAE_NU, MORPH_VAE_NT))
        n_vae = 0
        if n_want > 0:
            gk = torch.Generator(device=dev)
            gk.manual_seed(seed)
            Xk = self._morph_physics_kernels(MORPH_PHYS_KERNELS, gk, uv, Wt)
            t0 = time.time()
            vae, hist = self._train_morph_prior(Xk, Xe, int(Pc.morphology_vae_steps), seed + 1)
            n_vae = int(sum(p_.numel() for p_ in vae.parameters()))
            Lr, lst = self._sample_learned_morphologies(vae, Xk, Xe, n_want, float(Pc.morphology_novelty), seed + 2, uv)
            lst["prior_training_seconds"] = float(time.time() - t0)
            vae_state = {k_: v_.detach().cpu() for k_, v_ in vae.state_dict().items()}
            if Lr.shape[0]:
                Wu_up = _interp_matrix(MORPH_U, uv, outside="zero")                            # (U, NU)
                Wt_up = _interp_matrix(np.minimum(np.arange(T, dtype=np.float64), kt - 1.0),
                                       np.linspace(0.0, kt - 1.0, MORPH_VAE_NT))                  # (T, NT)
                for i, Pm in enumerate(np.einsum("ua,nab,tb->nut", Wu_up, Lr, Wt_up)):
                    Pm = _unit(Pm)
                    if Pm is not None:
                        pats.append(Pm)
                        sig.append(1.0)
                        src.append(2)
                        tsh.append(0.0)
                        names.append(f"learned {i + 1}")
            del vae
        P_ = np.stack(pats).astype(np.float32) if pats else np.zeros((0, MORPH_U_POINTS, T), np.float32)
        envu = np.abs(P_).max(axis=2) if P_.size else np.zeros((0, MORPH_U_POINTS))
        uext = np.array([max(float(np.abs(MORPH_U[e_ > 0.02]).max()) if np.any(e_ > 0.02) else 1.0, 1.0) for e_ in envu])
        tpk = np.array([int(np.argmax(np.abs(p_[:, :kt]).max(axis=0))) for p_ in P_], dtype=np.int64)
        fe = morphology_filters(Xe, uv) if len(Xe) else None
        rep = {"sources": {"measured": n_meas, "simulated_conductors": self.mb_n_sim, "simulated_patches": n_simp,
                           "learned": int(sum(1 for v_ in src if v_ == 2))},
               "mix_requested": dict(self.mb_mix), "patch_grid": {"u_half": MORPH_U_HALF, "u_points": MORPH_U_POINTS,
                                                                  "time": "measured grid", "prior_u_points": MORPH_VAE_NU,
                                                                  "prior_time_nodes": MORPH_VAE_NT},
               "prior": {"model": "beta-VAE (conv encoder / decoder)", "latent": MORPH_VAE_LATENT, "beta": MORPH_VAE_BETA,
                         "pixel_sd": MORPH_VAE_PIX_SD, "steps": int(Pc.morphology_vae_steps), "parameters": n_vae,
                         "physics_kernels": int(Xk.shape[0]), "kernel_shares": list(MORPH_PHYS_SHARES),
                         "example_share_per_batch": MORPH_EXAMPLE_SHARE if len(Xe) else 0.0,
                         "loss_history_recon_kl": hist},
               "learned": lst, "filters": MORPH_FILTER,
               "examples_admissible": ([bool(v_) for v_ in fe["ok"]] if fe is not None else []),
               "time_shift_rule": "measured / simulated shapes: t -> s^2 t for a size factor s (diffusion), peak kept in "
                                  "the measured window; learned: as decoded",
               "size_rule": "measured / simulated: width = s x own sigma, s log-uniform in MEASURED_LIKE_SIZE_RANGE (at "
                            "least one station spacing); learned: width = the analytic kinds' distance law",
               "entries": [{"name": n_, "source": ("measured", "simulated", "learned")[s_], "sigma_m": float(g_),
                            "u_extent": float(e_), "peak_time_ms": float(self.meas.grid_ms[t_])}
                           for n_, s_, g_, e_, t_ in zip(names, src, sig, uext, tpk)]}
        return {"patches": P_, "sigma": np.asarray(sig, dtype=np.float32), "src": np.asarray(src, dtype=np.int64),
                "uext": uext.astype(np.float32), "tpk": tpk, "tshift": np.asarray(tsh, dtype=np.float32),
                "names": names, "report": rep, "vae_state": vae_state,
                "prior_examples": Xe.astype(np.float32), "prior_kernels_sample": Xk[:256].astype(np.float32)}

    def _set_morph_bank(self, bank: Dict[str, Any]) -> None:
        dev = self.device
        P_ = np.asarray(bank["patches"], dtype=np.float32)
        src = np.asarray(bank["src"], dtype=np.int64)
        cnt = np.array([int(np.sum(src == k_)) for k_ in range(3)], dtype=np.int64)
        p = np.array([self.mb_mix["measured"], self.mb_mix["sim"], self.mb_mix["learned"]], dtype=np.float64) * (cnt > 0)
        if P_.shape[0] == 0 or p.sum() <= 0:
            raise RuntimeError("[MORPH] the morphology bank is empty (no measured anomaly, no simulated conductor and no "
                               "learned morphology passed the filters): run with --morphology_bank legacy or check the "
                               "inputs / --morphology_novelty")
        p = p / p.sum()
        self.t_mb = torch.tensor(P_, device=dev).contiguous()
        self.mb_n = int(P_.shape[0])
        self.t_mb_sigma = torch.tensor(np.asarray(bank["sigma"], dtype=np.float32), device=dev)
        self.t_mb_src = torch.tensor(src, device=dev)
        self.t_mb_uext = torch.tensor(np.asarray(bank["uext"], dtype=np.float32), device=dev)
        self.t_mb_tpk = torch.tensor(np.asarray(bank["tpk"], dtype=np.int64), device=dev)
        self.t_mb_shift = torch.tensor(np.asarray(bank["tshift"], dtype=np.float32), device=dev)
        self.t_mb_src_p = torch.tensor(p, dtype=torch.float32, device=dev)
        self.t_mb_cnt = torch.tensor(cnt, device=dev)
        self.t_mb_off = torch.tensor(np.concatenate([[0], np.cumsum(cnt)[:-1]]), dtype=torch.long, device=dev)
        self.mb_names = list(bank["names"])
        self.mb_du = float(MORPH_U[1] - MORPH_U[0])
        lg = np.log10(np.asarray(self.meas.grid_ms, dtype=np.float64))
        self.mb_dlg = float(np.median(np.diff(lg))) if lg.size > 1 else 1.0
        self.morph_bank = bank
        self.morph_report = dict(bank["report"])
        self.morph_report["mix_used"] = {"measured": float(p[0]), "sim": float(p[1]), "learned": float(p[2])}
        self.morph_report["key"] = bank.get("key")
        ls = self.morph_report.get("learned", {})
        print(f"[MORPH] morphology bank: {cnt[0]} measured + {cnt[1]} simulated ({self.mb_n_sim} conductors x "
              f"{len(MORPH_SIM_TAUS)} time constants) + {cnt[2]} learned morphologies | drawn by source "
              f"{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}"
              + (f" | prior: {ls.get('drawn', 0)} latent draws, admissible {ls.get('admissible_rate', float('nan')):.1%}, "
                 f"on the training manifold {ls.get('on_manifold_rate', float('nan')):.1%}, novel (>= "
                 f"{ls.get('novelty_threshold', float('nan')):.2f} rel. L2 from every example) "
                 f"{ls.get('novel_rate', float('nan')):.1%}; kept novelty min / median "
                 f"{ls.get('novelty_vs_examples_min', float('nan')):.2f} / {ls.get('novelty_vs_examples_median', float('nan')):.2f}"
                 if cnt[2] else ""))

    def _bank_draw(self, v: torch.Tensor, vs: torch.Tensor, zp: torch.Tensor, r: torch.Tensor, rlo: torch.Tensor,
                   ml: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The morphology of each kind-3 slot (B, K): its source by the bank's shares and an entry uniform within the
        source (both from v), its size factor (from vs): a measured / simulated shape x a factor log-uniform in
        MEASURED_LIKE_SIZE_RANGE of its own width (at least one local station spacing wide); a learned one (sigma 1
        m) the width r of the analytic kinds.  Returns (entry, source, size factor, its floor, width)."""
        p = self.t_mb_src_p
        cum = torch.cumsum(p, 0).clone()
        cum[-1] = 2.0
        sidx = torch.searchsorted(cum, v.contiguous().reshape(-1), right=True).clamp(max=2).view_as(v)
        lo = torch.cat([torch.zeros(1, device=p.device), torch.cumsum(p, 0)[:-1]])[sidx]
        frac = ((v - lo) / p[sidx].clamp_min(1e-12)).clamp(0.0, 1.0 - 1e-6)
        cnt = self.t_mb_cnt[sidx]
        idx = self.t_mb_off[sidx] + torch.minimum((frac * cnt.float()).floor().long(), (cnt - 1).clamp_min(0))
        sg = self.t_mb_sigma[idx]
        s_lo, s_hi = MEASURED_LIKE_SIZE_RANGE
        f_sh = torch.clamp(self.anom_w_lo_sp * self._local_spacing(zp) / sg.clamp_min(1e-6), min=s_lo)
        s_top = torch.maximum(torch.full_like(f_sh, s_hi), f_sh)
        s_sh = torch.exp(torch.log(f_sh) + vs * (torch.log(s_top) - torch.log(f_sh)))
        lrn = sidx == 2
        s = torch.where(lrn, r, s_sh)
        return idx.float(), sidx, s, torch.where(lrn, rlo, f_sh), torch.where(ml, s * sg, r)

    def _bank_log(self, depth: torch.Tensor, j: torch.Tensor, s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Bank morphology j (decades, (B, K, W, T), max |.| = 1 over the measured window) centred at z0 with the
        along-hole size factor s (width s x the entry's own sigma), at depths (B, W).  Measured / simulated shapes
        also move in time by the diffusion scaling of their size, t -> s^2 t (a body s times larger decays s^2 times
        later), their peak kept inside the measured window; 0 off the patch and off the generation axis."""
        B, Kn = j.shape
        W, T = depth.shape[1], self.T
        sg = (self.t_mb_sigma[j] * s).clamp_min(1e-6)
        zq = depth.float().unsqueeze(1)                                                          # (B, 1, W)
        pu = ((zq - z0.unsqueeze(-1)) / sg.unsqueeze(-1) + MORPH_U_HALF) / self.mb_du            # (B, K, W)
        inside = (pu >= 0.0) & (pu <= MORPH_U_POINTS - 1.0) & (zq >= self.d0 - 1e-6) & (zq <= self.d1 + 1e-6)
        pu = pu.clamp(0.0, MORPH_U_POINTS - 1.0 - 1e-4)
        i0 = pu.floor().long()
        wu = (pu - i0.float()).unsqueeze(-1)
        flat = self.t_mb.view(-1, T)
        base = (j * MORPH_U_POINTS).unsqueeze(-1)
        M = ((1.0 - wu) * flat[(base + i0).reshape(-1)].view(B, Kn, W, T)
             + wu * flat[(base + i0 + 1).reshape(-1)].view(B, Kn, W, T))
        shf = self.t_mb_shift[j]
        if bool((shf > 0).any()) and T > 1:
            sh = shf * 2.0 * torch.log10(s.clamp_min(1e-6)) / max(self.mb_dlg, 1e-12)              # grid samples
            tp = self.t_mb_tpk[j].float()
            sh = torch.minimum(torch.maximum(sh, -tp), (self.kt - 1.0) - tp)
            tq = (torch.arange(T, device=depth.device, dtype=torch.float32).view(1, 1, T) - sh.unsqueeze(-1)).clamp(0.0, T - 1.0)
            t0 = tq.floor().long().clamp(max=T - 2)
            wt = (tq - t0.float()).unsqueeze(2)
            ix = t0.unsqueeze(2).expand(B, Kn, W, T)
            M = (1.0 - wt) * M.gather(-1, ix) + wt * M.gather(-1, ix + 1)
        return M * inside.unsqueeze(-1).float()

    # ------------------------------------------------------------------ batch construction
    def _starts(self, W: int) -> np.ndarray:
        if W not in self._window_starts:
            k = self.keep_mask
            ok = [s for s in range(0, self.D - W + 1) if bool(np.all(k[s:s + W]))]
            if not ok:  # fall back to the longest window that fits a known segment
                ok = [0]
            self._window_starts[W] = np.asarray(ok, dtype=np.int64)
        return self._window_starts[W]

    def _measured_at(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pos: fractional station coordinate (B, W) -> (L (B, W, T), depth (B, W))."""
        D = self.D
        p = pos.clamp(0.0, D - 1.0)
        k = p.floor().long().clamp(0, max(D - 2, 0))
        s = (p - k.float()).unsqueeze(-1)
        dk, dk1 = self.t_depth[k], self.t_depth[(k + 1).clamp(max=D - 1)]
        h = (dk1 - dk).unsqueeze(-1)
        depth = dk + s.squeeze(-1) * (dk1 - dk)
        y0, y1 = self.t_Lk[k], self.t_Lk[(k + 1).clamp(max=D - 1)]
        m0, m1 = self.t_mk[k], self.t_mk[(k + 1).clamp(max=D - 1)]
        s2, s3 = s * s, s * s * s
        L = (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * m1
        return L, depth

    def _meas_anomalies_at(self, pos: torch.Tensor) -> torch.Tensor:
        """The measured anomaly templates at fractional station coordinates pos (B, W): (n, B, W, T), the same
        Hermite / PCHIP interpolation as the anchor (_measured_at)."""
        D = self.D
        p = pos.clamp(0.0, D - 1.0)
        k = p.floor().long().clamp(0, max(D - 2, 0))
        k1 = (k + 1).clamp(max=D - 1)
        s = (p - k.float()).unsqueeze(-1)
        h = (self.t_depth[k1] - self.t_depth[k]).unsqueeze(-1)
        y0, y1 = self.t_MA[:, k], self.t_MA[:, k1]
        m0, m1 = self.t_mMA[:, k], self.t_mMA[:, k1]
        s2, s3 = s * s, s * s * s
        return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * m1

    def _meas_anomaly_one(self, j: int, pos: torch.Tensor) -> torch.Tensor:
        """Measured anomaly template j at fractional station coordinates pos (any shape): pos.shape + (T,), the
        Hermite / PCHIP interpolation of _meas_anomalies_at."""
        D = self.D
        p = pos.clamp(0.0, D - 1.0)
        k = p.floor().long().clamp(0, max(D - 2, 0))
        k1 = (k + 1).clamp(max=D - 1)
        s = (p - k.float()).unsqueeze(-1)
        h = (self.t_depth[k1] - self.t_depth[k]).unsqueeze(-1)
        y0, y1 = self.t_MA[j][k], self.t_MA[j][k1]
        m0, m1 = self.t_mMA[j][k], self.t_mMA[j][k1]
        s2, s3 = s * s, s * s * s
        return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * m1

    def _measured_like_log(self, zq: torch.Tensor, tpl: torch.Tensor, s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Measured anomaly template j (decades, zq.shape + (T,)) moved to centre z0 with along-hole size factor s,
        at query depths zq (B, K, Q): A_j(c_j + (z - z0) / s, t), the measured shape as it is - both lobes and the
        time behaviour; 0 where the moved support does not reach z or z is off the generation axis.  (Dividing the
        anomalous field by the destination's background instead changed the lobes' ratio with depth: Liuyuan's 184 m
        mid-time dip doubled at 241 m.)"""
        c = self.t_ma_c[tpl]
        zs = c.unsqueeze(-1) + (zq - z0.unsqueeze(-1)) / s.unsqueeze(-1)
        inside = ((zs >= self.t_ma_lo[tpl].unsqueeze(-1) - 1e-6) & (zs <= self.t_ma_hi[tpl].unsqueeze(-1) + 1e-6)
                  & (zq >= self.d0 - 1e-6) & (zq <= self.d1 + 1e-6))
        ps = self._depth_to_pos(zs)
        A = torch.zeros(zq.shape + (self.T,), device=zq.device)
        for j in range(self.n_ma):
            mj = tpl == j
            if bool(mj.any()):           # interpolate template j only where it is used
                A[mj] = self._meas_anomaly_one(j, ps[mj])
        return A * inside.unsqueeze(-1).float()

    def _measured_like_fields(self, depth: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Relative response change (B, K, W, T) of the measured_like anomalies (kind 3) at depths (B, W): the moved
        template (_measured_like_log) times |A_k| / the template's peak over the measured window, in decades (as the
        measured anomalies are scaled), so its largest |value| on the template's stations is |A_k| and its sign that
        of the template's dominant lobe (the dense labels report the exact peak).  On a late-time extension it is
        held within +-|A_k|, as the conductors' time factor is."""
        B, Kn = P["an_A"].shape
        s = P["an_s"].clamp_min(1e-3)
        a = P["an_A"].abs()
        if getattr(self, "anom_bank_on", False):
            # a bank morphology (measured, simulated or learned), max |.| = 1 over the measured window
            dA = a.unsqueeze(-1).unsqueeze(-1) * self._bank_log(depth, P["an_tpl"].long().clamp(0, self.mb_n - 1), s,
                                                                P["an_z0"])
        else:
            tpl = P["an_tpl"].long().clamp(0, self.n_ma - 1)
            dA = (a / self.t_ma_peak[tpl]).unsqueeze(-1).unsqueeze(-1) * self._measured_like_log(
                depth.float().unsqueeze(1).expand(B, Kn, depth.shape[1]), tpl, s, P["an_z0"])
        if self.kt < self.T:
            a4 = a.unsqueeze(-1).unsqueeze(-1)
            dA = torch.cat([dA[..., :self.kt], torch.maximum(torch.minimum(dA[..., self.kt:], a4), -a4)], dim=-1)
        return torch.pow(10.0, dA) - 1.0

    # ------------------------------------------------------------------ per-profile random parameters
    def _depth_to_pos(self, depth: torch.Tensor) -> torch.Tensor:
        """Physical depth (m) -> fractional station coordinate on the measured axis."""
        d = depth.clamp(self.d0, self.d1).contiguous()
        j = (torch.searchsorted(self.t_depth, d.reshape(-1), right=True) - 1).clamp(0, max(self.D - 2, 0))
        j = j.reshape(d.shape)
        dj, dj1 = self.t_depth[j], self.t_depth[(j + 1).clamp(max=self.D - 1)]
        return j.float() + (d - dj) / (dj1 - dj).clamp_min(1e-9)

    def _local_spacing(self, z: torch.Tensor) -> torch.Tensor:
        pos = self._depth_to_pos(z)
        k = pos.round().long().clamp(0, self.D - 1)
        return self.t_spacing[k]

    def _alpha_base(self, B: int, mode: str, gen: Optional[torch.Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-profile alpha and kind (0 bridge, 1 measured endpoint, 2 simulated endpoint) of the union sampler."""
        dev = self.device
        cfg, R = self.cfg, self.cfg.reproduction
        kind = torch.zeros(B, dtype=torch.long, device=dev)
        if mode == "train":
            if not self.has_sim:
                return torch.ones(B, device=dev), torch.ones_like(kind)
            g = cfg.generation
            rw, sw, bw = max(g.union_real_weight, 0.0), max(g.union_sim_weight, 0.0), max(g.union_bridge_weight, 0.0)
            tot = max(rw + sw + bw, 1e-9)
            u = torch.rand(B, device=dev, generator=gen)
            kind = torch.where(u < rw / tot, torch.ones_like(kind),
                               torch.where(u < (rw + sw) / tot, torch.full_like(kind, 2), kind))
            v = torch.rand(B, device=dev, generator=gen)
            bs = torch.distributions.Beta
            a_rn = bs(torch.tensor(4.0, device=dev), torch.tensor(0.8, device=dev)).sample((B,))
            a_mid = bs(torch.tensor(2.0, device=dev), torch.tensor(2.0, device=dev)).sample((B,))
            a_sn = bs(torch.tensor(0.8, device=dev), torch.tensor(4.0, device=dev)).sample((B,))
            fr, fs = R.bridge_real_near_frac, R.bridge_sim_near_frac
            a_b = torch.where(v < fs, a_sn, torch.where(v < fs + fr, a_rn, a_mid)).clamp(R.bridge_alpha_min, R.bridge_alpha_max)
            a = torch.where(kind == 1, torch.ones_like(a_b), torch.where(kind == 2, torch.zeros_like(a_b), a_b))
        elif mode in ("gen", "val") and self.has_sim:
            ws = torch.tensor([w for _, _, w in self.gen_alpha_mix], device=dev)
            comp = torch.multinomial(ws, B, replacement=True, generator=gen)
            lo = torch.tensor([l for l, _, _ in self.gen_alpha_mix], device=dev)[comp]
            hi = torch.tensor([h for _, h, _ in self.gen_alpha_mix], device=dev)[comp]
            a = lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
        else:
            a = torch.ones(B, device=dev)
            kind[:] = 1
        mgm = cfg.generation.mode
        if mgm == "real_only" or not self.has_sim:
            a = torch.ones_like(a)
        elif mgm == "simulate_only":
            a = torch.zeros_like(a)
        elif mgm == "interpolate" and mode != "train":
            a = torch.full_like(a, float(cfg.generation.interpolate_alpha))
        return a, kind

    def _anomaly_params(self, B: int, gen: Optional[torch.Generator]) -> Dict[str, torch.Tensor]:
        """Random anomaly set of each profile: count K ~ anomaly_count_probs (0 = no anomaly); per anomaly a sign,
        a log-uniform |peak| in the measured-anchored range, a uniform centre depth over the measured span,
        a log-uniform footprint sigma (>= anomaly_width_range[0] local station spacings), a log-uniform
        eddy-current time constant tau and the type (conductor decay or time-flat near-hole zone)."""
        dev = self.device
        Kmax = self.anom_kmax
        K = torch.multinomial(self.t_anom_p, B, replacement=True, generator=gen)
        K_all = K
        ma_on = None
        if self.n_ma > 0 and self.ma_count:
            # the count includes the measured anomalies: m of the K are measured ones, m uniform over 0 .. min(K,
            # n_measured), which ones at random; the other K - m are synthetic
            um = torch.rand(B, 1 + self.n_ma, device=dev, generator=gen)
            mmax = torch.clamp(K, max=self.n_ma)
            m = torch.minimum((um[:, 0] * (mmax + 1).float()).floor().long(), mmax)
            rank = torch.argsort(torch.argsort(um[:, 1:], dim=1), dim=1)
            ma_on = (rank < m.unsqueeze(1)).float()
            K = K_all - m
        elif self.n_ma > 0:
            # a profile with anomalies carries the measured anomaly set in its first slot, synthetic ones in the rest
            K = (K_all - 1).clamp_min(0)
        active = (torch.arange(Kmax, device=dev)[None, :] < K[:, None]).float()
        u = torch.rand(B, Kmax, 6, device=dev, generator=gen)
        lo, hi = self.anom_amp
        amp = torch.exp(math.log(lo) + u[..., 0] * (math.log(hi) - math.log(lo)))
        sign = torch.where(u[..., 1] < self.anom_neg_frac, -torch.ones_like(amp), torch.ones_like(amp))
        if self.anom_phys:
            out = self._physical_anomaly_params(u, amp, sign, active, K, gen, ma_on)
            if self.n_ma > 0:
                out["ma_any"] = (K_all >= 1).float()
                if ma_on is not None:
                    out["ma_on_draw"] = ma_on
            return out
        z0 = self.d0 + u[..., 2] * (self.d1 - self.d0)
        ltau = self.anom_ltau[0] + u[..., 4] * (self.anom_ltau[1] - self.anom_ltau[0])
        if self.anom_family == "buried":
            # footprint of a conductor under a line: a fraction of the measured lateral correlation length at the
            # anomaly's time scale (clamped to the measured window), l(tau) = l1 (tau / t1)^beta, at least one station
            # spacing and at most a quarter of the line
            lw = self.lat_law
            t_lo, t_hi = float(self.meas.grid_ms[0]), float(self.meas.grid_ms[self.kt - 1])
            tau_eff = torch.pow(10.0, ltau).clamp(t_lo, t_hi)
            ell = lw["l1_m"] * torch.pow(tau_eff / lw["t1_ms"], lw["beta"])
            f0, f1 = self.anom_fp
            w0 = ell * torch.exp(math.log(f0) + u[..., 3] * (math.log(f1) - math.log(f0)))
            w0 = torch.minimum(torch.maximum(w0, self._local_spacing(z0)),
                               torch.full_like(w0, 0.25 * max(self.m1 - self.m0, 1e-6)))
        else:
            wlo = self.anom_w_lo_sp * self._local_spacing(z0)
            whi = torch.maximum(torch.full_like(wlo, self.anom_w_hi), 1.5 * wlo)
            w0 = torch.exp(torch.log(wlo) + u[..., 3] * (torch.log(whi) - torch.log(wlo)))
        typ = (u[..., 5] < self.anom_bb_frac).float()
        out = {"an_K": K, "an_A": sign * amp * active, "an_z0": z0, "an_w0": w0, "an_ltau": ltau, "an_type": typ}
        if self.n_ma > 0:
            out["ma_any"] = (K_all >= 1).float()
            if ma_on is not None:
                out["ma_on_draw"] = ma_on
        return out

    def _physical_anomaly_params(self, u: torch.Tensor, amp: torch.Tensor, sign: torch.Tensor, active: torch.Tensor,
                                 K: torch.Tensor, gen: Optional[torch.Generator],
                                 ma_on: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Synthetic borehole anomalies from conductor bodies, from seven uniform draws per anomaly:
          kind  u5: 0 the hole passes through a conductor (single positive peak), 1 a conductor beside the hole (axial
                dipole field), 2 a near-hole zone; shares --borehole_anomaly_kinds;
          r     u3: distance of the body from the hole (kinds 0 / 2: its half size), log-uniform from
                anomaly_width_range[0] local station spacings to anomaly_width_range[1] of the measured span;
          z0    u2: centre depth, uniform from max(first station, anomaly_min_depth_m + r) to the last station (no
                synthetic anomaly in the shallow part of the hole);
          tau   u4: eddy-current time constant, log-uniform from max(t_first, t_first (r / station spacing)^2) to the top
                of the range: farther from the hole only a larger body is detectable (coupling ~ (size / r)^3) and a
                body's own diffusion time grows with its size squared (sphere: tau = mu0 sigma a^2 / pi^2), so distant
                bodies give late-time anomalies;
          |peak| u0: log-uniform over the visible range; sign u1 (u1 < anomaly_neg_frac = -1): kind 2 its sign, kind 1
                the orientation of the dipole kernel (+1: at small dips the negative-centred three-peak anomaly of a flat
                body at the edge of the hole; the labelled sign is that of the lobe that reaches |peak|), kind 0 always
                positive (inside a conductor its secondary field adds to the decay);
          dip   -90..90 deg (kind 1).
        borehole_anomaly_model measured, after the dip, one more draw per anomaly for each of:
          kind 3 measured_like (share from --borehole_anomaly_kinds): which measured anomaly's shape (uniform), its
                along-hole size factor s (log-uniform MEASURED_LIKE_SIZE_RANGE; width s x the measured sigma); |peak| as
                drawn, the sign of the shape's own largest lobe;
          centre: SEPARATION_CANDIDATES candidates over the anomaly's centre range; the first whose core (+-
                SEPARATION_CORE_SIGMAS x its width) stays clear of the cores of the profile's earlier synthetic anomalies
                and of the measured anomalies it carries (their part of the inventory run); where none does (a crowded
                profile), the first clear at 3/4 and then 1/2 of those cores, else the candidate with the largest
                clearance (overlap only where the profile has no room)."""
        dev = u.device
        B, Kmax = u.shape[0], u.shape[1]
        c0, c1 = float(self.anom_kind_p[0]), float(self.anom_kind_p[0] + self.anom_kind_p[1])
        kind = (u[..., 5] >= c0).long() + (u[..., 5] >= c1).long()
        if float(self.anom_kind_p[3]) > 0:      # measured_like
            kind = kind + (u[..., 5] >= c1 + float(self.anom_kind_p[2])).long()
        zlo0 = max(self.d0, min(self.anom_zmin, self.d1))
        zp = zlo0 + u[..., 2] * (self.d1 - zlo0)            # provisional centre: the local station spacing there
        rlo = self.anom_w_lo_sp * self._local_spacing(zp)
        rhi = torch.maximum(torch.full_like(rlo, self.anom_w_hi), 1.5 * rlo)
        r = torch.exp(torch.log(rlo) + u[..., 3] * (torch.log(rhi) - torch.log(rlo)))
        if self.anom_zmin > 0:
            zlo = torch.clamp(self.anom_zmin + r, min=self.d0, max=self.d1)
        else:
            zlo = torch.full_like(r, self.d0)
        z0 = zlo + u[..., 2] * (self.d1 - zlo)
        lt0, lt1 = self.anom_ltau
        ltlo = (lt0 + 2.0 * torch.log10((r / max(self.anom_r_ref, 1e-9)).clamp_min(1.0))).clamp(max=max(lt1 - 0.5, lt0))
        ltau = ltlo + u[..., 4] * (lt1 - ltlo)
        sign = torch.where(kind == 0, torch.ones_like(sign), sign)
        dip = (2.0 * torch.rand(B, Kmax, device=dev, generator=gen) - 1.0) * (0.5 * math.pi)
        if not self.anom_meas:
            return {"an_K": K, "an_A": sign * amp * active, "an_z0": z0, "an_w0": r, "an_ltau": ltau,
                    "an_type": (kind == 2).float(), "an_kind": kind.float(), "an_dip": dip}
        # measured model: measured shapes at other depths, separated centres
        C = int(SEPARATION_CANDIDATES)
        u2 = torch.rand(B, Kmax, 2 + C, device=dev, generator=gen)
        ml = kind == 3
        src = None
        if self.anom_bank_on:
            tpl, src, s, s_floor, width = self._bank_draw(u2[..., 0], u2[..., 1], zp, r, rlo, ml)
        elif self.anom_cal:
            tpl = (u2[..., 0] * self.n_ma).floor().clamp(max=self.n_ma - 1)
            s_lo, s_hi = MEASURED_LIKE_SIZE_RANGE
            sig_t = self.t_ma_sigma[tpl.long()]
            # at least one local station spacing wide, like the other kinds (a narrower copy falls between stations)
            s_floor = torch.clamp(self.anom_w_lo_sp * self._local_spacing(zp) / sig_t.clamp_min(1e-6), min=s_lo)
            s_top = torch.maximum(torch.full_like(s_floor, s_hi), s_floor)
            s = torch.exp(torch.log(s_floor) + u2[..., 1] * (torch.log(s_top) - torch.log(s_floor)))
            width = torch.where(ml, s * sig_t, r)
        else:
            tpl, s, width = torch.zeros_like(r), torch.ones_like(r), r
            s_floor = torch.ones_like(r)
        sign = torch.where(ml, torch.ones_like(sign), sign)
        h = SEPARATION_CORE_SIGMAS * width
        d0f, d1f = float(self.d0), float(self.d1)
        zlo_m = torch.clamp(self.anom_zmin + h, min=d0f, max=d1f)
        zlo_k = torch.where(ml, zlo_m, zlo)
        zhi_k = torch.where(ml, torch.maximum(torch.clamp(d1f - h, max=d1f), zlo_k), torch.full_like(zlo, d1f))
        cand = zlo_k.unsqueeze(-1) + u2[..., 2:] * (zhi_k - zlo_k).unsqueeze(-1)            # (B, Kmax, C)
        on_m = None
        if self.n_ma > 0:
            # the measured anomalies the profile carries (mode all: all of them, the conservative choice)
            on_m = (ma_on > 0.5) if ma_on is not None else torch.ones(B, self.n_ma, dtype=torch.bool, device=dev)
        z0n = z0.clone()
        for k in range(Kmax):
            ck = cand[:, k]
            pick = None
            # cores of +-2 widths apart; in a crowded profile (four or five anomalies) +-1.5 widths (two peaks still show
            # a dip between them), then +-1; else the candidate with the largest clearance at +-2
            for lvl in (1.0, 0.75, 0.5):
                hk = lvl * h[:, k:k + 1]
                clear = torch.full_like(ck, float("inf"))
                if on_m is not None:
                    for j in range(self.n_ma):
                        lo_j, hi_j = float(self.ma_runs[j]["from_m"]), float(self.ma_runs[j]["to_m"])
                        dj = torch.maximum(lo_j - (ck + hk), (ck - hk) - hi_j)
                        clear = torch.where(on_m[:, j:j + 1], torch.minimum(clear, dj), clear)
                for k2 in range(k):
                    d2 = (ck - z0n[:, k2:k2 + 1]).abs() - (hk + lvl * h[:, k2:k2 + 1])
                    clear = torch.where(active[:, k2:k2 + 1] > 0, torch.minimum(clear, d2), clear)
                if pick is None:
                    clear0, okany, pick = clear, torch.zeros_like(ck[:, 0], dtype=torch.bool), clear.argmax(dim=1)
                okc = clear >= 0
                new_ok = okc.any(dim=1) & ~okany
                pick = torch.where(new_ok, okc.float().argmax(dim=1), pick)
                okany = okany | new_ok
            z0n[:, k] = ck.gather(1, pick.unsqueeze(1)).squeeze(1)
        out = {"an_K": K, "an_A": sign * amp * active, "an_z0": z0n, "an_w0": width, "an_ltau": ltau,
               "an_type": (kind == 2).float(), "an_kind": kind.float(), "an_dip": dip, "an_tpl": tpl, "an_s": s}
        if src is not None:          # the bank source of each morphology slot (0 measured, 1 simulated, 2 learned)
            out["an_src"] = torch.where(ml, src.float(), torch.full_like(r, -1.0))
        # an anomaly is an undulation: a labelled synthetic anomaly must make the profile rise against its decay trend
        # by at least half the visibility floor (the rule that finds the measured anomalies, at the visible level)
        req = 0.5 * float(self.anom_amp[0])
        lo_a, hi_a = self.anom_amp
        bad = (active > 0) & (self._undulation_strength(out, active > 0) < req)
        for _ in range(int(UNDULATION_REDRAWS)):
            if not bool(bad.any()):
                break
            v = torch.rand(B, Kmax, 3, device=dev, generator=gen)
            # stronger (|peak| between its own and the top of the range) and narrower: each pass can only help
            la = torch.log(out["an_A"].abs().clamp(min=lo_a, max=hi_a))
            amp_n = torch.exp(la + v[..., 0] * (math.log(hi_a) - la))
            sg = torch.where(out["an_A"] < 0, -torch.ones_like(amp_n), torch.ones_like(amp_n))
            out["an_A"] = torch.where(bad, sg * amp_n, out["an_A"])
            out["an_ltau"] = torch.where(bad & ~ml, ltlo + v[..., 1] * (lt1 - ltlo), out["an_ltau"])
            if self.anom_cal or self.anom_bank_on:
                s_n = torch.exp(torch.log(s_floor) + v[..., 2] * (torch.log(torch.maximum(out["an_s"], s_floor))
                                                                    - torch.log(s_floor)))
                out["an_s"] = torch.where(bad & ml, s_n, out["an_s"])
                w_ml = out["an_s"] * (self.t_mb_sigma if self.anom_bank_on else self.t_ma_sigma)[out["an_tpl"].long()]
            else:
                w_ml = out["an_w0"]
            r_n = torch.exp(torch.log(rlo) + v[..., 2] * (torch.log(out["an_w0"].clamp_min(1e-6)) - torch.log(rlo)).clamp_min(0.0))
            out["an_w0"] = torch.where(bad, torch.where(ml, w_ml, r_n), out["an_w0"])
            bad = bad & (self._undulation_strength(out, bad) < req)          # only the redrawn ones are re-checked
        if bool(bad.any()):
            # still no undulation (Liuyuan: ~0.3 %, on the steep 60-75 m bend): not a visible anomaly, so not labelled -
            # the slot is dropped and the profile's count is one less (the remaining slots keep their order)
            keep = (active > 0) & ~bad
            perm = torch.sort((~keep).long(), dim=1, stable=True).indices
            for key_ in [k_ for k_, v_ in out.items() if torch.is_tensor(v_) and v_.dim() == 2 and v_.shape[1] == Kmax]:
                out[key_] = out[key_].gather(1, perm)
            out["an_K"] = keep.sum(dim=1).to(K.dtype)
            out["an_A"] = out["an_A"] * (torch.arange(Kmax, device=dev)[None, :] < out["an_K"][:, None]).float()
        return out

    @torch.no_grad()
    def _undulation_strength(self, P: Dict[str, torch.Tensor], which: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K): how far each synthetic anomaly alone makes the background trend rise against its decay direction
        along the profile (largest rise above the running minimum, decades, measured window).  Evaluated on a fine
        local axis of UNDULATION_LOCAL_POINTS points over the anomaly's own extent (Gaussian +-4 widths, dipole +-6
        r, a measured shape its moved support; at least +-3 local station spacings): the trend decays along the
        profile, so the rise inside that window is the rise along the whole profile, and an anomaly narrower than the
        station spacing is resolved as in the exported dense profiles.  which (B, K) bool: only those anomalies (the
        others: +inf)."""
        B, Kn = P["an_A"].shape
        dev = self.t_depth.device
        if getattr(self, "bg_trend", False):
            dn = self.t_bg_down[:self.kt].view(1, 1, -1)
        else:
            dn = (self.t_Lk[-1, :self.kt] <= self.t_Lk[0, :self.kt]).view(1, 1, -1)
        lin = torch.linspace(-1.0, 1.0, int(UNDULATION_LOCAL_POINTS), device=dev)
        t_last = float(self.t_time_ms[self.kt - 1])
        out = torch.full((B, Kn), float("inf"), device=dev)
        sel = torch.ones(B, Kn, dtype=torch.bool, device=dev) if which is None else which
        bi, ki = torch.nonzero(sel, as_tuple=True)
        if bi.numel() == 0:
            return out
        # every selected anomaly as its own one-anomaly profile: one evaluation for all of them
        Pk = {key: v[bi, ki].unsqueeze(1) for key, v in P.items()
              if key.startswith("an_") and torch.is_tensor(v) and v.dim() == 2}
        z0, w = Pk["an_z0"][:, 0], Pk["an_w0"][:, 0]
        kind = Pk["an_kind"][:, 0] if "an_kind" in Pk else torch.zeros_like(w)
        half = 4.0 * w
        if self.anom_diffusion:         # a footprint widening sqrt(1 + t / tau) reaches its largest width at the last time
            half = half * torch.sqrt(1.0 + t_last / torch.pow(10.0, Pk["an_ltau"][:, 0]))
        half = torch.where(kind == 1, 6.0 * w, half)
        if "an_tpl" in Pk and getattr(self, "anom_bank_on", False):
            jb = Pk["an_tpl"][:, 0].long().clamp(0, self.mb_n - 1)
            half = torch.where(kind == 3, Pk["an_s"][:, 0] * self.t_mb_sigma[jb] * self.t_mb_uext[jb], half)
        elif "an_tpl" in Pk and self.n_ma > 0:
            tpl = Pk["an_tpl"][:, 0].long().clamp(0, self.n_ma - 1)
            c = self.t_ma_c[tpl]
            ext = Pk["an_s"][:, 0] * torch.maximum(c - self.t_ma_lo[tpl], self.t_ma_hi[tpl] - c)
            half = torch.where(kind == 3, ext, half)
        half = torch.minimum(torch.maximum(half, 3.0 * self._local_spacing(z0)),
                             torch.full_like(half, max(self.d1 - self.d0, 1e-6)))
        zq = (z0.unsqueeze(1) + half.unsqueeze(1) * lin.unsqueeze(0)).clamp(self.d0, self.d1)         # (N, Q)
        F_ = self.anomaly_fields(zq, Pk)[:, 0, :, :self.kt]
        X = self._measured_at(self._depth_to_pos(zq))[0][..., :self.kt] + torch.log10((1.0 + F_).clamp_min(1e-12))
        X = torch.where(dn, X, torch.flip(X, dims=[1]))
        out[bi, ki] = (X - torch.cummin(X, dim=1).values).amax(dim=(1, 2))
        return out

    def anomaly_ratio(self, depth: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Relative response change r(z, t) of the profile's anomalies, (B, W, T): the sum of anomaly_fields."""
        return self.anomaly_fields(depth, P).sum(1)

    def anomaly_fields(self, depth: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Relative response change of each anomaly, (B, K, W, T); log10(1 + sum_k r_k) is added to the background.
        Per anomaly k:
            r_k(z, t) = (10^A_k - 1) u_k(t) exp(-(z - z_k)^2 / (2 w_k(t)^2)),   w_k(t) = w0_k sqrt(1 + t / tau_k)
        The footprint widens as the induced current system diffuses (sqrt of time) and its amplitude spreads as
        w0 / w(t).  Conductor: u_k is the eddy-current decay exp(-t / tau_k) superposed on the measured
        background at the centre, u ~ exp(-t / tau) (w0 / w) / V_meas(z_k, t), normalised to peak 1, so the
        centre curve is V_meas + c exp(-t / tau): a sum of two decays (still a decay), peaking relative to the
        background near t ~ (p - 0.5) tau for a t^-p background.  Near-hole zone: u_k = w0 / w(t) (time-flat
        apart from the diffusive spreading).  The peak relative change at the centre is exactly A_k decades.
        Conductor beside the hole (an_kind 1): r_k = c_k K((z - z_k) / r_k; dip_k) u_k(t) with the axial dipole
        kernel (dipole_axial_kernel, shape fixed in time: the late-time field of a compact body) and u_k = exp(-t / tau) /
        V_meas(z_k, t) normalised to peak 1; c_k is the largest factor with which neither lobe exceeds |A_k| decades
        (log10(1 + c K+) <= |A|, -log10(1 - c K-) <= |A|): the strongest lobe is exactly |A_k| decades and the total
        response never changes sign."""
        t = self.t_time_ms.view(1, 1, -1)
        tau = torch.pow(10.0, P["an_ltau"]).unsqueeze(-1)                              # (B, K, 1)
        tt = t / tau
        w0 = P["an_w0"].unsqueeze(-1)
        if self.anom_diffusion:
            w = (w0 * torch.sqrt(1.0 + tt)).clamp(max=0.5 * (self.m1 - self.m0))
        else:
            w = w0.expand(-1, -1, self.T)
        lspread = torch.log10(w0 / w)
        Lc, _ = self._measured_at(self._depth_to_pos(P["an_z0"]))                     # (B, K, T)
        lu_c = -tt * math.log10(math.e) + lspread - Lc
        lu = torch.where(P["an_type"].unsqueeze(-1) > 0.5, lspread, lu_c)
        off = (P["an_kind"] == 1) if "an_kind" in P else None
        if off is not None:   # beside the hole: no footprint spreading (the dipole field's extent is set by r)
            lu = torch.where(off.unsqueeze(-1), -tt * math.log10(math.e) - Lc, lu)
        # peak 1 over the MEASURED window, so the label A_k is the peak a measurement would see; on a late-time extension
        # a conductor still rising at the last gate holds that peak (never above the label)
        un = torch.pow(10.0, lu - lu[..., :self.kt].max(dim=-1, keepdim=True).values)
        if self.kt < self.T:
            un = torch.cat([un[..., :self.kt], un[..., self.kt:].clamp(max=1.0)], dim=-1)
        r0 = (torch.pow(10.0, P["an_A"]) - 1.0).unsqueeze(-1) * un                     # (B, K, T)
        dz = depth.float()[:, None, :, None] - P["an_z0"][:, :, None, None]            # (B, K, W, 1)
        shape = torch.exp(-0.5 * (dz / w[:, :, None, :]) ** 2)                         # (B, K, W, T)
        F_ = r0[:, :, None, :] * shape
        if off is not None and bool(off.any()):
            A = P["an_A"]
            s = torch.where(A < 0, -torch.ones_like(A), torch.ones_like(A))
            ug = torch.linspace(-6.0, 6.0, 1201, device=A.device).view(1, 1, -1)
            Kg = s.unsqueeze(-1) * dipole_axial_kernel(ug, P["an_dip"].unsqueeze(-1))   # (B, K, G): lobes of s K
            kp, km = Kg.clamp_min(0.0).amax(-1), (-Kg).clamp_min(0.0).amax(-1)
            a = A.abs()
            c = torch.minimum((torch.pow(10.0, a) - 1.0) / kp.clamp_min(1e-9), (1.0 - torch.pow(10.0, -a)) / km.clamp_min(1e-9))
            Kz = dipole_axial_kernel(dz / P["an_w0"][:, :, None, None], P["an_dip"][:, :, None, None])   # (B, K, W, 1)
            F1 = (s * c)[:, :, None, None] * Kz * un[:, :, None, :]
            F_ = torch.where(off[:, :, None, None], F1, F_)
        if "an_tpl" in P and (self.n_ma > 0 or getattr(self, "anom_bank_on", False)):
            ml = P["an_kind"] == 3
            if bool(ml.any()):    # a measured anomaly's shape at another depth (evaluated for those slots only)
                bi, ki = torch.nonzero(ml, as_tuple=True)
                Ps = {key: v[bi, ki].unsqueeze(1) for key, v in P.items()
                      if key.startswith("an_") and torch.is_tensor(v) and v.dim() == 2}
                F_ = F_.clone()
                F_[bi, ki] = self._measured_like_fields(depth[bi], Ps)[:, 0]
        return F_

    @torch.no_grad()
    def anomaly_labels(self, P: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Labels of each synthetic anomaly from its own field alone (anomaly_fields on a dense depth axis, measured
        window): the signed peak (largest |log10(1 + r_k)|: the drawn amplitude, for a conductor beside the hole with
        the sign of the lobe that reaches it), where (m) and when (ms) it is reached, and the visible time window:
        the first and last time at which the anomaly's largest |change| along the profile is at least half the
        visibility floor (anom_amp[0]).  (B, K) each, NaN for empty slots."""
        if "an_A" not in P:
            return {}
        B, Kmax = P["an_A"].shape
        dev = P["an_A"].device
        A = P["an_A"]
        pk_exact = A.clone()
        if "an_kind" in P:
            # the lobe that binds the scale factor of anomaly_fields: + if (10^a - 1) / K+ <= (1 - 10^-a) / K-
            s = torch.where(A < 0, -torch.ones_like(A), torch.ones_like(A))
            ug = torch.linspace(-6.0, 6.0, 1201, device=dev).view(1, 1, -1)
            Kg = s.unsqueeze(-1) * dipole_axial_kernel(ug, P["an_dip"].unsqueeze(-1))
            kp, km = Kg.clamp_min(0.0).amax(-1), (-Kg).clamp_min(0.0).amax(-1)
            a = A.abs()
            pos = (torch.pow(10.0, a) - 1.0) / kp.clamp_min(1e-9) <= (1.0 - torch.pow(10.0, -a)) / km.clamp_min(1e-9)
            pk_exact = torch.where(P["an_kind"] == 1, torch.where(pos, a, -a), A)
        n = int(np.clip(math.ceil((self.d1 - self.d0) / max(0.25 * float(self.t_spacing.min()), 1e-6)) + 1, 200, 2000))
        ax = torch.linspace(self.d0, self.d1, n, device=dev)
        tms = self.t_time_ms[:self.kt]
        thr = 0.5 * float(self.anom_amp[0])
        keys = ("lab_peak", "lab_peak_at", "lab_peak_ms", "lab_vis_from_ms", "lab_vis_to_ms")
        rise_lab = "an_tpl" in P          # measured model: the undulation each anomaly makes on the trend
        if rise_lab:
            keys = keys + ("lab_rise",)
            base_d = self._measured_at(self._depth_to_pos(ax))[0][:, :self.kt]          # (n, kt) trend on the axis
            dn_d = (self.t_bg_down[:self.kt] if getattr(self, "bg_trend", False) else
                    (base_d[-1] <= base_d[0])).view(1, 1, -1)
        out = {k_: torch.full((B, Kmax), float("nan"), device=dev) for k_ in keys}
        act = torch.arange(Kmax, device=dev)[None, :] < P["an_K"][:, None]
        for k in range(Kmax):     # one anomaly at a time: B x n x T
            Pk = {key: v[:, k:k + 1] for key, v in P.items() if key.startswith("an_") and key != "an_K"}
            dl = torch.log10((1.0 + self.anomaly_fields(ax.unsqueeze(0).expand(B, -1), Pk)[:, 0, :, :self.kt])
                             .clamp_min(1e-12))                                         # (B, n, kt)
            if rise_lab:
                Xr = base_d.unsqueeze(0) + dl
                Xr = torch.where(dn_d, Xr, torch.flip(Xr, dims=[1]))
                out["lab_rise"][:, k] = torch.where(act[:, k], (Xr - torch.cummin(Xr, dim=1).values).amax(dim=(1, 2)),
                                                    torch.full_like(Xr[:, 0, 0], float("nan")))
            flat = dl.abs().reshape(B, -1)
            j = flat.argmax(dim=1)
            iz, it = j // self.kt, j % self.kt
            prof = dl.abs().amax(dim=1)                                                 # (B, kt)
            vis = prof >= thr
            first = torch.where(vis.any(dim=1), vis.float().argmax(dim=1), torch.zeros_like(it))
            last = torch.where(vis.any(dim=1), self.kt - 1 - vis.flip(-1).float().argmax(dim=1), torch.zeros_like(it))
            ok = act[:, k]
            okv = ok & vis.any(dim=1)
            nanv = torch.full_like(prof[:, 0], float("nan"))
            pk_k = pk_exact[:, k]
            if "an_tpl" in P:
                # a measured shape at another depth: its signed peak as evaluated on the dense axis
                pk_k = torch.where(P["an_kind"][:, k] == 3, dl.reshape(B, -1).gather(1, j.unsqueeze(1)).squeeze(1), pk_k)
            out["lab_peak"][:, k] = torch.where(ok, pk_k, nanv)
            out["lab_peak_at"][:, k] = torch.where(ok, ax[iz], nanv)
            out["lab_peak_ms"][:, k] = torch.where(ok, tms[it], nanv)
            out["lab_vis_from_ms"][:, k] = torch.where(okv, tms[first], nanv)
            out["lab_vis_to_ms"][:, k] = torch.where(okv, tms[last], nanv)
        return out

    def sample_profile_params(self, B: int, mode: str, gen: Optional[torch.Generator]) -> Dict[str, torch.Tensor]:
        """Every random choice of a profile, so that the same profile can be evaluated at any depths."""
        dev = self.device
        cfg = self.cfg
        a, kind = self._alpha_base(B, mode, gen)
        P: Dict[str, torch.Tensor] = {"alpha": a, "kind": kind}
        sig = float(cfg.profile.depth_alpha_sigma) if (cfg.profile.enabled and self.has_sim) else 0.0
        if sig > 0:
            P.update({f"af_{k}": v for k, v in fourier_params_t(B, 1, [self.latent_bands[0]], [1.0], 4, gen, dev).items()})
        if self.has_sim and self.sim.depth_mode != "physical" and mode != "canonical":
            zmin = float(np.clip(cfg.profile.registration_zoom_min, 0.1, 1.0))
            zoom = zmin + (1.0 - zmin) * torch.rand(B, device=dev, generator=gen)
            off = (1.0 - zoom) * torch.rand(B, device=dev, generator=gen)
        else:
            zoom, off = torch.ones(B, device=dev), torch.zeros(B, device=dev)
        spread = max(float(cfg.profile.structure_gain_spread), 1.0)
        if not self.has_sim:
            gain = torch.zeros(B, device=dev)
        elif mode == "canonical":
            gain = torch.full((B,), float(self.sim.gain), device=dev)
        else:
            gain = float(self.sim.gain) * torch.exp((2 * torch.rand(B, device=dev, generator=gen) - 1) * math.log(spread))
        P.update({"zoom": zoom, "offset": off, "gain": gain})
        if self.anom_enabled and mode != "canonical":
            P.update(self._anomaly_params(B, gen))
        if self.bg_enabled and mode != "canonical":
            P["bg_level"] = float(cfg.profile.bg_level_sigma) * torch.randn(B, device=dev, generator=gen)
            P["bg_tilt"] = float(cfg.profile.bg_tilt_sigma) * torch.randn(B, device=dev, generator=gen)
        # departures from the anchor rule where nothing is measured
        if self.n_unk_eps > 0 and mode != "canonical":
            P["unk_eps"] = torch.randn(B, self.n_unk_eps, device=dev, generator=gen)
        if self.t_time_spread is not None and mode != "canonical":
            P["text_eps"] = torch.randn(B, device=dev, generator=gen)
        if self.bg_pure and self.bg_enabled and mode != "canonical" and self.bg_trend:
            # background_trend monotone: the regional variation is the trend's contrast about its middle, exp(c0 + c1
            # (tau - 1/2)) (a decay along the profile stays one)
            P["bg_contrast"] = float(self.bg_contrast_sigma[0]) * torch.randn(B, device=dev, generator=gen)
            P["bg_contrast_t"] = float(self.bg_contrast_sigma[1]) * torch.randn(B, device=dev, generator=gen)
        elif self.bg_pure and self.bg_enabled and mode != "canonical":
            # the background's regional trend differs from profile to profile (the measured background is one
            # realisation of the host response): level slope and curvature and decay-rate slope along the profile, as
            # large as the per-profile level / decay-rate modulation.  Quadratic along the profile, so it has no
            # anomaly-scale content.
            P["bg_lslope"] = float(cfg.profile.bg_level_sigma) * torch.randn(B, device=dev, generator=gen)
            P["bg_lcurv"] = float(cfg.profile.bg_level_sigma) * torch.randn(B, device=dev, generator=gen)
            P["bg_tslope"] = float(cfg.profile.bg_tilt_sigma) * torch.randn(B, device=dev, generator=gen)
        if self.n_ma > 0 and "ma_any" in P:
            # which measured anomalies the profile carries and their amplitude factors (drawn last)
            u = torch.rand(B, self.n_ma, 2, device=dev, generator=gen)
            if "ma_on_draw" in P:      # count mode: drawn with the anomaly count (_anomaly_params)
                P["ma_on"] = P.pop("ma_on_draw")
            else:
                P["ma_on"] = P["ma_any"].unsqueeze(1) * (u[..., 0] < self.ma_prob).float()
            # each anomaly's own factor range
            P["ma_scale"] = torch.stack([torch.exp(math.log(lo_) + u[:, j, 1] * (math.log(hi_) - math.log(lo_)))
                                         for j, (lo_, hi_) in enumerate(self.ma_scale)], dim=1)
        return P

    def _dist_pow(self, d: torch.Tensor, gamma: float) -> torch.Tensor:
        """d^gamma for distances d >= 0 (metres) with a bounded slope: exact beyond eps = 1 % of the median station
        spacing, linear inside it (eps^(gamma - 1) d, continuous at eps), so value and derivative are finite
        everywhere. d^gamma alone has the slope gamma d^(gamma - 1): infinite at d = 0 for gamma < 1, and its
        backward gave 0 x inf = NaN wherever the distance is clamped to 0.  No unwarped station lies within eps of an
        edge."""
        eps = 0.01 * max(float(self.spacing_med), 1e-6)
        far = d > eps
        return torch.where(far, torch.where(far, d, torch.full_like(d, eps)).pow(gamma), d * (eps ** (float(gamma) - 1.0)))

    def _unmeasured_departure(self, depth: torch.Tensor,
                              P: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(B, W, T) departure of the measured-background anchor from its rule where nothing is measured: per profile
        one N(0, 1) draw per extended end, per interior gap and for the late-time extension, times the calibrated
        spread a_t dist^gamma (ends: distance to the last known station; gaps: bridge distance d1 d2 / (d1 + d2),
        zero at both known stations; late times: a dlog t^gamma after the last known time).  Returns (profile-axis
        part, late-time part), each None when it does not apply."""
        out: Optional[torch.Tensor] = None
        if "unk_eps" in P and self.n_unk_eps > 0:
            e = P["unk_eps"]
            z = depth.float()
            lo_e, hi_e = self.unk_edges
            dlo, dhi = (lo_e - z).clamp_min(0.0), (z - hi_e).clamp_min(0.0)
            # _dist_pow: bounded slope at the last measured station
            out = (e[:, 0, None, None] * self.t_unk_a[0].view(1, 1, -1)
                   * self._dist_pow(dlo, self.unk_gamma[0]).unsqueeze(-1)
                   + e[:, 1, None, None] * self.t_unk_a[1].view(1, 1, -1)
                   * self._dist_pow(dhi, self.unk_gamma[1]).unsqueeze(-1))
            for g, (za, zb) in enumerate(self.unk_gaps):
                d1, d2 = (z - za).clamp_min(0.0), (zb - z).clamp_min(0.0)
                inside = (z > za) & (z < zb)
                deff = torch.where(inside, self._dist_pow(d1 * d2 / (d1 + d2).clamp_min(1e-9), self.unk_gamma[2]),
                                   torch.zeros_like(z))
                out = out + e[:, 2 + g, None, None] * self.t_unk_a[2].view(1, 1, -1) * deff.unsqueeze(-1)
        tt: Optional[torch.Tensor] = None
        if "text_eps" in P and self.t_time_spread is not None:
            tt = P["text_eps"].view(-1, 1, 1) * self.t_time_spread.view(1, 1, -1)
        return out, tt

    def _sim_values(self, depth: torch.Tensor, zoom: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        B, W = depth.shape
        if self.sim.depth_mode == "physical":
            sd = depth.clamp(self.sim_s0, self.sim_s1)
        else:
            # registered on the MEASURED span (the auto gain was matched there); virtual stations of a profile extension
            # continue the registration beyond it and take the sim's end rows past its ends
            u = (depth - self.m0) / max(self.m1 - self.m0, 1e-9)
            sd = self.sim_s0 + (off.unsqueeze(1) + zoom.unsqueeze(1) * u) * (self.sim_s1 - self.sim_s0)
        sdc = sd.clamp(self.sim_s0, self.sim_s1).reshape(-1).contiguous()
        j = (torch.searchsorted(self.t_sim_depth, sdc, right=True) - 1).clamp(0, self.t_sim_depth.numel() - 2)
        w = ((sdc - self.t_sim_depth[j]) / (self.t_sim_depth[j + 1] - self.t_sim_depth[j]).clamp_min(1e-9)).unsqueeze(-1)
        src = self.t_sim_anom if self.sim.level_mode == "structure" else self.t_sim_L
        return ((1 - w) * src[j] + w * src[j + 1]).reshape(B, W, self.T)

    def compose_anchor(self, depth: torch.Tensor, P: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """log10 profiles at the given depths (B, W) for the parameter set P:
        L_M measured (PCHIP along depth), L_S sim-structured, background L_bg = a(d) L_M + (1 - a(d)) L_S,
        anchor L_A = L_bg + log10(1 + r) + background modulation (level + decay-rate tilt).  The measured /
        sim windows with the same modulation (L_Mx, L_Sx) are the critics' real samples, L_base is the anchor
        without its anomalies.  Every modulated profile is projected onto non-increasing decays (cummin)."""
        B, W = depth.shape
        L_M, _ = self._measured_at(self._depth_to_pos(depth))
        dP, dT = self._unmeasured_departure(depth, P)
        if dP is not None or dT is not None:
            if dP is not None:
                L_M = L_M + dP
                if self.bg_trend:
                    # where nothing is measured the drawn departure keeps the decay along the profile: an unmeasured
                    # station lies between its measured neighbours' values (cross-validation hold-outs, profile
                    # extensions; measured stations are not changed)
                    L_M = self._trend_fill(L_M, dP.abs().amax(dim=-1) > 0)
            if dT is not None:
                L_M = L_M + dT
            if self.meas.positive and self.decay_hard:
                # departures stay decays: the whole curve where a station is unmeasured (dP != 0), the extended late times
                # elsewhere; measured curves are untouched
                sel = torch.zeros_like(L_M, dtype=torch.bool)
                if dP is not None:
                    sel = sel | (dP.abs().amax(dim=-1, keepdim=True) > 0)
                if dT is not None:
                    sel = sel | (~torch.as_tensor(self.time_known, device=L_M.device)).view(1, 1, -1)
                L_M = torch.where(sel, torch.cummin(L_M, dim=-1).values, L_M)
        a = P["alpha"]
        if "af_lam" in P:
            f = fourier_field(depth, {"lam": P["af_lam"], "phase": P["af_phase"]})[..., 0]
            a_d = (a.unsqueeze(1) + float(self.cfg.profile.depth_alpha_sigma) * f).clamp(0.0, 1.0)
            a_d = torch.where((P["kind"] > 0).unsqueeze(1), a.unsqueeze(1).expand_as(a_d), a_d)
        else:
            a_d = a.unsqueeze(1).expand(B, W).clone()
        if self.has_sim:
            S = self._sim_values(depth, P["zoom"], P["offset"])
            cov = self.t_cov.view(1, 1, -1)
            if self.sim.level_mode == "structure":
                L_S = L_M + cov * P["gain"].view(-1, 1, 1) * S
            else:
                L_S = cov * S + (1.0 - cov) * L_M
        else:
            L_S = L_M
        ad = a_d.unsqueeze(-1)
        L_bg = ad * L_M + (1.0 - ad) * L_S
        dA = None
        if "an_A" in P:
            r = self.anomaly_ratio(depth, P)
            dA = torch.log10((1.0 + r).clamp_min(10.0 ** (-self.anom_cap))).clamp(-self.anom_cap, self.anom_cap)
        dM = None
        if self.n_ma > 0 and "ma_on" in P:
            # the measured anomalies (labelled): template x on/off x amplitude factor, added like any anomaly
            Am = self._meas_anomalies_at(self._depth_to_pos(depth))                  # (n, B, W, T)
            wgt = (P["ma_on"] * P["ma_scale"]).t().unsqueeze(-1).unsqueeze(-1)        # (n, B, 1, 1)
            dM = (wgt * Am).sum(dim=0)
            dA = dM if dA is None else dA + dM
        dB = None
        if "bg_level" in P:
            dB = P["bg_level"].view(-1, 1, 1) + P["bg_tilt"].view(-1, 1, 1) * (self.t_tau.view(1, 1, -1) - 0.5)
            if "bg_lslope" in P:     # pure: regional trend along the profile (u in [-1/2, 1/2] on the measured span)
                u = ((depth.float() - 0.5 * (self.m0 + self.m1)) / max(self.m1 - self.m0, 1e-9)).unsqueeze(-1)
                dB = (dB + P["bg_lslope"].view(-1, 1, 1) * u + P["bg_lcurv"].view(-1, 1, 1) * (u * u - 1.0 / 12.0)
                      + P["bg_tslope"].view(-1, 1, 1) * u * (self.t_tau.view(1, 1, -1) - 0.5))
            elif "bg_contrast" in P:  # trend: its contrast about the middle of the measured span
                kap = P["bg_contrast"].view(-1, 1, 1) + P["bg_contrast_t"].view(-1, 1, 1) * (self.t_tau.view(1, 1, -1) - 0.5)
                dB = dB + (torch.exp(kap) - 1.0) * (L_bg - self.t_Lmid.view(1, 1, -1))
        if dA is None and dB is None:
            return {"L_M": L_M, "L_S": L_S, "L_A": L_bg, "L_Mx": L_M, "L_Sx": L_S, "L_base": L_bg, "L_baseM": L_bg,
                    "alpha_d": a_d, "dA": None}
        delta = (dA if dA is not None else 0.0) + (dB if dB is not None else 0.0)

        def _cm(X: torch.Tensor) -> torch.Tensor:
            return self._tcm(X, dim=-1)

        L_base = _cm(L_bg + dB) if dB is not None else L_bg
        L_A = _cm(L_bg + delta)
        if getattr(self, "anom_fixed", False):
            # With the synthetic labels free the network's own departure on them was station-scale texture on the
            # anomalies (Liuyuan, Stage-1B: departure depth texture median 0.21 once most profiles carry synthetic
            # anomalies; 0.006 without anomalies)
            L_baseM = L_A
        else:
            # L_baseM: the base plus the measured anomalies, the base the regional departure head blends on
            # (background_scale_head): their shape is the measured one, only their amplitude factor varies
            L_baseM = L_base if dM is None else _cm(L_bg + dM + (dB if dB is not None else 0.0))
        return {"L_M": L_M, "L_S": L_S, "L_A": L_A, "L_Mx": _cm(L_M + delta), "L_Sx": _cm(L_S + delta),
                "L_base": L_base, "L_baseM": L_baseM, "alpha_d": a_d, "dA": dA}

    def build_profile_batch(self, n_windows: int, W: int, mode: str = "train", station_jitter: bool = True,
                            gen: Optional[torch.Generator] = None, starts: Optional[np.ndarray] = None,
                            params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Profile windows on the measured stations: measured profile, registered sim-structured profile and the
        anchor with its random anomaly set (normalised units), plus the anomaly-modulated measured / sim windows.
        `params` re-uses a profile's random parameters (same profile, e.g. on another depth axis)."""
        dev = self.device
        W = int(min(W, self.D))
        if starts is None:
            allowed = self._starts(W)
            st = torch.as_tensor(allowed[self.np_rng.integers(0, len(allowed), n_windows)], device=dev, dtype=torch.long)
        else:
            st = torch.as_tensor(starts, device=dev, dtype=torch.long)
        pos = st.unsqueeze(1).float() + torch.arange(W, device=dev).float().unsqueeze(0)
        if station_jitter and mode == "train" and W < self.D:
            jit = (torch.rand(n_windows, 1, device=dev, generator=gen) < self.cfg.profile.station_jitter_prob).float()
            pos = (pos + jit * torch.rand(n_windows, 1, device=dev, generator=gen) * 0.999).clamp(max=self.D - 1)
        _, depth = self._measured_at(pos)
        P = params if params is not None else self.sample_profile_params(n_windows, mode, gen)
        c = self.compose_anchor(depth, P)
        lo, sp = float(self.norm.min_val), float(self.norm.max_val - self.norm.min_val)
        return {"y_M": (c["L_M"] - lo) / sp, "y_S": (c["L_S"] - lo) / sp, "y_A": (c["L_A"] - lo) / sp,
                "y_Mx": (c["L_Mx"] - lo) / sp, "y_Sx": (c["L_Sx"] - lo) / sp, "L_base": c["L_base"],
                "L_baseM": c["L_baseM"], "dA": c["dA"],
                "depth": depth, "alpha": P["alpha"], "alpha_d": c["alpha_d"], "kind": P["kind"], "starts": st,
                "reg_zoom": P["zoom"], "reg_offset": P["offset"], "reg_gain": P["gain"], "params": P}

    # ------------------------------------------------------------------ latent + forward
    def sample_latent(self, depth: torch.Tensor, gen: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        B, W = depth.shape
        zd, J = int(self.cfg.model.z_dim), int(self.cfg.profile.latent_basis)
        if self.cfg.profile.enabled:
            A = fourier_field(depth, fourier_params_t(B, J * zd, self.latent_bands, self.latent_probs, 8, gen, depth.device))
        else:
            A = torch.randn(B, W, J * zd, device=depth.device, generator=gen)
        lat = {"A": A}
        if float(self.cfg.profile.latent_white_sigma) > 0:
            # off by default: white noise per station and time sample is noise in a signal generator
            lat["eps"] = torch.randn(B, W, self.T, zd, device=depth.device, generator=gen)
        lat["code"] = torch.randn(B, int(self.cfg.model.profile_code_dim), device=depth.device, generator=gen)
        return lat

    def _cond(self, y: torch.Tensor) -> torch.Tensor:
        return self.condition_extractor(y)

    def _phys5(self, y_norm: torch.Tensor) -> torch.Tensor:
        L = torch_L_from_norm(y_norm[..., 0] if y_norm.dim() == 3 else y_norm, self.norm)[:, self.t_gate_cols]
        z = (self.phys.torch(L) - self.t_phys_mu) / self.t_phys_sd
        return z[:, [0, 2, 4, 5, 7]].clamp(-8, 8)

    def forward_profile(self, batch: Dict[str, torch.Tensor], lat: Dict[str, torch.Tensor],
                        modules: Optional[Dict[str, nn.Module]] = None, train: bool = True) -> Dict[str, torch.Tensor]:
        mods = modules or self._g_modules()
        G, sup, rec = mods["generator"], self.supervisor, mods["recovery"]
        ctx, cmg, mixer = mods.get("profile_ctx"), mods.get("cmg"), mods.get("mixer")
        y_M, y_S, y_A = batch["y_M"], batch["y_S"], batch["y_A"]
        B, W, T = y_A.shape
        N = B * W
        xr, xs, xa = y_M.reshape(N, T, 1), y_S.reshape(N, T, 1), y_A.reshape(N, T, 1)
        a = batch["alpha_d"].reshape(N, 1)
        with torch.no_grad():
            c_real = self._cond(xr)
            c_sim = self._cond(xs) if self.has_sim else c_real
            # the generator / critics / CMG are conditioned on the anchor's own features: with random anomalies the
            # anchor is no longer the real-sim interpolation a c_real + (1 - a) c_sim, and a condition without the
            # anomalies would pull the generated profile back onto the measured
            c_bridge = self._cond(xa)
            h_anchor = self.embedder(xa)
        c_anchor = G.condition_with_amp_token(c_bridge, xa)
        zd, J = int(self.cfg.model.z_dim), int(self.cfg.profile.latent_basis)
        A = lat["A"].view(B, W, J, zd)
        z = torch.einsum("bwjz,jt->bwtz", A, self.t_psi) / math.sqrt(J)
        if "eps" in lat:
            z = z + float(self.cfg.profile.latent_white_sigma) * lat["eps"]
        if ctx is not None:
            dfeat = depth_features(batch["depth"], self.d0, self.d1, self.spacing_med)
            k = ctx(c_bridge.view(B, W, -1).detach(), dfeat, batch["alpha_d"].unsqueeze(-1), lat["code"])
            z = z + k.unsqueeze(2)
        z_seq = z.reshape(N, T, zd)
        h0, gate_t, _ = G(z_seq, c_anchor, c_real, c_sim, a, h_anchor)
        h = h0
        # anchor-relative decoding: the recovery maps the generator's latent DEPARTURE from the anchor to a signal
        # departure, y = cummin_t(anchor + R(h0) - R(h_anchor)).
        y_pair = rec(torch.cat([h0, h_anchor.to(h0.dtype)], dim=0))
        y_decA = y_pair[N:]
        y_nn = self._tcm(xa.to(y_pair.dtype) + (y_pair[:N] - y_decA), dim=1)
        y_fake = y_nn
        out: Dict[str, torch.Tensor] = {"y_decA": y_decA}
        if cmg is not None and mixer is not None:
            with torch.no_grad():
                hf = self.embedder(y_nn.detach())
            cf = self._cond(y_nn)
            # CMG reference = the anchor (its memory bank holds anchor conditions, anomalies included), so the
            # manifold projection cannot erase an anomaly by projecting onto the measured stations
            p5f, p5r = self._phys5(y_nn), self._phys5(xa)
            g = cmg(cf.float(), c_bridge.float(), c_sim.float(), a.float(), p5f, p5r)
            if int(cmg.memory_bank.filled.item()) >= 2:
                pi = cmg.memory_bank.dllr_project(cf.float(), K=cmg.dllr_K)
                c_t = cf.float() + g * (pi - cf.float())
            else:
                c_t = c_bridge.float()
            y_m = rec(torch.cat([mixer(hf, c_t.to(hf.dtype)), hf], dim=0))
            fw = float(self.cfg.reproduction.film_residual)
            y_fake = self._tcm(y_nn + fw * g.view(-1, 1, 1).to(y_nn.dtype) * (y_m[:N] - y_m[N:]), dim=1)
            if train:
                cmg.memory_bank.update(c_bridge.detach().float(), p5r.detach())
                out["cmg_loss"] = cmg.gate_loss(cf.float(), c_bridge.float(), c_sim.float(), a.float(), p5f, p5r)
            out["cmg_gate"] = g.detach()
            out["c_nn"] = cf
        if train:
            out["h_sup"] = sup(h0)
        y_fake = y_fake.reshape(B, W, T)
        if self.bg_pure and W >= 5 and bool(self.cfg.profile.pure_departure_head):
            y_fake, extras = self.background_scale_head(y_fake, batch, return_extras=True)
            out.update(extras)      # warped anchor y_Aw, warp coefficients, largest shift (warp background only)
        out.update({"y_fake": y_fake, "h0": h0, "h": h, "h_anchor": h_anchor, "c_real": c_real,
                    "c_sim": c_sim, "c_bridge": c_bridge, "gate_t": gate_t, "z_seq": z_seq})
        return out

    @staticmethod
    def _split_out(outb: Dict[str, torch.Tensor], nb: int, W: int, first: bool) -> Dict[str, torch.Tensor]:
        """Split a merged forward (nb windows + extra windows) into the main part or the extra part."""
        n_curves = nb * W
        res: Dict[str, torch.Tensor] = {}
        for k, v in outb.items():
            if not torch.is_tensor(v) or v.dim() == 0:
                res[k] = v
            elif k in ("y_fake", "y_Aw", "warp_b", "warp_umax_m"):     # per window (warp outputs too)
                res[k] = v[:nb] if first else v[nb:]
            elif v.shape[0] >= n_curves and k not in ("cmg_loss",):
                res[k] = v[:n_curves] if first else v[n_curves:]
            else:
                res[k] = v
        return res

    # ------------------------------------------------------------------ curve pools (Stage 0 / 0B / 1A)
    def build_curve_pool(self, n_meas: int, n_sim: int, n_bridge: int, seed: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        ys, dom, al = [], [], []
        W = 1   # single-station draws: every depth (incl. the shallow / deep edge stations) equally likely
        # batches of at most pool_draw_chunk windows
        chunk = int(getattr(self.cfg.data, "pool_draw_chunk", 0) or 0)

        def _draw(n: int, which: str) -> None:
            left = n
            while left > 0:
                nw = max(1, math.ceil(left / W))
                if chunk > 0:
                    nw = min(nw, chunk)
                b = self.build_profile_batch(nw, W, mode="train", gen=gen)
                if which == "measured":
                    y, d, a = b["y_M"], 0, torch.ones_like(b["alpha_d"])
                elif which == "sim":
                    y, d, a = b["y_S"], 1, torch.zeros_like(b["alpha_d"])
                else:
                    y, d, a = b["y_A"], 2, b["alpha_d"]
                y = y.reshape(-1, self.T)[:left]
                a = a.reshape(-1)[:left]
                ys.append(y)
                dom.append(torch.full((y.shape[0],), d, dtype=torch.long, device=self.device))
                al.append(a)
                left -= y.shape[0]

        with torch.no_grad():
            _draw(n_meas, "measured")
            if self.has_sim:
                _draw(n_sim, "sim")
            if self.has_sim or self.anom_enabled or self.bg_enabled:
                _draw(n_bridge, "bridge")
        y = torch.cat(ys).clamp(0.0, 1.0).unsqueeze(-1)
        return {"y": y, "domain": torch.cat(dom), "alpha": torch.cat(al).unsqueeze(-1)}

    def _pool_loader(self, pool: Dict[str, torch.Tensor], shuffle: bool = True):
        n = pool["y"].shape[0]
        bs = int(self.cfg.runtime.batch_size)
        perm = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for s in range(0, n - bs + 1 if n >= bs else 1, bs):
            idx = perm[s:s + bs]
            yield {k: v[idx] for k, v in pool.items()}

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, stage: str, epoch: int, metrics: Dict[str, Any], elapsed: float) -> None:
        row = {"stage": stage, "epoch": int(epoch), "global_epoch": int(self.global_epoch), "elapsed_s": float(elapsed)}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()})
        self.history.append(row)

    @staticmethod
    def _set_cosine_lr(opt: Optional[torch.optim.Optimizer], base_lr: float, prog: float, floor_frac: float) -> float:
        """Resume-safe cosine schedule: the learning rate is a function of the stage progress only."""
        lr = base_lr * (floor_frac + (1.0 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0))))
        if opt is not None:
            for g in opt.param_groups:
                g["lr"] = lr
        return lr

    AMP_SCALE_FLOOR = 2.0 ** -10

    def _clip_step(self, params: List[nn.Parameter], scaler, opt) -> bool:
        """Unscale, clip, step; False when the step was skipped.  A step whose unscaled gradient norm is not finite
        moves no weight, with or without the AMP scaler: the scaler tests the SCALED gradients, so once its scale had
        fallen below 2^-128 (reciprocal inf) a clean step was unscaled to inf / NaN and Adam wrote NaN into the
        weights; on the CPU the scaler is off and Adam took any NaN gradient.  The scale backs off as before (x 0.5
        on the scaler's own inf) but stays >= AMP_SCALE_FLOOR."""
        on = bool(scaler.is_enabled())
        if on:
            scaler.unscale_(opt)
        clip = float(self.cfg.train.grad_clip)
        gn = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"))
        ok = bool(torch.isfinite(gn))
        if not ok:
            opt.zero_grad(set_to_none=True)      # no gradient left: the step below moves no weight
            self.skipped_steps += 1
        scaler.step(opt)                         # the scaler's own inf test skips it as before
        scaler.update()
        if on and scaler.get_scale() < self.AMP_SCALE_FLOOR:
            scaler.update(self.AMP_SCALE_FLOOR)
        return ok

    # ------------------------------------------------------------------ Stage 0
    def stage0_autoencoder(self, pool: Dict[str, torch.Tensor]) -> None:
        E = self.cfg.train.stage0_epochs
        s0 = self.stage_progress["stage0"]
        if s0 >= E:
            print(f"[Stage-0] already completed ({s0}/{E}); skipping.")
            return
        print(f"Start Stage-0: embedder/recovery autoencoder on {pool['y'].shape[0]} curves (measured / sim-structured / bridge)")
        t0 = time.time()
        T = self.T
        w_d = float(T * T) / 100.0
        n_batches = max(1, pool["y"].shape[0] // max(1, int(self.cfg.runtime.batch_size)))
        for ep in range(s0, E):
            self.embedder.train()
            self.recovery.train()
            mt, ml = AverageMeter(), AverageMeter()
            for bi, b in enumerate(self._pool_loader(pool)):
                # hold the learning rate for the first half of Stage-0, cosine decay in the second half
                self._set_cosine_lr(self.opt_er, self.cfg.train.lr_er,
                                    max(0.0, 2.0 * (ep + bi / n_batches) / max(E, 1) - 1.0), 0.05)
                y = b["y"]
                self.opt_er.zero_grad(set_to_none=True)
                with autocast(self.use_amp):
                    yr = self.recovery(self.embedder(y))
                lvl = F.mse_loss(yr.float(), y)
                loss = lvl + w_d * F.mse_loss(torch.diff(yr.float(), dim=1), torch.diff(y, dim=1))
                self.scaler_er.scale(loss).backward()
                self._clip_step(list(self.embedder.parameters()) + list(self.recovery.parameters()), self.scaler_er, self.opt_er)
                mt.update(loss.item(), y.shape[0])
                ml.update(lvl.item(), y.shape[0])
            self.stage_progress["stage0"] = ep + 1
            self.global_epoch += 1
            rmse_dec = math.sqrt(max(ml.avg, 0.0)) * (self.norm.max_val - self.norm.min_val)
            self._record("stage0", ep + 1, {"recon": mt.avg, "recon_level_rmse_decades": rmse_dec}, time.time() - t0)
            print(f"[Stage-0] epoch {ep + 1}/{E} | recon={mt.avg:.6g} | level RMSE={rmse_dec:.4f} decades | {time.time() - t0:.1f}s")
            self.maybe_save("stage0", ep, {"recon": mt.avg})
        self.save_stage_end("stage0")

    # ------------------------------------------------------------------ Stage 0B
    def stage0b_condition(self, pool: Dict[str, torch.Tensor]) -> None:
        E = self.cfg.train.stage0b_epochs
        s0 = self.stage_progress["stage0B"]
        if s0 >= E:
            print(f"[Stage-0B] already completed ({s0}/{E}); skipping.")
            self._freeze_condition_frame()
            return
        print("Start Stage-0B: supervised condition frame (domain / alpha / physics statistics), then frozen")
        t0 = time.time()
        set_module_trainable(self.condition_extractor, True)
        for ep in range(s0, E):
            self.condition_extractor.train()
            self.cond_head.train()
            mt = {k: AverageMeter() for k in ("total", "domain", "alpha", "stats", "var")}
            for b in self._pool_loader(pool):
                y, dom, al = b["y"], b["domain"], b["alpha"]
                with torch.no_grad():
                    L = torch_L_from_norm(y[..., 0], self.norm)[:, self.t_gate_cols]
                    tgt = ((self.phys.torch(L) - self.t_phys_mu) / self.t_phys_sd).clamp(-8, 8)
                self.opt_c.zero_grad(set_to_none=True)
                with autocast(self.use_amp):
                    c = self.condition_extractor(y)
                    logits, ap, sp = self.cond_head(c)
                l_dom = F.cross_entropy(logits.float(), dom)
                l_al = F.smooth_l1_loss(ap.float(), al)
                l_st = F.smooth_l1_loss(sp.float(), tgt)
                l_var = F.relu(0.03 - F.normalize(c.float(), dim=1).var(dim=0, unbiased=False).mean())
                loss = l_dom + 2.0 * l_al + 2.0 * l_st + 0.5 * l_var
                self.scaler_c.scale(loss).backward()
                self._clip_step(list(self.condition_extractor.parameters()) + list(self.cond_head.parameters()),
                                self.scaler_c, self.opt_c)
                for k, v in (("total", loss), ("domain", l_dom), ("alpha", l_al), ("stats", l_st), ("var", l_var)):
                    mt[k].update(v.item(), y.shape[0])
            self.stage_progress["stage0B"] = ep + 1
            self.global_epoch += 1
            met = {k: v.avg for k, v in mt.items()}
            self._record("stage0B", ep + 1, met, time.time() - t0)
            print(f"[Stage-0B] epoch {ep + 1}/{E} | " + " | ".join(f"{k}={v:.4g}" for k, v in met.items())
                  + f" | {time.time() - t0:.1f}s")
            self.maybe_save("stage0B", ep, {"cond": met["total"]})
        self._freeze_condition_frame()
        self.save_stage_end("stage0B")

    def _freeze_condition_frame(self) -> None:
        if self.cfg.reproduction.freeze_condition_extractor:
            self.condition_extractor.eval()
            set_module_trainable(self.condition_extractor, False)

    # ------------------------------------------------------------------ Stage 1A
    def stage1a_supervisor(self, pool: Dict[str, torch.Tensor]) -> None:
        E = self.cfg.train.stage1_supervisor_epochs
        s0 = self.stage_progress["stage1A"]
        if s0 >= E:
            print(f"[Stage-1A] already completed ({s0}/{E}); skipping.")
            return
        print("Start Stage-1A: supervisor (latent next-gate dynamics)")
        t0 = time.time()
        self.embedder.eval()
        for ep in range(s0, E):
            self.supervisor.train()
            mt = AverageMeter()
            for b in self._pool_loader(pool):
                with torch.no_grad():
                    h = self.embedder(b["y"])
                self.opt_s.zero_grad(set_to_none=True)
                with autocast(self.use_amp):
                    hs = self.supervisor(h)
                loss = F.mse_loss(hs[:, :-1].float(), h[:, 1:].float())
                self.scaler_s.scale(loss).backward()
                self._clip_step(list(self.supervisor.parameters()), self.scaler_s, self.opt_s)
                mt.update(loss.item(), h.shape[0])
            self.stage_progress["stage1A"] = ep + 1
            self.global_epoch += 1
            self._record("stage1A", ep + 1, {"sup": mt.avg}, time.time() - t0)
            print(f"[Stage-1A] epoch {ep + 1}/{E} | sup={mt.avg:.6g} | {time.time() - t0:.1f}s")
            self.maybe_save("stage1A", ep, {"sup": mt.avg})
        self.supervisor.eval()
        set_module_trainable(self.supervisor, False)
        self.save_stage_end("stage1A")

    # ------------------------------------------------------------------ generator objective
    def _window_plan(self) -> Tuple[int, int]:
        mix = _parse_mix(self.cfg.profile.window_mix, "16:1.0") if self.cfg.profile.enabled else [(str(self.cfg.profile.window), 1.0)]
        r = random.random()
        acc = 0.0
        choice = mix[-1][0]
        for k, w in mix:
            acc += w
            if r <= acc:
                choice = k
                break
        W = self.D if choice == "full" else int(choice)
        W = int(max(4, min(W, self.D, self.max_window)))
        nb = int(max(2, round(self.cfg.runtime.batch_size / W)))
        return nb, W

    def station_weights(self, W: int) -> torch.Tensor:
        """Importance weight per station for windows of length W so that, over the window mix, every
        station contributes equally to the anchor terms (uniform window starts over-sample the middle:
        with W=16 on 54 stations an edge station is covered by 1/39 of the windows, a central one by 16/39)."""
        key = ("_sw", int(W))
        if key not in self._window_starts:
            mix = _parse_mix(self.cfg.profile.window_mix, "16:1.0") if self.cfg.profile.enabled else [(str(self.cfg.profile.window), 1.0)]
            count = np.zeros(self.D)
            for k_, p_ in mix:
                Wk = int(max(4, min(self.D if k_ == "full" else int(k_), self.D, self.max_window)))
                nb = max(2, round(self.cfg.runtime.batch_size / Wk))
                st = self._starts(Wk)
                cov = np.zeros(self.D)
                for s0 in st:
                    cov[s0:s0 + Wk] += 1.0
                count += p_ * nb * cov / max(len(st), 1)
            count = np.maximum(count, 1e-9)
            w = np.where(self.keep_mask, count[self.keep_mask].mean() / count, 0.0)
            self._window_starts[key] = torch.tensor(np.clip(w, 0.0, 8.0), dtype=torch.float32, device=self.device)
        return self._window_starts[key]

    def _instance_noise(self, epoch_in_stage: int) -> float:
        R = self.cfg.reproduction
        frac = min(1.0, epoch_in_stage / max(1, R.instance_noise_decay_epochs))
        return float(R.instance_noise_sigma_start + (R.instance_noise_sigma_floor - R.instance_noise_sigma_start) * frac)

    def generator_losses(self, batch, out, out2, stage: str, scales: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        Lc, P = self.cfg.losses, self.cfg.profile
        y_f, y_A, y_M = out["y_fake"].float(), batch["y_A"].float(), batch["y_M"].float()
        # the warped anchor (warp background manifold: the window's trend re-sampled by its own warp, the labels in
        # place): the reference of the label, structure, roughness and trend terms, whose departure would otherwise
        # count the warp.
        y_Ar = out["y_Aw"].float() if out.get("y_Aw") is not None else y_A
        B, W, T = y_f.shape
        st: Dict[str, float] = {}
        terms: Dict[str, torch.Tensor] = {}
        sw = self.station_weights(W)[batch["starts"].unsqueeze(1) + torch.arange(W, device=self.device).unsqueeze(0)]
        sw = (sw / sw.mean().clamp_min(1e-6)).unsqueeze(-1)                        # (B, W, 1) station balance
        terms["wave"] = Lc.lambda_bridge_wave * scales["anchor"] * (((y_f - y_A).abs() * self.t_wseg) * sw).sum(-1).mean()
        terms["shape"] = Lc.lambda_shape * scales["anchor"] * T * (((torch.diff(y_f, dim=-1) - torch.diff(y_A, dim=-1)).abs()
                                                                    * self.t_wseg_d) * sw).sum(-1).mean()
        if Lc.lambda_bridge_feat > 0:
            c_gen = out["c_nn"] if "c_nn" in out else self._cond(out["y_fake"].reshape(-1, T, 1))
            terms["feat"] = Lc.lambda_bridge_feat * scales["anchor"] * (
                F.normalize(c_gen.float(), dim=1) - F.normalize(out["c_bridge"].float().detach(), dim=1)).abs().mean()
        if "h_sup" in out:   # learned latent dynamics: the frozen supervisor must predict the generated latent
            terms["sup"] = Lc.lambda_sup * F.mse_loss(out["h_sup"][:, :-1].float(), out["h"][:, 1:].float())
        yf2 = y_f.reshape(-1, T)
        ya2 = y_A.reshape(-1, T)
        terms["mom"] = Lc.lambda_mom * ((yf2.mean(0) - ya2.mean(0)).abs().mean() + (yf2.std(0) - ya2.std(0)).abs().mean())
        Lf = torch_L_from_norm(yf2[:, self.t_gate_cols], self.norm)
        pf_raw = self.phys.torch(Lf)
        zf = ((pf_raw - self.t_phys_mu) / self.t_phys_sd).clamp(-12, 12)
        if self.fam is not None:
            # field shell: the background curves (no labelled anomaly above fam_bg_tol at the station) stay inside the
            # field-admissible manifold of the measured background curves, squared excess in robust sd
            ex = self.fam.excess_torch(pf_raw)
            bgm = ((batch["dA"].abs().amax(dim=-1) <= self.fam_bg_tol).reshape(-1) if batch.get("dA") is not None
                   else torch.ones_like(ex, dtype=torch.bool))
            if not self.fam_all_known:
                # known stations only, as the acceptance gate (both stations around a jittered window position)
                pos = self._depth_to_pos(batch["depth"].float())
                kn = (self.t_fam_span[pos.floor().long().clamp(0, self.D - 1)]
                      & self.t_fam_span[pos.ceil().long().clamp(0, self.D - 1)])
                bgm = bgm & kn.reshape(-1)
            if bool(bgm.any()):
                if self.fam_on and Lc.lambda_field_shell > 0:
                    terms["fam"] = Lc.lambda_field_shell * F.relu(ex[bgm]).pow(2).mean()
                st["fam_admissible"] = float((ex[bgm].detach() <= 0).float().mean().item())
        if out.get("cmg_gate") is not None:
            st["cmg_gate_mean"] = float(out["cmg_gate"].float().mean().item())
        if out.get("warp_umax_m") is not None:
            st["warp_umax_m"] = float(out["warp_umax_m"].float().median().item())
        if Lc.lambda_guid > 0:
            terms["guid"] = Lc.lambda_guid * torch.log1p(F.relu(self.t_phys_lo - zf) + F.relu(zf - self.t_phys_hi)).mean()
        if Lc.lambda_dist > 0:
            # reference = the batch's own anchors (the admissible set the batch is drawn from, its random anomalies
            # included), paired with their generated curves.  the fixed union reference (measured + canonical sim + 24
            # composed profiles) left most anomaly configurations without a near reference, and the chamfer's recall
            # half pulled anomalous curves toward anomaly-free references.
            with torch.no_grad():
                La = torch_L_from_norm(ya2[:, self.t_gate_cols], self.norm)
                za = ((self.phys.torch(La) - self.t_phys_mu) / self.t_phys_sd).clamp(-12, 12)
            sel = torch.randperm(zf.shape[0], device=self.device)[:256]
            terms["dist"] = Lc.lambda_dist * soft_chamfer(zf[sel], za[sel], self.phys_radius)
        if Lc.lambda_over_expand > 0:
            terms["env"] = Lc.lambda_over_expand * (F.relu(self.t_env_lo - y_f) + F.relu(y_f - self.t_env_hi)).mean() * 10.0
        if Lc.lambda_tsmooth > 0:
            # time-direction texture: high-frequency part of the log-log curvature must not exceed the anchor's
            def _hf(y: torch.Tensor) -> torch.Tensor:
                d2 = y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]
                flat = d2.reshape(-1, 1, d2.shape[-1])
                sm = F.avg_pool1d(F.pad(flat, (2, 2), mode="replicate"), 5, stride=1).reshape(d2.shape)
                return ((d2 - sm) ** 2).mean(-1).add(1e-14).sqrt()
            Hf, Ha = _hf(y_f), _hf(y_A)
            terms["tsmooth"] = Lc.lambda_tsmooth * F.relu(Hf - 1.2 * Ha).mean() / (Ha.mean().detach() + 1e-6)
        if self.decay_hard and not self.cfg.model.recovery_monotone_head:
            terms["mono"] = 10.0 * F.relu(torch.diff(y_f, dim=-1)).mean() * T
        elif self.decay_soft and float(Lc.lambda_decay_soft) > 0:
            # soft decay constraint: a rise in time beyond the measured gate-to-gate noise (per grid step) is penalised;
            # nothing is projected, so a real late-time rise of the data is not erased
            terms["decay_soft"] = float(Lc.lambda_decay_soft) * F.relu(torch.diff(y_f, dim=-1) - self.decay_tol_n).mean() * T
        if Lc.lambda_recon > 0:   # keep recovery an inverse of the (frozen) embedder on the anchors
            terms["recon"] = Lc.lambda_recon * F.mse_loss(out["y_decA"].float(), batch["y_A"].reshape(-1, T, 1).float())
        if "cmg_loss" in out and Lc.lambda_mechanism_anchor > 0:
            terms["cmg"] = Lc.lambda_mechanism_anchor * out["cmg_loss"]
        span = float(self.norm.max_val - self.norm.min_val)
        an_a = an_sup = None
        if batch.get("dA") is not None:
            # support of the window's intended anomalies: |field| >= 10 % of its peak, peak >= the tested amplitude
            an_a = batch["dA"].float() / span
            amax = an_a.abs().amax(dim=(1, 2), keepdim=True)
            thr = max(float(self.anom_test_amp), float(self.cfg.generation.qc_anomaly_min_decades))
            an_sup = (an_a.abs() >= 0.1 * amax).float() * (amax * span >= thr).float()
        if out2 is not None and (Lc.lambda_div + Lc.lambda_prof_ms) > 0:
            nb = out2["y_fake"].shape[0]
            dd = (out2["y_fake"].float() - y_f[:nb]).abs()
            # the two latent draws are compared, and pushed apart, OFF the anomaly supports only: there the departure is
            # the profiles' diversity, on a support it blurs the anomaly's shape
            off = (1.0 - an_sup[:nb]) if an_sup is not None else torch.ones_like(dd)
            d12 = (dd * off).sum(dim=(1, 2)) / off.sum(dim=(1, 2)).clamp_min(1.0)
            terms["ms"] = (Lc.lambda_div + Lc.lambda_prof_ms) * F.relu(self.div_target - d12).mean() / max(self.div_target, 1e-6)
            st["ms_pair_dist_decades"] = float(d12.mean().item()) * span
        if (Lc.lambda_anomaly > 0 or Lc.lambda_anomaly_shape > 0) and an_sup is not None:
            # anomaly fidelity: the departure y_f - y_A must not project onto the window's intended anomaly field (its
            # projection coefficient is the retention slope - 1 of the acceptance check).  Departures orthogonal to the
            # anomaly (the network's own diversity) stay free; a systematic rescaling of the designed anomaly does not.
            a, sup = an_a, an_sup
            den = (a * a * sup).sum(dim=(1, 2))
            ok = den > 0
            if bool(ok.any()):
                beta_dev = ((y_f - y_Ar) * a * sup).sum(dim=(1, 2))[ok] / den[ok]
                if Lc.lambda_anomaly > 0:
                    terms["anom"] = Lc.lambda_anomaly * F.smooth_l1_loss(beta_dev, torch.zeros_like(beta_dev), beta=0.1)
                st["anom_retention"] = float((1.0 + beta_dev).mean().item())
                if Lc.lambda_anomaly_shape > 0:
                    # ... nor blur its shape: on the support the departure's part orthogonal to the anomaly (centred; it
                    # sets the shape correlation of the acceptance check) is held below ANOMALY_SHAPE_TOL of the
                    # anomaly's energy, per draw.
                    r_perp = anomaly_shape_residual(y_f - y_Ar, a, sup)[ok]
                    x = F.relu(r_perp - ANOMALY_SHAPE_TOL) / ANOMALY_SHAPE_TOL
                    terms["anom_shape"] = Lc.lambda_anomaly_shape * (x / (1.0 + x)).mean()
                    st["anom_blur"] = float(r_perp.median().item())
        if self.bg_pure and Lc.lambda_purity > 0 and W >= 5:
            # background purity: off its labelled anomalies a generated profile is smooth geological background, no bump
            # or dip at anomaly scale.  The anchors carry none (purified background + labels).
            E = y_A * span + float(self.norm.min_val) - batch["L_base"].float()   # labelled anomalies' effect, decades
            base = batch["L_baseM"] if batch.get("L_baseM") is not None else batch["L_base"]
            m2 = self.synthetic_label_weight(y_A * span + float(self.norm.min_val) - base.float())  # head's labels
            hp, off = self.purity_map((y_f - y_Ar) * span, E, batch["depth"], m_syn=m2)
            tol = float(self.purity_thr)
            xp = F.relu(hp - PURITY_LOSS_MARGIN * tol) / tol
            rms = (xp.pow(2).sum(dim=(1, 2)) / off.sum(dim=(1, 2)).clamp_min(1.0) + 1e-12).sqrt()
            terms["purity"] = Lc.lambda_purity * rms.mean()
            with torch.no_grad():
                pmax = hp.amax(dim=(1, 2))
                st["purity_max_decades"] = float(pmax.median().item())
                st["purity_within_tol"] = float((pmax <= tol).float().mean().item())
        if Lc.lambda_copy_guard > 0:
            near = (batch["alpha_d"].min(dim=1).values >= 0.95)
            if bool(near.any()):
                dm = (y_f - y_M).abs().mean(dim=(1, 2))[near]
                terms["copy"] = Lc.lambda_copy_guard * F.relu(self.copy_min - dm).mean() / max(self.copy_min, 1e-6)
        if self.cfg.profile.enabled and W >= 3:
            # the hinge bounds the local depth structure of the DEPARTURE from the anchor: |LS(y_f - y_A)| <= sqrt(tol^2
            # - 1) |LS(y_A)| + floor keeps the total ratio |LS(y_f)| / |LS(y_A)| <= tol for an uncorrelated departure.
            rd = local_structure_torch(y_f - y_Ar, batch["depth"])
            ra = local_structure_torch(y_Ar, batch["depth"])
            Rd = (rd.pow(2).mean(dim=(1, 2)) + 1e-12).sqrt()
            Ra = (ra.pow(2).mean(dim=(1, 2)) + 1e-12).sqrt()
            excess = F.relu(Rd - math.sqrt(max(P.rough_tol ** 2 - 1.0, 0.0)) * Ra - self.rough_floor)
            terms["rough"] = Lc.lambda_prof_rough * excess.mean() / (Ra.detach().mean() + self.rough_floor + 1e-5)
            with torch.no_grad():
                Rf = (local_structure_torch(y_f, batch["depth"]).pow(2).mean(dim=(1, 2)) + 1e-12).sqrt()
            st["rough_ratio"] = float((Rf / (Ra + self.rough_floor + 1e-9)).mean().item())
            if Lc.lambda_structure > 0:
                # structure fidelity: the departure's local depth structure must not project onto the anchor's own
                # (measured narrow features, sim structure, anomalies), per window and time band. Per band, so the late
                # gates cannot be traded against the larger early ones; departures orthogonal to the anchor's structure
                # (diversity) stay free.
                beta_s, ok_s = structure_projection(rd, ra, self.struct_thr_n)
                if out2 is not None:
                    # only the SYSTEMATIC part (the centre over latent draws) is constrained, on the windows that carry
                    # a second latent draw (the mode-seeking pair): centred_smooth_l1.
                    nb2 = int(out2["y_fake"].shape[0])
                    y_Ar2 = out2["y_Aw"].float() if out2.get("y_Aw") is not None else y_A[:nb2]   # the draw's own warp
                    rd2 = local_structure_torch(out2["y_fake"].float() - y_Ar2, batch["depth"][:nb2])
                    ra2 = local_structure_torch(y_Ar2, batch["depth"][:nb2]) if out2.get("y_Aw") is not None else ra[:nb2]
                    beta2, _ = structure_projection(rd2, ra2, self.struct_thr_n)
                    ok2 = ok_s[:nb2]
                    if bool(ok2.any()):
                        b1, b2 = beta_s[:nb2][ok2], beta2[ok2]
                        terms["struct"] = Lc.lambda_structure * centred_smooth_l1(b1, b2)
                        st["struct_retention"] = float((1.0 + 0.5 * (b1 + b2)).mean().item())
                elif bool(ok_s.any()):      # no second draw (mode seeking off): every draw's coefficient is penalised
                    bsel = beta_s[ok_s]
                    terms["struct"] = Lc.lambda_structure * F.smooth_l1_loss(bsel, torch.zeros_like(bsel), beta=0.1)
                    st["struct_retention"] = float((1.0 + bsel).mean().item())
            lp = depth_lowpass_torch(y_f - y_Ar, float(P.trend_sigma_stations))
            terms["trend"] = Lc.lambda_prof_trend * F.relu(lp.abs() - P.trend_tol).mean()
        if stage == "stage2" and scales.get("adv", 0.0) > 0:
            a = batch["alpha_d"].reshape(-1, 1)
            adv = ns_g_loss(self.discriminator(out["h"], out["c_bridge"]))
            fm = torch.zeros((), device=self.device)
            if self.sequence_discriminator is not None:
                sx, fx = self.sequence_discriminator(out["y_fake"].reshape(-1, T, 1), out["c_bridge"], a, return_features=True)
                adv = adv + ns_g_loss(sx)
                if self.cfg.losses.lambda_fm > 0:
                    with torch.no_grad():
                        _, fr = self.sequence_discriminator(y_Ar.detach().reshape(-1, T, 1), out["c_bridge"], a,
                                                            return_features=True)
                    fm = fm + (fx.float().mean(0) - fr.float().mean(0)).abs().mean()
            terms["adv"] = Lc.lambda_adv * scales["adv"] * adv
            if self.profile_discriminator is not None:
                x4 = ProfilePatchDiscriminator.make_input(out["y_fake"], batch["depth"], self.d0, self.d1, self.spacing_med)
                sp, fp = self.profile_discriminator(x4, return_features=True)
                terms["prof_adv"] = Lc.lambda_prof_adv * scales["adv"] * ns_g_loss(sp)
                if self.cfg.losses.lambda_fm > 0:
                    with torch.no_grad():
                        xr4 = ProfilePatchDiscriminator.make_input(self._warp_real(batch["y_Mx"], batch, out), batch["depth"],
                                                                   self.d0, self.d1, self.spacing_med)
                        _, fpr = self.profile_discriminator(xr4, return_features=True)
                    fm = fm + (fp.float().mean(0) - fpr.float().mean(0)).abs().mean()
            if self.cfg.losses.lambda_fm > 0:
                terms["fm"] = self.cfg.losses.lambda_fm * scales["adv"] * fm
        total = sum(terms.values())
        for k, v in terms.items():
            st[k] = float(v.detach().item())
        st["total"] = float(total.detach().item())
        return total, st

    def discriminator_step(self, batch, out, epoch_in_stage: int, step: int) -> Dict[str, float]:
        st: Dict[str, float] = {}
        if self.opt_d is None:
            return st
        T = self.T
        sigma = self._instance_noise(epoch_in_stage)
        a = batch["alpha_d"].reshape(-1, 1)
        c = out["c_bridge"].detach()
        do_r1 = (step % max(1, self.cfg.train.r1_every) == 0) and self.cfg.train.r1_gamma > 0
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            h_real = self.embedder(batch["y_A"].reshape(-1, T, 1))
        with autocast(self.use_amp):
            loss = ns_d_loss(self.discriminator(h_real, c), self.discriminator(out["h"].detach(), c))
            if self.sequence_discriminator is not None:
                xr = self._warp_real(batch["y_A"], batch, out).reshape(-1, T, 1)
                xf = out["y_fake"].detach().reshape(-1, T, 1)
                if sigma > 0:
                    xr = xr + sigma * torch.randn_like(xr)
                    xf = xf + sigma * torch.randn_like(xf)
                loss = loss + ns_d_loss(self.sequence_discriminator(xr, c, a), self.sequence_discriminator(xf, c, a))
            if self.profile_discriminator is not None:
                y_real = self._critic_real_windows(batch, out)
                yr4 = ProfilePatchDiscriminator.make_input(y_real, batch["depth"], self.d0, self.d1, self.spacing_med)
                yf4 = ProfilePatchDiscriminator.make_input(out["y_fake"].detach(), batch["depth"], self.d0, self.d1, self.spacing_med)
                if sigma > 0:
                    yr4 = yr4 + sigma * torch.randn_like(yr4)
                    yf4 = yf4 + sigma * torch.randn_like(yf4)
                loss = loss + ns_d_loss(self.profile_discriminator(yr4), self.profile_discriminator(yf4))
        st["D"] = float(loss.detach().item())
        if do_r1:
            r1 = torch.zeros((), device=self.device)
            with autocast(False):
                if self.sequence_discriminator is not None:
                    xr = self._warp_real(batch["y_A"], batch, out).reshape(-1, T, 1).detach().float().requires_grad_(True)
                    r1 = r1 + r1_penalty(self.sequence_discriminator(xr, c.float(), a.float()), xr)
                if self.profile_discriminator is not None:
                    yr4 = ProfilePatchDiscriminator.make_input(self._critic_real_windows(batch, out).float(), batch["depth"],
                                                               self.d0, self.d1, self.spacing_med).detach().requires_grad_(True)
                    r1 = r1 + r1_penalty(self.profile_discriminator(yr4), yr4)
            loss = loss + 0.5 * self.cfg.train.r1_gamma * self.cfg.train.r1_every * r1
            st["R1"] = float(r1.detach().item())
        self.scaler_d.scale(loss).backward()
        self._clip_step(self._d_params(), self.scaler_d, self.opt_d)
        return st

    def _warp_real(self, y: torch.Tensor, batch, out: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """A real window (B, W, T) with the generated window's own background warp (detached), kept a decay in time:
        the critics then compare texture and coherence on the same warped trend and never judge the warp, which the
        anchor-fidelity terms bound.  Unchanged without a warp."""
        if out is None or out.get("y_Aw") is None:
            return y
        return self._tcm(y + (out["y_Aw"] - batch["y_A"]).detach().to(y.dtype), dim=-1)

    def _critic_real_windows(self, batch, out: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Real profile windows: measured / sim windows carrying the SAME anomaly set and background modulation as
        the anchor, or the anchor itself — the critic judges texture and depth coherence, never 'anomaly or not'.
        with the generated window's background warp (_warp_real)."""
        mix = dict(_parse_mix(self.cfg.profile.critic_real_mix, "measured:1.0"))
        B = batch["y_Mx"].shape[0]
        pm = mix.get("measured", 1.0)
        ps = mix.get("sim", 0.0) if self.has_sim else 0.0
        pa = mix.get("anchor", 0.0)
        tot = max(pm + ps + pa, 1e-9)
        u = torch.rand(B, device=self.device) * tot
        sel = torch.where(u < pm, 0, torch.where(u < pm + ps, 1, 2)).view(B, 1, 1)
        return self._warp_real(torch.where(sel == 0, batch["y_Mx"], torch.where(sel == 1, batch["y_Sx"], batch["y_A"])),
                               batch, out)

    # ------------------------------------------------------------------ Stage 1B / 2 loop
    def _set_train_mode(self, flag: bool) -> None:
        for n, mmod in self._g_modules().items():
            mmod.train(flag)
        self.embedder.eval()
        self.condition_extractor.eval()
        # frozen supervisor (no dropout): train mode only because cuDNN cannot back-propagate an eval-mode RNN
        self.supervisor.train(flag)

    def train_generator_stage(self, stage: str) -> None:
        E = self.cfg.train.stage1_epochs if stage == "stage1B" else self.cfg.train.stage2_epochs
        s0 = self.stage_progress[stage]
        if s0 >= E:
            print(f"[{stage}] already completed ({s0}/{E}); skipping.")
            return
        title = ("Stage-1B: profile union-bridge pretraining (anchor + physics + profile constraints)" if stage == "stage1B"
                 else "Stage-2: adversarial refinement (latent / sequence / profile critics, EMA)")
        print(f"Start {title}")
        if stage == "stage1B":
            print(f"[stage1B] best-checkpoint selection from epoch {self.stage1_select_from()} of {E} "
                  f"(min(--stage1_min_epochs, half of the stage): the latent spread develops first)")
        t0 = time.time()
        set_module_trainable(self.supervisor, False)
        steps = int(max(1, self.cfg.train.steps_per_epoch))
        R = self.cfg.reproduction
        patience = int(self.cfg.train.early_stop_patience)
        bad = 0
        self._freeze_condition_frame()
        if stage == "stage1B" and s0 == 0:
            self.ema.reset(self._g_modules())   # EMA starts from the pretrained recovery / supervisor
        critics = [d for d in (self.discriminator, self.sequence_discriminator, self.profile_discriminator) if d is not None]
        for ep in range(s0, E):
            self._set_train_mode(True)
            prog = (ep + 1) / max(E, 1)
            if stage == "stage2":
                anchor_scale = 1.0 - (1.0 - R.stage2_anchor_floor) * prog if R.curriculum else 1.0
                adv_scale = min(1.0, (ep + 1) / max(1.0, 0.2 * E)) if R.curriculum else 1.0
                lr_now = self._set_cosine_lr(self.opt_g, self.cfg.train.lr_g, ep / max(E, 1), 0.1)
                self._set_cosine_lr(self.opt_d, self.cfg.train.lr_d, ep / max(E, 1), 0.1)
            else:
                anchor_scale, adv_scale = 1.0, 0.0
                lr_now = self._set_cosine_lr(self.opt_g, self.cfg.train.lr_g, 0.0, 1.0)
            scales = {"anchor": anchor_scale, "adv": adv_scale}
            meters: Dict[str, AverageMeter] = {}
            use_ms = R.mode_seek_two_z_enabled and (self.cfg.losses.lambda_div + self.cfg.losses.lambda_prof_ms) > 0
            # throughput and peak memory of the training steps (server sizing: real-step self-test)
            t_ep, n_cv = time.time(), 0
            n_ok = n_bad_loss = n_bad_grad = 0      # finite / skipped generator steps of the epoch
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            for it in range(steps):
                nb, W = self._window_plan()
                batch = self.build_profile_batch(nb, W, mode="train")
                lat = self.sample_latent(batch["depth"])
                half = max(1, nb // 2) if use_ms else 0
                if half:
                    # one pass over [windows | first half of the windows with a second latent draw]
                    lat2 = self.sample_latent(batch["depth"][:half])
                    def _dup(v):    # the profile parameters too (the warp head re-evaluates the trend)
                        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == nb:
                            return torch.cat([v, v[:half]])
                        return {k2: _dup(v2) for k2, v2 in v.items()} if isinstance(v, dict) else v
                    big = {k: _dup(v) for k, v in batch.items()}
                    latb = {k: torch.cat([lat[k], lat2[k]]) for k in lat}
                else:
                    big, latb = batch, lat
                n_cv += int(big["y_A"].shape[0]) * int(W)
                with autocast(self.use_amp):
                    outb = self.forward_profile(big, latb, train=True)
                out = self._split_out(outb, nb, W, first=True)
                out2 = self._split_out(outb, nb, W, first=False) if half else None
                if stage == "stage2":
                    for d in critics:
                        set_module_trainable(d, True)
                        d.train()
                    dst = self.discriminator_step(batch, out, ep + 1, it)
                    for k, v in dst.items():
                        meters.setdefault(k, AverageMeter()).update(v)
                    for d in critics:
                        set_module_trainable(d, False)
                self.opt_g.zero_grad(set_to_none=True)
                with autocast(self.use_amp):
                    total, st = self.generator_losses(batch, out, out2, stage, scales)
                if not torch.isfinite(total):
                    n_bad_loss += 1
                    if n_bad_loss <= 3:
                        bad_t = [k for k, v in st.items() if not math.isfinite(float(v))]
                        print(f"[{stage}] non-finite generator loss at epoch {ep + 1} step {it}; batch skipped"
                              + (f" (terms: {', '.join(bad_t[:8])})" if bad_t else ""))
                    continue
                self.scaler_g.scale(total).backward()
                if self._clip_step(self._g_params(), self.scaler_g, self.opt_g):
                    n_ok += 1
                else:
                    n_bad_grad += 1
                self.ema.update(self._g_modules())
                for k, v in st.items():
                    meters.setdefault(k, AverageMeter()).update(v)
            if n_bad_loss or n_bad_grad:
                print(f"[{stage}] epoch {ep + 1}: {n_bad_loss} non-finite loss(es) and {n_bad_grad} non-finite gradient(s) "
                      f"in {steps} steps, skipped (generator weights unchanged by them)")
            amp_scale = float(self.scaler_g.get_scale()) if self.use_amp else 1.0
            verdict = generator_epoch_verdict(n_ok, n_bad_loss, n_bad_grad, self.use_amp, amp_scale)
            if verdict == "stop":
                raise RuntimeError(
                    f"[{stage}] epoch {ep + 1}: no finite generator step in {steps} steps ({n_bad_loss} non-finite losses, "
                    f"{n_bad_grad} non-finite gradients): the generator cannot train - stopped instead of training on "
                    f"(the lines above name the loss terms; AMP scale {amp_scale:.3g})")
            if verdict == "calibrating":   # the scaler is still finding its scale (short epochs: --smoke)
                print(f"[{stage}] epoch {ep + 1}: all {steps} steps went to the AMP loss-scale calibration (scale now "
                      f"{amp_scale:.3g} >= {AMP_CALIBRATION_MIN_SCALE:g}); the generator trains from the next step on")
            self.stage_progress[stage] = ep + 1
            self.global_epoch += 1
            met = {k: v.avg for k, v in meters.items()}
            met["lr_g"] = lr_now
            met["skipped_steps"] = float(n_bad_loss + n_bad_grad)
            met["curves_per_s"] = n_cv / max(time.time() - t_ep, 1e-9)
            if self.device.type == "cuda":
                met["vram_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            q = self.validate(n_profiles=int(R.validation_profiles))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()   # hand the validation pass's cache back: the GPU is often shared
            met.update({f"val_{k}": v for k, v in q["summary"].items()})
            self._record(stage, ep + 1, met, time.time() - t0)
            loss_keys = [k for k in met if not k.startswith("val_") and k not in ("total", "D", "R1", "ms_pair_dist_decades",
                                                                                     "rough_ratio", "lr_g", "curves_per_s",
                                                                                     "vram_peak_gb", "skipped_steps")]
            core = f"total={met.get('total', float('nan')):.4g} " + " ".join(
                f"{k}={met[k]:.3g}" for k in sorted(loss_keys, key=lambda k: -abs(met[k]))[:9])
            if "D" in met:
                core += f" | D={met['D']:.3g}"
            print(f"[{stage}] epoch {ep + 1}/{E} | {core} | val score={q['score']:.4f} rough/anchor="
                  f"{q['summary']['rough_ratio_vs_anchor']:.3f} cover={q['summary']['measured_coverage']:.3f} "
                  f"div={q['summary']['diversity_decades']:.4f} copy={q['summary']['copy_fraction']:.3f} "
                  f"prec={q['summary']['precision']:.3f} rec={q['summary']['recall']:.3f} "
                  f"accept={q['summary']['accept_rate']:.2f} anom-ret={q['summary']['anomaly_retention']:.3f} "
                  f"anom-corr={q['summary']['anomaly_corr']:.3f} struct-ret={q['summary']['struct_retention']:.3f} "
                  f"feature-ret={q['summary']['feature_retention_first']:.2f}/{q['summary']['feature_retention_mid']:.2f}/"
                  f"{q['summary']['feature_retention_last']:.2f} (min {q['summary']['feature_retention_min']:.2f}) "
                  f"texture={q['summary']['depth_texture']:.3f} "
                  + (f"trend={q['summary']['trend_rate']:.2f} " if math.isfinite(q['summary']['trend_rate']) else "")
                  + (f"fam={q['summary']['field_admissible']:.3f} " if math.isfinite(q['summary']['field_admissible']) else "")
                  + (f"warp={q['summary']['warp_shift_m']:.1f}m " if math.isfinite(q['summary']['warp_shift_m']) else "")
                  + "| "
                  f"{time.time() - t0:.1f}s | {met['curves_per_s']:.0f} curves/s"
                  + (f", peak {met['vram_peak_gb']:.2f} GB" if "vram_peak_gb" in met else ""))
            improved = self.maybe_save(stage, ep, {"score": q["score"]}, quality=q,
                                       eligible=(stage != "stage1B" or ep + 1 >= self.stage1_select_from()))
            if stage == "stage2":
                if improved:
                    bad = 0
                else:
                    bad += 1
                    if patience > 0 and bad >= patience and (ep + 1) >= max(4, int(0.3 * E)):
                        print(f"[{stage}] early stop: no validation improvement for {patience} epochs")
                        self.stage_progress[stage] = E
                        break
        self.save_stage_end(stage)

    # ------------------------------------------------------------------ validation / generation
    def _eval_modules(self, use_ema: bool) -> Dict[str, nn.Module]:
        mods = self.ema.shadow if use_ema else self._g_modules()
        for mmod in mods.values():
            mmod.eval()
        self.embedder.eval()
        self.condition_extractor.eval()
        return mods

    def _band_rough(self, L: torch.Tensor, n_bands: int = 8) -> torch.Tensor:
        """RMS local depth structure per time band, (B, n_bands) — the torch twin of profile_quality_metrics."""
        if L.shape[1] < 3:
            return torch.zeros(L.shape[0], n_bands, device=L.device)
        r = L[:, 1:-1] - ((1.0 - self.t_ls_w) * L[:, :-2] + self.t_ls_w * L[:, 2:])
        return torch.stack([c.pow(2).mean(dim=(1, 2)).sqrt() for c in torch.tensor_split(r, n_bands, dim=2)], dim=1)

    def purity_map(self, dep: torch.Tensor, E: Optional[torch.Tensor],
                   depth: Optional[torch.Tensor] = None,
                   m_syn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Anomaly-scale content (decades, (B, W, T)) of the network's departure from the anchor, dep = L - L_A, off
        the labelled anomalies, and the cells it counts on (1 = off them).  E = the labelled anomalies' effect on the
        anchor, L_A - L_base (the field after the decay projection, which carries a dip that makes a decay rise on
        into later times).  Cells with |E| >= a quarter of the tolerance, widened by one station and one time sample,
        belong to the labels: they neither enter the regional fit (local quadratic at sigma_R) nor are tested; there
        the retention / shape checks hold the label (elsewhere a label's retention error, <= 0.3 |E|, stays below the
        threshold)."""
        if depth is None:
            depth = self.t_depth
        off = self._label_off_mask(E, dep, m_syn)
        hp = masked_quadratic_residual_torch(dep, depth, self.purity_sigma_m, off)
        return hp.abs() * off, off

    def _label_off_mask(self, E: Optional[torch.Tensor], like: torch.Tensor,
                        m_syn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """1 on the cells off the labelled anomalies: |E| < a quarter of the tolerance, also at the neighbouring
        station and time sample (E = L_A - L_base, decades, (B, W, T)). m_syn (the regional departure head's label
        weight, synthetic_label_weight): every cell with m_syn > 0 is on a label too, so the labels are the union of
        the anomalies' footprints and of the cells where the head keeps the network's own departure.  Where a
        synthetic anomaly overlaps a measured one of the other sign their sum E can fall below the cut while the head
        keeps the network's departure on the synthetic label; with E alone those cells were tested as background."""
        if E is None:
            return torch.ones_like(like, dtype=torch.float32)
        cut = 0.25 * float(self.purity_tol)
        core = E.detach().float().abs() >= cut
        if m_syn is not None:
            core = core | (m_syn.detach() > 0)
        core = core.float()
        Tn, Wn = core.shape[-1], core.shape[1]
        c = F.max_pool1d(core.reshape(-1, 1, Tn), kernel_size=3, stride=1, padding=1).reshape(core.shape)
        c = F.max_pool1d(c.transpose(1, 2).reshape(-1, 1, Wn), kernel_size=3, stride=1,
                         padding=1).reshape(c.shape[0], Tn, Wn).transpose(1, 2)
        return (c <= 0).float()

    def synthetic_label_weight(self, E: torch.Tensor) -> torch.Tensor:
        """Label weight of the regional departure head, (B, W, T) in [0, 1], from the synthetic labels' own effect E
        = L_A - L_baseM (decades): s = smoothstep((|E| - tol / 4) / (tol / 4)), then along the log-spaced time grid a
        moving maximum and a moving average over +-label_k samples (LABEL_WEIGHT_SMOOTH_DECADES).  E can cross zero
        in time (anomalies of both signs overlapping, the decay projection of the anchor) or end within a sample; s
        alone then switched between the network's departure and the regional trend within one or two samples and left
        a kink in the decay.  The weight reaches up to 2 label_k samples beyond the label; the purity test counts
        every cell with weight > 0 as a label (_label_off_mask), so it never tests a cell where the head keeps the
        network's departure."""
        q = 0.25 * float(self.purity_tol)
        x = ((E.detach().float().abs() - q) / q).clamp(0.0, 1.0)
        s = x * x * (3.0 - 2.0 * x)
        k = int(getattr(self, "label_k", 0))
        if k > 0 and s.shape[-1] > 1:
            shp = s.shape
            z = s.reshape(-1, 1, shp[-1])
            z = F.max_pool1d(F.pad(z, (k, k), mode="replicate"), kernel_size=2 * k + 1, stride=1)
            z = F.avg_pool1d(F.pad(z, (k, k), mode="replicate"), kernel_size=2 * k + 1, stride=1)
            s = z.reshape(shp)
        return s

    @staticmethod
    def _bg_params(P: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """A profile's parameters without its anomalies (synthetic an_*, measured ma_*): compose_anchor(.)["L_base"]."""
        return {k: v for k, v in P.items() if not (k.startswith("an_") or k.startswith("ma_"))}

    def _warp_phi(self, depth: torch.Tensor, J: int) -> torch.Tensor:
        """Warp modes sin(pi j s) / (pi j), s = (z - m0) / span clamped to [0, 1]: (B, W, J), zero beyond the span."""
        span = max(self.m1 - self.m0, 1e-9)
        s = ((depth.float() - self.m0) / span).clamp(0.0, 1.0)
        j = torch.arange(1, J + 1, device=depth.device, dtype=torch.float32)
        return torch.sin(math.pi * s.unsqueeze(-1) * j) / (math.pi * j)

    def warp_shift(self, depth: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """u(z) = span sum_j b_j sin(pi j s) / (pi j) (metres, (B, W)) for bounded coefficients b (B, J)."""
        return max(self.m1 - self.m0, 1e-9) * (self._warp_phi(depth, int(b.shape[1])) * b.unsqueeze(1)).sum(-1)

    def _warp_bound(self, b: torch.Tensor) -> torch.Tensor:
        """Smooth bound |b|_1 < warp_max_strain: b s tanh(n / s) / n with n the smoothed L1 norm (~b for small b), so
        |du/dz| = |sum_j b_j cos(pi j s)| < s < 1 and z + u(z) is strictly increasing."""
        s_ = float(self.warp_strain)
        n = torch.sqrt(b * b + (1e-6 * s_) ** 2).sum(dim=1, keepdim=True)
        return b * (s_ * torch.tanh(n / s_) / n)

    def _warp_background(self, dep: torch.Tensor, ref: torch.Tensor, w: torch.Tensor,
                         batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Warp of the background trend chosen by the network's departure dep (B, W, T, normalised): the mode
        coefficients b are the weighted least-squares (MAP, ridge warp_ridge) fit of dep by the warp tangents G_j =
        span phi_j(s) d ref / dz after the per-time level and contrast directions {1, ref - mean} are removed (they
        stay with the affine fit; variable projection), bounded by _warp_bound; the warped background is the trend
        itself at the warped depths, compose_anchor(z + u, P without anomalies)["L_base"].  Returns (warped ref
        (normalised), b (B, J), u (B, W) metres).  Differentiable in dep; float32."""
        P = self._bg_params(batch["params"])
        depth = batch["depth"].float()
        lo, sp = float(self.norm.min_val), float(self.norm.max_val - self.norm.min_val)
        J = int(self.warp_J)
        with torch.autocast(device_type=depth.device.type, enabled=False):
            with torch.no_grad():
                h = 0.25 * float(self.spacing_med)
                dref = (self.compose_anchor(depth + h, P)["L_base"].float()
                        - self.compose_anchor(depth - h, P)["L_base"].float()) / (2.0 * h * sp)      # (B, W, T) 1/m
                G = (max(self.m1 - self.m0, 1e-9) * self._warp_phi(depth, J).unsqueeze(2)
                     * dref.unsqueeze(-1))                                                          # (B, W, T, J)
                wf = w.float().unsqueeze(-1)
                sw = wf.sum(dim=1, keepdim=True).clamp_min(1e-6)
                rf = ref.float().unsqueeze(-1)
                rc = rf - (wf * rf).sum(dim=1, keepdim=True) / sw
                G = G - (wf * G).sum(dim=1, keepdim=True) / sw
                G = G - rc * (wf * rc * G).sum(dim=1, keepdim=True) / ((wf * rc * rc).sum(dim=1, keepdim=True) + 1e-9)
                WG = wf * G
                A = torch.einsum("bwtj,bwtk->bjk", WG, G)
                A = A + (self.warp_ridge + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)) \
                    * torch.eye(J, device=A.device, dtype=A.dtype)
            # G is orthogonal to {1, rc} in the weighted product, so the projection of dep needs no explicit removal
            r = torch.einsum("bwtj,bwt->bj", WG, dep.float())
            b = self._warp_bound(torch.linalg.solve(A, r.unsqueeze(-1))[..., 0])
            u = self.warp_shift(depth, b)
            ref_w = (self.compose_anchor(depth + u, P)["L_base"].float() - lo) / sp
        return ref_w, b, u

    def background_scale_head(self, y_fake: torch.Tensor, batch: Dict[str, torch.Tensor],
                              return_extras: bool = False):
        """Regional departure head (background_mode pure): off the labelled anomalies the network's departure from
        its anchor, y_fake - y_A, is replaced by its weighted least-squares quadratic along the window / profile, per
        time sample (masked_quadratic_trend_torch, weights 1 - m); on the labels (m = 1) the departure is kept (the
        anomaly retention / shape terms hold it); the result is projected onto non-increasing decays.  A profile off
        its labels is then the purified background anchor plus a regional trend whose level, slope and curvature the
        network sets at every time sample: the purity of a "0 anomalies" profile is a property of the architecture."""
        y_A = batch["y_A"].float()
        sp, lo = float(self.norm.max_val - self.norm.min_val), float(self.norm.min_val)
        # label weight from the synthetic labels' own effect E = L_A - L_baseM (synthetic_label_weight): 0 on every cell
        # the purity test checks (_label_off_mask counts the weight's cells as labels), 1 on the labels, filled and
        # ramped along time.
        base = batch["L_baseM"] if batch.get("L_baseM") is not None else batch["L_base"]
        m = self.synthetic_label_weight(y_A * sp + lo - base.float())
        dep = y_fake.float() - y_A
        extras: Dict[str, torch.Tensor] = {}
        if getattr(self, "bg_trend", False):
            # background_trend monotone: off the labels the departure is the level and depth contrast of the profile's
            # own trend (masked_trend_fit_torch), so a "0 anomalies" profile stays a decay along the hole.
            ref = (batch["L_base"].float() - lo) / sp
            if getattr(self, "bg_warp", False) and batch.get("params") is not None:
                # the trend is first re-sampled along the hole by the network's monotone warp (_warp_background); the
                # labelled anomalies stay where they are, on the warped background; level and contrast then follow on
                # the warped trend
                ref_w, wb, u = self._warp_background(dep, ref, 1.0 - m, batch)
                y_A = y_A + (ref_w - ref)
                dep = y_fake.float() - y_A
                ref = ref_w
                extras = {"y_Aw": y_A, "warp_b": wb, "warp_umax_m": u.abs().amax(dim=1)}
            fit = masked_trend_fit_torch(dep, ref, 1.0 - m)
        else:
            fit = masked_quadratic_trend_torch(dep, batch["depth"], 1.0 - m)
        y = self._tcm(y_A + fit + m * (dep - fit), dim=-1)
        return (y, extras) if return_extras else y

    @torch.no_grad()
    def _profile_pass(self, P: Dict[str, torch.Tensor], mods: Dict[str, nn.Module],
                      gen: torch.Generator) -> Dict[str, Optional[torch.Tensor]]:
        """Generate full profiles (all measured stations) for the parameter sets P (fresh latent draws)."""
        nb = int(P["alpha"].shape[0])
        b = self.build_profile_batch(nb, self.D, mode="gen", station_jitter=False, gen=gen,
                                     starts=np.zeros(nb, dtype=np.int64), params=P)
        lat = self.sample_latent(b["depth"], gen=gen)
        with autocast(self.use_amp):
            out = self.forward_profile(b, lat, modules=mods, train=False)
        lo, sp = float(self.norm.min_val), float(self.norm.max_val - self.norm.min_val)
        res = {"L": torch_L_from_norm(out["y_fake"].float(), self.norm), "L_A": b["y_A"].float() * sp + lo,
               "L_base": b["L_base"].float(), "L_baseM": b["L_baseM"].float(),
               "dA": (b["dA"].float() if b["dA"] is not None else None)}
        if out.get("y_Aw") is not None:
            # the profile's anchor set on its warped trend (labels in place): the acceptance's retention, contrast,
            # roughness, densify and purity refer to it; the unwarped anchor L_A0 and background L_base0 are kept for
            # the warp's own content (warp_ok) and the sampler's diversity
            dW = (out["y_Aw"].float() - b["y_A"].float()) * sp
            res["L_A0"], res["L_base0"] = res["L_A"], res["L_base"]
            res["L_A"], res["L_base"], res["L_baseM"] = res["L_A"] + dW, res["L_base"] + dW, res["L_baseM"] + dW
            res["warp_b"], res["warp_umax_m"] = out["warp_b"].float(), out["warp_umax_m"].float()
        if out.get("cmg_gate") is not None:
            res["cmg_gate"] = out["cmg_gate"].float().view(nb, -1).mean(dim=1)
        return res

    @torch.no_grad()
    def profile_acceptance(self, L: torch.Tensor, L_A: torch.Tensor, L_base: torch.Tensor,
                           dA: Optional[torch.Tensor], L_baseM: Optional[torch.Tensor] = None,
                           L_A0: Optional[torch.Tensor] = None,
                           L_base0: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Per-profile checks on the measured stations (log10, (B, D, T)): finite; depth roughness per time band <=
        qc_rough_max x max(anchor, measured) (median over bands); physics descriptors at the measured gates inside
        the union envelope for >= qc_envelope_min of the curves."""
        g = self.cfg.generation
        B, D, T = L.shape
        dev = L.device
        finite = torch.isfinite(L).all(dim=2).all(dim=1)
        Ls = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
        rg, ra = self._band_rough(Ls), self._band_rough(L_A)
        rough = (rg / torch.maximum(ra, self.t_rough_meas_band.clamp_min(1e-6).unsqueeze(0))).median(dim=1).values
        pf = self.phys.torch(Ls[:, :, self.t_gate_cols].reshape(B * D, -1))
        zf = (pf - self.t_phys_mu) / self.t_phys_sd
        inside = ((zf >= self.t_phys_lo) & (zf <= self.t_phys_hi)).all(dim=1).float().view(B, D)
        env = inside.mean(dim=1)
        nan = torch.full((B,), float("nan"), device=dev)
        ret, con, corr, amax = nan.clone(), nan.clone(), nan.clone(), torch.zeros(B, device=dev)
        if dA is not None:
            x, a = Ls - L_base, dA
            amax = a.abs().amax(dim=(1, 2))
            sup = (a.abs() >= 0.1 * amax.view(-1, 1, 1)).float() * (amax.view(-1, 1, 1) > 0).float()
            n = sup.sum(dim=(1, 2)).clamp_min(1.0)
            r_ = (x * a * sup).sum(dim=(1, 2)) / (a * a * sup).sum(dim=(1, 2)).clamp_min(1e-12)
            ma, mx = (a * sup).sum(dim=(1, 2)) / n, (x * sup).sum(dim=(1, 2)) / n
            ac, xc = (a - ma.view(-1, 1, 1)) * sup, (x - mx.view(-1, 1, 1)) * sup
            c_ = (ac * xc).sum(dim=(1, 2)) / ((ac * ac).sum(dim=(1, 2)) * (xc * xc).sum(dim=(1, 2))).clamp_min(1e-20).sqrt()
            k_ = (ac * xc).sum(dim=(1, 2)) / (ac * ac).sum(dim=(1, 2)).clamp_min(1e-12)
            valid = amax >= max(float(self.anom_test_amp), float(g.qc_anomaly_min_decades))
            ret, con, corr = torch.where(valid, r_, nan), torch.where(valid, k_, nan), torch.where(valid, c_, nan)
        lo_r, hi_r = (float(v) for v in str(g.qc_retention_range).split(","))
        lo_k, hi_k = (float(v) for v in str(g.qc_contrast_range).split(","))
        ok_an = torch.isnan(ret) | ((ret >= lo_r) & (ret <= hi_r) & (con >= lo_k) & (con <= hi_k)
                                    & (corr >= float(g.qc_anomaly_corr_min)))
        ok = finite & (rough <= float(g.qc_rough_max)) & (env >= float(g.qc_envelope_min)) & ok_an
        pur = None
        if self.bg_pure and D >= 5:
            # the label check of the background: off its labelled anomalies the network's departure has no anomaly-scale
            # content above the calibrated threshold (a "0 anomalies" profile is pure smooth background: its anchor is
            # the purified background, its departure regional)
            m2 = self.synthetic_label_weight(L_A - (L_baseM if L_baseM is not None else L_base))
            hp, _ = self.purity_map(Ls - L_A, L_A - L_base, m_syn=m2)
            pur = hp.amax(dim=(1, 2))
            ok = ok & (pur <= float(self.purity_thr))
            if L_A0 is not None and L_base0 is not None:
                # the warp's own anomaly-scale content, bounded explicitly: the warped background is the measured trend
                # re-sampled with 1 - s < phi' < 1 + s, whose local-quadratic residual grows at most like phi'^4
                E_ = L_A - L_base
                hw, _ = self.purity_map(L_base, E_, m_syn=m2)
                h0, _ = self.purity_map(L_base0, E_, m_syn=m2)
                cw, c0 = hw.amax(dim=(1, 2)), h0.amax(dim=(1, 2))
                wlim = (1.0 + float(self.warp_strain)) ** 4 * c0 + float(self.purity_thr)
                hd, _ = self.purity_map(L_A - L_A0, E_, m_syn=m2)
                wq = {"warp_content_ratio": cw / c0.clamp_min(1e-9), "warp_limit_decades": wlim,
                      "warp_content_decades": hd.amax(dim=(1, 2)), "warp_ok": (cw <= wlim).float()}
                wsl = (wlim - cw) / wlim.clamp_min(1e-12)
                ok = ok & (cw <= wlim)
        t_step = t_ok = None
        if self.bg_trend and D >= 2:
            # a "0 anomalies" profile is the overall decay trend: no step against it along the profile beyond a quarter
            # of the purity tolerance (its anchor has none, the network's departure is regional and small)
            zero = amax <= 0
            st = self._trend_steps(Ls)
            good = st <= PURITY_ANCHOR_SHARE * float(self.purity_tol)
            ok = ok & (good | ~zero)
            t_step, t_ok = torch.where(zero, st, nan), torch.where(zero, good.float(), nan)
        fam_frac = None
        if getattr(self, "fam", None) is not None:
            # field-admissible fraction of the profile's background stations (measured, inside the span, no labelled
            # anomaly above fam_bg_tol); NaN when the labels cover every station.  Gate with fam_enabled, else reported
            ex = self.fam.excess_torch(pf).view(B, D)
            bgs = self.t_fam_span[:D].view(1, D).expand(B, D)
            if dA is not None:
                bgs = bgs & (dA.abs().amax(dim=-1) <= self.fam_bg_tol)
            nbg = bgs.float().sum(dim=1)
            fam_frac = torch.where(nbg > 0, ((ex <= 0) & bgs).float().sum(dim=1) / nbg.clamp_min(1.0), nan)
            if self.fam_on:
                ok = ok & (torch.isnan(fam_frac) | (fam_frac >= float(self.cfg.profile.fam_min_frac)))
        dist = (Ls - self.t_Lk.unsqueeze(0)).pow(2).mean(dim=(1, 2)).sqrt()
        res = {"pass": ok, "finite": finite, "rough_ratio": rough, "envelope_frac": env, "anomaly_retention": ret,
               "anomaly_contrast": con, "anomaly_corr": corr, "max_abs_anomaly_decades": amax,
               "dist_to_measured_decades": dist}
        if fam_frac is not None:
            res["field_admissible_frac"] = fam_frac
        if pur is not None:
            res["purity_max_decades"] = pur
            res["purity_ok"] = (pur <= float(self.purity_thr)).float()
            if L_A0 is not None and L_base0 is not None:
                res.update(wq)
        if t_ok is not None:
            res["trend_step_decades"], res["trend_ok"] = t_step, t_ok
        # the acceptance margin: the smallest slack of the gates that apply, each normalised by its tolerance band (0 at
        # a threshold, < 0 failed); a pass with a margin below --qc_borderline_margin is borderline (qc_export split)
        one = torch.ones_like(env)
        sl = [(float(g.qc_rough_max) - rough) / max(float(g.qc_rough_max), 1e-9)]
        if float(g.qc_envelope_min) < 1.0:
            sl.append((env - float(g.qc_envelope_min)) / max(1.0 - float(g.qc_envelope_min), 1e-9))
        if dA is not None:
            okv = ~torch.isnan(ret)
            sl.append(torch.where(okv, torch.minimum(ret - lo_r, hi_r - ret) / max(hi_r - lo_r, 1e-9), one))
            sl.append(torch.where(okv, torch.minimum(con - lo_k, hi_k - con) / max(hi_k - lo_k, 1e-9), one))
            cmin = float(g.qc_anomaly_corr_min)
            sl.append(torch.where(okv, (corr - cmin) / max(1.0 - cmin, 1e-9), one))
        if pur is not None:
            sl.append((float(self.purity_thr) - pur) / max(float(self.purity_thr), 1e-12))
            if "warp_ok" in res:
                sl.append(wsl)
        if t_step is not None:
            tt_ = PURITY_ANCHOR_SHARE * float(self.purity_tol)
            sl.append(torch.where(torch.isnan(t_step), one, (tt_ - t_step) / max(tt_, 1e-12)))
        if fam_frac is not None and self.fam_on:
            fm_ = float(self.cfg.profile.fam_min_frac)
            sl.append(torch.where(torch.isnan(fam_frac), one, (fam_frac - fm_) / max(1.0 - fm_, 1e-9)))
        mg = torch.stack([torch.nan_to_num(x_.float(), nan=-1.0) for x_ in sl], dim=0).amin(dim=0).clamp(-1.0, 1.0)
        res["qc_margin"] = torch.where(finite, mg, -one)
        return res

    def _trend_steps(self, L: torch.Tensor) -> torch.Tensor:
        """Largest step of L (B, D, T) against the background trend's direction along the profile (decades)."""
        d = L[:, 1:, :] - L[:, :-1, :]
        return torch.where(self.t_bg_down.view(1, 1, -1), d, -d).amax(dim=(1, 2))

    def _trend_fill(self, L: torch.Tensor, unk: torch.Tensor) -> torch.Tensor:
        """L (B, W, T) with the stations unk (B, W) made to follow the trend's direction along the profile: from the
        top each unmeasured station is no higher than the station above, from the bottom no lower than the station
        below (directions per time from the trend).  Measured stations keep their values; an unmeasured stretch ends
        up between its measured neighbours and decays across it. One tensor per station and no in-place write: in
        training the anchor is differentiable, and writing into one (B, W, T) tensor changed the slices torch.minimum
        / maximum had saved for the backward pass (cross-validation hold-outs failed in Stage-1B).  The values are
        unchanged."""
        dn = self.t_bg_down.view(1, 1, -1)
        X = list(torch.where(dn, L, -L).unbind(1))
        u = unk.unsqueeze(-1)
        for w in range(1, len(X)):
            X[w] = torch.where(u[:, w], torch.minimum(X[w], X[w - 1]), X[w])
        for w in range(len(X) - 2, -1, -1):
            X[w] = torch.where(u[:, w], torch.maximum(X[w], X[w + 1]), X[w])
        Xs = torch.stack(X, dim=1)
        return torch.where(dn, Xs, -Xs)

    def _trend_project(self, L: torch.Tensor) -> torch.Tensor:
        """Greatest monotone minorant of L (B, D, T) along the profile in the trend's direction per time (the last
        step of a "0 anomalies" profile: it decays along the profile like its anchor; changes are reported)."""
        dn = torch.cummin(L, dim=1).values
        up = torch.flip(torch.cummin(torch.flip(L, dims=[1]), dim=1).values, dims=[1])
        return torch.where(self.t_bg_down.view(1, 1, -1), dn, up)

    @torch.no_grad()
    def densify(self, L: torch.Tensor, L_A: torch.Tensor, P: Dict[str, torch.Tensor],
                dense_depth: torch.Tensor, warp_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense-depth profile = anchor evaluated analytically on the dense axis (same parameters: alpha field, sim
        registration, anomalies, background) + the generated departure from the anchor, PCHIP-interpolated between
        the stations.  Exact at the measured stations (they are nodes of the dense axis). warp_b: the profile's
        background warp, applied analytically on the dense axis too (L_A is then the warped anchor on the stations)."""
        nb = L.shape[0]
        R = (L - L_A).transpose(1, 2)                                         # (nb, T, D)
        m = torch_pchip_slopes(self.t_depth, R)
        Rd = torch_hermite_dense(self.t_depth, R, m, dense_depth).transpose(1, 2)    # (nb, Dd, T)
        dd = dense_depth.unsqueeze(0).expand(nb, -1).contiguous()
        LAd = self.compose_anchor(dd, P)["L_A"]
        if warp_b is not None:
            Pb = self._bg_params(P)
            LAd = LAd + (self.compose_anchor(dd + self.warp_shift(dd, warp_b), Pb)["L_base"]
                         - self.compose_anchor(dd, Pb)["L_base"])
        return LAd + Rd

    @torch.no_grad()
    def generate_profiles(self, n_profiles: int, seed: int, mode: str = "gen", use_ema: bool = True,
                          chunk: Optional[int] = None, return_anchor: bool = False, accept: bool = False,
                          dense_depth: Optional[np.ndarray] = None, sink=None,
                          progress: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate n full profiles.  Returns (L (N, D, T) log10 on the measured stations, trace). accept:
        per-profile acceptance (profile_acceptance) with up to qc_max_tries passes — failing profiles are regenerated
        with fresh latents, the last pass also redraws the profile's random parameters; the last attempt is kept and
        flagged when it still fails. dense_depth / sink: every profile is densified on dense_depth and handed to
        sink(i0, L_dense) as float32 log10 (nb, Dd, T), after the time-monotone projection (counted in the trace).
        accept with --qc_export split: _generate_split (accepted / review / rejected, no parameter redraw)."""
        if accept and str(self.cfg.generation.qc_export).strip().lower() == "split":
            return self._generate_split(n_profiles, seed, mode, use_ema, chunk, return_anchor, dense_depth, sink,
                                        progress)
        mods = self._eval_modules(use_ema)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        gen_r = torch.Generator(device=self.device)
        gen_r.manual_seed(int(seed) + 1)
        # GPU memory scales with curves per pass (transformer blocks at T gates): cap at ~512 curves
        cap = max(1, 512 // max(self.D, 1))
        chunk = int(max(1, min(int(chunk or self.cfg.generation.gen_chunk_profiles), cap)))
        tries_max = max(1, int(self.cfg.generation.qc_max_tries)) if accept else 1
        dd = torch.tensor(dense_depth, dtype=torch.float32, device=self.device) if dense_depth is not None else None
        outs: List[np.ndarray] = []
        anchors: List[np.ndarray] = []
        anchors0: List[np.ndarray] = []
        pr: Dict[str, List[np.ndarray]] = {}
        qr: Dict[str, List[np.ndarray]] = {}
        n_proj, n_vals, max_proj = 0, 0, 0.0
        tz = {"station": [0, 0, 0.0], "dense": [0, 0, 0.0]}     # trend projection of "0 anomalies" profiles
        done, t0, next_report = 0, time.time(), 0.1
        while done < n_profiles:
            nb = min(chunk, n_profiles - done)
            P = self.sample_profile_params(nb, mode, gen)
            res = self._profile_pass(P, mods, gen)
            q = self.profile_acceptance(res["L"], res["L_A"], res["L_base"], res["dA"], res["L_baseM"], L_A0=res.get("L_A0"),
                                        L_base0=res.get("L_base0"))
            tries = torch.ones(nb, dtype=torch.long, device=self.device)
            for attempt in range(1, tries_max):
                bad = (~q["pass"]).nonzero().view(-1)
                if bad.numel() == 0:
                    break
                # latent redraws first (they keep the profile's parameters, so the configured anomaly / background
                # distribution is not filtered); the last ceil(tries / 3) passes redraw the profile itself
                if attempt >= tries_max - max(1, math.ceil(tries_max / 3)):
                    Pn = self.sample_profile_params(int(bad.numel()), mode, gen_r)
                    for k in P:
                        P[k][bad] = Pn[k]
                rb = self._profile_pass({k: v[bad] for k, v in P.items()}, mods, gen_r)
                qb = self.profile_acceptance(rb["L"], rb["L_A"], rb["L_base"], rb["dA"], rb["L_baseM"], L_A0=rb.get("L_A0"),
                                             L_base0=rb.get("L_base0"))
                for k, v in rb.items():
                    if v is not None:
                        res[k][bad] = v
                for k, v in qb.items():
                    q[k][bad] = v
                tries[bad] += 1
            if self.bg_trend:
                # "0 anomalies" profiles end on the overall decay trend along the profile (the steps against it are
                # bounded by the acceptance; their projection is counted)
                zero = q["max_abs_anomaly_decades"] <= 0
                if bool(zero.any()):
                    Lz = res["L"][zero]
                    Lp = self._trend_project(Lz)
                    tz["station"][0] += int((Lp < Lz).sum())
                    tz["station"][1] += int(Lz.numel())
                    tz["station"][2] = max(tz["station"][2], float((Lz - Lp).max()))
                    res["L"][zero] = Lp
            outs.append(res["L"].cpu().numpy().astype(np.float32))
            if return_anchor:
                anchors.append(res["L_A"].cpu().numpy().astype(np.float32))
                if "L_A0" in res:
                    anchors0.append(res["L_A0"].cpu().numpy().astype(np.float32))
            for k, v in P.items():
                pr.setdefault(k, []).append(v.detach().cpu().numpy())
            for k in ("warp_b", "warp_umax_m"):          # the network's background warp of each profile
                if k in res:
                    pr.setdefault(k, []).append(res[k].detach().cpu().numpy())
            for k, v in self.anomaly_labels(P).items():      # per-anomaly peak / where / when / visible window
                pr.setdefault(k, []).append(v.detach().cpu().numpy())
            for k, v in q.items():
                qr.setdefault(k, []).append(v.detach().cpu().numpy())
            qr.setdefault("tries", []).append(tries.cpu().numpy())
            if "cmg_gate" in res:                        # mean CMG contraction strength of each profile
                qr.setdefault("cmg_gate_mean", []).append(res["cmg_gate"].cpu().numpy())
            if dd is not None and sink is not None:
                Ldt = self.densify(res["L"], res["L_A"], P, dd, warp_b=res.get("warp_b"))
                if self.bg_trend:
                    zero = q["max_abs_anomaly_decades"] <= 0
                    if bool(zero.any()):
                        Lz = Ldt[zero]
                        Lp = self._trend_project(Lz)
                        tz["dense"][0] += int((Lp < Lz).sum())
                        tz["dense"][1] += int(Lz.numel())
                        tz["dense"][2] = max(tz["dense"][2], float((Lz - Lp).max()))
                        Ldt[zero] = Lp
                Ld = Ldt.cpu().numpy()
                Lm = np.minimum.accumulate(Ld, axis=2) if self.decay_hard else Ld
                ch = Lm < Ld
                n_proj += int(ch.sum())
                n_vals += int(Ld.size)
                if ch.any():
                    max_proj = max(max_proj, float((Ld - Lm).max()))
                sink(done, Lm.astype(np.float32))
            done += nb
            if progress and (done >= next_report * n_profiles or done == n_profiles):
                acc = float(np.mean(np.concatenate(qr["pass"])))
                print(f"[GENERATION] {done}/{n_profiles} profiles | accepted {acc:.1%} | {time.time() - t0:.0f}s")
                next_report += 0.1
        params = {k: np.concatenate(v) for k, v in pr.items()}
        res_tr: Dict[str, Any] = {"alpha": params["alpha"], "zoom": params["zoom"], "offset": params["offset"],
                                  "gain": params["gain"], "params": params,
                                  "qc": {k: np.concatenate(v) for k, v in qr.items()},
                                  "extension_sides": self.ext_sides}
        if self.n_ma:
            res_tr["measured_anomalies"] = [dict(r_) for r_ in self.ma_runs]
            # for the measured anomalies' visible windows in the index: each template's largest |change| along the
            # profile per time (measured window), the visibility floor and the times
            res_tr["measured_anomaly_time_profile"] = self.t_MA.abs().amax(dim=1)[:, :self.kt].cpu().numpy()
            res_tr["visibility_floor_decades"] = float(self.anom_amp[0])
            res_tr["time_ms"] = self.t_time_ms[:self.kt].cpu().numpy()
        if getattr(self, "anom_bank_on", False):
            res_tr["morphology_bank_names"] = list(self.mb_names)
        if dd is not None and sink is not None:
            res_tr["dense_projection"] = {"changed_fraction": n_proj / max(n_vals, 1), "max_change_decades": max_proj}
        if self.bg_trend:
            res_tr["zero_anomaly_trend_projection"] = {
                k: {"changed_fraction": v[0] / max(v[1], 1), "max_change_decades": v[2]} for k, v in tz.items() if v[1]}
        if return_anchor:
            res_tr["anchor_L"] = np.concatenate(anchors)
            if anchors0:                                  # the unwarped anchors (the sampler's own diversity)
                res_tr["anchor_L0"] = np.concatenate(anchors0)
        return np.concatenate(outs), res_tr

    @torch.no_grad()
    def _generate_split(self, n_profiles: int, seed: int, mode: str, use_ema: bool, chunk: Optional[int],
                        return_anchor: bool, dense_depth: Optional[np.ndarray], sink,
                        progress: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """qc_export split.  Every slot draws its parameters once from the configured law.  A failing slot gets up to
        qc_max_tries - 1 latent redraws (same profile, same labels); then it is
          accepted  - passes every gate with a margin >= --qc_borderline_margin (qc_margin): the data set;
          review    - passes, but within the margin of a gate: exported apart (qc_review/), not in the data set;
          rejected  - fails after the last redraw: exported apart (qc_rejected/) with the gates it failed.
        Slots are drawn until n_profiles are accepted or --qc_budget_factor x n_profiles slots are used.  The parameters
        of every slot are kept for the selection audit (qc_selection_audit: accepted against drawn)."""
        g = self.cfg.generation
        mods = self._eval_modules(use_ema)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        gen_r = torch.Generator(device=self.device)
        gen_r.manual_seed(int(seed) + 1)
        cap = max(1, 512 // max(self.D, 1))
        chunk = int(max(1, min(int(chunk or g.gen_chunk_profiles), cap)))
        tries_max = max(1, int(g.qc_max_tries))
        m_min = float(g.qc_borderline_margin)
        budget = int(math.ceil(max(float(g.qc_budget_factor), 1.0) * int(n_profiles)))
        dd = torch.tensor(dense_depth, dtype=torch.float32, device=self.device) if dense_depth is not None else None
        outs: List[np.ndarray] = []
        anchors: List[np.ndarray] = []
        anchors0: List[np.ndarray] = []
        pr: Dict[str, List[np.ndarray]] = {}
        qr: Dict[str, List[np.ndarray]] = {}
        allp: Dict[str, List[np.ndarray]] = {}
        allq: Dict[str, List[np.ndarray]] = {}
        stat: List[np.ndarray] = []
        side = {1: {"L": [], "p": {}, "q": {}, "n": 0}, 2: {"L": [], "p": {}, "q": {}, "n": 0}}
        n_proj, n_vals, max_proj = 0, 0, 0.0
        tz = {"station": [0, 0, 0.0], "dense": [0, 0, 0.0]}
        n_acc, n_slots, t0, next_report = 0, 0, time.time(), 0.1
        while n_acc < n_profiles and n_slots < budget:
            nb = min(chunk, budget - n_slots)
            P = self.sample_profile_params(nb, mode, gen)
            res = self._profile_pass(P, mods, gen)
            q = self.profile_acceptance(res["L"], res["L_A"], res["L_base"], res["dA"], res["L_baseM"], L_A0=res.get("L_A0"),
                                        L_base0=res.get("L_base0"))
            tries = torch.ones(nb, dtype=torch.long, device=self.device)
            for _ in range(1, tries_max):
                bad = (~q["pass"]).nonzero().view(-1)
                if bad.numel() == 0:
                    break
                rb = self._profile_pass({k: v[bad] for k, v in P.items()}, mods, gen_r)    # latents only
                qb = self.profile_acceptance(rb["L"], rb["L_A"], rb["L_base"], rb["dA"], rb["L_baseM"], L_A0=rb.get("L_A0"),
                                             L_base0=rb.get("L_base0"))
                for k, v in rb.items():
                    if v is not None:
                        res[k][bad] = v
                for k, v in qb.items():
                    q[k][bad] = v
                tries[bad] += 1
            if self.bg_trend:
                zero = q["max_abs_anomaly_decades"] <= 0
                if bool(zero.any()):
                    Lz = res["L"][zero]
                    Lp = self._trend_project(Lz)
                    tz["station"][0] += int((Lp < Lz).sum())
                    tz["station"][1] += int(Lz.numel())
                    tz["station"][2] = max(tz["station"][2], float((Lz - Lp).max()))
                    res["L"][zero] = Lp
            lab = self.anomaly_labels(P)
            st = torch.where(~q["pass"], torch.full_like(tries, 2),
                             torch.where(q["qc_margin"] < m_min, torch.ones_like(tries), torch.zeros_like(tries)))
            for k, v in list(P.items()) + list(lab.items()):
                allp.setdefault(k, []).append(v.detach().cpu().numpy())
            for k, v in q.items():
                allq.setdefault(k, []).append(v.detach().cpu().numpy())
            allq.setdefault("tries", []).append(tries.cpu().numpy())
            stat.append(st.cpu().numpy())
            ia = (st == 0).nonzero().view(-1)[: max(int(n_profiles) - n_acc, 0)]
            if ia.numel():
                Pa = {k: v[ia] for k, v in P.items()}
                outs.append(res["L"][ia].cpu().numpy().astype(np.float32))
                if return_anchor:
                    anchors.append(res["L_A"][ia].cpu().numpy().astype(np.float32))
                    if res.get("L_A0") is not None:
                        anchors0.append(res["L_A0"][ia].cpu().numpy().astype(np.float32))
                for k, v in Pa.items():
                    pr.setdefault(k, []).append(v.detach().cpu().numpy())
                for k in ("warp_b", "warp_umax_m"):
                    if res.get(k) is not None:
                        pr.setdefault(k, []).append(res[k][ia].detach().cpu().numpy())
                for k, v in lab.items():
                    pr.setdefault(k, []).append(v[ia].detach().cpu().numpy())
                for k, v in q.items():
                    qr.setdefault(k, []).append(v[ia].detach().cpu().numpy())
                qr.setdefault("tries", []).append(tries[ia].cpu().numpy())
                if res.get("cmg_gate") is not None:
                    qr.setdefault("cmg_gate_mean", []).append(res["cmg_gate"][ia].cpu().numpy())
                if dd is not None and sink is not None:
                    Ldt = self.densify(res["L"][ia], res["L_A"][ia], Pa, dd,
                                       warp_b=(res["warp_b"][ia] if res.get("warp_b") is not None else None))
                    if self.bg_trend:
                        zero = q["max_abs_anomaly_decades"][ia] <= 0
                        if bool(zero.any()):
                            Lz = Ldt[zero]
                            Lp = self._trend_project(Lz)
                            tz["dense"][0] += int((Lp < Lz).sum())
                            tz["dense"][1] += int(Lz.numel())
                            tz["dense"][2] = max(tz["dense"][2], float((Lz - Lp).max()))
                            Ldt[zero] = Lp
                    Ld = Ldt.cpu().numpy()
                    Lm = np.minimum.accumulate(Ld, axis=2) if self.decay_hard else Ld
                    ch = Lm < Ld
                    n_proj += int(ch.sum())
                    n_vals += int(Ld.size)
                    if ch.any():
                        max_proj = max(max_proj, float((Ld - Lm).max()))
                    sink(n_acc, Lm.astype(np.float32))
                n_acc += int(ia.numel())
            for code, box in side.items():          # review / rejected: kept apart (at most n_profiles each)
                ib = (st == code).nonzero().view(-1)[: max(int(n_profiles) - box["n"], 0)]
                if ib.numel():
                    box["L"].append(res["L"][ib].cpu().numpy().astype(np.float32))
                    for k, v in list(P.items()) + list(lab.items()):
                        box["p"].setdefault(k, []).append(v[ib].detach().cpu().numpy())
                    for k, v in q.items():
                        box["q"].setdefault(k, []).append(v[ib].detach().cpu().numpy())
                    box["q"].setdefault("tries", []).append(tries[ib].cpu().numpy())
                    box["n"] += int(ib.numel())
            n_slots += nb
            if progress and (n_acc >= next_report * n_profiles or n_acc >= n_profiles or n_slots >= budget):
                print(f"[GENERATION] accepted {n_acc}/{n_profiles} | slots drawn {n_slots} (budget {budget}) | "
                      f"review {side[1]['n']} rejected {side[2]['n']} | {time.time() - t0:.0f}s")
                next_report = max(next_report + 0.1, n_acc / max(n_profiles, 1) + 0.1)
        status = np.concatenate(stat) if stat else np.zeros(0, dtype=np.int64)
        ap = {k: np.concatenate(v) for k, v in allp.items()}
        aq = {k: np.concatenate(v) for k, v in allq.items()}
        if n_acc == 0:
            raise RuntimeError(f"[QC-SPLIT] no profile passed the acceptance with a margin >= {m_min:g} in {n_slots} slots "
                               f"(review {side[1]['n']}, rejected {side[2]['n']}): see the acceptance gates")
        params = {k: np.concatenate(v) for k, v in pr.items()}
        res_tr: Dict[str, Any] = {"alpha": params["alpha"], "zoom": params["zoom"], "offset": params["offset"],
                                  "gain": params["gain"], "params": params,
                                  "qc": {k: np.concatenate(v) for k, v in qr.items()},
                                  "extension_sides": self.ext_sides}
        if self.n_ma:
            res_tr["measured_anomalies"] = [dict(r_) for r_ in self.ma_runs]
            res_tr["measured_anomaly_time_profile"] = self.t_MA.abs().amax(dim=1)[:, :self.kt].cpu().numpy()
            res_tr["visibility_floor_decades"] = float(self.anom_amp[0])
            res_tr["time_ms"] = self.t_time_ms[:self.kt].cpu().numpy()
        if getattr(self, "anom_bank_on", False):
            res_tr["morphology_bank_names"] = list(self.mb_names)
        if dd is not None and sink is not None:
            res_tr["dense_projection"] = {"changed_fraction": n_proj / max(n_vals, 1), "max_change_decades": max_proj}
        if self.bg_trend:
            res_tr["zero_anomaly_trend_projection"] = {
                k: {"changed_fraction": v[0] / max(v[1], 1), "max_change_decades": v[2]} for k, v in tz.items() if v[1]}
        if return_anchor:
            res_tr["anchor_L"] = np.concatenate(anchors)
            if anchors0:
                res_tr["anchor_L0"] = np.concatenate(anchors0)
        fails = qc_fail_reasons(aq, status, self)
        res_tr["qc_split"] = {
            "rule": {"accepted": f"passes every gate with qc_margin >= {m_min:g}", "review": "passes, qc_margin below it",
                     "rejected": f"fails after {tries_max - 1} latent redraws (parameters never redrawn)",
                     "budget_slots": budget},
            "requested": int(n_profiles), "accepted": int(n_acc), "complete": bool(n_acc >= n_profiles),
            "slots_drawn": int(status.size), "rates": {"accepted": float(np.mean(status == 0)) if status.size else 0.0,
                                                       "review": float(np.mean(status == 1)) if status.size else 0.0,
                                                       "rejected": float(np.mean(status == 2)) if status.size else 0.0},
            "mean_tries_all": float(np.mean(aq["tries"])) if "tries" in aq else float("nan"),
            "first_pass_rate": float(np.mean((aq["tries"] == 1) & (status != 2))) if "tries" in aq else float("nan"),
            "rejected_gate_failures": fails,
            "audit": qc_selection_audit(ap, status),
            "review": {"L": np.concatenate(side[1]["L"]) if side[1]["L"] else None,
                       "params": {k: np.concatenate(v) for k, v in side[1]["p"].items()},
                       "qc": {k: np.concatenate(v) for k, v in side[1]["q"].items()}},
            "rejected": {"L": np.concatenate(side[2]["L"]) if side[2]["L"] else None,
                         "params": {k: np.concatenate(v) for k, v in side[2]["p"].items()},
                         "qc": {k: np.concatenate(v) for k, v in side[2]["q"].items()}}}
        return np.concatenate(outs), res_tr

    @torch.no_grad()
    def validate(self, n_profiles: int = 16, seed: int = 20260) -> Dict[str, Any]:
        n_profiles = max(8, int(n_profiles))
        Lg, tr = self.generate_profiles(n_profiles, seed=seed, mode="val", use_ema=True, return_anchor=True)
        # the checkpoint score is measured on the measured stations of the axis
        mc = self.meas_cols
        # with the measured anomalies kept, the reference is the measured profile itself
        Lref_v = self.L_meas_np if (self.bg_pure and self.n_ma > 0) else self.L_anchor_np
        Lg, LAg, LKa, dpa = Lg[:, mc], tr["anchor_L"][:, mc], Lref_v[mc], self.depth_np[mc]
        nonfinite = float(np.mean(~np.isfinite(Lg)))     # reported; a non-finite profile set never becomes best
        keep = self.keep_mask[mc]
        gc = self.gate_cols
        cm = curve_quality_metrics(Lg.reshape(-1, self.T)[:, gc], LKa[keep][:, gc], self.L_union_np[:, gc],
                                   self.phys, max_gen=2000)
        pm = profile_quality_metrics(Lg, LKa, dpa, P_anchor=LAg, L_floor=(self.L_meas_np[mc] if self.bg_pure else None))
        pm.update(acceptance_summary(tr["qc"]))
        sm, _ = structure_fidelity_metrics(Lg, LAg, LKa, dpa, gc, self.struct_thr, keep=keep)
        pm["struct_retention"] = sm.get("struct_retention", float("nan"))
        pm["anchor_pairwise_distance_decades_median"] = pairwise_profile_distance(
            tr["anchor_L0"][:, mc] if "anchor_L0" in tr else LAg)[0]
        top = sm.get("strongest_feature", {})
        fr = top.get("retention_departure_first_mid_last", [float("nan")] * 3)
        score = composite_quality_score(cm, pm) if nonfinite == 0.0 else float("inf")
        summary = {"score": score, "rough_ratio_vs_anchor": pm.get("rough_ratio_vs_anchor_median", float("nan")),
                   "accept_rate": pm.get("accept_rate", float("nan")),
                   "anomaly_retention": pm.get("anomaly_retention_median", float("nan")),
                   "anomaly_corr": pm.get("anomaly_corr_median", float("nan")),
                   "rough_ratio_vs_measured": pm.get("rough_ratio_vs_measured_median", float("nan")),
                   "measured_coverage": pm.get("measured_coverage", float("nan")),
                   "diversity_decades": pm.get("pairwise_profile_distance_decades_median", 0.0),
                   "copy_fraction": pm.get("copy_fraction", 0.0), "precision": cm.get("descriptor_precision", 0.0),
                   "recall": cm.get("descriptor_recall", 0.0), "monotone": cm.get("monotone_fraction", 0.0),
                   "struct_retention": pm["struct_retention"],
                   "anchor_diversity_decades": pm["anchor_pairwise_distance_decades_median"],
                   "feature_retention_first": fr[0], "feature_retention_mid": fr[1], "feature_retention_last": fr[2],
                   "feature_retention_min": top.get("retention_departure_min", float("nan")),
                   "depth_texture": pm.get("departure_depth_texture_ratio_median", float("nan")),
                   "purity_rate": pm.get("purity_rate", float("nan")),
                   "purity_max": pm.get("purity_max_decades_median", float("nan")),
                   "trend_rate": pm.get("trend_rate", float("nan")),
                   "field_admissible": pm.get("field_admissible_occupancy", float("nan")),
                   "warp_shift_m": (float(np.median(tr["params"]["warp_umax_m"])) if "warp_umax_m" in tr["params"]
                                    else float("nan")),
                   "nonfinite_fraction": nonfinite}
        return {"score": score, "curve": cm, "profile": pm, "summary": summary, "structure": sm}

    # ------------------------------------------------------------------ checkpoints
    def _payload(self, tag: str, reason: str) -> Dict[str, Any]:
        mods = {n: getattr(self, n) for n in ("embedder", "recovery", "supervisor", "condition_extractor", "cond_head",
                                              "generator", "profile_ctx", "discriminator", "sequence_discriminator",
                                              "profile_discriminator", "cmg", "mixer")}
        opts = {n: getattr(self, n) for n in ("opt_er", "opt_s", "opt_c", "opt_g", "opt_d")}
        return {
            "meta": {"program": RFMG_PROGRAM_NAME, "version": RFMG_VERSION, "schema": RFMG_SCHEMA_VERSION, "tag": tag,
                     "reason": reason, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "global_epoch": self.global_epoch, "stage_progress": dict(self.stage_progress),
                     "best_metric": dict(self.best_stage_metric), "contract_key": self.contract.get("contract_key", ""),
                     "ablation": self.cfg.reproduction.ablation_mode},
            "modules": {n: m.state_dict() for n, m in mods.items() if m is not None},
            "ema": self.ema.state_dict(),
            "optimizers": {n: o.state_dict() for n, o in opts.items() if o is not None},
            "history": self.history, "cfg": self.cfg.to_dict(), "contract": self.contract,
            "normalizer": self.norm.to_dict(), "best_quality": self.best_quality,
        }

    def save(self, tag: str, reason: str = "") -> Optional[str]:
        if not self.cfg.export.save_checkpoint:
            return None
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        atomic_torch_save(self._payload(tag, reason), path)
        return path

    def stage1_select_from(self) -> int:
        """First Stage-1B epoch whose checkpoint may become best_stage1B: min(--stage1_min_epochs, ceil(E / 2)). the
        generator's latent spread develops during the first half of Stage-1B (local pilot p2: spread of the structure
        coefficient over latent draws 0.07 at epoch 4, 0.18 at epoch 12, profile diversity 0.058 -> 0.063 on the same
        256 anchors)."""
        E = int(self.cfg.train.stage1_epochs)
        return int(max(1, min(int(self.cfg.train.stage1_min_epochs), math.ceil(E / 2))))

    def maybe_save(self, stage: str, epoch_idx: int, metrics: Dict[str, float], quality: Optional[Dict[str, Any]] = None,
                   eligible: bool = True) -> bool:
        if not self.cfg.export.save_checkpoint:
            return False
        every = max(1, int(self.cfg.checkpoint.save_every_epochs))
        if (epoch_idx + 1) % every == 0:
            self.save("latest", f"{stage}_epoch_{epoch_idx + 1}")
        if not eligible:
            return False
        key = "score" if "score" in metrics else next(iter(metrics))
        cur = float(metrics[key])
        best = self.best_stage_metric.get(stage)
        improved = best is None or cur < best
        if improved:
            self.best_stage_metric[stage] = cur
            if quality is not None:
                self.best_quality = {"stage": stage, "epoch": epoch_idx + 1, "summary": quality.get("summary", {})}
            if self.cfg.checkpoint.save_best_per_stage:
                self.save(f"best_{stage}", f"{stage}_epoch_{epoch_idx + 1}_{key}={cur:.6g}")
        return improved

    def save_stage_end(self, stage: str) -> None:
        if self.cfg.checkpoint.save_stage_end:
            self.save(f"{stage}_end", f"{stage} finished")
        self.save("latest", f"{stage} finished")

    def load_checkpoint(self, path: str, strict_contract: bool = True, prefer_best: bool = False) -> Dict[str, Any]:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        meta = ck.get("meta", {})
        if meta.get("program") != RFMG_PROGRAM_NAME or int(meta.get("schema", 0)) != RFMG_SCHEMA_VERSION:
            raise RuntimeError(f"[CHECKPOINT] {path} is not an RFMG-Gen v16 checkpoint (meta={meta}); "
                               f"v15 checkpoints use another data contract and cannot be resumed — choose [R].")
        key_ck, key_now = meta.get("contract_key", ""), self.contract.get("contract_key", "")
        if key_ck != key_now:
            msg = (f"[CHECKPOINT] data contract mismatch: checkpoint {key_ck} vs current {key_now} "
                   f"(measured axes, target_len or normaliser changed)")
            if strict_contract and not self.cfg.checkpoint.allow_mismatch_resume:
                raise RuntimeError(msg + " — retrain with [R] or pass --allow_mismatch_resume")
            print(msg + " — continuing because --allow_mismatch_resume is set")
        for n, sd in ck.get("modules", {}).items():
            mod = getattr(self, n, None)
            if isinstance(mod, nn.Module):
                missing, unexpected = mod.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[CHECKPOINT] {n}: missing={len(missing)} unexpected={len(unexpected)}")
        self.ema.load_state_dict(ck.get("ema", {}))
        for n, sd in ck.get("optimizers", {}).items():
            opt = getattr(self, n, None)
            if opt is not None:
                try:
                    opt.load_state_dict(sd)
                except Exception as e:
                    print(f"[CHECKPOINT] optimizer {n} not restored ({e})")
        self.stage_progress.update(meta.get("stage_progress", {}))
        self.global_epoch = int(meta.get("global_epoch", 0))
        self.best_stage_metric = dict(meta.get("best_metric", {}))
        self.history = list(ck.get("history", []))
        self.best_quality = ck.get("best_quality", {})
        self._freeze_condition_frame()
        print(f"[CHECKPOINT] loaded {path} | tag={meta.get('tag')} saved={meta.get('saved_at')} "
              f"stages={_format_stage_progress(self.stage_progress)}")
        return meta
