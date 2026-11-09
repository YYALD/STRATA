"""Command line: the argument parser, capacity presets, the configuration built from the arguments, the
manifold-constraint presets, the configuration summary and the input / output paths of a run."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Sequence

from . import _LOG, RFMG_PROGRAM_NAME, RFMG_VERSION
from .common import _drive_available, project_root, sanitize_path
from .config import (ALLOWED_TARGET_LEN, DESIGN_TARGET_LEN, LOCAL_REAL_NAMES, LOCAL_SIM_NAMES, SERVER_LOG_DIR,
                     SERVER_MODEL_DIR, SERVER_OUTPUT_DIR, SERVER_PLOT_DIR, SERVER_REAL_FILE, SERVER_ROOT,
                     SERVER_SIM_FILE, AppConfig, CheckpointConfig, CVConfig, DataConfig, ExportConfig, GenerationConfig,
                     LossConfig, ModelConfig, NormalizationConfig, ProfileConfig, ReproductionConfig, RuntimeConfig,
                     TrainConfig, VisualizationConfig)


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{RFMG_PROGRAM_NAME} v{RFMG_VERSION}")
    A = p.add_argument
    A("extra_paths", nargs="*", help="optional positional override: measured_csv [simulata_csv]")
    # ---- data ----
    A("--data_name", choices=["real", "simulata", "combined", "tem_simulate", "combined+sine"], default="combined")
    A("--task_mode", choices=["signal", "noise", "hybrid"], default="hybrid")
    A("--module", choices=["gru", "lstm", "lstmLN"], default="lstm")
    A("--real_data_file", type=str, default=SERVER_REAL_FILE)
    A("--simulata_data_file", type=str, default=SERVER_SIM_FILE)
    A("--input_csv", type=str, default="")
    A("--read_by", choices=["column", "row", "auto"], default="auto")
    A("--target_len", type=int, default=DESIGN_TARGET_LEN, choices=[0] + list(ALLOWED_TARGET_LEN),
      help="generation length on the measured time span: 64 / 128 / 256 (0 = auto = 256, the original design length)")
    A("--real_time_unit", choices=["auto", "s", "ms", "us"], default="auto")
    A("--sim_time_unit", choices=["auto", "s", "ms", "us"], default="auto")
    A("--sim_depth_match", choices=["auto", "physical", "relative"], default="auto")
    A("--sim_level_anchor", choices=["structure", "gate_median", "global_median", "none"], default="structure")
    A("--sim_time_extrapolation", choices=["loglinear", "edge"], default="loglinear")
    A("--sim_structure_gain", type=str, default="auto", help="'auto' (measured-anchored) or a number")
    A("--sim_reservoir_rows", type=int, default=8192)
    A("--measured_pool_size", type=int, default=4096)
    A("--pool_draw_chunk", type=int, default=256,
      help="curve pool drawn in batches of at most this many windows (0 = all at once as v16-v21; memory peak)")
    # ---- airborne / ground line files and long profiles ----
    A("--gate_time_file", type=str, default="",
      help="gate times of a line file or station table (STATION, CH1..CHn) whose gate columns carry no times; gate n = "
           "the n-th time (default: the one gate-time file next to it)")
    A("--line_id", type=str, default="auto", help="line of a multi-line survey file (auto = the line with the most stations)")
    A("--component_block", type=int, default=1, help="1-based block of gate columns (zoff1..zoffN: block 1 = first component)")
    A("--gate_window", type=str, default="auto", help="1-based measured gates 'a-b' or auto (positive, falling decays)")
    A("--min_positive_frac", type=float, default=0.99, help="usable-gate rule: share of stations with a positive value")
    A("--max_stations", type=int, default=-1,
      help="long profiles are stacked to at most this many stations (-1 auto: line files 256, matrix files as read; 0 = off)")
    A("--sensor_height", type=str, default="auto", help="airborne sensor clearance in m (auto = median of the height column)")
    A("--sim_survey_check", choices=["auto", "off"], default="auto",
      help="auto: a simulated file of another survey type is not used as structure prior")
    # ---- model ----
    A("--hidden_dim", type=int, default=64)
    A("--cond_dim", type=int, default=64)
    A("--z_dim", type=int, default=32)
    A("--num_layer", type=int, default=3)
    A("--dropout", type=float, default=0.2)
    A("--inception_base_ch", type=int, default=32)
    A("--condition_deep_dim", type=int, default=128)
    A("--condition_branch_dim", type=int, default=32)
    A("--disc_branch_ch", type=int, default=64)
    A("--gen_shared_dim", type=int, default=512)
    A("--gen_static_dim", type=int, default=512)
    A("--gen_z_proj_dim", type=int, default=128)
    A("--gen_temporal_hidden", type=int, default=128)
    A("--capacity_profile", choices=["baseline", "enhanced", "high", "large", "xl"], default="baseline",
      help="enhanced / high: x1.5 / x2 generator widths (v15); [v22] large / xl: every width of the network scaled "
           "(CAPACITY_PRESETS; baseline 5.6 M parameters)")
    A("--recovery_monotone_head", dest="recovery_monotone_head", action="store_true", default=True)
    A("--no_recovery_monotone_head", dest="recovery_monotone_head", action="store_false")
    A("--profile_ctx_hidden", type=int, default=64)
    A("--profile_code_dim", type=int, default=16)
    A("--profile_disc_ch", type=int, default=48)
    # ---- runtime ----
    A("--batch_size", type=int, default=128)
    A("--num_workers", type=int, default=0)
    A("--no_pin_memory", action="store_true")
    A("--gpu_devices", type=str, default="")
    A("--device_index", type=int, default=0)
    A("--seed", type=int, default=42)
    A("--use_amp", action="store_true", default=True)
    A("--no_amp", action="store_false", dest="use_amp")
    # ---- generation ----
    A("--generated_data_count", type=int, default=500000, help="curves; profiles = ceil(N / measured depths)")
    A("--generated_profile_count", type=int, default=1000,
      help="number of generated profiles (default 1000; 0 = ceil(generated_data_count / measured depths))")
    A("--profile_depth_points", type=int, default=1000,
      help="depth samples per exported profile; the measured stations are embedded exactly (0 = stations only)")
    A("--qc_max_tries", type=int, default=12)
    A("--qc_rough_max", type=float, default=2.0)
    A("--qc_envelope_min", type=float, default=0.95)
    A("--qc_retention_range", type=str, default="0.5,1.5")
    A("--qc_anomaly_corr_min", type=float, default=0.8)
    A("--qc_contrast_range", type=str, default="0.7,1.3",
      help="accepted contrast of an anomaly against its local background, relative to its labelled amplitude")
    A("--qc_anomaly_min_decades", type=float, default=0.05)
    A("--gen_chunk_profiles", type=int, default=8, help="profiles per generation pass (capped at ~512 curves)")
    A("--metric_iteration", type=int, default=1)
    A("--generation_mode", choices=["union_guided", "bridge_guided", "real_guided", "balanced", "interpolate",
                                    "real_only", "simulate_only", "train_mixed"], default="union_guided")
    A("--interpolate_alpha", type=float, default=0.5)
    A("--paired_eval_limit", type=int, default=12288)
    A("--sim_prior_k", type=int, default=8)
    A("--real_bootstrap_jitter", type=float, default=0.015)
    A("--sim_prior_strength", type=float, default=0.25)
    A("--real_guided_adv_weight", type=float, default=0.05)
    A("--disable_generation_postprocess", action="store_true")
    A("--gen_alpha_mix", type=str, default="0.85-1.0:0.6,0.55-0.85:0.3,0.30-0.55:0.1")
    # ---- training ----
    A("--stage0_epochs", type=int, default=20)
    A("--stage0b_epochs", type=int, default=4)
    A("--stage1_supervisor_epochs", type=int, default=8)
    A("--stage1_epochs", type=int, default=40)
    A("--stage2_epochs", type=int, default=60)
    A("--steps_per_epoch", type=int, default=100)
    A("--lr_er", type=float, default=1e-3)
    A("--lr_s", type=float, default=1e-3)
    A("--lr_g", type=float, default=1.5e-4)
    A("--lr_d", type=float, default=1.0e-4)
    A("--k_d", type=int, default=1)
    A("--grad_clip", type=float, default=0.5)
    A("--ema_decay", type=float, default=0.999)
    A("--r1_gamma", type=float, default=2.0)
    A("--early_stop_patience", type=int, default=20)
    A("--stage1_min_epochs", type=int, default=20,
      help="first Stage-1B epoch whose checkpoint can be selected as best (capped at half of --stage1_epochs)")
    # ---- losses ----
    for name, default in (("lambda_bridge_wave", 25.0), ("lambda_bridge_feat", 5.0), ("lambda_sup", 10.0),
                          ("lambda_recon", 10.0), ("lambda_mom", 5.0), ("lambda_dist", 0.5), ("lambda_div", 1.0),
                          ("lambda_guid", 2.0), ("lambda_spec", 0.0), ("lambda_shape", 2.0), ("lambda_late", 1.0),
                          ("lambda_adv", 1.0), ("lambda_copy_guard", 2.0), ("lambda_over_expand", 0.5),
                          ("lambda_mechanism_anchor", 0.5), ("lambda_fm", 1.0), ("lambda_tsmooth", 5.0), ("lambda_prof_adv", 1.0),
                          ("lambda_prof_rough", 20.0), ("lambda_prof_trend", 10.0), ("lambda_prof_ms", 6.0), ("lambda_anomaly", 5.0),
                          ("lambda_anomaly_shape", 1.0), ("lambda_structure", 5.0), ("lambda_purity", 5.0), ("lambda_gp", 10.0), ("lambda_cond", 10.0), ("lambda_union_cover", 0.0),
                          ("lambda_segment", 0.0), ("lambda_union_div", 0.0), ("lambda_field_shell", 2.0)):
        A(f"--{name}", type=float, default=default)
    A("--stage1_use_adaptive_loss", action="store_true")
    A("--late_start_frac", type=float, default=0.50)
    A("--mmd_sigma", type=float, default=1.0)
    # ---- profile level ----
    A("--profile_window", type=int, default=16)
    A("--profile_window_mix", type=str, default="16:0.7,32:0.2,full:0.1")
    A("--profile_max_window", type=int, default=0,
      help="training windows hold at most this many stations (0 auto: 64 on airborne / ground lines, no cap on boreholes)")
    A("--depth_alpha_sigma", type=float, default=0.10)
    A("--registration_zoom_min", type=float, default=0.70)
    A("--structure_gain_spread", type=float, default=2.0)
    A("--prof_rough_tol", type=float, default=1.03)
    A("--latent_white_sigma", type=float, default=0.0, help="white latent noise per station and time sample (v16.0-v16.2: 0.10)")
    A("--latent_scale_probs", type=str, default="0.5,0.5,0",
      help="long / mid / short depth-scale shares of the latent field (v16.0-v16.2: 0.4,0.4,0.2)")
    A("--prof_trend_tol", type=float, default=0.02)
    A("--copy_guard_min_dist", type=float, default=0.004)
    A("--critic_real_mix", type=str, default="measured:0.4,sim:0.2,anchor:0.4")
    A("--no_anomalies", action="store_true", help="profiles without the random anomaly composition")
    A("--anomaly_count_probs", type=str, default="auto",
      help="probability of 0, 1, 2, ... anomalies per profile, e.g. 0:0.2,1:0.2,2:0.2,3:0.2,4:0.1,5:0.1; auto = every "
           "count 0 .. --max_anomalies equally often (--measured_anomaly_mode all: v19-v20.1 0:0.2,1:0.35,2:0.25,3:0.12,4:0.08)")
    A("--max_anomalies", type=int, default=5,
      help="largest number of labelled anomalies (measured + synthetic) in one profile (measured_anomaly_mode count)")
    A("--anomaly_amp_range", type=str, default="auto", help="|peak amplitude| range in decades 'lo,hi' or auto")
    A("--anomaly_neg_frac", type=float, default=0.35)
    A("--anomaly_width_range", type=str, default="1.0,0.12",
      help="sigma range: 'min in local station spacings, max as fraction of the depth span'")
    A("--anomaly_tau_range", type=str, default="auto", help="eddy-current time constant range (ms) 'lo,hi' or auto")
    A("--anomaly_broadband_frac", type=float, default=0.30)
    A("--no_anomaly_diffusion", action="store_true")
    A("--bg_level_sigma", type=float, default=0.03)
    A("--bg_tilt_sigma", type=float, default=0.03)
    A("--survey_mode", choices=["ask", "auto", "borehole", "airborne", "ground"], default="ask",
      help="ask = menu; auto = detect from the measured file; borehole / airborne / ground")
    # ---- extensions beyond the measured coverage ----
    A("--extend_profile_frac", type=float, default=0.0,
      help="profile-axis extension beyond the measured stations as a fraction of the span (0 = off)")
    A("--extend_side", choices=["auto", "low", "high", "both"], default="auto",
      help="auto: borehole the deeper end, airborne / ground both line ends")
    A("--extend_time_ms", type=str, default="0",
      help="late-time window extension: target time in ms, 'depth' (borehole: last gate x (d_end/d_last)^2) or 0 = off")
    A("--unmeasured_spread", type=float, default=1.0,
      help="x the calibrated spread of the anchor where nothing is measured (0 = the rule alone)")
    A("--background_mode", choices=["pure", "measured"], default="pure",
      help="pure: anchors on the measured profile's smooth background (no unlabelled anomalies); measured: v17 behaviour")
    A("--no_pure_departure_head", action="store_true",
      help="pure mode: do not project the network's departure onto the background scale (purity by the loss only)")
    A("--background_sigma", type=str, default="auto",
      help="background smoothing scale in m (auto: borehole 2 x station spacing, lines 2 x lateral correlation length)")
    A("--purity_tol", type=str, default="auto",
      help="largest anomaly-scale departure outside labels in decades (auto: half the smallest anomaly amplitude)")
    A("--undulation_sides", choices=["two", "one"], default="two",
      help="background_trend measured: two = rises and dips against the trimmed least-squares decay trend are "
           "undulations (v22); one = rises above the lowest supporting chord only (v20.2-v21)")
    A("--background_trend", choices=["auto", "measured", "monotone", "smooth"], default="auto",
      help="pure mode, '0 anomalies' background: measured = the measured profile itself off its undulations, passed "
           "under each undulation, the measured anomalies = the undulations (v20.2); monotone = the smooth background "
           "made to decay along the profile (v20.1); smooth = v18-v20; auto = measured in a borehole, smooth on lines")
    A("--inventory_noise", choices=["auto", "quiet", "mad"], default="auto",
      help="station-scale noise of the measured residual for the anomaly inventory: quiet = from the quietest quarter of "
           "the station-to-station changes, mad = v19-v20; auto = quiet in a borehole, mad on lines")
    A("--measured_anomaly_prob", type=float, default=1.0,
      help="pure mode: probability that a profile with anomalies carries each measured anomaly (shape kept; 0 = v18; "
           "measured_anomaly_mode count: > 0 = the measured anomalies take part in the count)")
    A("--measured_anomaly_mode", choices=["auto", "count", "all"], default="auto",
      help="count: each measured anomaly is one of a profile's anomalies (m of them uniform over 0 .. min(count, number "
           "measured), the rest synthetic; v20.2); all: every profile with anomalies carries all measured ones (v19-v20.1); "
           "auto = count in a borehole, all on airborne / ground lines")
    A("--measured_anomaly_scale", type=str, default="auto",
      help="pure mode: amplitude factor range of the measured anomalies, 'lo,hi' (log-uniform) or auto (per anomaly 0.5 .. "
           "max(2, twice the synthetic visibility floor / its peak); v19: 0.5,2.0)")
    A("--borehole_anomaly_model", choices=["measured", "physical", "v19"], default="measured",
      help="borehole synthetic anomalies: measured (v20.2: the physical law calibrated on the measured anomalies - "
           "amplitude, width, no widening, separated centres - plus measured anomaly shapes at other depths), physical "
           "(v20: conductor bodies at a distance r from the hole, tau >= t1 (r / spacing)^2, visible amplitudes) or v19 "
           "(the v16-v19 Gaussian law)")
    A("--borehole_anomaly_kinds", type=str, default="auto",
      help="borehole: shares of measured_like (measured model only), intersected, off_hole and zone anomalies; auto = "
           "measured_like:0.50,intersected:0.35,off_hole:0.15,zone:0 (measured model with measured anomalies), else "
           "intersected:0.35,off_hole:0.50,zone:0.15")
    A("--anomaly_min_depth_m", type=float, default=50.0,
      help="borehole: synthetic anomalies are centred at least this deep plus their distance r (0 = off)")
    # ---- field-admissible manifold constraints ----
    A("--manifold_constraint", choices=["v21", "v20"], default="v21",
      help="v21: background manifold (warp), FAM shell + acceptance gate, credibility x trust-region CMG; v20 = the v20.2.0 "
           "mechanisms (affine head, credibility-only gate, no FAM)")
    A("--background_manifold", choices=["auto", "warp", "affine"], default="auto",
      help="regional head: warp = monotone depth reparametrisation of the trend x level / contrast; affine = level / "
           "contrast only (v20.1-v20.2); auto = warp for a decay trend along the profile")
    A("--warp_modes", type=int, default=3, help="sine modes of the depth reparametrisation over the measured span")
    A("--warp_max_strain", type=float, default=0.25, help="bound of |du/dz| (< 1 keeps the reparametrisation monotone)")
    A("--no_fam", action="store_true", help="no field-admissible manifold (shell term, acceptance gate)")
    A("--fam_k", type=int, default=5, help="FAM: ball radius = kappa x distance to the k-th nearest anchor")
    A("--fam_kappa", type=float, default=1.25)
    A("--fam_min_frac", type=float, default=0.95, help="FAM: share of a profile's background curves that must be admissible")
    A("--cmg_form", choices=["eaai", "v20"], default="eaai",
      help="CMG gate: eaai = credibility x trust region (learned radius); v20 = credibility only")
    # ---- joint anomaly morphology, decay constraint, QC export, validation ----
    A("--sim_role", choices=["auto", "morphology", "texture", "off"], default="auto",
      help="simulated profile: morphology = its anomalies (relative to its own background) are labelled examples of the "
           "anomaly-morphology prior; texture = v17 structure prior on the anchors (background_mode measured); auto = "
           "morphology in pure mode")
    A("--sim_exclude_m", type=str, default="",
      help="depth intervals 'lo-hi,lo-hi' (m): simulated anomalies overlapping them are withheld (leave-out protocols)")
    A("--morphology_bank", choices=["auto", "bank", "legacy"], default="auto",
      help="anomalies of the morphology kind: bank = measured + simulated + learned morphologies; legacy = the v20.2-v21 "
           "measured_like kind (measured shapes only)")
    A("--morphology_mix", type=str, default="auto",
      help="source shares of the bank 'measured:a,sim:b,learned:c' (auto = 0.25 / 0.25 / 0.5 of the sources present)")
    A("--morphology_learned", type=int, default=192, help="learned morphologies in the bank (0 = no learned prior)")
    A("--morphology_novelty", type=float, default=0.20,
      help="relative L2 distance below which a learned morphology is a copy of a measured / simulated one")
    A("--morphology_vae_steps", type=int, default=3000, help="training steps of the morphology prior (beta-VAE)")
    A("--decay_constraint", choices=["auto", "hard", "soft", "off"], default="auto",
      help="time-monotone constraint: hard = projection (cummin + monotone head), soft = hinge penalty, off = none; auto = "
           "hard only when the measured quantity is positive and non-increasing within its noise")
    A("--signal_quantity", choices=["auto", "dbdt", "b", "signed"], default="auto",
      help="measured quantity, recorded with the decay-constraint evidence (signed forces the constraint off)")
    A("--lambda_decay_soft", type=float, default=5.0, help="decay_constraint soft: weight of the hinge on rises in time")
    A("--qc_export", choices=["split", "legacy"], default="split",
      help="split: accepted / review / rejected sets, drawn until the accepted set is complete; legacy = v21 (redraws, "
           "failures kept and flagged)")
    A("--qc_budget_factor", type=float, default=3.0, help="split: at most this x generated_profile_count profiles drawn")
    A("--qc_borderline_margin", type=float, default=0.10,
      help="split: a pass within this share of a gate's tolerance band is borderline (review set)")
    A("--cv_buffer_m", type=str, default="auto",
      help="cross-validation: stations within this distance (m) of a held-out block are not trained on either (auto = "
           "the practical range of the measured residual's autocorrelation along the profile, 0 = off)")
    A("--identifiability_trials", type=int, default=0,
      help="injection-recovery test of the measured-profile decomposition with conductive-sphere responses (an "
           "independent forward model): number of trials, 1/6 of them noise-only nulls (0 = off)")
    A("--evidence_only", action="store_true",
      help="write the data-side evidence (decay constraint, decomposition stability, residual dependence, "
           "identifiability bench) and stop before training")
    A("--diversity_max_profiles", type=int, default=2048,
      help="profiles (random subset) in the diversity / effective-count report of the generated set (0 = off)")
    A("--no_ablation_subdir", action="store_true",
      help="an ablation writes into the folders it is given (no <folder>/ablation/<mode> below them)")
    A("--gpu_memory_cap_gb", type=float, default=0.0,
      help="cap this process's CUDA memory in GB (0 = none; for a GPU shared with other jobs)")
    A("--anomaly_footprint_range", type=str, default="0.25,1.0",
      help="airborne / ground anomalies: sigma0 as a fraction of the measured lateral correlation length at tau")
    # ---- session control ----
    A("--run_scope", choices=["ask", "full", "test"], default="ask")
    A("--ablation", type=str, default="ask", help="ask | mode | comma list (batch) | report")
    A("--cv_folds", type=int, default=0, help="leave-depth-block-out cross-validation folds (0 = off)")
    A("--cv_scale", choices=["quick", "medium", "full"], default="quick",
      help="epochs per fold: quick = 1/4 of the stage epochs, medium = 1/2 (both at half steps/epoch), full = as configured")
    A("--cv_generate_profiles", type=int, default=64)
    A("--cv_ablations", type=str, default="", help="comma list of ablation modes to cross-validate besides full")
    A("--cv_folds_select", type=str, default="", help="1-based folds to run, comma list (empty = all)")
    A("--cv_time_holdout", type=float, default=0.0,
      help="time-direction fold: hide the last N decades of the measured gates and extend into them (0 = off)")
    # ---- output / visualisation ----
    A("--generated_data_path", type=str, default=SERVER_OUTPUT_DIR)
    A("--model_dir", type=str, default=SERVER_MODEL_DIR)
    A("--normalization_mode", choices=["auto", "log_minmax", "symlog_minmax", "signed_log1p"], default="auto")
    A("--symlog_scale", type=float, default=1.0)
    A("--normalization_percentile", type=float, default=99.5)
    A("--normalization_clip_min", type=float, default=0.0)
    A("--normalization_clip_max", type=float, default=1.0)
    A("--normalization_eps", type=float, default=1e-6)
    A("--no_learned_embedding_eval", action="store_true")
    A("--visualization", choices=["pca", "tsne", "all", "none"], default="all")
    A("--max_embedding_samples", type=int, default=12000)
    A("--tsne_perplexity", type=float, default=30.0)
    A("--gallery_count", type=int, default=12)
    A("--plot_dpi", type=int, default=200)
    A("--export_profile_samples", type=int, default=0, help="also write the first K profiles as separate files")
    A("--export_legacy_csv", action="store_true", help="also write the v15 flat / matrix CSVs on the measured stations")
    A("--no_export_npy", action="store_true")
    A("--no_export_measured_gates", action="store_true")
    A("--no_export_csv", action="store_true",
      help="[v21] skip the two large CSVs (decay curves / profiles); the array, the index and the measured-gate CSV stay")
    A("--log_dir", type=str, default="", help="[v21] run-log folder (default <model_dir>/logs; server: <root>/log/rfmg)")
    A("--plot_dir", type=str, default="",
      help="[v21] plot folder (default <model_dir>/analysis/plots; server: <root>/plot/rfmg)")
    A("--report_dir", type=str, default="", help="[v21] report folder (default <model_dir>/analysis/reports)")
    for flag in ("no_save_training_curves", "no_save_embedding_plots", "no_save_curve_gallery",
                 "no_save_envelope_plot", "no_save_spectrum_plot", "no_save_checkpoint", "no_save_config_json",
                 "no_save_history_csv", "no_save_history_json", "no_save_metrics_json", "no_save_metrics_csv",
                 "no_save_stage_end", "no_save_best_per_stage", "no_save_interrupt_checkpoint"):
        A(f"--{flag}", action="store_true")
    # ---- checkpoints ----
    A("--train_action", choices=["auto", "restart", "resume", "ask", "generate_only"], default="ask")
    A("--resume_checkpoint", type=str, default="")
    A("--checkpoint_subdir", type=str, default="checkpoints")
    A("--save_every_epochs", type=int, default=1)
    A("--allow_mismatch_resume", action="store_true")
    A("--keep_checkpoints", choices=["all", "final"], default="all",
      help="final: after generation keep only final.pt (the generation model) and the morphology bank; cross-validation "
           "folds keep no checkpoint once scored")
    # ---- union bridge / reproduction ----
    A("--repro_mode", choices=["stable_appendix", "strict_quality"], default="stable_appendix")
    A("--bridge_alpha_min", type=float, default=0.15)
    A("--bridge_alpha_max", type=float, default=0.85)
    A("--bridge_sampling", choices=["uniform", "beta_balanced", "beta_bimodal"], default="beta_balanced")
    A("--bridge_mid_frac", type=float, default=0.20)
    A("--bridge_real_near_frac", type=float, default=0.75)
    A("--bridge_sim_near_frac", type=float, default=0.05)
    A("--union_real_weight", type=float, default=0.35)
    A("--union_sim_weight", type=float, default=0.35)
    A("--union_bridge_weight", type=float, default=0.30)
    A("--freeze_condition_extractor", dest="freeze_condition_extractor", action="store_true", default=True)
    A("--train_condition_extractor", dest="freeze_condition_extractor", action="store_false")
    A("--quality_feedback_strength", type=float, default=0.25)
    A("--validation_fraction", type=float, default=0.05)
    A("--validation_min_samples", type=int, default=256)
    A("--validation_profiles", type=int, default=128)
    A("--no_dual_discriminator", action="store_true")
    A("--no_feature_matching", action="store_true")
    A("--no_bridge_generator", action="store_true")
    A("--no_cmg", action="store_true")
    A("--no_profile", action="store_true", help="disable the profile (depth-direction) model level")
    A("--no_profile_critic", action="store_true")
    A("--mc_instance_noise_sigma_start", type=float, default=0.02)
    A("--mc_instance_noise_sigma_floor", type=float, default=0.0)
    A("--mc_instance_noise_decay_epochs", type=int, default=30)
    A("--mc_feature_moment_weight", type=float, default=1.0)
    for name, kind, default in (("switch_ma_window", int, 10), ("switch_loss_delta", float, 1e-5),
                                ("switch_loss_level", float, 1e-3), ("union_radius_quantile", float, 0.75),
                                ("union_radius_scale", float, 1.25), ("over_expand_scale", float, 1.75),
                                ("union_dataset_multiplier", float, 2.0), ("union_stage2_adv_max_weight", float, 0.10),
                                ("mc_late_log_monotonic_weight", float, 10.0), ("mc_pull_warmup_fraction", float, 0.5),
                                ("mc_pull_floor", float, 0.01), ("mc_pull_hold_epochs", int, 10),
                                ("mc_independent_real_anchor_count", int, 86), ("large_generation_chunk_size", int, 65536),
                                ("large_pilot_count", int, 12288), ("pairing_knn_pool", int, 50000),
                                ("pairing_usage_penalty", float, 1.0), ("real_jitter_min_duplicate_fraction", float, 0.0),
                                ("real_jitter_scale", float, 0.0), ("synthetic_count", int, 2048)):
        A(f"--{name}", type=kind, default=default)
    A("--mc_real_anchor_compaction", choices=["declared", "exact", "off"], default="declared")
    A("--v33_skip_legacy_real_guided_patch", action="store_true")
    for name, kind, default in (("copy_guard_wave_l1", float, 0.004), ("generation_local_blend", float, 0.55),
                                ("generation_rejection_tries", int, 4), ("generation_samples_per_real", int, 1),
                                ("identity_loss_weight", float, 80.0), ("lambda_learned_cond", float, 3.0),
                                ("local_precision_weight", float, 4.0), ("local_residual_scale", float, 0.035),
                                ("novelty_feature_frac", float, 0.18), ("novelty_loss_weight", float, 3.0),
                                ("novelty_wave_l1", float, 0.006), ("pairing_candidate_pool", int, 50000),
                                ("pairing_knn_k", int, 8), ("pairing_matched_prob", float, 0.7),
                                ("pairing_max_repeat_fraction", float, 0.05), ("pairing_mid_prob", float, 0.2),
                                ("pairing_min_unique_usage", float, 0.25), ("pairing_p95_p50_max_ratio", float, 3.5),
                                ("quality_raw_copy_max", float, 0.3), ("quality_raw_cover_min", float, 0.85),
                                ("quality_raw_mono_min", float, 0.55), ("quality_raw_over_max", float, 0.55),
                                ("quality_raw_peak_min", float, 0.75), ("quality_raw_real_cover_min", float, 0.8),
                                ("quality_raw_sim_cover_min", float, 0.5), ("real_memory_capacity", int, 16384),
                                ("real_memory_recall_weight", float, 5.0), ("sim_guidance_start_strength", float, 0.08),
                                ("sim_prior_max_strength", float, 0.08), ("sim_prior_warmup_frac", float, 0.35),
                                ("sim_to_real_phase_frac", float, 0.55), ("supervisor_residual_weight", float, 0.0),
                                ("union_freeze_recovery_epochs", int, 5), ("union_metrics_max_samples", int, 4096),
                                ("union_stage2_gate_batches", int, 8), ("union_stage2_gate_cover_min", float, 0.08),
                                ("union_stage2_gate_mono_min", float, 0.15), ("union_stage2_gate_over_max", float, 0.95),
                                ("union_stage2_gate_peak_min", float, 0.45)):
        A(f"--{name}", type=kind, default=default)
    for name in ("enable_real_guided_stage2", "use_sim_prior_at_export"):
        A(f"--{name}", action="store_true")
    for name in ("disable_matched_real_peak_calibration", "final_real_only_generation"):
        A(f"--{name}", action="store_true", default=True)
    A("--metrics_fixed_memory_bank", dest="metrics_fixed_memory_bank", action="store_true", default=True)
    A("--no_metrics_fixed_memory_bank", dest="metrics_fixed_memory_bank", action="store_false")
    A("--union_stage2_gate_enable", dest="union_stage2_gate_enable", action="store_true", default=True)
    A("--no_union_stage2_gate", dest="union_stage2_gate_enable", action="store_false")
    A("--fulltrain_profile", choices=["debug", "medium", "strict"], default="debug")
    A("--pairing_mode", choices=["matched_mix", "random", "aligned_cyclic_v35", "aligned_cyclic"],
      default="aligned_cyclic_v35")
    A("--smoke_test", action="store_true")
    A("--smoke_samples", type=int, default=64)
    return p


_CAPACITY = {"baseline": 1.0, "enhanced": 1.5, "high": 2.0}
# every width of the network; the parameter counts of each preset are printed at start ([MODEL PARAMETER REPORT]) and
# written to config_and_device.json
CAPACITY_PRESETS: Dict[str, Dict[str, int]] = {
    "large": {"hidden_dim": 128, "cond_dim": 128, "z_dim": 48, "inception_base_ch": 48, "condition_deep_dim": 256,
              "condition_branch_dim": 48, "disc_branch_ch": 96, "gen_shared_dim": 1024, "gen_static_dim": 1024,
              "gen_z_proj_dim": 192, "gen_temporal_hidden": 256, "profile_ctx_hidden": 128, "profile_code_dim": 24,
              "profile_disc_ch": 96},
    "xl": {"hidden_dim": 192, "cond_dim": 192, "z_dim": 64, "inception_base_ch": 64, "condition_deep_dim": 384,
           "condition_branch_dim": 64, "disc_branch_ch": 128, "gen_shared_dim": 1536, "gen_static_dim": 1536,
           "gen_z_proj_dim": 256, "gen_temporal_hidden": 384, "profile_ctx_hidden": 192, "profile_code_dim": 32,
           "profile_disc_ch": 128},
}


def apply_capacity_preset(cfg: "AppConfig") -> None:
    """--capacity_profile large / xl: the preset's widths replace the individual width flags."""
    pre = CAPACITY_PRESETS.get(str(cfg.model.capacity_profile))
    if pre:
        for k, v in pre.items():
            setattr(cfg.model, k, int(v))


def build_app_config(args: argparse.Namespace) -> AppConfig:
    cap = _CAPACITY.get(str(args.capacity_profile), 1.0)
    tl = int(args.target_len) if int(args.target_len) > 0 else DESIGN_TARGET_LEN
    cfg = AppConfig(
        runtime=RuntimeConfig(seed=args.seed, gpu_devices=args.gpu_devices, device_index=args.device_index,
                              use_amp=bool(args.use_amp), batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=not args.no_pin_memory, gpu_memory_cap_gb=float(args.gpu_memory_cap_gb)),
        data=DataConfig(data_name=args.data_name, task_mode=args.task_mode, read_by=args.read_by, target_len=tl,
                        real_data_file=args.real_data_file, simulata_data_file=args.simulata_data_file,
                        input_csv=args.input_csv, synthetic_count=args.synthetic_count,
                        generated_data_count=args.generated_data_count, metric_iteration=args.metric_iteration,
                        real_time_unit=args.real_time_unit, sim_time_unit=args.sim_time_unit,
                        sim_depth_match=args.sim_depth_match, sim_level_anchor=args.sim_level_anchor,
                        sim_time_extrapolation=args.sim_time_extrapolation,
                        sim_structure_gain=str(args.sim_structure_gain),
                        sim_reservoir_rows=args.sim_reservoir_rows, measured_pool_size=args.measured_pool_size,
                        pool_draw_chunk=max(0, int(args.pool_draw_chunk)),
                        gate_time_file=sanitize_path(args.gate_time_file), line_id=str(args.line_id),
                        component_block=int(args.component_block), gate_window=str(args.gate_window),
                        min_positive_frac=float(args.min_positive_frac), max_stations=int(args.max_stations),
                        sensor_height=str(args.sensor_height), sim_survey_check=str(args.sim_survey_check),
                        sim_role=str(args.sim_role), sim_exclude_m=str(args.sim_exclude_m)),
        model=ModelConfig(module=args.module, hidden_dim=args.hidden_dim, cond_dim=args.cond_dim, z_dim=args.z_dim,
                          num_layer=args.num_layer, dropout=args.dropout, inception_base_ch=args.inception_base_ch,
                          condition_deep_dim=args.condition_deep_dim, condition_branch_dim=args.condition_branch_dim,
                          disc_branch_ch=args.disc_branch_ch,
                          gen_shared_dim=int(round(args.gen_shared_dim * cap)),
                          gen_static_dim=int(round(args.gen_static_dim * cap)),
                          gen_z_proj_dim=args.gen_z_proj_dim,
                          gen_temporal_hidden=int(round(args.gen_temporal_hidden * cap)),
                          capacity_profile=args.capacity_profile,
                          recovery_monotone_head=bool(args.recovery_monotone_head),
                          profile_ctx_hidden=args.profile_ctx_hidden, profile_code_dim=args.profile_code_dim,
                          profile_disc_ch=args.profile_disc_ch),
        train=TrainConfig(stage0_epochs=args.stage0_epochs, stage0b_epochs=args.stage0b_epochs,
                          stage1_supervisor_epochs=args.stage1_supervisor_epochs, stage1_epochs=args.stage1_epochs,
                          stage2_epochs=args.stage2_epochs, steps_per_epoch=args.steps_per_epoch, lr_er=args.lr_er,
                          lr_s=args.lr_s, lr_g=args.lr_g, lr_d=args.lr_d, k_d=args.k_d, grad_clip=args.grad_clip,
                          ema_decay=args.ema_decay, r1_gamma=args.r1_gamma,
                          early_stop_patience=args.early_stop_patience, stage1_min_epochs=args.stage1_min_epochs),
        losses=LossConfig(**{k: getattr(args, k) for k in (
            "lambda_bridge_wave", "lambda_bridge_feat", "lambda_sup", "lambda_recon", "lambda_mom", "lambda_dist",
            "lambda_div", "lambda_guid", "lambda_spec", "lambda_shape", "lambda_late", "lambda_adv",
            "lambda_copy_guard", "lambda_over_expand", "lambda_mechanism_anchor", "lambda_fm", "lambda_tsmooth", "lambda_prof_adv",
            "lambda_prof_rough", "lambda_prof_trend", "lambda_prof_ms", "lambda_anomaly", "lambda_anomaly_shape",
            "lambda_structure", "lambda_purity", "lambda_gp", "lambda_cond",
            "lambda_union_cover", "lambda_segment", "lambda_union_div", "stage1_use_adaptive_loss",
            "late_start_frac", "mmd_sigma", "lambda_field_shell", "lambda_decay_soft")}),
        profile=ProfileConfig(enabled=not args.no_profile, critic=not (args.no_profile or args.no_profile_critic),
                              window=args.profile_window, window_mix=args.profile_window_mix,
                              max_window=int(args.profile_max_window),
                              depth_alpha_sigma=args.depth_alpha_sigma, registration_zoom_min=args.registration_zoom_min,
                              structure_gain_spread=args.structure_gain_spread, rough_tol=args.prof_rough_tol,
                              latent_white_sigma=args.latent_white_sigma, latent_scale_probs=args.latent_scale_probs,
                              trend_tol=args.prof_trend_tol, copy_min_dist=args.copy_guard_min_dist,
                              gen_alpha_mix=args.gen_alpha_mix, critic_real_mix=args.critic_real_mix,
                              survey_mode=("borehole" if args.survey_mode == "ask" else args.survey_mode),
                              anomaly_enabled=not args.no_anomalies, anomaly_count_probs=args.anomaly_count_probs,
                              max_anomalies=int(args.max_anomalies),
                              measured_anomaly_mode=str(args.measured_anomaly_mode),
                              anomaly_amp_range=args.anomaly_amp_range, anomaly_neg_frac=args.anomaly_neg_frac,
                              anomaly_width_range=args.anomaly_width_range, anomaly_tau_range=args.anomaly_tau_range,
                              anomaly_broadband_frac=args.anomaly_broadband_frac,
                              anomaly_diffusion=not args.no_anomaly_diffusion, bg_level_sigma=args.bg_level_sigma,
                              bg_tilt_sigma=args.bg_tilt_sigma, profile_depth_points=args.profile_depth_points,
                              extend_profile_frac=float(args.extend_profile_frac), extend_side=str(args.extend_side),
                              extend_time_ms=str(args.extend_time_ms), unmeasured_spread=float(args.unmeasured_spread),
                              anomaly_footprint_range=str(args.anomaly_footprint_range),
                              background_mode=str(args.background_mode), background_sigma=str(args.background_sigma),
                              pure_departure_head=not bool(args.no_pure_departure_head),
                              purity_tol=str(args.purity_tol), background_trend=str(args.background_trend),
                              undulation_sides=str(args.undulation_sides),
                              inventory_noise=str(args.inventory_noise),
                              measured_anomaly_prob=float(args.measured_anomaly_prob),
                              measured_anomaly_scale=str(args.measured_anomaly_scale),
                              borehole_anomaly_model=str(args.borehole_anomaly_model),
                              borehole_anomaly_kinds=str(args.borehole_anomaly_kinds),
                              anomaly_min_depth_m=float(args.anomaly_min_depth_m),
                              manifold_constraint=str(args.manifold_constraint),
                              background_manifold=str(args.background_manifold), warp_modes=int(args.warp_modes),
                              warp_max_strain=float(args.warp_max_strain), fam_enabled=not bool(args.no_fam),
                              fam_k=int(args.fam_k), fam_kappa=float(args.fam_kappa),
                              fam_min_frac=float(args.fam_min_frac), morphology_bank=str(args.morphology_bank),
                              morphology_mix=str(args.morphology_mix), morphology_learned=int(args.morphology_learned),
                              morphology_novelty=float(args.morphology_novelty),
                              morphology_vae_steps=int(args.morphology_vae_steps)),
        visualization=VisualizationConfig(mode=args.visualization, max_embedding_samples=args.max_embedding_samples,
                                          tsne_perplexity=args.tsne_perplexity, gallery_count=args.gallery_count,
                                          dpi=args.plot_dpi, save_training_curves=not args.no_save_training_curves,
                                          save_embedding_plots=not args.no_save_embedding_plots,
                                          save_curve_gallery=not args.no_save_curve_gallery,
                                          save_envelope_plot=not args.no_save_envelope_plot,
                                          save_spectrum_plot=not args.no_save_spectrum_plot),
        export=ExportConfig(generated_data_path=args.generated_data_path, model_dir=args.model_dir,
                            save_checkpoint=not args.no_save_checkpoint, save_config_json=not args.no_save_config_json,
                            save_history_csv=not args.no_save_history_csv, save_history_json=not args.no_save_history_json,
                            save_metrics_json=not args.no_save_metrics_json, save_metrics_csv=not args.no_save_metrics_csv,
                            export_measured_gates=not args.no_export_measured_gates,
                            export_profile_samples=args.export_profile_samples,
                            export_legacy_csv=bool(args.export_legacy_csv), export_npy=not args.no_export_npy,
                            export_csv=not bool(args.no_export_csv), log_dir=str(args.log_dir),
                            plot_dir=str(args.plot_dir), report_dir=str(args.report_dir),
                            ablation_subdir=not bool(args.no_ablation_subdir)),
        checkpoint=CheckpointConfig(train_action=args.train_action, resume_checkpoint=args.resume_checkpoint,
                                    checkpoint_subdir=args.checkpoint_subdir, save_every_epochs=args.save_every_epochs,
                                    save_stage_end=not args.no_save_stage_end,
                                    save_best_per_stage=not args.no_save_best_per_stage,
                                    save_interrupt_checkpoint=not args.no_save_interrupt_checkpoint,
                                    allow_mismatch_resume=bool(args.allow_mismatch_resume),
                                    keep=str(args.keep_checkpoints)),
        normalization=NormalizationConfig(mode=args.normalization_mode, symlog_scale=args.symlog_scale,
                                          robust_percentile=args.normalization_percentile,
                                          clip_min=args.normalization_clip_min, clip_max=args.normalization_clip_max,
                                          eps=args.normalization_eps),
        generation=GenerationConfig(mode=args.generation_mode, interpolate_alpha=args.interpolate_alpha,
                                    learned_embedding_eval=not args.no_learned_embedding_eval,
                                    paired_eval_limit=args.paired_eval_limit, union_real_weight=args.union_real_weight,
                                    union_sim_weight=args.union_sim_weight, union_bridge_weight=args.union_bridge_weight,
                                    generated_profile_count=args.generated_profile_count,
                                    gen_chunk_profiles=args.gen_chunk_profiles, qc_max_tries=args.qc_max_tries,
                                    qc_rough_max=args.qc_rough_max, qc_envelope_min=args.qc_envelope_min,
                                    qc_retention_range=args.qc_retention_range,
                                    qc_anomaly_corr_min=args.qc_anomaly_corr_min, qc_contrast_range=args.qc_contrast_range,
                                    qc_anomaly_min_decades=args.qc_anomaly_min_decades,
                                    disable_generation_postprocess=bool(args.disable_generation_postprocess),
                                    decay_constraint=str(args.decay_constraint),
                                    signal_quantity=str(args.signal_quantity), qc_export=str(args.qc_export),
                                    qc_budget_factor=float(args.qc_budget_factor),
                                    qc_borderline_margin=float(args.qc_borderline_margin),
                                    sim_prior_k=args.sim_prior_k, real_bootstrap_jitter=args.real_bootstrap_jitter,
                                    sim_prior_strength=args.sim_prior_strength,
                                    real_guided_adv_weight=args.real_guided_adv_weight),
        reproduction=ReproductionConfig(mode=args.repro_mode, bridge_alpha_min=args.bridge_alpha_min,
                                        bridge_alpha_max=args.bridge_alpha_max, bridge_sampling=args.bridge_sampling,
                                        bridge_mid_frac=args.bridge_mid_frac,
                                        bridge_real_near_frac=args.bridge_real_near_frac,
                                        bridge_sim_near_frac=args.bridge_sim_near_frac,
                                        freeze_condition_extractor=bool(args.freeze_condition_extractor),
                                        use_dual_discriminator=not args.no_dual_discriminator,
                                        use_bridge_generator=not args.no_bridge_generator,
                                        use_feature_matching=not args.no_feature_matching,
                                        cmg_enabled=not args.no_cmg, cmg_form=str(args.cmg_form),
                                        instance_noise_sigma_start=args.mc_instance_noise_sigma_start,
                                        instance_noise_sigma_floor=args.mc_instance_noise_sigma_floor,
                                        instance_noise_decay_epochs=args.mc_instance_noise_decay_epochs,
                                        quality_feedback_strength=args.quality_feedback_strength,
                                        validation_profiles=args.validation_profiles),
        cv=CVConfig(folds=max(0, int(args.cv_folds)), scale=args.cv_scale,
                    generate_profiles=args.cv_generate_profiles, ablations=args.cv_ablations,
                    folds_select=str(args.cv_folds_select), time_holdout_decades=float(args.cv_time_holdout),
                    buffer_m=str(args.cv_buffer_m), identifiability_trials=max(0, int(args.identifiability_trials)),
                    evidence_only=bool(args.evidence_only), diversity_max=max(0, int(args.diversity_max_profiles))),
        extra_paths=[p for p in (sanitize_path(x) for x in (args.extra_paths or [])) if p],
    )
    cfg.losses.lambda_fm = float(args.lambda_fm if args.lambda_fm != 1.0 else args.mc_feature_moment_weight)
    if not cfg.reproduction.use_feature_matching:
        cfg.losses.lambda_fm = 0.0
    apply_capacity_preset(cfg)
    apply_manifold_constraint(cfg)
    if int(args.target_len) == 0:
        print(f"{_LOG} target_len=0 (auto) -> {DESIGN_TARGET_LEN} = original design length")
    return cfg


def apply_manifold_constraint(cfg: AppConfig) -> None:
    """--manifold_constraint v20: affine regional head, credibility-only gate and no field-admissible manifold; v21
    (default) keeps the individual flags as given."""
    if str(cfg.profile.manifold_constraint).lower() == "v20":
        cfg.profile.background_manifold = "affine"
        cfg.profile.fam_enabled = False
        cfg.losses.lambda_field_shell = 0.0
        cfg.reproduction.cmg_form = "v20"
        apply_v21_generator(cfg)


def apply_v21_generator(cfg: "AppConfig") -> None:
    """Generator settings implied by --manifold_constraint v20: no simulated shapes, the legacy shape bank, one-sided
    undulations, the hard decay projection and a single export set."""
    cfg.data.sim_role = "off"
    cfg.profile.morphology_bank = "legacy"
    cfg.profile.undulation_sides = "one"
    cfg.generation.decay_constraint = "hard"
    cfg.generation.qc_export = "legacy"
    cfg.data.pool_draw_chunk = 0


def print_config_summary(cfg: AppConfig, device_info: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"{_LOG} CONFIG SUMMARY  ({cfg.program_version})")
    print(f"  data      : name={cfg.data.data_name} target_len={cfg.data.target_len} | measured={cfg.data.real_data_file}")
    print(f"              simulated={cfg.data.simulata_data_file}")
    print(f"              sim depth match={cfg.data.sim_depth_match} level anchor={cfg.data.sim_level_anchor} "
          f"structure gain={cfg.data.sim_structure_gain}")
    print(f"  model     : hidden={cfg.model.hidden_dim} cond={cfg.model.cond_dim} z={cfg.model.z_dim} "
          f"gen_temporal={cfg.model.gen_temporal_hidden} monotone_head={cfg.model.recovery_monotone_head} "
          f"capacity={cfg.model.capacity_profile}")
    print(f"  profile   : enabled={cfg.profile.enabled} critic={cfg.profile.critic} window={cfg.profile.window} "
          f"mix={cfg.profile.window_mix} survey={cfg.profile.survey_mode}")
    print(f"  extension : profile {cfg.profile.extend_profile_frac:g} x span ({cfg.profile.extend_side}) | late time "
          f"{cfg.profile.extend_time_ms} | unmeasured spread x{cfg.profile.unmeasured_spread:g} | line file: gates "
          f"{cfg.data.gate_window}, line {cfg.data.line_id}, block {cfg.data.component_block}, max stations "
          f"{cfg.data.max_stations if cfg.data.max_stations >= 0 else 'auto (line files 256, matrix files as read)'} | "
          f"training window cap {cfg.profile.max_window if cfg.profile.max_window > 0 else 'auto (lines 64, boreholes none)'}")
    print(f"  fidelity  : lambda_anomaly={cfg.losses.lambda_anomaly:g} lambda_anomaly_shape={cfg.losses.lambda_anomaly_shape:g} "
          f"lambda_structure={cfg.losses.lambda_structure:g} "
          f"lambda_prof_rough={cfg.losses.lambda_prof_rough:g}")
    print(f"  manifold  : constraint={cfg.profile.manifold_constraint} background={cfg.profile.background_manifold} "
          f"(warp modes {cfg.profile.warp_modes}, strain <= {cfg.profile.warp_max_strain:g}) | FAM "
          f"{'on' if cfg.profile.fam_enabled else 'off'} (k={cfg.profile.fam_k}, kappa={cfg.profile.fam_kappa:g}, "
          f"min frac {cfg.profile.fam_min_frac:g}, shell lambda {cfg.losses.lambda_field_shell:g}) | CMG "
          f"{'on' if cfg.reproduction.cmg_enabled else 'off'} form={cfg.reproduction.cmg_form}")
    print(f"  train     : epochs s0={cfg.train.stage0_epochs} s0B={cfg.train.stage0b_epochs} "
          f"s1A={cfg.train.stage1_supervisor_epochs} s1B={cfg.train.stage1_epochs} s2={cfg.train.stage2_epochs} "
          f"| steps/epoch={cfg.train.steps_per_epoch} batch={cfg.runtime.batch_size} lr_g={cfg.train.lr_g:g} "
          f"lr_d={cfg.train.lr_d:g}")
    print(f"  generation: mode={cfg.generation.mode} curves={cfg.data.generated_data_count} "
          f"profiles={cfg.generation.generated_profile_count or 'auto'} alpha mix={cfg.profile.gen_alpha_mix}")
    print(f"  output    : model_dir={cfg.export.model_dir} | generated={cfg.export.generated_data_path}")
    print(f"  ablation  : {cfg.reproduction.ablation_mode} | cv folds={cfg.cv.folds} ({cfg.cv.scale})")
    print(f"  device    : {json.dumps(device_info, ensure_ascii=False)}")
    print("=" * 88)


def resolve_io_paths(cfg: AppConfig) -> Dict[str, Any]:
    """Measured / simulated inputs: explicit path > server default > <project>/input fallbacks.
    Output folders: the server drive when it exists, else <project>/model and <project>/output."""
    root = project_root()
    notes: Dict[str, Any] = {"project_root": root}

    def _pick(explicit: str, server_default: str, local_names: Sequence[str], role: str) -> str:
        explicit = sanitize_path(explicit)
        server_default_n = sanitize_path(server_default)
        if explicit and explicit != server_default_n:
            if not os.path.isfile(explicit):
                raise FileNotFoundError(f"[PATH] {role} file given explicitly does not exist: {explicit}")
            notes[role] = {"path": explicit, "source": "explicit"}
            return explicit
        tried = [server_default_n]
        if server_default_n and os.path.isfile(server_default_n):
            notes[role] = {"path": server_default_n, "source": "server default"}
            return server_default_n
        for nm in local_names:
            cand = os.path.join(root, "input", nm)
            tried.append(cand)
            if os.path.isfile(cand):
                notes[role] = {"path": cand, "source": "local project input/"}
                return cand
        notes[role] = {"path": "", "source": "missing", "tried": tried}
        return ""

    if cfg.data.input_csv:
        cfg.data.real_data_file = cfg.data.input_csv
    if cfg.extra_paths:
        if len(cfg.extra_paths) >= 1:
            cfg.data.real_data_file = cfg.extra_paths[0]
        if len(cfg.extra_paths) >= 2:
            cfg.data.simulata_data_file = cfg.extra_paths[1]
    if cfg.data.data_name != "tem_simulate":
        cfg.data.real_data_file = _pick(cfg.data.real_data_file, SERVER_REAL_FILE, LOCAL_REAL_NAMES, "measured")
        if cfg.data.data_name in ("combined", "combined+sine", "simulata"):
            cfg.data.simulata_data_file = _pick(cfg.data.simulata_data_file, SERVER_SIM_FILE, LOCAL_SIM_NAMES, "simulated")
        if not cfg.data.real_data_file and cfg.data.data_name != "simulata":
            raise FileNotFoundError("[PATH] measured file not found; tried " + json.dumps(notes.get("measured"), ensure_ascii=False)
                                    + " — pass --real_data_file PATH")
    for attr, server_default, local in (("model_dir", SERVER_MODEL_DIR, "model"),
                                        ("generated_data_path", SERVER_OUTPUT_DIR, "output")):
        cur = sanitize_path(getattr(cfg.export, attr))
        if cur == sanitize_path(server_default) and not _drive_available(cur):
            alt = os.path.join(root, local)
            setattr(cfg.export, attr, alt)
            notes[attr] = {"path": alt, "source": "server drive unavailable -> project folder"}
        else:
            notes[attr] = {"path": cur, "source": "as configured"}
    # on the paper server the run logs and plots go to <root>/log/rfmg and <root>/plot/rfmg unless set
    if os.path.isdir(SERVER_ROOT):
        if not cfg.export.log_dir:
            cfg.export.log_dir = os.path.join(SERVER_LOG_DIR, "rfmg")
        if not cfg.export.plot_dir:
            cfg.export.plot_dir = os.path.join(SERVER_PLOT_DIR, "rfmg")
    notes["log_dir"] = {"path": cfg.export.log_dir or os.path.join(cfg.export.model_dir, "logs")}
    notes["plot_dir"] = {"path": cfg.export.plot_dir or os.path.join(cfg.export.model_dir, "analysis", "plots")}
    return notes
