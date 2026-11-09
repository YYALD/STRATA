"""Configuration of a run: one dataclass per part (runtime, data, model, training, losses, profile, export,
checkpoints, normalisation, generation, reproduction, cross-validation); these dataclasses and the command-line
defaults (cli.py) are the single source of truth.  The ablation switches read by the networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from . import RFMG_PROGRAM_NAME, RFMG_VERSION


SERVER_ROOT = "/root/autodl-tmp/progect_rfgan"
SERVER_REAL_FILE = SERVER_ROOT + "/input/liuyuan_denoised.csv"
SERVER_SIM_FILE = SERVER_ROOT + "/input/simulate.csv"
LOCAL_REAL_NAMES = ("liuyuan_denoised.csv",)
LOCAL_SIM_NAMES = ("simulate.csv",)
SERVER_MODEL_DIR = SERVER_ROOT + "/model"
SERVER_OUTPUT_DIR = SERVER_ROOT + "/outputdata"
SERVER_LOG_DIR = SERVER_ROOT + "/log"
SERVER_PLOT_DIR = SERVER_ROOT + "/plot"
DESIGN_TARGET_LEN = 256          # original design length (real_01.csv / simulate.csv carried 256 gates)
ALLOWED_TARGET_LEN = (64, 128, 256)


@dataclass
class RuntimeConfig:
    seed: int = 42
    gpu_devices: str = ""
    device_index: int = 0
    use_amp: bool = True
    batch_size: int = 128               # curves per generator step (profile windows x depths)
    num_workers: int = 0
    pin_memory: bool = True
    enable_tf32: bool = False
    enable_cudnn_benchmark: bool = False
    auto_max_throughput_on_large_gpu: bool = True
    gpu_memory_cap_gb: float = 0.0      # cap of this process's CUDA memory (0 = none): on a GPU shared with other jobs
                                        # the caching allocator otherwise grows into WDDM-paged memory and training
                                        # stalls


@dataclass
class DataConfig:
    data_name: str = "combined"
    task_mode: str = "hybrid"
    read_by: str = "auto"
    target_len: int = DESIGN_TARGET_LEN
    real_data_file: str = SERVER_REAL_FILE
    simulata_data_file: str = SERVER_SIM_FILE
    input_csv: str = ""
    synthetic_count: int = 2048
    generated_data_count: int = 500000
    metric_iteration: int = 1
    real_time_unit: str = "auto"
    sim_time_unit: str = "auto"
    sim_depth_match: str = "auto"
    sim_level_anchor: str = "structure"
    sim_time_extrapolation: str = "loglinear"
    sim_structure_gain: str = "auto"
    sim_reservoir_rows: int = 8192
    measured_pool_size: int = 4096
    # curve pool drawn in batches of at most this many single-station windows: the anomaly visibility check evaluates
    # every drawn anomaly on its own local axis x every time node, so one batch of 4096 windows peaked at several GB
    # before training started and set how many runs fit on one GPU
    pool_draw_chunk: int = 256
    # airborne / ground line files (geometry columns + gate columns <prefix><n>) and long profiles
    gate_time_file: str = ""            # gate times of a line file (default: the one gate-time file next to it)
    line_id: str = "auto"               # line of a multi-line file (auto = the line with the most stations)
    component_block: int = 1            # 1-based block of gate columns (zoff1..zoff50 = Z gates 1..26 + ...: block 1)
    gate_window: str = "auto"           # 1-based gates 'a-b' | auto = usable-gate rule (positive, falling decay)
    min_positive_frac: float = 0.99     # usable-gate rule: share of stations with a positive value at a kept gate
    max_stations: int = -1              # long profiles are stacked to at most this many stations: -1 auto = line files
                                        # 256, matrix files as read; an explicit N applies to both; 0 = off
    sensor_height: str = "auto"         # airborne sensor clearance (m) | auto = median of the height column
    sim_survey_check: str = "auto"      # auto: a simulated file of another survey type (borehole depth axis under a line
                                        # survey, or the reverse) is not used as structure prior | off
    # role of the simulated profile: morphology = its anomalies (relative to its own background, read at the measured
    # stations, decomposed by the measured rule) are labelled examples of the anomaly-morphology prior (background_mode
    # pure); off = not used; auto = morphology in pure mode, texture otherwise
    sim_role: str = "auto"
    sim_exclude_m: str = ""             # "lo-hi,lo-hi" (m): simulated anomalies overlapping these are withheld


@dataclass
class ModelConfig:
    module: str = "lstm"
    hidden_dim: int = 64
    cond_dim: int = 64
    z_dim: int = 32
    num_layer: int = 3
    dropout: float = 0.2
    inception_base_ch: int = 32
    condition_deep_dim: int = 128
    condition_branch_dim: int = 32
    disc_branch_ch: int = 64
    gen_shared_dim: int = 512
    gen_static_dim: int = 512
    gen_z_proj_dim: int = 128
    gen_temporal_hidden: int = 128
    capacity_profile: str = "baseline"
    recovery_monotone_head: bool = True
    profile_ctx_hidden: int = 64
    profile_code_dim: int = 16
    profile_disc_ch: int = 48


@dataclass
class TrainConfig:
    stage0_epochs: int = 20
    stage0b_epochs: int = 4
    stage1_supervisor_epochs: int = 8
    stage1_epochs: int = 40
    stage2_epochs: int = 60
    steps_per_epoch: int = 100
    lr_er: float = 1e-3
    lr_s: float = 1e-3
    lr_g: float = 1.5e-4
    lr_d: float = 1.0e-4
    k_d: int = 1
    grad_clip: float = 0.5
    ema_decay: float = 0.999
    r1_gamma: float = 2.0
    r1_every: int = 4
    early_stop_patience: int = 20
    stage1_min_epochs: int = 20


@dataclass
class LossConfig:
    # --- curve level ---
    lambda_bridge_wave: float = 25.0     # segment-balanced L1, generated vs union-bridge anchor
    lambda_bridge_feat: float = 5.0      # condition-frame anchoring (unit-norm L1)
    lambda_sup: float = 10.0             # supervisor (latent next-gate) consistency
    lambda_recon: float = 10.0           # recovery reconstruction regulariser during G steps
    lambda_mom: float = 5.0              # per-gate mean / std matching against the anchors
    lambda_dist: float = 0.5             # physics-descriptor symmetric Chamfer (coverage / precision)
    lambda_field_shell: float = 2.0      # FAM soft shell: squared excess of a background curve over its nearest anchor
                                         # ball (robust-standardised descriptor units)
    lambda_div: float = 1.0              # curve-level two-z mode seeking
    lambda_guid: float = 2.0             # physics guidance: log-log slope / curvature envelope
    lambda_spec: float = 0.0
    lambda_shape: float = 2.0            # derivative (shape) consistency with the anchor
    lambda_late: float = 1.0             # extra weight of the late segment in the wave term
    lambda_adv: float = 1.0              # latent + sequence critics
    lambda_copy_guard: float = 2.0       # profile anti-copy hinge
    lambda_over_expand: float = 0.5      # union-envelope hinge
    lambda_mechanism_anchor: float = 0.5 # CMG gate loss
    lambda_fm: float = 1.0               # critic feature matching
    lambda_tsmooth: float = 5.0          # time-direction curvature texture vs the anchor
    # --- profile level ---
    lambda_prof_adv: float = 1.0
    lambda_prof_rough: float = 20.0
    lambda_prof_trend: float = 10.0
    lambda_prof_ms: float = 6.0                 # 2 -> 6 with the push off the anomaly supports
    lambda_anomaly: float = 5.0                 # anomaly fidelity: departure must not rescale the anchor's anomalies
    lambda_anomaly_shape: float = 1.0           # ... nor blur their shape beyond ANOMALY_SHAPE_TOL (on the support)
    lambda_structure: float = 5.0               # structure fidelity: departure must not rescale the anchor's local
                                                # depth structure (measured narrow features included), per time band
    lambda_purity: float = 5.0                  # background purity: the departure from the anchor carries no
                                                # anomaly-scale structure (high-pass at the background scale <=
                                                # tolerance)
    lambda_decay_soft: float = 5.0              # decay_constraint soft: hinge on rises in time (normalised units)
    # --- legacy fields accepted for CLI compatibility ---
    lambda_gp: float = 10.0
    lambda_cond: float = 10.0
    lambda_union_cover: float = 0.0
    lambda_segment: float = 0.0
    lambda_union_div: float = 0.0
    lambda_learned_cond: float = 0.0
    stage1_use_adaptive_loss: bool = False
    late_start_frac: float = 0.50
    mmd_sigma: float = 1.0


@dataclass
class ProfileConfig:
    enabled: bool = True
    critic: bool = True
    window: int = 16
    window_mix: str = "16:0.7,32:0.2,full:0.1"
    max_window: int = 0                         # training windows hold at most this many stations ("full" = the whole
                                                # profile up to it): 0 auto = 64 on airborne / ground lines, no cap on
                                                # boreholes; an explicit N applies to every survey
    latent_basis: int = 4
    # the generated profiles are signals: no white latent noise and no short depth band; the latent varies smoothly (>=
    # 6 spacings, log-time cosines)
    latent_white_sigma: float = 0.0
    latent_scale_probs: str = "0.5,0.5,0"       # long / mid / short depth scales of the latent field
    depth_alpha_sigma: float = 0.10
    registration_zoom_min: float = 0.70
    structure_gain_spread: float = 2.0          # per-profile structure gain ~ LogU[1/s, s]
    rough_tol: float = 1.03                     # departure local structure <= 0.25 x the anchor's (+ floor); 1.5
                                                # allowed 1.12 x
    trend_sigma_stations: float = 3.0
    trend_tol: float = 0.02
    copy_min_dist: float = 0.004
    # real windows of the profile critic; measured / sim windows carry the same random anomaly set and
    # background modulation as the anchor, so the critic never learns "anomaly = fake"
    critic_real_mix: str = "measured:0.4,sim:0.2,anchor:0.4"
    station_jitter_prob: float = 0.5
    gen_alpha_mix: str = "0.85-1.0:0.6,0.55-0.85:0.3,0.30-0.55:0.1"
    survey_mode: str = "borehole"
    # random anomaly composition of every training / generated profile (count, sign, amplitude, depth, width,
    # time behaviour); amplitudes are anchored to the measured profile's own local structure
    anomaly_enabled: bool = True
    # the number of labelled anomalies of a profile (measured + synthetic) is at most max_anomalies; auto = every count
    # 0 .. max_anomalies equally often
    anomaly_count_probs: str = "auto"
    max_anomalies: int = 5
    anomaly_amp_range: str = "auto"             # |peak| in decades "lo,hi" | auto
    anomaly_neg_frac: float = 0.35              # share of negative (response-reducing) anomalies
    anomaly_width_range: str = "1.0,0.12"       # Gaussian sigma: min in local station spacings, max as span fraction
    anomaly_tau_range: str = "auto"             # eddy-current time constant range in ms "lo,hi" | auto = [t1, 2 tn]
    anomaly_broadband_frac: float = 0.30        # share of anomalies with a time-flat signature (near-hole zones)
    anomaly_diffusion: bool = True              # footprint width grows as sqrt(1 + t / tau) (TEM diffusion)
    bg_level_sigma: float = 0.03                # per-profile background level modulation (decades)
    bg_tilt_sigma: float = 0.03                 # per-profile background decay-rate modulation (decades over the window)
    profile_depth_points: int = 1000            # depth samples of every exported profile (0 = measured stations)
    # survey-adaptive extensions beyond the measured coverage
    extend_profile_frac: float = 0.0            # profile-axis extension beyond the measured stations (fraction of the span)
    extend_side: str = "auto"                   # auto: borehole the deeper end, airborne / ground both line ends | low |
                                                # high | both
    extend_time_ms: str = "0"                   # late-time window extension: target time in ms | depth (borehole: last
                                                # gate x (d_end / d_last)^2, diffusion depth ~ sqrt t) | 0 = off
    unmeasured_spread: float = 1.0              # x the calibrated spread of the anchor where nothing is measured (0 = rule)
    # background of the anchors: "pure" = the measured profile's smooth geological background (robust regional /
    # residual separation, the measured anomalies are not copied into the generated profiles: a profile with no labelled
    # anomaly is pure background), "measured" = the measured profile itself
    background_mode: str = "pure"
    pure_departure_head: bool = True            # pure: the network's departure off the labels is projected onto the
                                                # background scale in the forward pass (--no_pure_departure_head: off)
    background_sigma: str = "auto"              # background smoothing scale in m | auto = borehole 2 x station spacing,
                                                # lines 2 x the measured lateral correlation length
    purity_tol: str = "auto"                    # largest anomaly-scale departure allowed outside labels (decades) | auto =
                                                # half the smallest labelled anomaly amplitude
    # "0 anomalies" = the overall decay trend, no rise or dip: "monotone" = the smooth background is also made to decay
    # along the profile (where it rises, the trend passes under the rise: a late-time response wider than sigma_B that
    # the robust fit kept), and the profiles' regional modulation keeps that shape
    background_trend: str = "auto"
    undulation_sides: str = "two"
    # station-scale noise of the measured residual for the anomaly inventory (threshold = max(4 noise, tolerance)):
    # "quiet" = from the quietest quarter of the station-to-station changes (holds when the anomalies cover more than
    # half of the stations: Liuyuan's 4 x MAD reached 0.025 decades at 3.7 ms and left out a 6-station late feature of
    # 0.024)
    inventory_noise: str = "auto"
    # the measured profile's own anomalies (inventory of the background decomposition) are anchor anomalies: every
    # profile with anomalies carries each of them with this probability, at its measured position, with its measured
    # shape, its amplitude scaled by a log-uniform factor; profiles with 0 anomalies stay pure background
    measured_anomaly_prob: float = 1.0
    # "count": each measured anomaly is one of a profile's anomalies - a profile with K anomalies carries m of the
    # measured ones (m uniform over 0 .. min(K, number measured), which ones at random) and K - m synthetic ones, so the
    # count is 0 .. max_anomalies and a profile can show exactly the measured set
    measured_anomaly_mode: str = "auto"
    measured_anomaly_scale: str = "auto"        # "lo,hi" | auto = per anomaly 0.5 .. max(2, the factor that lifts its
                                                # peak to twice the synthetic visibility floor)
    anomaly_footprint_range: str = "0.25,1.0"   # buried family (airborne / ground): sigma0 as a fraction of the measured
                                                # line's lateral correlation length at the anomaly's time scale
    # borehole synthetic anomalies from conductor bodies at a distance r from the hole ("physical"): the hole through a
    # conductor (single positive peak), a conductor beside the hole (axial dipole field: negative-centred three-peak to
    # positive / negative pair with the body's dip) or a near-hole zone (all times)
    borehole_anomaly_model: str = "measured"
    borehole_anomaly_kinds: str = "auto"        # auto: measured model
                                                # measured_like:0.50,intersected:0.35,off_hole:0.15, zone:0
    anomaly_min_depth_m: float = 50.0           # borehole: no synthetic anomaly centred above this depth + its distance r
    manifold_constraint: str = "v21"
    # background manifold of the regional head: "warp" = the trend under a monotone depth reparametrisation (warp_modes
    # sine modes over the measured span, strain |du/dz| <= warp_max_strain) composed with the level / contrast map;
    # "affine" = level / contrast only
    background_manifold: str = "auto"
    warp_modes: int = 3
    warp_max_strain: float = 0.25
    # field-admissible manifold (FAM) of the background curves: union of anchor balls in the robust-standardised
    # decay-curve descriptor space, radius fam_kappa x the distance to the fam_k-th nearest anchor (floor: half the
    # median nearest-anchor spacing); fam_min_frac of a profile's background curves must be admissible (acceptance)
    fam_enabled: bool = True
    fam_k: int = 5
    fam_kappa: float = 1.25
    fam_min_frac: float = 0.95
    # anomaly-morphology bank of the "morphology" kind: measured, simulated and learned morphologies (MorphologyPrior);
    # auto = bank
    morphology_bank: str = "auto"
    morphology_mix: str = "auto"                # source shares "measured:a,sim:b,learned:c" | auto = 0.25 / 0.25 / 0.5
                                                # (renormalised over the sources present)
    morphology_learned: int = 192               # learned morphologies in the bank (0 = no learned prior)
    morphology_novelty: float = 0.20            # a learned morphology this close (relative L2) to a measured / simulated
                                                # example is a copy and is not used
    morphology_vae_steps: int = 3000            # training steps of the morphology prior (beta-VAE, batch 256)


@dataclass
class VisualizationConfig:
    mode: str = "all"
    max_embedding_samples: int = 12000
    tsne_perplexity: float = 30.0
    gallery_count: int = 12
    dpi: int = 200
    save_training_curves: bool = True
    save_embedding_plots: bool = True
    save_curve_gallery: bool = True
    save_envelope_plot: bool = True
    save_spectrum_plot: bool = True


@dataclass
class ExportConfig:
    generated_data_path: str = SERVER_OUTPUT_DIR
    model_dir: str = SERVER_MODEL_DIR
    save_checkpoint: bool = True
    save_config_json: bool = True
    save_history_csv: bool = True
    save_history_json: bool = True
    save_metrics_json: bool = True
    save_metrics_csv: bool = True
    export_measured_gates: bool = True
    export_profile_samples: int = 0
    export_legacy_csv: bool = False
    export_npy: bool = True                     # float32 .npy of all profiles + axes (fast reading)
    # the two large CSVs (decay curves, profiles) are optional: experiment runs keep the array + index + the
    # measured-gate CSV; log / plot / report folders may be set apart from the model folder (server layout)
    export_csv: bool = True
    log_dir: str = ""                           # "" = <model_dir>/logs (server: SERVER_LOG_DIR/rfmg)
    plot_dir: str = ""                          # "" = <model_dir>/analysis/plots (server: SERVER_PLOT_DIR/rfmg)
    report_dir: str = ""                        # "" = <model_dir>/analysis/reports
    ablation_subdir: bool = True                # False: an ablation writes into the folders it is given


@dataclass
class CheckpointConfig:
    train_action: str = "ask"
    resume_checkpoint: str = ""
    strict_resume: bool = True
    checkpoint_subdir: str = "checkpoints"
    save_every_epochs: int = 1
    save_stage_end: bool = True
    save_best_per_stage: bool = True
    save_interrupt_checkpoint: bool = True
    allow_mismatch_resume: bool = False
    # final: after generation only final.pt (the generation model) and the morphology bank stay; cross-validation folds
    # keep nothing once scored (an xl run writes ~2.5 GB of stage checkpoints, a fold as much)
    keep: str = "all"


@dataclass
class NormalizationConfig:
    mode: str = "auto"                  # auto -> log_minmax for positive decays, symlog_minmax otherwise
    symlog_scale: float = 1.0
    robust_percentile: float = 99.5
    clip_min: float = 0.0
    clip_max: float = 1.0
    eps: float = 1e-6
    margin: float = 0.05


@dataclass
class GenerationConfig:
    mode: str = "union_guided"
    interpolate_alpha: float = 0.5
    learned_embedding_eval: bool = True
    paired_eval_limit: int = 12288
    union_real_weight: float = 0.35
    union_sim_weight: float = 0.35
    union_bridge_weight: float = 0.30
    generated_profile_count: int = 1000
    gen_chunk_profiles: int = 8
    qc_max_tries: int = 12                      # per-profile acceptance passes: latent redraws first, the last third
                                                # redraw the profile itself (3 -> 12: 7 failing profiles were shipped;
                                                # 12 passes: none, fewer parameter redraws than with 3)
    qc_rough_max: float = 2.0                   # station roughness <= qc_rough_max x anchor (per band, median)
    qc_envelope_min: float = 0.95               # share of the profile's curves inside the physics envelope
    qc_retention_range: str = "0.5,1.5"         # generated / intended anomaly amplitude (regression slope)
    qc_anomaly_corr_min: float = 0.8            # correlation of generated and intended anomaly fields (0.5 -> 0.8: 10 %
                                                # of the anomalous profiles carried a blurred anomaly)
    qc_contrast_range: str = "0.7,1.3"          # the anomaly's contrast against its local background (centred slope
                                                # over the support) / its labelled amplitude
    qc_anomaly_min_decades: float = 0.05        # anomalies weaker than this (2.5 x the 0.02-decade diversity floor)
                                                # sit inside the generator's own variation and are not tested
    disable_generation_postprocess: bool = False
    # the hard time-monotone projection only where the measured quantity supports it (decay_constraint.json)
    decay_constraint: str = "auto"              # auto | hard | soft | off
    signal_quantity: str = "auto"               # dbdt | b | signed | auto (recorded with the evidence)
    # export: split = accepted / review / rejected sets, profiles drawn until the accepted set is complete, no parameter
    # redraw
    qc_export: str = "split"
    qc_budget_factor: float = 3.0               # at most this x generated_profile_count profiles are drawn
    qc_borderline_margin: float = 0.10          # a pass within this share of a gate's tolerance band is borderline
    sim_prior_k: int = 8
    real_bootstrap_jitter: float = 0.015
    sim_prior_strength: float = 0.25
    real_guided_adv_weight: float = 0.05


@dataclass
class ReproductionConfig:
    mode: str = "stable_appendix"
    bridge_alpha_min: float = 0.15
    bridge_alpha_max: float = 0.85
    bridge_sampling: str = "beta_balanced"
    bridge_mid_frac: float = 0.20
    bridge_real_near_frac: float = 0.75
    bridge_sim_near_frac: float = 0.05
    freeze_condition_extractor: bool = True
    use_dual_discriminator: bool = True
    use_bridge_generator: bool = True
    use_feature_matching: bool = True
    cmg_enabled: bool = True
    cmg_form: str = "eaai"
    film_residual: float = 0.25
    mode_seek_two_z_enabled: bool = True
    instance_noise_sigma_start: float = 0.02
    instance_noise_sigma_floor: float = 0.0
    instance_noise_decay_epochs: int = 30
    curriculum: bool = True
    stage2_anchor_floor: float = 0.35
    quality_feedback_strength: float = 0.25
    validation_profiles: int = 128              # 32 -> 128: 32-profile medians misranked checkpoints
    ablation_mode: str = "full"


@dataclass
class CVConfig:
    folds: int = 0
    scale: str = "quick"
    generate_profiles: int = 64
    ablations: str = ""
    folds_select: str = ""              # 1-based folds to run (comma list), empty = all
    time_holdout_decades: float = 0.0   # time-direction fold: the last N decades of the measured gates are hidden
    buffer_m: str = "auto"              # stations within this distance of a held-out block are not trained on either
    identifiability_trials: int = 0     # sphere injection-recovery trials of the decomposition (0 = off)
    evidence_only: bool = False         # write the data-side evidence and stop (no training)
    diversity_max: int = 2048           # profiles in the diversity / effective-count report (0 = off)


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    extra_paths: List[str] = field(default_factory=list)
    program_version: str = f"{RFMG_PROGRAM_NAME} v{RFMG_VERSION}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_RFMG_ABLATION_FLAGS = {"no_film": False, "no_condition": False, "plain_export": False}
