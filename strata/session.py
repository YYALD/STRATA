"""Interactive session and runner: menus (run scope, ablation, survey mode, checkpoint action), device setup, the run
of a session (run_session), the self-check (smoke_main) and main()."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import _LOG, RFMG_PROGRAM_NAME, RFMG_VERSION
from .ablation import (_RFMG_ABLATION_MODES, _rfmg_ablation_redirect_dirs, _rfmg_ablation_report, _rfmg_apply_ablation,
                       run_standalone_baseline)
from .checkpoints import (_format_best_metric, _format_stage_progress, _scan_all_checkpoints, find_latest_checkpoint,
                          prune_checkpoints, select_best_checkpoint)
from .cli import build_app_config, build_argument_parser, print_config_summary, resolve_io_paths
from .common import ensure_dir, project_root, sanitize_path, save_json, set_seed, start_run_log
from .config import LOCAL_REAL_NAMES, SERVER_REAL_FILE, AppConfig
from .data.contract import build_dense_depth_axis
from .data.time_grid import depth_interp_profile
from .evidence import decomposition_evidence, diversity_report, identifiability_bench
from .export import (GeneratedProfileStore, build_profile_index, disk_preflight, export_history, export_qc_split,
                     export_size_estimate, manifold_report, print_qc_table, save_generated_outputs)
from .figures import (_axis_names, background_purity_report, extension_diagnostics, print_structure_summary,
                      save_anomaly_figures, save_background_figure, save_extension_figure, save_figures,
                      save_structure_figure)
from .manifold.joint import _DECAY, _tmono_np, anomaly_count_law, measured_anomaly_mode
from .pipeline import load_data_contract
from .quality import (acceptance_summary, curve_quality_metrics, profile_quality_metrics, qc_gate_status,
                      structure_fidelity_metrics)
from .trainer import RFMGGenTrainer
from .validation import _train_all_stages, run_cross_validation


_RFMG_TEST_PROFILE = {"stage0_epochs": 3, "stage0b_epochs": 1, "stage1_supervisor_epochs": 2, "stage1_epochs": 4,
                      "stage2_epochs": 5, "steps_per_epoch": 25, "generated_profile_count": 48,
                      "sim_reservoir_rows": 2048, "measured_pool_size": 1024, "validation_profiles": 8}
_RFMG_TEST_RUN = False


_RFMG_ABLATION_QUEUE: List[str] = []
_RFMG_ABLATION_COMPLETED: List[str] = []


_RFMG_SURVEY_MODES = ("borehole", "airborne", "ground")


def _stdin_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def _ask(prompt: str) -> Optional[str]:
    """input that returns None on EOF (piped / non-interactive stdin) instead of raising."""
    try:
        return input(prompt)
    except EOFError:
        return None
    except Exception:
        return None


def _rfmg_prompt_run_scope() -> bool:
    SEP = "=" * 88
    print(SEP)
    print("  RFMG-Gen  —  RUN SCOPE SELECTION")
    print(SEP)
    print("  [F]  Full        — full data, full epochs (production; hours)")
    print("  [T]  Test        — small-sample smoke test: full Stage-0 -> 2 pipeline,")
    print("                     reduced pools and epochs, completes in minutes")
    print()
    raw = _ask("  Run scope [F/T] (default F): ")
    if raw is not None and raw.strip().lower() in {"t", "test", "smoke", "trial"}:
        print("  ✓  TEST RUN selected.")
        print(SEP)
        return True
    print("  ✓  FULL RUN selected.")
    print(SEP)
    return False


def _rfmg_apply_test_profile(cfg: AppConfig) -> None:
    p = _RFMG_TEST_PROFILE
    print("=" * 88)
    print("[RFMG TEST-RUN] small-sample test ACTIVE — full pipeline, reduced size/epochs.")
    for holder, keys in ((cfg.train, ("stage0_epochs", "stage0b_epochs", "stage1_supervisor_epochs", "stage1_epochs",
                                      "stage2_epochs", "steps_per_epoch")),
                         (cfg.generation, ("generated_profile_count",)),
                         (cfg.data, ("sim_reservoir_rows", "measured_pool_size")),
                         (cfg.reproduction, ("validation_profiles",))):
        for k in keys:
            old = getattr(holder, k)
            setattr(holder, k, int(p[k]))
            print(f"[RFMG TEST-RUN]   {k}: {old} -> {p[k]}")
    cfg.train.stage1_min_epochs = min(cfg.train.stage1_min_epochs, cfg.train.stage1_epochs)
    print("=" * 88)


def _rfmg_prompt_ablation_mode() -> str:
    if _RFMG_ABLATION_QUEUE:
        m = _RFMG_ABLATION_QUEUE.pop(0)
        print(f"[ABLATION] batch queue -> running '{m}' ({len(_RFMG_ABLATION_QUEUE)} remaining)")
        return m
    SEP = "=" * 88
    print(SEP)
    print("  RFMG-Gen — ABLATION SELECTION")
    print("  [0] full RFMG-Gen (default)      [1] no_film          [2] no_curriculum")
    print("  [3] single_critic                [4] no_manifold_gating (流形门控)")
    print("  [5] no_mechanism_anchor          [6] no_physics_pull   [7] no_diversity")
    print("  [8] no_real_endpoint             [9] no_bridge")
    print("  [10] cgan baseline (standalone)  [11] gan baseline (standalone)")
    print("  [12] no_profile (剖面层关闭: v15 逐道独立生成)  [13] no_profile_critic (无剖面判别器)")
    print("  [14] no_anomalies (无随机异常组合: 锚点只含实测/模拟结构)")
    print("  [15] no_field_manifold (无场可容许流形: 无壳约束/验收门)  [16] affine_background (背景流形只有电平/对比度)")
    print("  [17] cmg_credibility (CMG 只有可信度门, 无信赖域)      [18] v20_manifold (v21 流形机制全关 = v20.2.0)")
    print("  [B] batch — comma list, e.g. 0,12,13 (sequential; shared data/seed; per-mode dirs)")
    print("  [P] report only — aggregate completed variants into ablation_report.png")
    raw = _ask("  Ablation [0-18/B/P] (default 0): ")
    print(SEP)
    return _parse_ablation_answer(raw)


def _parse_ablation_answer(raw: Optional[str]) -> str:
    global _RFMG_ABLATION_QUEUE
    try:
        raw = (raw or "").strip().lower()
        if raw in ("p", "report"):
            return "__report__"
        if raw == "":
            return "full"
        if raw in _RFMG_ABLATION_MODES:
            return raw
        if raw.startswith("b") or ("," in raw):
            body = raw[1:] if raw.startswith("b") else raw
            toks = [s.strip() for s in body.replace(";", ",").split(",") if s.strip()]
            modes = [(_RFMG_ABLATION_MODES[int(s)] if s.isdigit() else s) for s in toks]
            modes = [m for m in modes if m in _RFMG_ABLATION_MODES]
            assert modes, "empty batch list"
            _RFMG_ABLATION_QUEUE = modes[1:]
            print(f"[ABLATION] batch accepted: {modes} (sequential, per-mode dirs)")
            return modes[0]
        return _RFMG_ABLATION_MODES[int(raw)]
    except Exception as e:
        print(f"[ABLATION] menu fallback to 'full' ({e})")
        return "full"


def _rfmg_prompt_survey_mode() -> str:
    """Survey menu.  Enter (or a non-interactive run) = detect the survey type from the measured file: a borehole
    matrix file (depth axis) -> borehole, so the default borehole workflow is unchanged."""
    env = str(os.environ.get("RFMG_SURVEY_MODE", "")).strip().lower()
    if env in _RFMG_SURVEY_MODES or env == "auto":
        print(f"[SURVEY] env RFMG_SURVEY_MODE -> '{env}'")
        return env
    SEP = "=" * 88
    print(SEP)
    print("  RFMG-Gen — SURVEY MODE SELECTION (勘测模式)")
    print("  [B] borehole 井中TEM — profile axis = depth along the hole (m); extension: deeper than the measured stations")
    print("  [A] airborne 航空TEM — profile axis = distance along the flight line (m); extension: line ends, later times")
    print("  [G] ground   地面TEM — profile axis = distance along the survey line (m); extension: line ends, later times")
    print("  [Enter] auto — detect from the measured file (depth matrix -> borehole; line file with a height column -> airborne)")
    raw = _ask("  Survey [B/A/G/Enter=auto]: ")
    print(SEP)
    r = (raw or "").strip().lower()
    if r in ("a", "airborne"):
        return "airborne"
    if r in ("g", "ground"):
        return "ground"
    if r in ("b", "borehole"):
        return "borehole"
    return "auto"


def _ask_checkpoint_path(prompt: str = "  Checkpoint path (.pt): ") -> str:
    raw = _ask(prompt)
    p_ = sanitize_path((raw or "").strip().strip('"').strip("'"))
    return p_ if (p_ and os.path.isfile(p_)) else ""


def choose_train_action(cfg: AppConfig, latest_ckpt: Optional[str], checkpoint_dir: str = "") -> str:
    """Interactive checkpoint menu: [R]estart / [A]uto / [C]ontinue / [S]elect / [P]ath / [G]enerate / [T]est run /
    [Q]uit."""
    global _RFMG_TEST_RUN
    action = cfg.checkpoint.train_action.lower().strip()
    if action == "restart":
        print("[CHECKPOINT] train_action=restart -> fresh training.")
        return "restart"
    if action in {"resume", "generate_only"}:
        if latest_ckpt and os.path.isfile(latest_ckpt):
            cfg.checkpoint.resume_checkpoint = cfg.checkpoint.resume_checkpoint or latest_ckpt
            print(f"[CHECKPOINT] train_action={action} -> {cfg.checkpoint.resume_checkpoint}")
            return action
        print(f"[CHECKPOINT] train_action={action} but no checkpoint found -> restart.")
        return "restart"
    if action == "auto":
        chosen = "resume" if latest_ckpt and os.path.isfile(latest_ckpt) else "restart"
        if chosen == "resume":
            cfg.checkpoint.resume_checkpoint = latest_ckpt
        print(f"[CHECKPOINT] train_action=auto -> {chosen}.")
        return chosen
    all_ckpts = _scan_all_checkpoints(checkpoint_dir or (os.path.dirname(latest_ckpt) if latest_ckpt else ""))
    SEP = "=" * 88
    has_ckpt = bool(all_ckpts)

    def _print_list() -> None:
        if not all_ckpts:
            print("  (no checkpoints found in checkpoint directory)")
            return
        print(f"  {'#':<3}  {'Tag':<22}  {'Saved At':<20}  {'Epoch':>5}  {'Stage Progress':<44}  {'Size':>8}")
        for i, ck in enumerate(all_ckpts, 1):
            mark = " ◄ latest" if ck["path"] == latest_ckpt else ""
            print(f"  {i:<3}  {ck['tag'][:22]:<22}  {ck['saved_at'][:20]:<20}  {ck['global_epoch']:>5}  "
                  f"{_format_stage_progress(ck['stage_progress'])[:44]:<44}  {ck['size_mb']:>6.1f}MB{mark}")

    print(SEP)
    print("  RFMG-Gen  —  TRAINING / GENERATION MODE SELECTION")
    print(SEP)
    _print_list()
    print()
    print("  OPTIONS:")
    print("  [R]  Restart   — discard existing checkpoints, train from scratch")
    print("  [A]  Auto      — continue the latest checkpoint if there is one, otherwise restart")
    if has_ckpt:
        print("  [C]  Continue  — resume from latest checkpoint (skip completed stages)")
        print("  [S]  Select    — pick a specific checkpoint from the list above")
    print("  [P]  Path      — type a checkpoint path and continue training from it")
    print("  [G]  Generate  — skip ALL training, run generation only (from the list or a typed path)")
    print("  [T]  Test run  — small-sample run of the whole pipeline, Stage-0 to export (minutes)")
    print("  [Q]  Quit      — exit program")
    print()
    while True:
        raw = _ask("  Your choice: ")
        if raw is None:
            print("[CHECKPOINT] Non-interactive environment detected -> auto mode.")
            if has_ckpt and latest_ckpt:
                cfg.checkpoint.resume_checkpoint = latest_ckpt
                return "resume"
            return "restart"
        raw = raw.strip().lower()
        if raw in {"r", "restart", "1", "fresh", "new"}:
            print("  ✓  Fresh training selected — all stages will run from epoch 1.")
            print(SEP)
            return "restart"
        if raw in {"a", "auto"}:
            if has_ckpt and latest_ckpt:
                cfg.checkpoint.resume_checkpoint = latest_ckpt
                print(f"  ✓  Auto: continuing the latest checkpoint {latest_ckpt}")
                print(SEP)
                return "resume"
            print("  ✓  Auto: no checkpoint found — training from scratch.")
            print(SEP)
            return "restart"
        if raw in {"t", "test", "smoke", "trial"}:
            _RFMG_TEST_RUN = True
            _rfmg_apply_test_profile(cfg)
            print("  ✓  Test run selected — small-sample run of the whole pipeline from scratch.")
            print(SEP)
            return "restart"
        if raw in {"p", "path", "manual"}:
            ck_ = _ask_checkpoint_path()
            if ck_:
                cfg.checkpoint.resume_checkpoint = ck_
                print(f"  ✓  Continuing from {ck_}")
                print(SEP)
                return "resume"
            print("  ✗  File not found. Please enter a valid .pt checkpoint path.")
            continue
        if has_ckpt and raw in {"c", "continue", "2", "resume"}:
            cfg.checkpoint.resume_checkpoint = latest_ckpt or all_ckpts[0]["path"]
            print(f"  ✓  Continuing from {cfg.checkpoint.resume_checkpoint}")
            print(SEP)
            return "resume"
        if has_ckpt and raw in {"s", "select", "3"}:
            sel = _ask("  Checkpoint #: ")
            try:
                ck = all_ckpts[int((sel or "").strip()) - 1]
                cfg.checkpoint.resume_checkpoint = ck["path"]
                print(f"  ✓  Selected {ck['path']} ({_format_stage_progress(ck['stage_progress'])}) "
                      f"{_format_best_metric(ck['best_metric'])}")
                print(SEP)
                return "resume"
            except Exception:
                print("  ✗  Invalid checkpoint number.")
                continue
        if raw in {"g", "generate", "4", "gen", "generate_only"}:
            if has_ckpt:
                sel = _ask("  Checkpoint # [Enter=latest, 0=type a path]: ")
                sel = (sel or "").strip()
                if sel == "0":
                    ck_ = _ask_checkpoint_path()
                    if not ck_:
                        print("  ✗  File not found.")
                        continue
                    cfg.checkpoint.resume_checkpoint = ck_
                else:
                    try:
                        ck = all_ckpts[0] if not sel else all_ckpts[int(sel) - 1]
                    except Exception:
                        print("  ✗  Invalid checkpoint number.")
                        continue
                    cfg.checkpoint.resume_checkpoint = ck["path"]
            else:
                ck_ = _ask_checkpoint_path()
                if not ck_:
                    print("  ✗  No valid checkpoint selected for generate-only.")
                    continue
                cfg.checkpoint.resume_checkpoint = ck_
            print(f"  ✓  Generate-only mode: {cfg.checkpoint.resume_checkpoint} — training stages will ALL be SKIPPED.")
            print(SEP)
            return "generate_only"
        if raw in {"q", "quit", "exit", "5"}:
            raise SystemExit("User cancelled from the training menu.")
        print(f"  ✗  Invalid input '{raw}'. Please enter: {'R/A/C/S/P/G/T/Q' if has_ckpt else 'R/A/P/G/T/Q'}")


# =====================================================================================
# Session runner
# =====================================================================================
def resolve_runtime_device(cfg: AppConfig) -> torch.device:
    if cfg.runtime.gpu_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.runtime.gpu_devices
    # is_available can be True while no device is visible (e.g. CUDA_VISIBLE_DEVICES=""): count the devices
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        idx = int(min(max(cfg.runtime.device_index, 0), torch.cuda.device_count() - 1))
        torch.cuda.set_device(idx)
        cap = float(cfg.runtime.gpu_memory_cap_gb)
        if cap > 0:
            # on a GPU shared with other jobs (Windows WDDM) a process whose caching allocator grows past the free
            # memory is paged and stalls without an error; the cap makes the allocator reuse its cache inside the limit
            # (the computation is unchanged; an over-sized step raises an out-of-memory error instead)
            total = float(torch.cuda.get_device_properties(idx).total_memory)
            frac = min(1.0, cap * 1024 ** 3 / total)
            torch.cuda.set_per_process_memory_fraction(frac, idx)
            print(f"[DEVICE] CUDA memory of this process capped at {cap:g} GB ({frac:.3f} of {total / 1024 ** 3:.1f} GB)")
        return torch.device(f"cuda:{idx}")
    print("[DEVICE] no CUDA device visible -> CPU")
    return torch.device("cpu")


def summarize_device(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": str(device), "torch": torch.__version__, "python": sys.version.split()[0]}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 1)})
    return info


def check_cuda_architecture(device: torch.device) -> Dict[str, Any]:
    """The PyTorch build must carry kernels (or PTX) for the GPU's architecture: RTX 50xx (Blackwell, compute
    capability 12.0) needs a CUDA 12.8 build (torch >= 2.7, e.g. pip install torch --index-url
    https://download.pytorch.org/whl/cu128); an older build fails at the first kernel with 'no kernel image is
    available'.  Stops with that instruction instead."""
    info: Dict[str, Any] = {}
    if device.type != "cuda":
        return info
    major, minor = torch.cuda.get_device_capability(device)
    arch = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
    want = f"sm_{major}{minor}"

    def _cc(a: str) -> Optional[Tuple[int, int]]:
        d = "".join(ch for ch in a.split("_", 1)[-1] if ch.isdigit())
        return (int(d[:-1]), int(d[-1])) if len(d) >= 2 else None
    # a cubin sm_XY runs on X.y >= X.Y of the same major (sm_86 on an RTX 4090, 8.9); PTX compute_XY is compiled at
    # load time for any device >= X.Y
    ok = (not arch) or want in arch or any(
        (c := _cc(a)) is not None and ((a.startswith("sm_") and c[0] == major and c[1] <= minor)
                                       or (a.startswith("compute_") and c <= (major, minor))) for a in arch)
    info = {"capability": f"{major}.{minor}", "arch_list": arch, "torch_cuda": torch.version.cuda,
            "kernels_for_device": bool(ok)}
    print(f"{_LOG} GPU {torch.cuda.get_device_name(device)} | compute capability {major}.{minor} | torch {torch.__version__} "
          f"CUDA {torch.version.cuda} | build archs {','.join(arch) if arch else 'unknown'}")
    if not ok:
        raise SystemExit(f"[DEVICE] this PyTorch build has no kernels for {want} ({torch.cuda.get_device_name(device)}); "
                         f"Blackwell GPUs (RTX 50xx, sm_120) need torch >= 2.7 built with CUDA 12.8: "
                         f"pip install torch --index-url https://download.pytorch.org/whl/cu128")
    return info


def apply_performance_backends(cfg: AppConfig, device: torch.device) -> None:
    tf32, bench = cfg.runtime.enable_tf32, cfg.runtime.enable_cudnn_benchmark
    if device.type == "cuda" and cfg.runtime.auto_max_throughput_on_large_gpu:
        if torch.cuda.get_device_properties(device).total_memory >= 24 * 1024 ** 3:
            # cuDNN autotuning only: TF32 matmuls keep ~1e-3 relative precision, ~4e-3 decades on the normalised
            # profiles - the size of the purity threshold (0.004 decades) the background tests and the regional head's
            # projections work at.  --enable_tf32 still switches it on explicitly.
            bench = True
    check_cuda_architecture(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(bench)
    else:
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        except Exception:
            pass
    print(f"{_LOG} backends: tf32={bool(tf32)} cudnn_benchmark={bool(bench)} amp={bool(cfg.runtime.use_amp and device.type == 'cuda')}")


def _scalar_items(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_scalar_items(v, prefix + k + "."))
        elif isinstance(v, (int, float, np.floating, np.integer, bool)) and not isinstance(v, str):
            out[prefix + k] = float(v)
    return out


def run_session(args: argparse.Namespace, ablation_mode: Optional[str] = None) -> Tuple[str, AppConfig]:
    global _RFMG_TEST_RUN
    cfg = build_app_config(args)
    # ---- 1. run scope ----
    test = _rfmg_prompt_run_scope() if args.run_scope == "ask" else (args.run_scope == "test")
    _RFMG_TEST_RUN = bool(test)
    if test:
        _rfmg_apply_test_profile(cfg)
    # ---- 2. ablation ----
    if ablation_mode is None:
        ablation_mode = _rfmg_prompt_ablation_mode() if args.ablation == "ask" else _parse_ablation_answer(args.ablation)
    io_notes = resolve_io_paths(cfg)
    if ablation_mode == "__report__":
        _rfmg_ablation_report(cfg, None)
        return ablation_mode, cfg
    # ---- 3. survey mode ----
    cfg.profile.survey_mode = _rfmg_prompt_survey_mode() if args.survey_mode == "ask" else args.survey_mode
    _rfmg_apply_ablation(cfg, ablation_mode)
    _rfmg_ablation_redirect_dirs(cfg, ablation_mode)
    set_seed(cfg.runtime.seed)
    device = resolve_runtime_device(cfg)
    analysis_dir = ensure_dir(os.path.join(cfg.export.model_dir, "analysis"))
    # report / plot / log folders apart from the model folder when set (server layout: <root>/log, <root>/plot)
    reports_dir = ensure_dir(cfg.export.report_dir or os.path.join(analysis_dir, "reports"))
    plots_dir = ensure_dir(cfg.export.plot_dir or os.path.join(analysis_dir, "plots"))
    ensure_dir(cfg.export.generated_data_path)
    start_run_log(cfg.export.log_dir or os.path.join(cfg.export.model_dir, "logs"), tag=f"rfmg_v{RFMG_VERSION}_{ablation_mode}")
    apply_performance_backends(cfg, device)
    device_info = summarize_device(device)
    print_config_summary(cfg, device_info)
    print(f"{_LOG} input/output resolution: {json.dumps(io_notes, ensure_ascii=False)}")
    # ---- 4. data contract ----
    meas, sim, norm, contract = load_data_contract(cfg)
    save_json(contract, os.path.join(reports_dir, "anchor_contract.json"))
    print(f"[DATA] anchor contract key={contract['contract_key']} -> {os.path.join(reports_dir, 'anchor_contract.json')}")
    # data-side evidence: decay constraint, decomposition stability and dependence, identifiability bench
    save_json(meas.report.get("decay_constraint") or {}, os.path.join(reports_dir, "decay_constraint.json"))
    save_json(decomposition_evidence(meas, cfg), os.path.join(reports_dir, "decomposition_evidence.json"))
    if int(cfg.cv.identifiability_trials) > 0:
        identifiability_bench(meas, cfg, int(cfg.cv.identifiability_trials), seed=cfg.runtime.seed + 4242,
                              reports_dir=reports_dir)
    if cfg.cv.evidence_only:
        print(f"[EVIDENCE] --evidence_only: data-side evidence written to {reports_dir}; no training")
        return ablation_mode, cfg
    if cfg.cv.folds > 0 or float(cfg.cv.time_holdout_decades) > 0:
        run_cross_validation(cfg, device, meas, sim, norm, contract, reports_dir, plots_dir)
        return ablation_mode, cfg
    trainer = RFMGGenTrainer(cfg, device, meas, sim, norm, contract)
    _n_prof0 = int(cfg.generation.generated_profile_count or math.ceil(cfg.data.generated_data_count / meas.D))
    _n_pts0 = int(cfg.profile.profile_depth_points) if int(cfg.profile.profile_depth_points) > 0 else int(meas.gen_depth.size)
    _exp_need = export_size_estimate(_n_prof0, max(_n_pts0, int(meas.gen_depth.size)), meas.T, int(meas.pm.time_ms.size),
                                     cfg)["total"]
    disk_preflight(cfg.export.generated_data_path, _exp_need, "before training", hard=False)
    rep = trainer.parameter_report()
    print("[MODEL PARAMETER REPORT] " + json.dumps({k: v["total"] for k, v in rep.items()}, ensure_ascii=False))
    if cfg.export.save_config_json:
        save_json({"config": cfg.to_dict(), "device": device_info, "parameter_report": rep, "normalizer": norm.to_dict(),
                   "contract_key": contract["contract_key"], "io": io_notes},
                  os.path.join(reports_dir, "config_and_device.json"))
    # ---- 5. training / checkpoint menu ----
    anchors = None
    store = None
    # the dense axis spans the generation axis (measured stations + virtual ones of a profile extension); every station
    # of that axis is a node, the measured ones are flagged measured_station = 1
    depth_dense, st_all = build_dense_depth_axis(meas.gen_depth, int(cfg.profile.profile_depth_points))
    st_index = st_all[trainer.meas_cols]
    if ablation_mode in ("cgan", "gan") and (meas.axis_depth is not None or meas.late_ext is not None):
        raise ValueError("[ABLATION] the standalone cgan / gan baselines generate on the measured coverage only; run them "
                         "without --extend_profile_frac / --extend_time_ms")
    if ablation_mode in ("cgan", "gan"):
        L_gen, trace = run_standalone_baseline(cfg, ablation_mode, trainer, device)
    else:
        # an explicit --resume_checkpoint wins, otherwise the newest file in the checkpoint folder
        rc = sanitize_path(cfg.checkpoint.resume_checkpoint)
        latest = rc if (rc and os.path.isfile(rc)) else find_latest_checkpoint(trainer.checkpoint_dir)
        if rc and not os.path.isfile(rc):
            raise FileNotFoundError(f"[CHECKPOINT] --resume_checkpoint does not exist: {rc}")
        action = choose_train_action(cfg, latest, trainer.checkpoint_dir)
        print(f"[CHECKPOINT] action={action} | latest={latest}")
        if action in ("resume", "generate_only"):
            trainer.load_checkpoint(cfg.checkpoint.resume_checkpoint or latest)
        if action == "generate_only":
            print("[CHECKPOINT] generate_only | training skipped")
        else:
            try:
                _train_all_stages(trainer, cfg)
            except KeyboardInterrupt:
                if cfg.checkpoint.save_interrupt_checkpoint:
                    p = trainer.save("interrupt", "KeyboardInterrupt")
                    print(f"[CHECKPOINT] interrupted -> {p}")
                raise
            best = select_best_checkpoint(trainer)
            if best:
                full_history = list(trainer.history)
                trainer.load_checkpoint(best)
                # the reloaded checkpoint carries the history up to its own epoch: keep the whole run's history
                trainer.history = full_history
        n_prof = int(cfg.generation.generated_profile_count or math.ceil(cfg.data.generated_data_count / meas.D))
        disk_preflight(cfg.export.generated_data_path, export_size_estimate(
            n_prof, int(depth_dense.size), meas.T, int(meas.pm.time_ms.size), cfg)["total"], "before generation", hard=True)
        print(f"[GENERATION] {n_prof} profiles x {depth_dense.size} {_axis_names(meas)['name']} points ({meas.D} measured stations embedded "
              f"exactly) x {meas.T} gates = {n_prof * depth_dense.size} decay curves | alpha mix "
              f"{cfg.profile.gen_alpha_mix} | per-profile acceptance up to {cfg.generation.qc_max_tries} passes")
        t0 = time.time()
        store = GeneratedProfileStore(cfg.export.generated_data_path, n_prof, depth_dense, meas.grid_ms, norm,
                                      keep_npy=cfg.export.export_npy)
        L_gen, trace = trainer.generate_profiles(n_prof, seed=cfg.runtime.seed + 2027, mode="gen", return_anchor=True,
                                                 accept=True, dense_depth=depth_dense, sink=store.put, progress=True)
        anchors = trace.pop("anchor_L", None)
        qs = trace.get("qc_split")
        if qs is not None:
            # accepted / review / rejected: the data set holds the accepted profiles only
            if not qs["complete"]:
                store.truncate(qs["accepted"])
                print(f"[QC-SPLIT] WARNING: {qs['accepted']} of {qs['requested']} profiles accepted within the budget of "
                      f"{qs['rule']['budget_slots']} slots: the data set holds {qs['accepted']} (raise --qc_budget_factor)")
            qrep = export_qc_split(qs, trace, meas, cfg, reports_dir)
            trace["qc_split"] = {k: v for k, v in qs.items() if k not in ("review", "rejected")}
            au = qs["audit"]
            print(f"[QC-SPLIT] {qs['slots_drawn']} slots drawn: accepted {qs['rates']['accepted']:.1%}, review "
                  f"{qs['rates']['review']:.1%}, rejected {qs['rates']['rejected']:.1%} (gates failed: "
                  + ", ".join(f"{k_} {v_}" for k_, v_ in qs["rejected_gate_failures"].items() if k_ != "rejected" and v_)
                  + f") | selection audit, accepted vs drawn: max TVD {au.get('max_tvd', 0.0):.3f} (counts / kinds / "
                  f"sources), max KS {au.get('max_ks', 0.0):.3f} (amplitudes, widths, centres, times, background) | "
                  f"review -> {(qrep.get('review_export') or {}).get('folder', '-')}, rejected -> "
                  f"{(qrep.get('rejected_export') or {}).get('folder', '-')}")
        dp = trace.get("dense_projection", {})
        print(f"[GENERATION] done in {time.time() - t0:.1f}s | dense time-monotone projection changed "
              f"{dp.get('changed_fraction', 0.0):.2e} of the values (max {dp.get('max_change_decades', 0.0):.2e} decades)")
        for k_, v_ in (trace.get("zero_anomaly_trend_projection") or {}).items():
            print(f"[GENERATION] '0 anomalies' profiles onto the overall decay trend along the profile ({k_}): changed "
                  f"{v_['changed_fraction']:.2e} of their values (max {v_['max_change_decades']:.2e} decades)")
    if (not cfg.model.recovery_monotone_head and not cfg.generation.disable_generation_postprocess
            and _DECAY["data_monotone"]):
        before = L_gen.copy()
        L_gen = np.minimum.accumulate(L_gen, axis=2)
        print(f"[POSTPROCESS] monotone head off -> running-minimum projection changed "
              f"{float(np.mean(np.any(before != L_gen, axis=2))):.2%} of curves")
    if store is None:   # standalone baselines generate on the stations: PCHIP along depth onto the dense axis
        store = GeneratedProfileStore(cfg.export.generated_data_path, L_gen.shape[0], depth_dense, meas.grid_ms, norm,
                                      keep_npy=cfg.export.export_npy)
        for i0 in range(0, L_gen.shape[0], 16):
            Ld = np.stack([depth_interp_profile(meas.depth, L_gen[k], depth_dense) for k in range(i0, min(i0 + 16, L_gen.shape[0]))])
            store.put(i0, _tmono_np(Ld, axis=2))
    # ---- 6. quality, export, figures ---- the QC compares with the measured profile on the measured coverage; the
    # extension bands are checked by extension_diagnostics
    L_all, A_all = L_gen, anchors
    mc = trainer.meas_cols
    kt = int(np.sum(meas.known_time_mask()))
    extended = (mc.size != L_all.shape[1]) or (kt != meas.T)
    if extended:
        L_gen = np.ascontiguousarray(L_all[:, mc, :kt])
        anchors = np.ascontiguousarray(A_all[:, mc, :kt]) if A_all is not None else None
    L_meas_q = meas.L[:, :kt]
    # background_mode pure: the profiles are tested against the measured BACKGROUND they are anchored on; the raw
    # measured profile, with the anomalies it carries, is a diagnostic (background_purity_report)
    pure = bool(getattr(trainer, "bg_pure", False)) and meas.background is not None
    # with the measured anomalies kept the reference is the measured profile itself
    L_ref_q = np.asarray(meas.background[:, :kt]) if (pure and not getattr(trainer, "n_ma", 0)) else L_meas_q
    ext_diag = extension_diagnostics(L_all, A_all, trainer, meas) if extended else {}
    gc = trainer.gate_cols
    cm = curve_quality_metrics(L_gen.reshape(-1, kt)[:, gc], L_ref_q[:, gc], trainer.L_union_np[:, gc], trainer.phys)
    pm = profile_quality_metrics(L_gen, L_ref_q, meas.depth, P_anchor=anchors, L_floor=(L_meas_q if pure else None))
    acc = acceptance_summary(trace.get("qc", {}))
    pm.update(acc)
    prm_c = trace.get("params", {})
    mode_c = measured_anomaly_mode(cfg, meas.survey.anomaly_family if meas.survey is not None else "borehole")
    if "an_K" in prm_c and mode_c == "count":
        # the labelled count of every profile within 0 .. max_anomalies, and every configured count present
        kk_ = np.asarray(prm_c["an_K"]).astype(int)
        if "ma_on" in prm_c:
            kk_ = kk_ + (np.asarray(prm_c["ma_on"]) > 0.5).sum(axis=1)
        want = [k_ for k_, w_ in anomaly_count_law(cfg.profile.anomaly_count_probs, cfg.profile.max_anomalies, mode_c)
                if w_ > 0]
        miss = sorted(set(want) - set(np.unique(kk_).tolist())) if kk_.size >= 10 * max(len(want), 1) else []
        pm["anomaly_count_max"] = int(kk_.max()) if kk_.size else 0
        pm["anomaly_counts_missing"] = [int(v_) for v_ in miss]
        pm["anomaly_count_ok"] = float(bool(kk_.size == 0 or (kk_.max() <= int(cfg.profile.max_anomalies) and not miss)))
    pm["field_admissible_gate"] = float(bool(getattr(trainer, "fam_on", False)))
    qc = qc_gate_status(cm, pm)
    print_qc_table(qc)
    if "departure_depth_texture_ratio_median" in pm:
        print(f"[TEXTURE] network departure from the anchors, relative to the anchors' own texture"
              f"{' (floor: the measured profile)' if pure else ''} (median [p90] over "
              f"profiles): station-scale depth {pm['departure_depth_texture_ratio_median']:.3f} "
              f"[{pm['departure_depth_texture_ratio_p90']:.3f}], high-frequency time "
              f"{pm.get('departure_time_texture_ratio_median', float('nan')):.4f} "
              f"[{pm.get('departure_time_texture_ratio_p90', float('nan')):.4f}] | departure RMS "
              f"{pm.get('anchor_deviation_rms_decades_median', float('nan')):.4f} decades")
    if "qc" in trace:
        q_ = trace["qc"]
        prm = trace.get("params", {})
        if "an_K" in prm:
            ktot = np.asarray(prm["an_K"]).astype(int)
            if "ma_on" in prm:
                nm = (np.asarray(prm["ma_on"]) > 0.5).sum(axis=1)
                ktot = ktot + nm
                print(f"[ANOMALY] profiles with the measured anomalies: {int((nm > 0).sum())} of {nm.size} (pure background: "
                      f"{int((ktot == 0).sum())})")
            kv, kc = np.unique(ktot, return_counts=True)
            print("[ANOMALY] profiles by anomaly count: " + ", ".join(f"{int(a)}: {int(b)}" for a, b in zip(kv, kc)))
        print(f"[ACCEPTANCE] accepted {acc.get('accept_rate', float('nan')):.1%} after up to "
              f"{cfg.generation.qc_max_tries} passes (mean passes {acc.get('mean_tries', float('nan')):.2f}) | "
              f"anomaly retention median {acc.get('anomaly_retention_median', float('nan')):.3f} "
              f"[p10 {acc.get('anomaly_retention_p10', float('nan')):.3f}, p90 {acc.get('anomaly_retention_p90', float('nan')):.3f}] | "
              f"contrast median {acc.get('anomaly_contrast_median', float('nan')):.3f} "
              f"[p10 {acc.get('anomaly_contrast_p10', float('nan')):.3f}, p90 {acc.get('anomaly_contrast_p90', float('nan')):.3f}] | "
              f"shape corr median {acc.get('anomaly_corr_median', float('nan')):.3f} "
              f"[p10 {acc.get('anomaly_corr_p10', float('nan')):.3f}] | distance to measured median "
              f"{acc.get('dist_to_measured_decades_median', float('nan')):.3f} decades")
        if not bool(np.all(q_["pass"])):
            print(f"[ACCEPTANCE] {int((~q_['pass'].astype(bool)).sum())} profiles still fail after the last pass: kept and "
                  f"flagged qc_pass=0 in generated_profile_index.csv")
    # structure fidelity: the measured profile's own narrow features and the per-station median departures
    bg_rep = background_purity_report(L_gen, trace, meas, trainer, kt, anchors=anchors) if pure else None
    struct, struct_tab = structure_fidelity_metrics(L_gen, anchors, L_ref_q, meas.depth, gc, trainer.struct_thr,
                                                    gate_ms=meas.grid_ms[gc])
    print_structure_summary(struct)
    if "med_dev_gen_anchor_mean" in struct_tab.columns:
        v = struct_tab["med_dev_gen_anchor_mean"].to_numpy()
        k = int(np.nanargmax(np.abs(v)))
        struct["max_abs_median_departure_gen_anchor"] = {_axis_names(meas)["key"]: float(meas.depth[k]),
                                                         "decades": float(v[k])}
        print(f"[STRUCTURE]   per-station median departure from the anchors (mean over gates): largest "
              f"{v[k]:+.4f} decades at {meas.depth[k]:.0f} m; median |.| over stations {np.nanmedian(np.abs(v)):.4f}")
    struct_tab.rename(columns={"depth_m": _axis_names(meas)["key"]}).to_csv(
        os.path.join(reports_dir, "structure_by_station.csv"), index=False)
    paths = save_generated_outputs(store, depth_dense, st_index, meas, trace, cfg, cfg.export.generated_data_path, norm,
                                   L_stations=(L_all[:, mc] if extended else L_gen))
    metrics = {"program": f"{RFMG_PROGRAM_NAME} v{RFMG_VERSION}", "ablation_mode": ablation_mode,
               "contract_key": contract["contract_key"], "n_profiles": int(L_gen.shape[0]), "n_stations": int(meas.D),
               "dense_depth_points": int(depth_dense.size), "target_len": int(meas.T), "curve": cm, "profile": pm,
               "qc": qc, "acceptance": acc, "structure": struct, "dense_projection": trace.get("dense_projection", {}),
               "zero_anomaly_trend_projection": trace.get("zero_anomaly_trend_projection", {}),
               "best_validation": trainer.best_quality, "outputs": paths,
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if meas.survey is not None:
        metrics["survey"] = meas.survey.to_dict()
    metrics["manifold"] = manifold_report(trainer, trace, acc)
    if trace.get("qc_split") is not None:
        metrics["qc_split"] = trace["qc_split"]
    if getattr(trainer, "morph_report", None):
        metrics["morphology_bank"] = {k: v for k, v in trainer.morph_report.items() if k != "entries"}
    metrics["decay_constraint"] = meas.report.get("decay_constraint")
    if int(cfg.cv.diversity_max) > 0 and L_gen.shape[0] >= 8:
        try:   # effective diversity / effective count of the generated set (diversity_report.json)
            dv = diversity_report(L_gen, anchors, trainer.gate_cols, trace, trainer, int(cfg.cv.diversity_max),
                                  seed=cfg.runtime.seed + 77, reports_dir=reports_dir)
            metrics["diversity"] = {k: v for k, v in dv.items() if k != "sets"}
            metrics["diversity"]["sets"] = {k: {kk: vv for kk, vv in v.items() if kk != "saturation"}
                                            for k, v in dv["sets"].items()}
        except (MemoryError, np.linalg.LinAlgError) as e_:
            print(f"[DIVERSITY] skipped: {type(e_).__name__}: {e_}")
    metrics["background"] = ({**(meas.background_info or {}), "purity": bg_rep} if pure else
                             {"mode": "measured", "note": "anchors on the measured profile itself (v16 / v17)"})
    if extended:
        metrics["extension"] = ext_diag
        metrics["qc_scope"] = (f"measured coverage: {mc.size} measured of {L_all.shape[1]} stations, {kt} of {meas.T} "
                               f"time nodes (the extensions are checked under 'extension')")
    if cfg.export.save_metrics_json:
        save_json(metrics, os.path.join(reports_dir, "generation_metrics.json"))
    if cfg.export.save_metrics_csv:
        pd.DataFrame([_scalar_items(metrics)]).T.to_csv(os.path.join(reports_dir, "generation_metrics.csv"), header=["value"])
    hist_paths = export_history(trainer.history, reports_dir) if (cfg.export.save_history_csv or cfg.export.save_history_json) else {}
    figs = {}
    if cfg.visualization.mode != "none":
        figs = save_figures(L_all[:, mc] if extended else L_gen, (A_all[:, mc] if A_all is not None else None)
                            if extended else anchors, meas, trainer.L_union_np, trainer.phys, trainer.history, plots_dir,
                            dpi=int(cfg.visualization.dpi), sim=sim, title_tag=f"[{ablation_mode}]")
        try:
            figs.update(save_structure_figure(struct, struct_tab, plots_dir, dpi=int(cfg.visualization.dpi),
                                              title_tag=f"[{ablation_mode}]", axis_label=_axis_names(meas)["label"]))
        except Exception as e:
            print(f"[PLOT] structure figure skipped: {e}")
        if pure:
            try:
                figs.update(save_background_figure(meas, plots_dir, dpi=int(cfg.visualization.dpi)))
            except Exception as e:
                print(f"[PLOT] background figure skipped: {e}")
        try:
            figs.update(save_anomaly_figures(store.log_sample(), depth_dense, meas,
                                             build_profile_index(trace, int(L_gen.shape[0]), meas.survey), plots_dir,
                                             dpi=int(cfg.visualization.dpi)))
        except Exception as e:
            print(f"[PLOT] anomaly figures skipped: {e}")
        if extended:
            try:
                figs.update(save_extension_figure(L_all, trainer, meas, ext_diag, plots_dir, dpi=int(cfg.visualization.dpi),
                                                  title_tag=f"[{ablation_mode}]"))
            except Exception as e:
                print(f"[PLOT] extension figure skipped: {e}")
    np.save(os.path.join(reports_dir, "generated_profiles_log10_sample.npy"), L_gen[: min(64, L_gen.shape[0])].astype(np.float32))
    store.close()
    ck = trainer.save("final", "generation finished") if ablation_mode not in ("cgan", "gan") else None
    if ck and str(cfg.checkpoint.keep) == "final":
        mb = prune_checkpoints(trainer.checkpoint_dir)
        print(f"[CHECKPOINT] --keep_checkpoints final: stage checkpoints removed ({mb:.0f} MB), generation model {ck}")
    print("=" * 88)
    print("[FINAL SUMMARY]")
    print(f"  profiles x points x gates   : {(int(L_gen.shape[0]), int(depth_dense.size), int(meas.T))} "
          f"(measured stations {meas.D} embedded)")
    print(f"  QC status                   : {qc['status']}")
    print(f"  outputs                     : {json.dumps(paths, ensure_ascii=False)}")
    print(f"  reports / plots             : {reports_dir} | {plots_dir}")
    print(f"  figures                     : {len(figs)} files")
    print(f"  final checkpoint            : {ck}")
    print(f"  history                     : {hist_paths.get('csv', '')}")
    print("=" * 88)
    return ablation_mode, cfg


def smoke_main() -> None:
    """End-to-end self-check on tiny settings (real files if found, else synthetic data)."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="rfmg_v16_smoke_")
    root = project_root()
    has_real = any(os.path.isfile(p) for p in [SERVER_REAL_FILE] + [os.path.join(root, "input", n) for n in LOCAL_REAL_NAMES])
    argv = ["--train_action", "restart", "--run_scope", "full", "--ablation", "full", "--survey_mode", "borehole",
            "--stage0_epochs", "1", "--stage0b_epochs", "1", "--stage1_supervisor_epochs", "1", "--stage1_epochs", "1",
            "--stage2_epochs", "2", "--steps_per_epoch", "3", "--generated_profile_count", "8",
            "--validation_profiles", "8", "--sim_reservoir_rows", "256", "--measured_pool_size", "256",
            "--batch_size", "64", "--export_profile_samples", "2", "--model_dir", os.path.join(tmp, "model"),
            "--generated_data_path", os.path.join(tmp, "output"), "--visualization", "all"]
    for i, a in enumerate(sys.argv[1:]):
        if a == "--target_len" and i + 2 <= len(sys.argv[1:]):
            argv += ["--target_len", sys.argv[1:][i + 1]]
    if not has_real:
        argv += ["--data_name", "tem_simulate"]
    report: Dict[str, Any] = {"tmp": tmp, "real_data": has_real}
    try:
        args = build_argument_parser().parse_args(argv)
        mode, cfg = run_session(args)
        out = cfg.export.generated_data_path
        rep = os.path.join(cfg.export.model_dir, "analysis", "reports")
        need = [os.path.join(out, "generated_decay_curves.csv"), os.path.join(out, "generated_profiles.csv"),
                os.path.join(out, "generated_profile_index.csv"), os.path.join(rep, "generation_metrics.json"),
                os.path.join(rep, "anchor_contract.json"), os.path.join(cfg.export.model_dir, "checkpoints", "final.pt")]
        report["files"] = {p: os.path.isfile(p) for p in need}
        with open(os.path.join(rep, "generation_metrics.json"), "r", encoding="utf-8") as f:
            js = json.load(f)
        report["qc_status"] = js["qc"]["status"]
        report["monotone_fraction"] = js["curve"]["monotone_fraction"]
        report["all_passed"] = all(report["files"].values()) and js["curve"]["monotone_fraction"] >= 0.999
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        report["error"] = f"{type(e).__name__}: {e}"
        report["all_passed"] = False
    print("RFMG_GEN_SELF_CHECK_PASS" if report.get("all_passed") else "RFMG_GEN_SELF_CHECK_FAILED")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if not report.get("all_passed"):
        raise SystemExit(2)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    args.real_data_file = sanitize_path(args.real_data_file)
    args.simulata_data_file = sanitize_path(args.simulata_data_file)
    if args.smoke_test:
        smoke_main()
        return
    print(f"{_LOG} {RFMG_PROGRAM_NAME} v{RFMG_VERSION} | python {sys.version.split()[0]} | torch {torch.__version__}")
    mode, cfg = run_session(args)
    if mode != "__report__":
        _RFMG_ABLATION_COMPLETED.append(mode)
    while _RFMG_ABLATION_QUEUE:
        nxt = _RFMG_ABLATION_QUEUE.pop(0)
        print(f"[ABLATION] === batch: starting variant {nxt} ===")
        args.run_scope = "test" if _RFMG_TEST_RUN else "full"
        args.survey_mode = cfg.profile.survey_mode
        if args.train_action == "ask":
            args.train_action = "auto"
        try:
            mode, cfg = run_session(args, ablation_mode=nxt)
            _RFMG_ABLATION_COMPLETED.append(mode)
        except Exception as e:
            print(f"[ABLATION] variant {nxt} failed: {type(e).__name__}: {e} — continuing")
    if len(set(_RFMG_ABLATION_COMPLETED)) >= 2:
        _rfmg_ablation_report(cfg, sorted(set(_RFMG_ABLATION_COMPLETED), key=_RFMG_ABLATION_MODES.index))
