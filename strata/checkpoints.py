"""Checkpoint files: listing and progress summaries, the latest and the best checkpoint of a run, pruning after a
finished run."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch


def _format_stage_progress(sp: Dict[str, Any]) -> str:
    try:
        return " ".join(f"{k}:{int(v)}" for k, v in sp.items())
    except Exception:
        return str(sp)


def _format_best_metric(bm: Any) -> str:
    try:
        return " ".join(f"{k}={float(v):.4g}" for k, v in (bm or {}).items())
    except Exception:
        return ""


def _scan_all_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return out
    for fn in os.listdir(checkpoint_dir):
        if not fn.endswith(".pt") or fn == "morphology_bank.pt":
            continue
        fp = os.path.join(checkpoint_dir, fn)
        info = {"path": fp, "tag": fn[:-3], "saved_at": "", "global_epoch": -1, "stage_progress": {},
                "best_metric": {}, "size_mb": os.path.getsize(fp) / 1e6, "mtime": os.path.getmtime(fp)}
        try:
            meta = torch.load(fp, map_location="cpu", weights_only=False).get("meta", {})
            info.update({"saved_at": str(meta.get("saved_at", "")), "global_epoch": int(meta.get("global_epoch", 0)),
                         "stage_progress": meta.get("stage_progress", {}), "best_metric": meta.get("best_metric", {})})
        except Exception:
            pass
        out.append(info)
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    for name in ("latest.pt",):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isfile(p):
            return p
    all_ck = _scan_all_checkpoints(checkpoint_dir)
    return all_ck[0]["path"] if all_ck else None


def prune_checkpoints(checkpoint_dir: str, keep: Tuple[str, ...] = ("final.pt", "morphology_bank.pt")) -> float:
    """--keep_checkpoints final: delete the stage checkpoints of a finished run (best_*, latest, *_end, interrupt)
    and keep the generation model (final.pt) and the morphology bank; returns the MB freed."""
    freed = 0.0
    if not os.path.isdir(checkpoint_dir):
        return freed
    for f in sorted(os.listdir(checkpoint_dir)):
        fp = os.path.join(checkpoint_dir, f)
        if f.endswith(".pt") and f not in keep and os.path.isfile(fp):
            freed += os.path.getsize(fp) / 1e6
            os.remove(fp)
    return freed


def select_best_checkpoint(trainer: "RFMGGenTrainer") -> Optional[str]:
    """Lowest validation score over the generator stages (Stage-1B and Stage-2 use the same fixed-seed
    validation profiles and the same composite score)."""
    cands = []
    for stg in ("stage1B", "stage2"):
        p = os.path.join(trainer.checkpoint_dir, f"best_{stg}.pt")
        if os.path.isfile(p) and stg in trainer.best_stage_metric:
            cands.append((float(trainer.best_stage_metric[stg]), stg, p))
    if not cands:
        return None
    score, stg, path = min(cands)
    print(f"[GENERATION] best validation checkpoint: {stg} (score {score:.4f}; "
          + ", ".join(f"{s_}={v_:.4f}" for v_, s_, _ in sorted(cands)) + f") -> {path}")
    return path
