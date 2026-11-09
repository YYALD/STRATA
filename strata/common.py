"""Shared helpers: mixed-precision (AMP) wrappers and the verdict of an epoch under AMP, seeds, JSON and checkpoint
files, digests, the run log, the program and project folders, mixture specifications."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import _LOG


warnings.filterwarnings("ignore")

# initial AMP loss scale of every scaler (torch's default 2^16 overflowed the generator's fp16 gradients on the first
# steps: 4 of 4 skipped at the smoke's 4 steps per epoch, 2^16 -> 2^12)
AMP_INIT_SCALE = 2.0 ** 10
# an epoch without a finite generator step is AMP calibration only while the scale is still >= this
AMP_CALIBRATION_MIN_SCALE = 1.0

# AMP compatibility (torch.cuda.amp.* is deprecated from torch 2.4 on).
try:
    from torch.amp import GradScaler as _TorchGradScaler
    from torch.amp import autocast as _torch_autocast

    def autocast(enabled: bool = True, dtype: Optional[torch.dtype] = None):
        return _torch_autocast("cuda", enabled=bool(enabled), dtype=dtype or torch.float16)

    def GradScaler(enabled: bool = True):
        return _TorchGradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=bool(enabled))
except Exception:  # pragma: no cover - old torch
    from torch.cuda.amp import GradScaler as _CudaGradScaler, autocast  # type: ignore

    def GradScaler(enabled: bool = True):  # type: ignore[no-redef]
        # torch.cuda.amp.GradScaler's first positional parameter is init_scale, not enabled
        return _CudaGradScaler(init_scale=AMP_INIT_SCALE, enabled=bool(enabled))


def generator_epoch_verdict(n_ok: int, n_bad_loss: int, n_bad_grad: int, amp: bool, scale: float) -> str:
    """A Stage-1B / Stage-2 epoch: "ok" (at least one finite generator step), "calibrating" or "stop". On the GPU the
    AMP scaler skips every step whose scaled fp16 gradient overflows and halves its scale; from AMP_INIT_SCALE it
    reaches a usable scale within a few steps, so an epoch shorter than that (e.g. 4 steps) can end without a finite
    step while nothing is wrong: every skip an overflow, no non-finite loss, scale still >=
    AMP_CALIBRATION_MIN_SCALE.  A gradient that is not finite at any scale halves the scale on every step and passes
    1 after 11 steps: "stop", as on the CPU (no scaler) and after a non-finite loss."""
    if n_ok > 0:
        return "ok"
    if amp and n_bad_loss == 0 and n_bad_grad > 0 and float(scale) >= AMP_CALIBRATION_MIN_SCALE:
        return "calibrating"
    return "stop"


# =====================================================================================
# Small utilities
# =====================================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        v = float(value)
        if math.isfinite(v):
            self.sum += v * n
            self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def sanitize_path(path: Optional[str]) -> str:
    if path is None:
        return ""
    p = str(path).strip().strip('"').strip("'")
    if not p:
        return ""
    return os.path.normpath(os.path.expanduser(p))


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def save_json(payload: Any, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return path


def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def file_sha256(path: str, limit_bytes: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit_bytes > 0:
            h.update(f.read(limit_bytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def array_digest(a: np.ndarray, decimals: int = 9) -> str:
    arr = np.round(np.asarray(a, dtype=np.float64), decimals)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def count_parameters(model: Optional[nn.Module]) -> Dict[str, int]:
    if model is None:
        return {"total": 0, "trainable": 0}
    return {"total": int(sum(p.numel() for p in model.parameters())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def set_module_trainable(module: Optional[nn.Module], flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(bool(flag))


class _TeeStream:
    """Duplicate stdout/stderr into a UTF-8 run log (console behaviour unchanged)."""

    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def __getattr__(self, item):
        return getattr(self._stream, item)


_RFMG_LOG_HANDLE = None


def start_run_log(log_dir: str, tag: str = "run") -> str:
    """Tee the console into <log_dir>/<tag>_<timestamp>.log (idempotent per process)."""
    global _RFMG_LOG_HANDLE
    if _RFMG_LOG_HANDLE is not None:
        return getattr(_RFMG_LOG_HANDLE, "name", "")
    ensure_dir(log_dir)
    path = os.path.join(log_dir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = open(path, "a", encoding="utf-8")
    _RFMG_LOG_HANDLE = fh
    sys.stdout = _TeeStream(sys.stdout, fh)
    sys.stderr = _TeeStream(sys.stderr, fh)
    print(f"{_LOG} run log -> {path}")
    return path


def program_dir() -> str:
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the folder of main.py
    except Exception:
        return os.getcwd()


def project_root() -> str:
    """Folder that owns input/ (the program folder's parent for <project>/<program folder>/main.py)."""
    here = program_dir()
    for cand in (os.path.dirname(here), here, os.getcwd(), os.path.dirname(os.getcwd())):
        if cand and os.path.isdir(os.path.join(cand, "input")):
            return cand
    return os.path.dirname(here) or here


def _drive_available(path: str) -> bool:
    """True when the path's drive / top folder exists (E:\\ on the server, maybe not locally)."""
    p = sanitize_path(path)
    if not p:
        return False
    drive, _ = os.path.splitdrive(p)
    if drive:
        return os.path.isdir(drive + os.sep)
    top = p.split(os.sep)[1] if p.startswith(os.sep) and len(p.split(os.sep)) > 1 else ""
    return (not top) or os.path.isdir(os.sep + top)


def _parse_mix(spec: str, default: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for tok in (spec or default).split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.rpartition(":")
        try:
            out.append((k.strip(), float(v)))
        except Exception:
            continue
    tot = sum(v for _, v in out) or 1.0
    return [(k, v / tot) for k, v in out]
