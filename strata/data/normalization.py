"""Adaptive normalisation: log10 min-max for positive decays, symlog min-max otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch


@dataclass
class NormalizationState:
    mode: str
    symlog_scale: float
    min_val: float
    max_val: float
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "symlog_scale": float(self.symlog_scale), "min_val": float(self.min_val),
                "max_val": float(self.max_val), "eps": float(self.eps)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizationState":
        return NormalizationState(mode=str(d["mode"]), symlog_scale=float(d.get("symlog_scale", 1.0)),
                                  min_val=float(d["min_val"]), max_val=float(d["max_val"]),
                                  eps=float(d.get("eps", 1e-6)))


def fit_normalizer_log(L_arrays: Sequence[np.ndarray], positive: bool, margin: float = 0.05,
                       eps: float = 1e-6) -> NormalizationState:
    """Normaliser in the log-amplitude domain the data contract already works in.
    positive -> L = log10 V ('log_minmax'); otherwise L = symlog(V) ('symlog_minmax')."""
    vals = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in L_arrays if a is not None and np.size(a)])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1e-6)
    lo, hi = lo - margin * span, hi + margin * span
    return NormalizationState(mode="log_minmax" if positive else "symlog_minmax", symlog_scale=1.0,
                              min_val=lo, max_val=hi, eps=eps)


def norm_from_L(L: np.ndarray, st: NormalizationState) -> np.ndarray:
    return ((np.asarray(L, dtype=np.float64) - st.min_val) / max(st.max_val - st.min_val, st.eps)).astype(np.float32)


def L_from_norm(y: np.ndarray, st: NormalizationState) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * (st.max_val - st.min_val) + st.min_val


def raw_from_L(L: np.ndarray, st: NormalizationState) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    if st.mode == "log_minmax":
        return np.power(10.0, L)
    return np.sign(L) * float(st.symlog_scale) * np.expm1(np.abs(L))


def torch_L_from_norm(y: torch.Tensor, st: NormalizationState) -> torch.Tensor:
    return y.float() * float(st.max_val - st.min_val) + float(st.min_val)
