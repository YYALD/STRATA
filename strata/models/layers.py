"""Building blocks: recurrent calls with a cuDNN fallback, FFT power, positional encoding, causal self-attention,
cross-attention, Transformer blocks, InceptionTime and FiLM."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import _RFMG_ABLATION_FLAGS


_RUNTIME_STATE: Dict[str, Any] = {"cudnn_rnn_off": False, "rfft_ok": None}
_DFT_CACHE: Dict[Tuple[int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _rnn_call(rnn: nn.Module, x: torch.Tensor):
    """Call a recurrent layer; a cuDNN failure is retried once with cuDNN disabled, and cuDNN then stays off for the
    recurrent layers of this run (the native kernels compute the same recurrence)."""
    if _RUNTIME_STATE["cudnn_rnn_off"] and x.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return rnn(x)
    try:
        return rnn(x)
    except RuntimeError as e:
        msg = str(e)
        if not x.is_cuda or "cudnn" not in msg.lower():
            raise
        _RUNTIME_STATE["cudnn_rnn_off"] = True
        print(f"[RUNTIME] cuDNN failure in a recurrent layer ({msg.splitlines()[0][:160]}) -> retried without cuDNN; "
              f"cuDNN stays off for recurrent layers in this run")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.backends.cudnn.flags(enabled=False):
            return rnn(x)


def _rfft_power(s: torch.Tensor) -> torch.Tensor:
    """|rfft(s)|^2 along dim 1 (s float32, last dim = time).  The first failure of torch.fft switches to the exact DFT by
    matrix product (cosine / sine tables, no run-time compilation) for the rest of the run."""
    if _RUNTIME_STATE["rfft_ok"] is not False:
        try:
            spec = torch.fft.rfft(s, dim=1)
            _RUNTIME_STATE["rfft_ok"] = True
            return spec.real ** 2 + spec.imag ** 2
        except RuntimeError as e:
            _RUNTIME_STATE["rfft_ok"] = False
            print(f"[RUNTIME] torch.fft.rfft failed ({str(e).splitlines()[0][:160]}) -> DFT by matrix product from now on")
    n = int(s.shape[1])
    key = (n, str(s.device), str(s.dtype))
    if key not in _DFT_CACHE:
        k = torch.arange(n // 2 + 1, device=s.device, dtype=torch.float64)
        t = torch.arange(n, device=s.device, dtype=torch.float64)
        ang = 2.0 * math.pi * torch.outer(t, k) / n
        _DFT_CACHE[key] = (torch.cos(ang).to(s.dtype), (-torch.sin(ang)).to(s.dtype))
    C, S_ = _DFT_CACHE[key]
    re, im = s @ C, s @ S_
    return re * re + im * im


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 400):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max(d_model, 1)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head self attention over the gate axis."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))


class CrossAttention(nn.Module):
    """Query = generator features, key/value = condition vector."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert q_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, _ = query.shape
        if context.dim() == 2:
            context = context.unsqueeze(1)
        q = self.q_proj(query).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.ffn(self.norm2(x))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        br = max(out_ch // 4, 8)
        self.pre = nn.Conv1d(in_ch, br, kernel_size=1)
        self.b1 = nn.Conv1d(br, br, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(br, br, kernel_size=5, padding=2)
        self.b3 = nn.Conv1d(br, br, kernel_size=7, padding=3)
        self.b4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1), nn.Conv1d(in_ch, br, kernel_size=1))
        self.proj = nn.Conv1d(br * 4, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pre(x)
        y = torch.cat([self.b1(p), self.b2(p), self.b3(p), self.b4(x)], dim=1)
        return self.act(self.bn(self.proj(y)) + self.res(x))


class InceptionTimeLite(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(InceptionBlock1D(in_ch, base_ch), InceptionBlock1D(base_ch, base_ch * 2),
                                 InceptionBlock1D(base_ch * 2, base_ch * 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_ch * 2, out_dim),
                                  nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)))


class FiLM(nn.Module):
    """x' = x * (1 + 0.1 gamma(c)) + beta(c)."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, max(128, feat_dim)), nn.GELU(),
                                 nn.Linear(max(128, feat_dim), feat_dim * 2))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if _RFMG_ABLATION_FLAGS.get("no_film"):
            return x
        gamma, beta = torch.chunk(self.mlp(c), 2, dim=1)
        return x * (1.0 + 0.1 * gamma.unsqueeze(1)) + beta.unsqueeze(1)
