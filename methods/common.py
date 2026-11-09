"""Shared code of the six comparison generators in this folder (gan_t, cgan_t, gan_p, cgan_p, ddpm_p, cddpm_p).

The profile readers, the depth interpolation, the anomaly-layout sampler, the network blocks, the WGAN-GP gradient
penalty and the command line used by the six programs.  Profiles are handled in log10 amplitude, standardised per gate
with the training profile's statistics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_PROGRAM = os.path.join(os.path.dirname(HERE), "main.py")   # the generator program: its PCHIP is used along depth


# ---- readers, depth interpolation, anomaly layouts, network blocks, gradient penalty ---------------------------------
_RF = None


def rf():
    """main.py as a module (imported once; its __main__ block does not run)."""
    global _RF
    if _RF is None:
        spec = importlib.util.spec_from_file_location("rfmg_main", GEN_PROGRAM)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rfmg_main"] = mod
        spec.loader.exec_module(mod)
        _RF = mod
    return _RF


def read_profile_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float)


def read_gate_ms(path: str) -> np.ndarray:
    g = pd.read_csv(path).iloc[:, 0].to_numpy(float)
    # the program's magnitude rule: 0.5 < t_max <= 2000 -> milliseconds (the Liuyuan file says seconds, holds ms)
    return g * 1e3 if g.max() <= 0.5 else g


def pchip_depth(z_knots: np.ndarray, L: np.ndarray, zq: np.ndarray) -> np.ndarray:
    """Shape-preserving (Fritsch-Carlson) interpolation along depth of (..., D, G) profiles at the depths zq, the
    program's own PCHIP (clamped outside the knots)."""
    m = rf()
    z_knots = np.asarray(z_knots, float)
    L = np.asarray(L, float)
    sh = L.shape
    D, G = sh[-2], sh[-1]
    Y = np.moveaxis(L.reshape(-1, D, G), 1, 2).reshape(-1, D)
    out = m.pchip_eval(z_knots, Y, m.pchip_slopes(z_knots, Y), np.asarray(zq, float))
    return np.moveaxis(out.reshape(-1, G, len(zq)), 1, 2).reshape(sh[:-2] + (len(zq), G))


WIN = 64                                       # training window of the profile baselines (126 m)


def weight_map(events: Sequence[Dict[str, float]], z: np.ndarray) -> np.ndarray:
    w = np.zeros(np.asarray(z).size)
    for e in events:
        w = np.maximum(w, np.exp(-0.5 * ((z - e["centre_m"]) / max(e["sigma_m"], 2.0)) ** 2))
    return w


def draw_layout(rng: np.random.Generator, sigmas: Sequence[float], z0: float, z1: float, k_max: int = 5,
                zmin: float = 50.0) -> List[Dict[str, float]]:
    K = int(rng.integers(0, k_max + 1))
    out: List[Dict[str, float]] = []
    for _ in range(K):
        for _try in range(64):
            s = float(rng.choice(sigmas)) * float(rng.uniform(0.75, 1.5))
            lo = max(z0, zmin) + 2 * s
            if lo >= z1 - 1.0:
                continue
            c = float(rng.uniform(lo, z1 - 1.0))
            if all(abs(c - e["centre_m"]) >= 2 * (s + e["sigma_m"]) for e in out):
                out.append({"centre_m": c, "sigma_m": s})
                break
    return sorted(out, key=lambda e: e["centre_m"])


class MLP(nn.Module):
    def __init__(self, i: int, o: int, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2),
                                 nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class ConvCritic(nn.Module):
    def __init__(self, c_in: int = 31, ch: Sequence[int] = (64, 128, 256), patch: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        c = c_in
        for k in ch:
            layers += [nn.Conv1d(c, k, 5, stride=2, padding=2), nn.LeakyReLU(0.2)]
            c = k
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(c, 1, 3, padding=1)
        self.patch = patch

    def forward(self, x):
        y = self.head(self.body(x))
        return y if self.patch else y.mean(dim=(1, 2))


def gradient_penalty(critic, real, fake):
    a = torch.rand(real.size(0), *([1] * (real.dim() - 1)), device=real.device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(critic(x).sum(), x, create_graph=True)[0]
    return ((g.flatten(1).norm(dim=1) - 1.0) ** 2).mean()


def _tt(x, dev) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def _dnorm(z: np.ndarray, z0: float, z1: float) -> np.ndarray:
    return (np.asarray(z, float) - 0.5 * (z0 + z1)) / max(0.5 * (z1 - z0), 1e-9)


def _windows(Lg, extra, n, rng):
    s = rng.integers(0, Lg.shape[0] - WIN + 1, size=n)
    return np.stack([Lg[a:a + WIN].T for a in s]), np.stack([extra[:, a:a + WIN] for a in s])


def _cond_channels(w, grid_n):
    return np.stack([w, grid_n])


# ---- command line shared by the six programs -------------------------------------------------------------------------
# training iterations of the manuscript's runs: 4000 for the GANs, 5 x 4000 for the diffusion models (one WGAN-GP
# iteration = 5 critic + 1 generator steps)
PAPER_ITERS = {"gan_t": 4000, "cgan_t": 4000, "gan_p": 4000, "cgan_p": 4000, "ddpm_p": 20000, "cddpm_p": 20000}
CONDITIONAL = ("cgan_t", "cgan_p", "cddpm_p")


def read_events(path: str) -> List[Dict[str, float]]:
    """Anomalies of the training profile: a JSON list of {"centre_m": ..., "sigma_m": ...} (sigma 5 m when missing),
    or an object holding that list under "events"."""
    with open(path, "r", encoding="utf-8") as fh:
        ev = json.load(fh)
    if isinstance(ev, dict):
        ev = ev.get("events", [])
    return [{"centre_m": float(e["centre_m"]), "sigma_m": float(e.get("sigma_m", 5.0))} for e in ev]


def build_parser(method: str, info: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{method}.py",
        description=f"{info}. Trained on one borehole profile; generates --n_profiles profiles on its stations.")
    ap.add_argument("--profile", required=True,
                    help="training profile CSV: depth (m) in the first column, one column per gate (amplitude > 0)")
    ap.add_argument("--gates", default="",
                    help="gate-time CSV (first column; seconds when every value <= 0.5, else ms), kept with the output")
    ap.add_argument("--events", default="",
                    help="JSON list of the training profile's anomalies {centre_m, sigma_m}: the condition of the "
                         "conditional methods; their widths set the drawn anomaly layouts")
    ap.add_argument("--stations", default="",
                    help="CSV whose first column holds the output depths (m); default: the training profile's")
    ap.add_argument("--iters", type=int, default=PAPER_ITERS[method],
                    help=f"training iterations (default {PAPER_ITERS[method]}, as in the manuscript)")
    ap.add_argument("--n_profiles", type=int, default=1000, help="profiles to generate (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4") or 4),
                    help="CPU threads of torch (default: OMP_NUM_THREADS, else 4)")
    ap.add_argument("--out", default="", help=f"output folder (default runs/{method})")
    return ap


def prepare(a: argparse.Namespace, method: str) -> Dict[str, Any]:
    """Training profile, output stations, 2 m depth grid and drawn anomaly layouts, set up as in the experiment
    program (one comparison run = one training profile x one seed)."""
    zt, V = read_profile_csv(a.profile)
    if not (np.all(np.isfinite(V)) and np.all(V > 0)):
        raise SystemExit(f"{a.profile}: every amplitude must be finite and > 0 (the methods work in log10)")
    L = np.log10(V)
    ev = read_events(a.events) if a.events else []
    zq = read_profile_csv(a.stations)[0] if a.stations else zt
    z0, z1 = float(zq.min()), float(zq.max())
    grid = np.arange(z0, z1 + 1e-9, 2.0)
    if method not in ("gan_t", "cgan_t") and grid.size < WIN:
        raise SystemExit(f"{method}: the profile spans {z1 - z0:.0f} m; the {WIN}-point training window on the "
                         f"2 m grid needs at least {2 * (WIN - 1)} m")
    if method in CONDITIONAL and not ev:
        raise SystemExit(f"{method}: a conditional model needs the anomalies of the training profile (--events)")
    gate = read_gate_ms(a.gates) if a.gates else np.arange(1.0, L.shape[1] + 1.0)
    rng = np.random.default_rng(a.seed + 100)
    sigmas = [e["sigma_m"] for e in ev] or [4.8, 8.7]
    layouts = [draw_layout(rng, sigmas, z0, z1) for _ in range(a.n_profiles)]
    rank_order = np.argsort(-pchip_depth(zt, L[None, :, :1], zq)[0, :, 0])
    return {"zt": zt, "L": L, "ev": ev, "zq": zq, "z0": z0, "z1": z1, "grid": grid, "gate": gate,
            "gate_unit": "ms" if a.gates else "gate index", "layouts": layouts, "rank_order": rank_order,
            "n": int(a.n_profiles), "iters": int(a.iters), "seed": int(a.seed), "gs": int(a.seed) * 1000 + 7}


def device_of(a: argparse.Namespace) -> torch.device:
    torch.set_num_threads(max(1, int(a.threads)))
    if a.device == "cpu" or (a.device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA device is available")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda:0")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def run(method: str, info: str, train_and_generate: Callable, a: argparse.Namespace) -> Dict[str, Any]:
    """Train, generate and write: profiles_raw.npz (L = log10 amplitude, (profiles, stations, gates), float32;
    depth_m; gate_ms), profiles_native.npz (profile models: the generator's own 2 m grid), layouts.json (conditional
    methods: the drawn anomaly layout of every profile), generator.pt and run.json (settings and training log)."""
    d = prepare(a, method)
    dev = device_of(a)
    out = a.out or os.path.join("runs", method)
    os.makedirs(out, exist_ok=True)
    hist: List[str] = []

    def say(msg: str) -> None:
        hist.append(msg)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    t0 = time.time()
    G, Pm, labs, nat = train_and_generate(d, dev, say)
    np.savez_compressed(os.path.join(out, "profiles_raw.npz"), L=Pm.astype(np.float32), depth_m=d["zq"],
                        gate_ms=d["gate"])
    if nat:
        np.savez_compressed(os.path.join(out, "profiles_native.npz"), L=np.asarray(nat["L"], np.float32),
                            depth_m=nat["depth_m"], gate_ms=d["gate"])
    if labs is not None:
        with open(os.path.join(out, "layouts.json"), "w", encoding="utf-8") as fh:
            json.dump({"layouts": labs}, fh, indent=1, default=_json_default)
    torch.save(G.state_dict(), os.path.join(out, "generator.pt"))
    res = {"method": method, "info": info, "profile": os.path.abspath(a.profile), "events": d["ev"],
           "seed": d["seed"], "profiles": d["n"], "iters": d["iters"], "stations": int(d["zq"].size),
           "gates": int(d["L"].shape[1]), "gate_axis": d["gate_unit"], "device": str(dev),
           "seconds": round(time.time() - t0, 1),
           "generator_parameters": int(sum(p_.numel() for p_ in G.parameters())), "training_log": hist}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=_json_default)
    say(f"[{method}] trained {d['iters']} iterations + generated {d['n']} profiles in {time.time() - t0:.0f} s "
        f"-> {out}")
    return res


def run_method(method: str, info: str, train_and_generate: Callable,
               argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    a = build_parser(method, info).parse_args(argv)
    return run(method, info, train_and_generate, a)


if __name__ == "__main__":
    sys.exit("common.py holds the shared code; run one of gan_t.py, cgan_t.py, gan_p.py, cgan_p.py, ddpm_p.py, "
             "cddpm_p.py")
