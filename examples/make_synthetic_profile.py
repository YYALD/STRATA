#!/usr/bin/env python
"""Synthetic borehole TEM profile for trying the programs without field data.

    python examples/make_synthetic_profile.py                    # -> examples/data/
    python examples/make_synthetic_profile.py --out my_data --stations 81 --gates 24

Writes three files in the layout the programs read:
  profile.csv     STATION (depth, m) and CH1..CHn (amplitude at each gate), one row per station
  gate_times.csv  time_ms, one gate per row (main.py finds a file named *gate*time* next to the profile)
  events.json     centre_m and sigma_m of the anomalies placed in the profile (the condition of the conditional
                  comparison methods)
The decays fall as a power law that steepens with time and weakens with depth; each anomaly is a Gaussian departure
along the hole whose width grows with time and whose time window is set by its time constant.  Every decay is
non-increasing in time.  It is a test signal, not a forward model."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

# anomalies: (centre m, width m, peak departure in decades, time constant ms)
ANOMALIES = ((110.0, 6.0, 0.10, 0.3), (215.0, 9.0, -0.08, 2.0))


def synthetic_profile(n_stations: int = 60, n_gates: int = 30, z0: float = 20.0, z1: float = 300.0,
                      t0_ms: float = 0.02, t1_ms: float = 10.0,
                      anomalies: Sequence[Tuple[float, float, float, float]] = ANOMALIES
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Depths (m), gate times (ms), amplitudes (stations, gates) and the anomaly list."""
    z = np.linspace(z0, z1, n_stations)
    t = np.geomspace(t0_ms, t1_ms, n_gates)
    u = (z - z0) / (z1 - z0)
    lt = np.log10(t / t0_ms)[None, :]                       # decades after the first gate
    L = (4.2 - 0.9 * u)[:, None] - (0.75 + 0.25 * u)[:, None] * lt - (0.35 - 0.10 * u)[:, None] * lt ** 2
    events: List[Dict[str, float]] = []
    for c, w, a, tau in anomalies:
        if not z0 <= c <= z1:
            continue
        width = w * np.sqrt(1.0 + t / tau)                  # footprint widening with time
        window = np.exp(-0.5 * (np.log10(t / tau) / 0.45) ** 2)
        L += a * window[None, :] * np.exp(-0.5 * ((z[:, None] - c) / width[None, :]) ** 2)
        events.append({"centre_m": float(c), "sigma_m": float(w)})
    L = np.minimum.accumulate(L, axis=1)                    # non-increasing in time
    return z, t, 10.0 ** L, events


def write_example(out: str, **kw) -> Dict[str, str]:
    """Writes profile.csv, gate_times.csv and events.json to out; returns their paths."""
    z, t, V, events = synthetic_profile(**kw)
    os.makedirs(out, exist_ok=True)
    paths = {"profile": os.path.join(out, "profile.csv"), "gates": os.path.join(out, "gate_times.csv"),
             "events": os.path.join(out, "events.json")}
    with open(paths["profile"], "w", encoding="utf-8", newline="\n") as fh:
        fh.write("STATION," + ",".join(f"CH{j + 1}" for j in range(t.size)) + "\n")
        for zi, row in zip(z, V):
            fh.write(f"{zi:.6g}," + ",".join(f"{v:.7g}" for v in row) + "\n")
    with open(paths["gates"], "w", encoding="utf-8", newline="\n") as fh:
        fh.write("time_ms\n" + "".join(f"{v:.7g}\n" for v in t))
    with open(paths["events"], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(events, fh, indent=1)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic borehole TEM profile (profile.csv, gate_times.csv, "
                                             "events.json)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--stations", type=int, default=60)
    ap.add_argument("--gates", type=int, default=30)
    ap.add_argument("--top", type=float, default=20.0, help="depth of the first station (m)")
    ap.add_argument("--bottom", type=float, default=300.0, help="depth of the last station (m)")
    a = ap.parse_args()
    paths = write_example(a.out, n_stations=a.stations, n_gates=a.gates, z0=a.top, z1=a.bottom)
    print("written: " + ", ".join(paths.values()))


if __name__ == "__main__":
    main()
