"""CDDPM-P, conditional profile diffusion: the DDPM-P model of ddpm_p.py conditioned on the anomaly-weight map and depth
(as CGAN-P).  Whole profiles are generated from drawn anomaly layouts, which are also their labels.

    python methods/cddpm_p.py --profile examples/data/profile.csv --events examples/data/events.json --out runs/cddpm_p
"""
from __future__ import annotations

from common import run_method
from ddpm_p import gen_ddpm_p, train_ddpm_p


INFO = "CDDPM-P (profile diffusion conditioned on the anomaly-weight map and depth)"


def train_and_generate(d, dev, say):
    """Trained on depth windows of the training profile and its anomaly-weight map; one whole profile per drawn
    layout."""
    nat = {}
    G, st = train_ddpm_p(d["L"], d["zt"], d["grid"], d["iters"], d["seed"], dev, say, events=d["ev"], z0=d["z0"],
                         z1=d["z1"])
    Pm = gen_ddpm_p(G, st, d["grid"], d["zq"], d["n"], d["gs"], dev, layouts=d["layouts"], z0=d["z0"], z1=d["z1"],
                    native=nat)
    return G, Pm, d["layouts"], nat


if __name__ == "__main__":
    run_method("cddpm_p", INFO, train_and_generate)
