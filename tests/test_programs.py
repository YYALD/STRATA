"""Command lines of the programs and the synthetic example."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

METHODS = ("gan_t", "cgan_t", "gan_p", "cgan_p", "ddpm_p", "cddpm_p")


def _run(*args, timeout=600):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)


def test_generator_help():
    r = _run("main.py", "--help")
    assert r.returncode == 0, r.stderr
    assert "RFMG-Gen v22.2.0" in r.stdout


def test_generator_loaded_by_path(tmp_path):
    """main.py loaded by path from another folder, as methods/common.py loads it."""
    code = ("import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('rfmg_main', sys.argv[1])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "print('missing:', [n for n in sys.argv[2:] if not hasattr(mod, n)])\n")
    names = ("pchip_slopes", "pchip_eval")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code, os.path.join(ROOT, "main.py"), *names], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env)
    assert r.returncode == 0, r.stderr
    assert "missing: []" in r.stdout


@pytest.mark.parametrize("method", METHODS)
def test_method_help(method):
    r = _run(os.path.join("methods", f"{method}.py"), "--help")
    assert r.returncode == 0, r.stderr
    assert "--profile" in r.stdout and "--n_profiles" in r.stdout


def test_example_profile(tmp_path):
    r = _run(os.path.join("examples", "make_synthetic_profile.py"), "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(tmp_path / "profile.csv")
    t = pd.read_csv(tmp_path / "gate_times.csv")["time_ms"].to_numpy()
    V = df.iloc[:, 1:].to_numpy(float)
    assert list(df.columns[:2]) == ["STATION", "CH1"] and V.shape == (60, 30) and t.size == 30
    assert np.all(V > 0) and np.all(np.diff(t) > 0)
    assert np.all(np.diff(np.log10(V), axis=1) <= 0)          # every decay non-increasing in time
    assert (tmp_path / "events.json").is_file()
