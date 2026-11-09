"""The comparison generators in methods/: each program trains and generates on a small synthetic profile."""
import importlib
import json

import numpy as np
import pytest

METHODS = ("gan_t", "cgan_t", "gan_p", "cgan_p", "ddpm_p", "cddpm_p")
CONDITIONAL = ("cgan_t", "cgan_p", "cddpm_p")
PROFILE_MODELS = ("gan_p", "cgan_p", "ddpm_p", "cddpm_p")


def _argv(example, out, seed=1, iters=2, n=6):
    return ["--profile", example["profile"], "--gates", example["gates"], "--events", example["events"],
            "--iters", str(iters), "--n_profiles", str(n), "--seed", str(seed), "--device", "cpu", "--threads", "2",
            "--out", str(out)]


@pytest.mark.parametrize("method", METHODS)
def test_method_trains_and_generates(method, example, tmp_path):
    common = importlib.import_module("common")
    mod = importlib.import_module(method)
    out = tmp_path / method
    res = common.run_method(method, mod.INFO, mod.train_and_generate, _argv(example, out))
    z, V = common.read_profile_csv(example["profile"])
    raw = np.load(out / "profiles_raw.npz")
    assert raw["L"].shape == (6, z.size, V.shape[1])
    assert np.all(np.isfinite(raw["L"]))
    assert np.allclose(raw["depth_m"], z)
    assert np.allclose(raw["gate_ms"], common.read_gate_ms(example["gates"]))
    assert (out / "layouts.json").is_file() == (method in CONDITIONAL)
    assert (out / "profiles_native.npz").is_file() == (method in PROFILE_MODELS)
    assert (out / "generator.pt").is_file()
    with open(out / "run.json", "r", encoding="utf-8") as fh:
        run = json.load(fh)
    assert run["method"] == method and run["iters"] == 2 and run["profiles"] == 6
    assert len(run["training_log"]) >= 1 and res["generator_parameters"] > 0


def test_conditional_methods_need_events(example, tmp_path):
    common = importlib.import_module("common")
    mod = importlib.import_module("cgan_t")
    argv = _argv(example, tmp_path / "x")
    i = argv.index("--events")
    del argv[i:i + 2]
    with pytest.raises(SystemExit):
        common.run_method("cgan_t", mod.INFO, mod.train_and_generate, argv)
