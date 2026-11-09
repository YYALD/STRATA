#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""RFMG-Gen 22.2.0: STRATA, labelled borehole transient electromagnetic (TEM) profiles generated from a few measured
profiles under joint decay and along-hole constraints.  PyTorch program with staged adversarial training: main.py is
the entry point, the modules are in the strata package.

Architecture
------------
  * Anchors (strata/data/, strata/morphology.py, strata/trainer.py).  The measured profile is split into a decay
    trend, non-increasing in time and along the hole, and the departures from it (background_decomposition,
    measured_trend_undulations).  A background anchor is the trend with its level, decay rate and along-hole contrast
    changed, under a running minimum over time; 0-5 anomalies drawn under physical priors (a conductor cut by the
    hole, a conductor beside it, a near-hole zone, measured-like shapes from a shape bank of measured, simulated and
    beta-VAE shapes) are added and are the labels (RFMGGenTrainer.compose_anchor, RFMGGenTrainer.anomaly_fields,
    MorphologyVAE).
  * Decay-manifold-constrained feature extraction (strata/models/decay_manifold.py).  An encoder (Embedder:
    Transformer and LSTM layers) maps each decay of the anchor to a latent sequence and a condition extractor
    (TEMConditionExtractor: time, frequency and shape features, InceptionTime and Transformer streams) to a
    condition vector; the decoder (Recovery) ends in a monotone head, so every decoded decay is non-increasing; a
    credibility gate (_FinCMG) moves the condition of a decoded decay far from those of the anchors toward its
    projection on their manifold (a memory bank of anchor conditions), and the decoded effect of the move corrects
    the decay.
  * Spatial-feature-constrained module (strata/models/spatial.py).  A random-feature field along the hole
    (depth_latent_field) and a bidirectional GRU over the stations (ProfileDepthContext) correlate the deformation
    from station to station; the regional head reduces the network's change to a monotone depth warp, a level and a
    contrast of the background (RFMGGenTrainer._warp_background), so the anomalies keep their depths, shapes and
    amplitudes; a critic judges profile windows (ProfilePatchDiscriminator).
  * Generator (strata/models/generator.py: BridgeResidualGenerator, Supervisor): bidirectional LSTMs with FiLM on
    the anchor condition, cross-attention and a residual path on the anchor latent; the supervisor learns the
    latent dynamics.
  * Critics and training (strata/models/critics.py, strata/trainer.py: ConditionalDiscriminator,
    SequencePatchDiscriminator, ProfilePatchDiscriminator, RFMGGenTrainer): critics of latent sequences, single
    decays and profile windows; stages from an autoencoder of single decays to adversarial training; generation
    with an exponential moving average of the generator.
  * Field-admissible manifold and quality control (strata/physics.py, strata/quality.py, strata/export.py:
    FieldAdmissibleManifold, PhysicsDescriptor, RFMGGenTrainer.profile_acceptance, export_qc_split): ten features
    of each background decay are tested against the manifold of the measured decays; purity, warp content,
    roughness, a physics envelope and the retention of every label sort the profiles into accepted, review and
    rejected sets.

Data
----
A measured profile as a CSV matrix (first row = axis name and the values of one axis, first column = the other axis)
or as a station table (STATION, CH1..CHn) with a gate-time file next to it (or --gate_time_file).  Orientation, time
unit and depth order are detected per file (--read_by, --real_time_unit, --sim_time_unit).  A simulated profile of the
hole (--simulata_data_file) adds conductor shapes to the shape bank.

Run
---
python main.py                      # menus
python main.py --train_action restart --run_scope full --ablation full --survey_mode borehole \
    --real_data_file <profile.csv>
python main.py --smoke_test         # self-check of the whole program on synthetic data"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:   # the strata package next to this file, also when another program loads main.py by path
    sys.path.insert(0, _HERE)

from strata import RFMG_PROGRAM_NAME, RFMG_VERSION

# Dependency-light help guard.
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        "usage: main.py [options]\n\n"
        f"{RFMG_PROGRAM_NAME} v{RFMG_VERSION} measured-anchored profile TEM generator.\n"
        "Common options:\n"
        "  --train_action {ask,restart,resume,auto,generate_only}\n"
        "  --run_scope {ask,full,test}      --ablation {ask,full,no_profile,...|comma list}\n"
        "  --survey_mode {ask,auto,borehole,airborne,ground}\n"
        "  --gate_time_file PATH --line_id ID --component_block N --max_stations N   (airborne / ground line files)\n"
        "  --extend_profile_frac F --extend_side {auto,low,high,both} --extend_time_ms {0,MS,depth}\n"
        "  --stage0_epochs N --stage1_epochs N --stage2_epochs N --steps_per_epoch N\n"
        "  --target_len {0,64,128,256}       (0/auto = 256, the original design length)\n"
        "  --generated_data_count N          (curves; profiles = ceil(N / measured depths))\n"
        "  --generated_profile_count N       (overrides the profile count)\n"
        "  --real_data_file PATH  --simulata_data_file PATH\n"
        "  --sim_depth_match {auto,physical,relative}  --sim_level_anchor {gate_median,global_median,none}\n"
        "  --cv_folds K                      (leave-block-out cross-validation along the profile, 0 = off)\n"
        "  --cv_folds_select 1,5  --cv_time_holdout DEC   (fold choice; time-direction fold)\n"
        "  --generation_mode union_guided    --smoke_test\n\n"
        "All v15 flags are still accepted; see the README in the program folder for the full list.\n"
        "Training requires a Python environment with torch, numpy, pandas and matplotlib.")
    raise SystemExit(0)

from strata.session import main

# read from this module by methods/common.py
from strata.data.time_grid import pchip_eval, pchip_slopes


if __name__ == "__main__":
    main()
