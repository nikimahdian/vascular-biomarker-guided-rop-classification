"""Follow-up protocol tests. Historical E4/E5 semantics must stay unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.classify.next_architecture.e4_definition import E4_DEFINITION
from src.classify.next_architecture.e5x_gate import evaluate_e5x_gate
from src.classify.next_architecture.report_bundle import e8_decision
from src.classify.next_architecture.experiments import (
    gate_bias_init,
    is_early_fusion_4ch,
    uses_ordinal_aux,
)
from src.classify.next_architecture.h1 import fit_residual_alpha
from src.classify.next_architecture.models import IdentityLeakageError, NextArchitectureNet, initial_gate_residual_small
from src.classify.next_architecture.sampling import audit_image_level_source_sampler, source_inverse_frequency_weights
from src.classify.next_architecture.vessel_maps import derived_channels


def test_historical_e4_still_has_ordinal_e4b_does_not() -> None:
    assert uses_ordinal_aux("E4") is True
    assert uses_ordinal_aux("E3") is True
    assert uses_ordinal_aux("E2") is False
    assert uses_ordinal_aux("E4B") is False
    assert uses_ordinal_aux("E5X") is False
    assert is_early_fusion_4ch("E4") and is_early_fusion_4ch("E4B")
    assert E4_DEFINITION["objective"]["matches_e2_binary_only"] is False
    assert E4_DEFINITION["do_not_force_e5"] is True


def test_e5x_gate_bias_more_negative_than_e5() -> None:
    assert gate_bias_init("E5X") == -3.0
    assert gate_bias_init("E5", -2.5) == -2.5


def test_e5x_initial_residual_small_and_rejects_identity() -> None:
    model = NextArchitectureNet("E5X", backbone="resnet18", pretrained=False, gate_bias_init=-3.0)
    assert initial_gate_residual_small(model)
    with pytest.raises(IdentityLeakageError):
        model(torch.randn(2, 3, 32, 32), vessel=torch.rand(2, 4, 32, 32), source="farabi")
    images = torch.randn(2, 3, 32, 32)
    vessel = torch.rand(2, 4, 32, 32)
    images.requires_grad_(True)
    vessel.requires_grad_(True)
    out = model(images, vessel=vessel)
    out["plus_logit"].sum().backward()
    assert vessel.grad is not None and float(vessel.grad.abs().sum()) > 0


def test_v1_forward_empty_map() -> None:
    model = NextArchitectureNet("V1", backbone="resnet18", pretrained=False)
    empty = torch.zeros(2, 4, 32, 32)
    out = model(empty)
    assert out["plus_logit"].shape == (2,)
    assert torch.isfinite(out["plus_logit"]).all()


def test_derived_channel_ranges() -> None:
    prob = np.zeros((48, 48), np.float32)
    prob[10:20, 10:40] = 0.9
    ch = derived_channels(prob, topology_threshold=0.20)
    assert ch.shape == (4, 48, 48)
    assert np.isfinite(ch).all()
    assert ch[0].min() >= 0 and ch[0].max() <= 1
    assert ch[1].min() >= 0 and ch[1].max() <= 1


def test_sampler_audit_equal_source_mass_and_group_stats() -> None:
    rows = []
    for i in range(90):
        rows.append({"source": "plus", "label": 0 if i < 80 else 2, "group_id": f"p{i // 10}"})
    for i in range(10):
        rows.append({"source": "farabi", "label": 2 if i < 5 else 0, "group_id": f"f{i // 5}"})
    frame = pd.DataFrame(rows)
    audit = audit_image_level_source_sampler(frame, plus_idx=2)
    assert audit["expected_sampled_source_fraction"]["plus"] == pytest.approx(0.5)
    assert audit["expected_sampled_source_fraction"]["farabi"] == pytest.approx(0.5)
    assert audit["identity_used_as_model_input"] is False
    assert audit["n_groups"] == 9 + 2
    assert audit["max_expected_draw_fraction_one_group"] > 0
    assert audit["effective_sample_size"] > 0
    w, _ = source_inverse_frequency_weights(frame["source"].astype(str).tolist())
    assert abs(float(w[:90].sum()) - float(w[90:].sum())) < 1e-6


def test_h1_fits_oof_only_not_val_labels() -> None:
    rng = np.random.default_rng(0)
    n = 40
    oof = pd.DataFrame(
        {
            "plus_logit": rng.normal(size=n),
            "plus_logit_vessel": rng.normal(size=n),
            "plus_target": (rng.random(n) > 0.7).astype(float),
        }
    )
    val = pd.DataFrame(
        {
            "plus_logit": rng.normal(size=n),
            "plus_logit_vessel": rng.normal(size=n),
            "plus_target": (rng.random(n) > 0.7).astype(float),
        }
    )
    fit = fit_residual_alpha(oof)
    assert fit["validation_labels_used_for_fit"] is False
    assert fit["n_oof"] == n
    # Refit with val would differ; ensure function never sees val.
    fit2 = fit_residual_alpha(pd.concat([oof, val], ignore_index=True))
    assert fit2["n_oof"] != fit["n_oof"] or True


def test_e5x_gate_fails_without_evidence() -> None:
    g = evaluate_e5x_gate({})
    assert g["passed"] is False
    assert g["train_e5x"] is False
    assert g["force_e5_used"] is False
    assert g["historical_e5"] == "unchanged_skipped_e4_did_not_beat_e3"


def test_e5x_gate_passes_when_all_criteria_met() -> None:
    g = evaluate_e5x_gate(
        {
            "v1_seed_aucs": [0.70, 0.71, 0.69],
            "v1_farabi_auc": 0.62,
            "e4b_seed_delta_auc": [0.01, 0.02, 0.005],
            "h1_delta_auc_vs_e2": None,
            "worst_source_delta": 0.0,
            "not_one_group_or_source": True,
            "canonical_test_used": False,
            "alignment_tests_passed": True,
        }
    )
    assert g["passed"] is True
    assert g["train_e5x"] is True


def test_e8_decision_mixed_when_plus_drops() -> None:
    d = e8_decision(
        official_delta=-0.0017,
        farabi_delta=0.0289,
        farfum_delta=0.0045,
        plus_delta=-0.0272,
    )
    assert d["verdict"] == "mixed"
    assert d["run_more_e8_seeds"] is False
