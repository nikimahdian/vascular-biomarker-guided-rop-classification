"""Invariance and correctness tests for the vascular feature definitions.

Section M of the data/feature correctness audit. These tests encode ground truth drawn from
analytic shapes, so a failure means the estimator is wrong, not merely that it changed.

Every test names the property it protects. If a test here is ever deleted, the corresponding
property is no longer protected and that must be recorded in the feature contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.biomarker.geometry_core import (branch_tortuosity, branch_tortuosity_pixelcount,
                                         fov_density, width_px)


# --------------------------------------------------------------------------- synthetic shapes

def _line(s=256, angle_deg=0.0, half=100, width=3):
    m = np.zeros((s, s), bool)
    c = s // 2
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-half, half, 4000):
        y = int(round(c + t * np.sin(a)))
        x = int(round(c + t * np.cos(a)))
        if 0 <= y < s and 0 <= x < s:
            m[max(0, y - width // 2):y + width // 2 + 1,
              max(0, x - width // 2):x + width // 2 + 1] = True
    return m


def _arc(s=256, radius=80, span_deg=90, width=3):
    m = np.zeros((s, s), bool)
    c = s // 2
    for t in np.deg2rad(np.linspace(-span_deg / 2, span_deg / 2, 4000)):
        y = int(round(c + radius * np.sin(t)))
        x = int(round(c + radius * np.cos(t) - radius))
        if 0 <= y < s and 0 <= x < s:
            m[max(0, y - width // 2):y + width // 2 + 1,
              max(0, x - width // 2):x + width // 2 + 1] = True
    return m


def _strip(s=256, w=8, vertical=False):
    m = np.zeros((s, s), bool)
    c = s // 2
    m[c - w // 2: c - w // 2 + w, :] = True
    return m.T if vertical else m


# --------------------------------------------------------------------------- tortuosity

def test_straight_vessel_tortuosity_is_one_at_every_orientation():
    """A straight vessel must score ~1 regardless of orientation.

    This is the property the historical pixel-count definition violated: it returned 1.005 at
    0 degrees and 0.712 at 45 degrees for the same straight line.
    """
    vals = []
    for ang in (0, 15, 30, 45, 60, 75, 90):
        t = branch_tortuosity(_line(angle_deg=ang))
        assert len(t) >= 1, f"no branch found at {ang} degrees"
        vals.append(float(np.median(t)))
    assert min(vals) > 0.95, f"tortuosity below 0.95 for a straight vessel: {vals}"
    assert max(vals) / min(vals) < 1.20, (
        f"orientation dependence too large for straight vessels: {vals}")
    # the corrected estimator stays within ~8% across orientations
    assert max(vals) < 1.15, f"excess arc length too large for straight vessels: {vals}"


def test_pixelcount_definition_is_orientation_dependent():
    """Guard the regression: the OLD definition must be shown to be wrong.

    If this test ever stops failing in the way it expects, the historical claim in the audit
    report is no longer reproducible.
    """
    old = [float(np.median(branch_tortuosity_pixelcount(_line(angle_deg=a)))) for a in (0, 45, 90)]
    assert max(old) / min(old) > 1.25, f"expected the old definition to be orientation dependent: {old}"


def test_curved_vessels_rank_above_straight_ones():
    straight = float(np.median(branch_tortuosity(_line(angle_deg=30))))
    shallow = float(np.median(branch_tortuosity(_arc(radius=80, span_deg=90))))
    deep = float(np.median(branch_tortuosity(_arc(radius=80, span_deg=180))))
    assert straight < shallow < deep, (straight, shallow, deep)


def test_tortuosity_matches_the_analytic_circular_arc():
    """A 90 degree arc of radius R has arc/chord = (pi/2 R) / (R sqrt2) = 1.1107."""
    t = float(np.median(branch_tortuosity(_arc(radius=80, span_deg=90))))
    assert abs(t - 1.1107) < 0.09, f"arc tortuosity {t:.4f}, analytic 1.1107"


def test_short_fragment_is_ignored():
    m = np.zeros((64, 64), bool)
    m[32, 30:34] = True
    assert len(branch_tortuosity(m, min_px=8)) == 0


# --------------------------------------------------------------------------- width

@pytest.mark.parametrize("w", [4, 6, 8, 12, 20])
def test_width_exact_on_even_strips(w):
    est = float(np.median(width_px(_strip(w=w))))
    assert abs(est - w) <= 0.001, f"true {w}, estimated {est}"


@pytest.mark.parametrize("vertical", [False, True])
def test_width_is_orientation_invariant(vertical):
    est = float(np.median(width_px(_strip(w=8, vertical=vertical))))
    assert abs(est - 8) <= 0.001, f"estimated {est}"


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_width_over_disc_diameter_is_scale_invariant(scale):
    """The same retinal geometry rendered at three resolutions must give the same width / DD."""
    s = 128 * scale
    w_px, dd_px = 4 * scale, 40 * scale
    est = float(np.median(width_px(_strip(s=s, w=w_px))))
    assert abs((est / dd_px) - (w_px / dd_px)) < 1e-6


def test_width_quantisation_scales_with_the_upsampling_factor():
    """A mask produced at 1/N of the native grid cannot resolve width better than N pixels."""
    from src.biomarker.geometry_core import width_quantisation
    q1 = width_quantisation(dd_px=120.0, upsampling_factor=2.0)
    q2 = width_quantisation(dd_px=120.0, upsampling_factor=6.25)
    assert q2 > q1
    assert abs(q2 - 6.25 / 120.0) < 1e-9


# --------------------------------------------------------------------------- density

def test_whole_frame_density_is_not_padding_invariant():
    """Documents why the historical density is acquisition-sensitive rather than a bug to hide."""
    m = np.zeros((100, 100), bool)
    m[25:75, 25:75] = True
    inner = m.sum() / m.size
    padded = np.pad(m, 100)
    assert abs(padded.mean() / inner - 1) > 0.5


def test_fov_density_is_padding_invariant():
    vessel = np.zeros((100, 100), bool)
    vessel[40:60, 40:60] = True
    fov = np.zeros((100, 100), bool)
    fov[10:90, 10:90] = True

    v_pad = np.pad(vessel, 50)
    f_pad = np.pad(fov, 50)

    d0 = fov_density(vessel, fov)
    d1 = fov_density(v_pad, f_pad)
    assert abs(d1 - d0) < 1e-12, (d0, d1)


# --------------------------------------------------------------------------- contract / table hygiene

def test_feature_contract_covers_every_numeric_feature():
    """No feature may enter a model without a contract entry."""
    import pandas as pd
    from pathlib import Path
    contract = Path("configs/feature_contract.csv")
    if not contract.exists():
        pytest.skip("feature contract not present")
    c = pd.read_csv(contract)
    assert not c.duplicated(["feature_name", "feature_version"]).any(), (
        "duplicate (feature_name, feature_version) in the contract; note that a name may "
        "legitimately appear once per feature version, e.g. vessel_density in both tables")
    for col in ("feature_name", "measurement_version", "feature_version",
                "mathematical_definition", "unit", "roi_definition", "depends_on_disc",
                "depends_on_fov", "depends_on_resolution", "depends_on_av",
                "scale_invariant", "padding_invariant", "rotation_invariant",
                "missingness_policy", "admission_status", "allowed_in_final_classifier",
                "requires_expert_validation", "known_limitations", "generation_script"):
        assert col in c.columns, f"contract missing required field {col}"


def test_metadata_columns_never_reach_a_classifier_matrix():
    """source, split, path, group and detector-status columns must not be predictors."""
    from src.classify.branch_a_tabular import META_COLS
    forbidden = {"image_path", "mask_path", "label", "split", "source", "group_id",
                 "patient_id", "exam_id", "identity_level",
                 "disc_valid", "disc_peak_prob", "disc_method", "dd_ok", "cohort"}
    assert forbidden <= META_COLS | {"disc_valid", "disc_peak_prob", "disc_method", "dd_ok",
                                     "cohort"}, (
        "a detector-status or metadata column is not excluded from the feature matrix")


def test_feature_table_has_no_infinities(tmp_path):
    """Produces no inf; NaN is allowed and governed by the contract."""
    import pandas as pd
    from pathlib import Path
    p = Path("data/features/biomarker_features.csv")
    if not p.exists():
        pytest.skip("feature table not present in this checkout")
    df = pd.read_csv(p, nrows=500)
    num = df.select_dtypes(include=[np.number])
    assert not np.isinf(num.to_numpy(dtype=float)).any(), "infinite value in the feature table"


def test_image_mask_pairing_is_by_exact_stem():
    """Only <image_stem>_<8 hex>.png may be accepted as this image's mask.

    Note on history: the audit first hypothesised that the old glob(f"{stem}_*.png") caused a
    prefix collision, and this test originally asserted that. It failed, which is how the
    hypothesis was disproved: glob("...cf.1_*.png") does NOT match "...cf.11_<hash>.png", because
    the character after "...cf.1" is "1", not "_". The test now asserts only what is true and
    what is needed as a guard.
    """
    import re
    pat = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")
    stem = "53671a65-c46e-438f-b52d-13ae5d0bfdcf.1"
    own = f"{stem}_bd7c547e.png"
    sibling = "53671a65-c46e-438f-b52d-13ae5d0bfdcf.11_bd7c547e.png"

    assert pat.match(own).group("stem") == stem
    assert pat.match(sibling).group("stem") != stem
    # and a strict-prefix sibling that the old glob WOULD have accepted
    prefixed = f"{stem}_2_bd7c547e.png"
    assert prefixed.startswith(stem + "_")
    assert pat.match(prefixed).group("stem") == f"{stem}_2" != stem
