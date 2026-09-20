"""CLINICAL_MEASUREMENT_V1 measurement-correctness tests.

Covers the invariances the frozen definitions claim, the guards that must produce NaN rather
than a biased number, and the separation between the measured table and the classifier matrix.
No test uses a label, an AUC or any predictive quantity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi
from skimage.morphology import disk, skeletonize

from src.biomarker.clinical_measurement_v1 import (
    FEATURE_COLUMNS, MIN_ROI_COVERAGE, _topology, measure, retinal_fov,
)
from src.biomarker.geometry_core import (
    branch_tortuosity, branch_tortuosity_pixelcount, branch_tortuosity_smoothed, fov_density,
    width_px,
)
from src.utils.common import ROOT

MEAS_YAML = ROOT / "configs" / "clinical_measurement_v1.yaml"
CLF_YAML = ROOT / "configs" / "clinical_measurement_v1_classifier_features.yaml"
CONTRACT = ROOT / "configs" / "feature_contract.csv"
TABLE = ROOT / "data" / "features" / "clinical_measurement_v1.csv"

META_AND_QC = {
    "image_path", "mask_path", "label", "split", "source", "group_id", "patient_id",
    "exam_id", "identity_level", "measurement_version", "work_resolution",
    "disc_valid", "disc_peak_prob", "disc_method", "dd_px_work", "dd_over_min_side",
    "disc_cx_frac", "disc_cy_frac", "disc_centre_offset_frac",
    "fov_valid", "fov_coverage_fraction", "fov_n_components", "fov_border_contact",
    "fov_centroid_offset", "fov_failure_reason", "roi_coverage_ring_0_2dd",
    "roi_coverage_ring_2_3dd", "roi_coverage_ring_3_6dd", "roi_coverage_annulus",
    "roi_coverage_pole", "n_skel_px_disc",
}


def rasterise(shape, points, width):
    m = np.zeros(shape, bool)
    for y, x in points:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            m[iy, ix] = True
    return ndi.binary_dilation(m, disk(int(width / 2))) if m.any() else m


def straight(shape, angle_deg, width, length=None):
    h, w = shape
    L = length or int(0.7 * min(h, w))
    t = np.linspace(-L / 2, L / 2, 4 * L)
    a = math.radians(angle_deg)
    return rasterise(shape, list(zip(h / 2 + t * math.sin(a), w / 2 + t * math.cos(a))), width)


def arc_mask(shape, radius, span_deg, width):
    h, w = shape
    t = np.radians(np.linspace(-span_deg / 2, span_deg / 2, int(6 * abs(span_deg)) + 8))
    return rasterise(shape, list(zip(h / 2 + radius * np.sin(t), w / 2 + radius * np.cos(t))),
                     width)


def median_tort(mask):
    v = branch_tortuosity_smoothed(mask)
    return float(np.median(v)) if len(v) else float("nan")


# ------------------------------------------------------------------ tortuosity
@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 75, 90])
def test_straight_tortuosity_is_one_at_every_orientation(angle):
    assert median_tort(straight((420, 420), angle, 9)) == pytest.approx(1.0, abs=0.01)


def test_straight_line_orientation_dispersion_is_very_small():
    vals = [median_tort(straight((420, 420), a, 9)) for a in (0, 15, 30, 45, 60, 75, 90)]
    assert max(vals) - min(vals) < 0.01


def test_pixelcount_definition_is_orientation_dependent():
    """The rejected historical definition must keep failing, so the difference stays visible."""
    vals = [float(np.median(branch_tortuosity_pixelcount(straight((420, 420), a, 9))))
            for a in (0, 45)]
    assert max(vals) - min(vals) > 0.2


def test_curved_vessel_ranks_above_straight():
    assert median_tort(arc_mask((420, 420), 90, 120, 9)) > median_tort(straight((420, 420), 45, 9))


@pytest.mark.parametrize("span,expected", [(30, 1.0115), (60, 1.0472), (90, 1.1107), (180, 1.5708)])
def test_tortuosity_matches_the_analytic_circular_arc(span, expected):
    assert median_tort(arc_mask((420, 420), 90, span, 9)) == pytest.approx(expected, abs=0.02)


def test_tortuosity_never_below_one():
    for mask in (straight((420, 420), 20, 9), arc_mask((420, 420), 80, 90, 9)):
        v = branch_tortuosity_smoothed(mask)
        assert len(v) and (v >= 1 - 1e-6).all()


def test_short_fragments_are_excluded_not_scored():
    frag = rasterise((120, 120), [(60, 60 + i) for i in range(5)], 5)
    assert len(branch_tortuosity_smoothed(frag)) == 0
    assert len(branch_tortuosity(frag)) == 0


# ------------------------------------------------------------------ width
def test_raw_width_is_close_to_truth_across_widths():
    for w in (9, 15, 21, 31):
        m = np.zeros((60, 300), bool)
        m[30 - w // 2:30 - w // 2 + w, 50:250] = True
        assert float(np.median(width_px(m))) == pytest.approx(w, abs=1.0)


def test_width_dd_is_scale_invariant_when_vessel_and_disc_scale_together():
    ratios = []
    for sc, w, dd in ((1, 9, 40), (2, 18, 80), (3, 36, 160)):
        m = np.zeros((40 * sc + 60, 400), bool)
        top = m.shape[0] // 2 - w // 2
        m[top:top + w, 50:350] = True
        ratios.append(float(np.median(width_px(m))) / dd)
    assert max(ratios) - min(ratios) < 0.03


def test_width_orientation_dispersion_is_bounded():
    vals = [float(np.median(width_px(straight((420, 420), a, 15))))
            for a in (0, 30, 45, 60, 90)]
    assert max(vals) - min(vals) < 3.0


# ------------------------------------------------------------------ density
def test_whole_frame_density_is_not_padding_invariant():
    m = straight((420, 420), 30, 9)
    base = fov_density(m, None)
    padded = np.zeros((m.shape[0] + 400, m.shape[1] + 400), bool)
    padded[200:-200, 200:-200] = m
    assert abs(fov_density(padded, None) - base) / base > 0.5


def test_fov_normalised_density_is_exactly_padding_invariant():
    m = straight((420, 420), 30, 9)
    fov = np.ones_like(m)
    base = fov_density(m, fov)
    for pad in (50, 100, 200, 300):
        mp = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), bool)
        mp[pad:-pad, pad:-pad] = m
        fp = np.zeros_like(mp)
        fp[pad:-pad, pad:-pad] = True
        assert fov_density(mp, fp) == pytest.approx(base, rel=1e-12)


def test_rectangular_padding_is_also_invariant_for_fov_density():
    m = straight((420, 420), 30, 9)
    base = fov_density(m, np.ones_like(m))
    mr = np.zeros((420, 420 + 400), bool)
    mr[:, 200:-200] = m
    fr = np.zeros_like(mr)
    fr[:, 200:-200] = True
    assert fov_density(mr, fr) == pytest.approx(base, rel=1e-12)


# ------------------------------------------------------------------ FOV
def test_fov_recovers_a_synthetic_circular_retina():
    h = w = 400
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    disc = (r < 0.45 * min(h, w)).astype(np.float32)
    green = disc * 0.6
    fov, qc = retinal_fov(green)
    assert qc["fov_valid"] is True
    assert fov.mean() == pytest.approx(disc.mean(), abs=0.01)


def test_fov_reports_a_reason_when_it_fails():
    _, qc = retinal_fov(np.zeros((64, 64), np.float32))
    assert qc["fov_valid"] is False
    assert qc["fov_failure_reason"]


# ------------------------------------------------------------------ guards
def _blank_rgb(h=200, w=200, level=0.5):
    a = np.full((h, w, 3), level, np.float32)
    return a


def test_disc_invalid_makes_every_disc_dependent_feature_nan():
    m = straight((200, 200), 30, 5)
    out = measure(_blank_rgb(), m.astype(np.uint8), {}, with_fractal=False)
    assert out["disc_valid"] == 0
    for c in ("width_p50_dd", "width_p90_dd", "width_mean_dd", "dd_px_work",
              "vessel_density_fov_ring_2_3dd", "frame_sector_ne_density"):
        assert not np.isfinite(out[c]) or out[c] != out[c], c


def test_disc_valid_is_required_not_assumed():
    m = straight((200, 200), 30, 5)
    good = {"disc_cx": 30.0, "disc_cy": 30.0, "disc_dd_px": 20.0, "peak_prob": 0.99}
    out = measure(_blank_rgb(), m.astype(np.uint8), good, with_fractal=False)
    assert out["disc_valid"] == 1
    assert np.isfinite(out["width_p90_dd"])


def test_low_detector_confidence_invalidates_the_disc():
    m = straight((200, 200), 30, 5)
    bad = {"disc_cx": 30.0, "disc_cy": 30.0, "disc_dd_px": 20.0, "peak_prob": 0.5}
    assert measure(_blank_rgb(), m.astype(np.uint8), bad, with_fractal=False)["disc_valid"] == 0


def test_implausible_disc_diameter_invalidates_the_disc():
    m = straight((200, 200), 30, 5)
    tiny = {"disc_cx": 100.0, "disc_cy": 100.0, "disc_dd_px": 200.0, "peak_prob": 0.99}
    assert measure(_blank_rgb(), m.astype(np.uint8), tiny, with_fractal=False)["disc_valid"] == 0


def test_roi_density_is_nan_when_coverage_is_below_the_minimum():
    """A truncated FOV must produce NaN, never a silently biased density."""
    m = straight((400, 400), 30, 9)
    disc = {"disc_cx": 200.0, "disc_cy": 200.0, "disc_dd_px": 40.0, "peak_prob": 0.99}
    rgb = _blank_rgb(400, 400)
    # FOV covering only a corner: the 2-3 DD ring is almost entirely outside it
    rgb[300:, 300:, :] = 0.9
    rgb[:300, :300, :] = 0.0
    out = measure(rgb, m.astype(np.uint8), disc, with_fractal=False)
    assert out["roi_coverage_ring_2_3dd"] < MIN_ROI_COVERAGE
    assert out["vessel_density_fov_ring_2_3dd"] != out["vessel_density_fov_ring_2_3dd"]


def test_roi_coverage_is_reported_for_every_region():
    m = straight((300, 300), 30, 9)
    disc = {"disc_cx": 150.0, "disc_cy": 150.0, "disc_dd_px": 30.0, "peak_prob": 0.99}
    out = measure(_blank_rgb(300, 300), m.astype(np.uint8), disc, with_fractal=False)
    for c in ("roi_coverage_ring_0_2dd", "roi_coverage_ring_2_3dd",
              "roi_coverage_ring_3_6dd", "roi_coverage_annulus", "roi_coverage_pole"):
        assert 0.0 <= out[c] <= 1.0


def test_roi_coordinates_scale_with_the_grid():
    """Doubling the frame must double the disc coordinates used, not keep them fixed."""
    m1 = straight((200, 200), 30, 5)
    m2 = straight((400, 400), 30, 10)
    d1 = {"disc_cx": 60.0, "disc_cy": 60.0, "disc_dd_px": 20.0, "peak_prob": 0.99}
    d2 = {"disc_cx": 120.0, "disc_cy": 120.0, "disc_dd_px": 40.0, "peak_prob": 0.99}
    o1 = measure(_blank_rgb(200, 200), m1.astype(np.uint8), d1, with_fractal=False)
    o2 = measure(_blank_rgb(400, 400), m2.astype(np.uint8), d2, with_fractal=False)
    assert o1["disc_cx_frac"] == pytest.approx(o2["disc_cx_frac"], abs=1e-9)
    assert o1["dd_over_min_side"] == pytest.approx(o2["dd_over_min_side"], abs=1e-9)


def test_topology_counts_scale_with_resolution():
    counts = []
    for sc in (1, 2, 4):
        m = np.zeros((200 * sc, 200 * sc), bool)
        m |= straight((200 * sc, 200 * sc), 20, 3 * sc)
        m |= straight((200 * sc, 200 * sc), 70, 5 * sc)
        t = _topology(skeletonize(m))
        counts.append((t["n_intersections"], float(m.sum())))
    assert counts[2][1] / counts[0][1] > 10           # area is scale dependent
    assert counts[2][0] != counts[0][0]               # intersections are too


# ------------------------------------------------------------------ contracts
def test_measurement_contract_declares_every_emitted_feature():
    yml = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    declared = set(yml["features"])
    emitted = set(FEATURE_COLUMNS)
    assert emitted <= declared, f"undeclared features: {sorted(emitted - declared)}"


def test_every_emitted_feature_has_an_admission_status():
    yml = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    allowed = {"PRIMARY_ALLOWED", "SECONDARY_ALLOWED", "EXPLORATORY_ONLY",
               "FORBIDDEN_FROM_FINAL_CLASSIFIER"}
    for name in FEATURE_COLUMNS:
        assert yml["features"][name]["admission"] in allowed, name


def test_av_features_are_exploratory_only():
    yml = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    assert yml["av_status"]["status"] == "EXPLORATORY_ONLY"
    for name in ("a_frac", "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90"):
        assert yml["features"][name]["admission"] == "EXPLORATORY_ONLY"


def test_no_fallback_disc_rule_is_declared():
    yml = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    assert yml["disc_validity"]["fallback"] == "NONE"


def test_classifier_matrix_contains_only_allowed_features():
    clf = yaml.safe_load(CLF_YAML.read_text(encoding="utf-8"))
    meas = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    kept = [f["name"] for f in clf["primary_allowed"]]
    kept += [f["name"] for f in clf["secondary_allowed_approved"]]
    for name in kept:
        adm = meas["features"][name]["admission"]
        assert adm in ("PRIMARY_ALLOWED", "SECONDARY_ALLOWED"), (name, adm)


def test_classifier_matrix_excludes_metadata_and_qc():
    clf = yaml.safe_load(CLF_YAML.read_text(encoding="utf-8"))
    kept = {f["name"] for f in clf["primary_allowed"]}
    kept |= {f["name"] for f in clf["secondary_allowed_approved"]}
    assert not (kept & META_AND_QC), sorted(kept & META_AND_QC)
    assert not (kept & set(meas_feature_names())), "exploratory/forbidden leaked"


def meas_feature_names():
    """Feature names the contract marks EXPLORATORY_ONLY or FORBIDDEN."""
    yml = yaml.safe_load(MEAS_YAML.read_text(encoding="utf-8"))
    return [n for n, s in yml["features"].items()
            if s["admission"] in ("EXPLORATORY_ONLY", "FORBIDDEN_FROM_FINAL_CLASSIFIER")]


def test_no_missingness_indicator_is_allowed_in_the_classifier():
    clf = yaml.safe_load(CLF_YAML.read_text(encoding="utf-8"))
    kept = {f["name"] for f in clf["primary_allowed"]}
    kept |= {f["name"] for f in clf["secondary_allowed_approved"]}
    assert not [n for n in kept if n.startswith("is_missing")]


# ------------------------------------------------------------------ table (skipped without data)
def _table():
    if not TABLE.exists():
        pytest.skip("clinical_measurement_v1.csv not present (git-ignored data directory)")
    import pandas as pd

    return pd.read_csv(TABLE)


def test_table_has_no_infinities():
    df = _table()
    num = df.select_dtypes("number")
    assert not np.isinf(num.to_numpy(dtype=float)).any()


def test_table_has_the_canonical_row_count():
    df = _table()
    assert len(df) == 8870
    assert df.image_path.nunique() == 8870


def test_table_carries_the_measurement_version():
    df = _table()
    assert set(df.measurement_version.unique()) == {"CLINICAL_MEASUREMENT_V1"}


def test_feature_contract_has_one_row_per_measured_feature():
    import pandas as pd

    if not CONTRACT.exists():
        pytest.skip("feature_contract.csv not present")
    c = pd.read_csv(CONTRACT)
    v1 = c[c.measurement_version == "CLINICAL_MEASUREMENT_V1"]
    assert set(FEATURE_COLUMNS) <= set(v1.feature_name), \
        sorted(set(FEATURE_COLUMNS) - set(v1.feature_name))


def test_feature_contract_has_the_required_columns():
    import pandas as pd

    if not CONTRACT.exists():
        pytest.skip("feature_contract.csv not present")
    required = {
        "feature_name", "measurement_version", "mathematical_definition", "unit",
        "roi_definition", "depends_on_disc", "depends_on_fov", "depends_on_resolution",
        "depends_on_av", "scale_invariant", "padding_invariant", "rotation_invariant",
        "missingness_policy", "admission_status", "allowed_in_final_classifier",
        "requires_expert_validation", "known_limitations",
    }
    cols = set(pd.read_csv(CONTRACT).columns)
    assert required <= cols, sorted(required - cols)


def test_feature_contract_admission_statuses_are_from_the_defined_set():
    import pandas as pd

    if not CONTRACT.exists():
        pytest.skip("feature_contract.csv not present")
    c = pd.read_csv(CONTRACT)
    allowed = {"PRIMARY_ALLOWED", "SECONDARY_ALLOWED", "EXPLORATORY_ONLY",
               "FORBIDDEN_FROM_FINAL_CLASSIFIER", "HISTORICAL_NOT_ADMITTED"}
    assert set(c.admission_status) <= allowed
