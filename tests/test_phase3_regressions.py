import numpy as np
import pandas as pd

from src.compare.stats_tests import _delong_paired
from src.utils import branch_eval
from src.utils.branch_eval import scores_from_classifier
from src.utils.common import (
    bootstrap_plus_ovr_ci,
    tune_binary_threshold,
    tune_plus_threshold,
)


def test_binary_threshold_does_not_silently_use_multiclass_plus_index() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.20, 0.90])
    threshold = tune_binary_threshold(labels, scores)
    assert threshold != 0.5
    try:
        tune_plus_threshold(labels, scores, plus_idx=2)
    except ValueError as error:
        assert "look binary" in str(error)
    else:
        raise AssertionError("binary/multiclass threshold misuse was not rejected")


def test_multiclass_plus_threshold_matches_binary_contract() -> None:
    labels = np.array([0, 1, 2, 2])
    scores = np.array([0.05, 0.10, 0.20, 0.90])
    assert tune_plus_threshold(labels, scores, plus_idx=2) == tune_binary_threshold(
        (labels == 2).astype(int), scores
    )


def test_delong_uses_variance_of_paired_difference() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    score_a = np.array([0.1, 0.4, 0.3, 0.8, 0.7, 0.9])
    score_b = np.array([0.2, 0.6, 0.1, 0.7, 0.5, 0.8])
    result = _delong_paired(labels, score_a, score_b)
    assert result["variance_delta"] > 0
    assert np.isclose(
        result["z"],
        result["delta_auc"] / np.sqrt(result["variance_delta"]),
    )


class _NonContiguousClassifier:
    classes_ = np.array([0, 2])

    def predict_proba(self, features):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_classifier_scores_map_through_classes() -> None:
    scores, predictions = scores_from_classifier(
        _NonContiguousClassifier(), np.zeros((2, 1)), plus_idx=2
    )
    assert np.allclose(scores, [0.2, 0.9])
    assert predictions.tolist() == [0, 2]


def test_group_bootstrap_is_deterministic() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    groups = np.array(["a", "a", "b", "b"])
    first = bootstrap_plus_ovr_ci(
        labels, scores, plus_idx=1, groups=groups, n_boot=50, seed=7
    )
    second = bootstrap_plus_ovr_ci(
        labels, scores, plus_idx=1, groups=groups, n_boot=50, seed=7
    )
    assert first == second


def test_f1_threshold_searches_extreme_decisions_and_rejects_nan() -> None:
    labels = np.array([1, 1, 1, 1])
    scores = np.array([0.10, 0.20, 0.30, 0.40])
    assert tune_binary_threshold(labels, scores, method="f1") <= scores.min()
    try:
        tune_binary_threshold(np.array([0, 1]), np.array([0.1, np.nan]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite scores must be rejected")


def test_delong_identical_scores_and_single_class_behavior() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    identical = _delong_paired(labels, scores, scores)
    assert identical["p"] == 1.0
    assert identical["z"] == 0.0
    try:
        _delong_paired(np.ones(4, dtype=int), scores, scores)
    except ValueError:
        pass
    else:
        raise AssertionError("single-class DeLong input must be rejected")


def test_tabular_selection_scores_test_only_for_winner() -> None:
    class FakeClassifier:
        classes_ = np.array([0, 1])

        def __init__(self, invert=False):
            self.invert = invert
            self.test_calls = 0

        def fit(self, features, labels):
            return self

        def predict_proba(self, features):
            if len(features) == 3:
                self.test_calls += 1
            score = np.asarray(features)[:, 0]
            if self.invert:
                score = 1.0 - score
            return np.column_stack([1.0 - score, score])

    winner = FakeClassifier()
    loser = FakeClassifier(invert=True)
    models = {"winner": winner, "loser": loser}
    train_features = np.array([[0.1], [0.2], [0.8], [0.9]])
    train_labels = np.array([0, 0, 1, 1])
    test_features = np.array([[0.1], [0.8], [0.9]])
    test_labels = np.array([0, 1, 1])
    test_meta = pd.DataFrame(
        {
            "source": ["a", "a", "b"],
            "group_id": ["g1", "g2", "g3"],
            "label": test_labels,
        }
    )
    summary, best_name, *_ = branch_eval.train_select_eval_tabular(
        models,
        train_features,
        train_labels,
        train_features.copy(),
        train_labels.copy(),
        test_features,
        test_labels,
        test_meta,
        plus_idx=1,
        class_names=["negative", "plus"],
        n_boot=10,
        seed=1,
    )
    assert best_name == "winner"
    assert winner.test_calls == 1
    assert loser.test_calls == 0
    assert "test" in summary["metrics"]["winner"]
    assert "test" not in summary["metrics"]["loser"]
