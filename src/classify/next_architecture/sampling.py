"""Source-balanced sampling for E8. Inverse-frequency weights, no identity as model input."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def source_inverse_frequency_weights(sources: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weight = N / (K * n_k) so expected draw rate is uniform across K sources."""
    labels = [str(s) for s in sources]
    if not labels:
        raise ValueError("source_inverse_frequency_weights: empty source list")
    counts = Counter(labels)
    n = len(labels)
    k = len(counts)
    weights = np.array([n / (k * counts[s]) for s in labels], dtype=np.float64)
    expected = {src: 1.0 / k for src in sorted(counts)}
    stats = {
        "method": "inverse_frequency",
        "n": n,
        "n_sources": k,
        "source_counts": dict(sorted(counts.items())),
        "expected_draw_fraction": expected,
        "weight_mean_by_source": {
            src: float(n / (k * counts[src])) for src in sorted(counts)
        },
    }
    return torch.as_tensor(weights, dtype=torch.double), stats


def make_source_balanced_sampler(
    sources: list[str],
    *,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    weights, stats = source_inverse_frequency_weights(sources)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, stats


def _expected_unique(p: np.ndarray, draws: int) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return float(np.sum(1.0 - np.power(1.0 - p, draws)))


def audit_image_level_source_sampler(
    frame: pd.DataFrame,
    *,
    plus_idx: int = 2,
    n_draws: int | None = None,
    top_k_groups: int = 20,
) -> dict[str, Any]:
    """Expected draw stats for E8 inverse-frequency weights. Group IDs are audit-only."""
    need = {"source", "label"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"sampler audit needs columns {sorted(need)}, missing {sorted(missing)}")
    sources = frame["source"].astype(str)
    labels = frame["label"].astype(int)
    plus = (labels == int(plus_idx)).astype(int)
    groups = frame["group_id"].astype(str) if "group_id" in frame.columns else pd.Series(["na"] * len(frame))
    weights, base = source_inverse_frequency_weights(sources.tolist())
    w = weights.detach().cpu().numpy().astype(np.float64)
    n = int(len(w))
    draws = int(n_draws or n)
    w_sum = float(w.sum())
    p = w / w_sum
    n_eff = float((w_sum ** 2) / float(np.square(w).sum()))

    orig_source = sources.value_counts(normalize=True).sort_index().to_dict()
    orig_plus_by_source = plus.groupby(sources).mean().to_dict()
    orig_plus = float(plus.mean())

    expected_source = {src: 1.0 / base["n_sources"] for src in sorted(sources.unique())}
    # Expected Plus prevalence under the sampler = sum_i p_i * plus_i
    expected_plus = float((p * plus.to_numpy()).sum())
    expected_plus_by_source: dict[str, float] = {}
    source_class: dict[str, dict[str, float]] = {}
    for src in sorted(sources.unique()):
        mask = (sources == src).to_numpy()
        p_src = p[mask]
        plus_src = plus.to_numpy()[mask]
        # Conditional Plus rate among draws from this source.
        expected_plus_by_source[str(src)] = float((p_src * plus_src).sum() / p_src.sum())
        for cls in sorted(labels.unique()):
            cls_mask = mask & (labels.to_numpy() == int(cls))
            source_class.setdefault(str(src), {})[str(int(cls))] = float(p[cls_mask].sum())

    group_mass = (
        pd.DataFrame({"group_id": groups.to_numpy(), "p": p, "source": sources.to_numpy(), "plus": plus.to_numpy()})
        .groupby("group_id", sort=False)
        .agg(expected_draw_fraction=("p", "sum"), n_images=("p", "size"), source=("source", "first"), n_plus=("plus", "sum"))
        .sort_values("expected_draw_fraction", ascending=False)
    )
    top = group_mass.head(int(top_k_groups))
    max_group_frac = float(group_mass["expected_draw_fraction"].iloc[0]) if len(group_mass) else 0.0
    unique_images = _expected_unique(p, draws)
    # Approximate unique groups: treat each group's total p as one item (with-replacement over group mass).
    gp = group_mass["expected_draw_fraction"].to_numpy(dtype=np.float64)
    unique_groups = _expected_unique(gp, draws)

    uniform_p = np.full(n, 1.0 / n)
    e2_unique_images = _expected_unique(uniform_p, draws)
    e2_n_eff = float(n)

    recommend_e8g = bool(max_group_frac >= 0.02 or n_eff < 0.5 * n)

    return {
        "method": "inverse_frequency_image_level",
        "n_images": n,
        "n_draws_per_epoch": draws,
        "original_source_fraction": {k: float(v) for k, v in orig_source.items()},
        "expected_sampled_source_fraction": expected_source,
        "original_plus_prevalence": orig_plus,
        "expected_sampled_plus_prevalence": expected_plus,
        "original_plus_prevalence_by_source": {k: float(v) for k, v in orig_plus_by_source.items()},
        "expected_sampled_plus_prevalence_by_source": expected_plus_by_source,
        "expected_source_by_class_draw_fraction": source_class,
        "n_groups": int(len(group_mass)),
        "max_expected_draw_fraction_one_group": max_group_frac,
        "effective_sample_size": n_eff,
        "effective_sample_size_over_n": float(n_eff / n) if n else 0.0,
        "expected_unique_images_per_epoch": unique_images,
        "expected_unique_groups_per_epoch": unique_groups,
        "e2_uniform_effective_sample_size": e2_n_eff,
        "e2_uniform_expected_unique_images_per_epoch": e2_unique_images,
        "top_oversampled_groups": [
            {
                "group_id": str(idx),
                "source": str(row["source"]),
                "n_images": int(row["n_images"]),
                "n_plus": int(row["n_plus"]),
                "expected_draw_fraction": float(row["expected_draw_fraction"]),
            }
            for idx, row in top.iterrows()
        ],
        "identity_used_as_model_input": False,
        "recommend_group_aware_e8g": recommend_e8g,
        "recommend_e8g_reason": (
            "image-level source weights concentrate mass on large groups or shrink n_eff"
            if recommend_e8g
            else "image-level source weights do not concentrate enough to require E8G before seeing LOSO"
        ),
        "weight_stats": base,
    }


def write_sampler_audit(payload: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return dest
