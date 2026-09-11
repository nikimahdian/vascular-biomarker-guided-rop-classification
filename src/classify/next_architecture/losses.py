"""Targets and losses for E0–E5. No third-party ordinal packages."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PLUS_INDEX = 2


def plus_target_from_labels(labels: torch.Tensor, plus_idx: int = PLUS_INDEX) -> torch.Tensor:
    return (labels.long() == plus_idx).float()


def ordinal_targets_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """CORAL-style cumulative targets for labels in {0,1,2}.

    Column 0: label >= 1 (Pre-Plus or Plus)
    Column 1: label >= 2 (Plus)
    """
    labels = labels.long()
    ge_pre = (labels >= 1).float()
    ge_plus = (labels >= 2).float()
    return torch.stack([ge_pre, ge_plus], dim=1)


def class_weights_from_labels(labels: torch.Tensor, n_cls: int) -> torch.Tensor:
    """Inverse-frequency weights. Call on the training fold only."""
    counts = torch.bincount(labels.long(), minlength=n_cls).float()
    return counts.sum() / (n_cls * counts.clamp(min=1.0))


def binary_pos_weight_from_labels(plus_targets: torch.Tensor) -> torch.Tensor:
    """BCE pos_weight = n_neg / n_pos from training fold only."""
    pos = plus_targets.sum().clamp(min=1.0)
    neg = (plus_targets.numel() - plus_targets.sum()).clamp(min=1.0)
    return (neg / pos).reshape(())


class CoralOrdinalLoss(nn.Module):
    """Independent BCE on cumulative thresholds (CORAL / cumulative-link)."""

    def forward(self, ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = ordinal_targets_from_labels(labels).to(ordinal_logits.device)
        return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def coral_probabilities(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """P(y >= k) for k in {1,2}. Monotonic if logits are constrained; we clip CDF."""
    return torch.sigmoid(ordinal_logits)


def coral_class_probs(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """P(y=0), P(y=1), P(y=2) from cumulative probabilities, clipped to be non-negative."""
    cum = coral_probabilities(ordinal_logits)
    p_ge1 = cum[:, 0]
    p_ge2 = torch.minimum(cum[:, 1], p_ge1)
    p0 = 1.0 - p_ge1
    p1 = p_ge1 - p_ge2
    p2 = p_ge2
    stacked = torch.stack([p0, p1, p2], dim=1).clamp(min=0.0)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp(min=1e-8)


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sources: list[str] | None = None,
    *,
    max_pairs: int = 64,
    prefer_within_source: bool = True,
    margin: float = 0.0,
) -> torch.Tensor:
    """Higher clinical severity must score higher. Training samples only.

    Pair construction uses `labels` (0<1<2). `sources` may balance pairs but is
    never a model input — callers must not concatenate it into `scores`.
    """
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    labels = labels.reshape(-1).long()
    n = int(scores.numel())
    if n < 2:
        return scores.new_zeros(())

    device = scores.device
    pair_i: list[int] = []
    pair_j: list[int] = []

    def add_pairs(indices: list[int]) -> None:
        labs = labels[indices]
        for a, ia in enumerate(indices):
            for ib in indices[a + 1 :]:
                if labels[ia] == labels[ib]:
                    continue
                if labels[ia] > labels[ib]:
                    pair_i.append(ia)
                    pair_j.append(ib)
                else:
                    pair_i.append(ib)
                    pair_j.append(ia)

    if prefer_within_source and sources is not None and len(sources) == n:
        by_source: dict[str, list[int]] = {}
        for idx, source in enumerate(sources):
            by_source.setdefault(str(source), []).append(idx)
        for idxs in by_source.values():
            add_pairs(idxs)
        if not pair_i:
            add_pairs(list(range(n)))
    else:
        add_pairs(list(range(n)))

    if not pair_i:
        return scores.new_zeros(())

    ii = torch.tensor(pair_i, device=device, dtype=torch.long)
    jj = torch.tensor(pair_j, device=device, dtype=torch.long)
    if ii.numel() > max_pairs:
        perm = torch.randperm(ii.numel(), device=device)[:max_pairs]
        ii, jj = ii[perm], jj[perm]
    # logistic pairwise: log(1+exp(-(s_high - s_low - margin)))
    delta = scores[ii] - scores[jj] - margin
    return F.softplus(-delta).mean()


def reverse_pair_loss_increases(score_high: float, score_low: float) -> bool:
    """Unit-test helper: swapping a correctly ordered pair must increase loss."""
    correct = F.softplus(torch.tensor(-(score_high - score_low))).item()
    reversed_ = F.softplus(torch.tensor(-(score_low - score_high))).item()
    return reversed_ > correct
