"""Lightweight unit tests for the metrics module.

These run on synthetic embeddings only -- no model download, no GPU, no
network access required. Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.metrics import evaluate_pair, sensitivity_specificity  # noqa: E402


def test_evaluate_pair_perfect_alignment():
    """If source[i] == target[i] exactly, cosine_gap and accuracy@1 should be maximal."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 16))
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    metrics = evaluate_pair(base, base, seed=0)
    assert metrics["accuracy_at_1"] == 1.0
    assert metrics["mean_gold_cosine"] > 0.999
    assert metrics["cosine_gap"] > 0.5


def test_evaluate_pair_random_alignment():
    """Purely random, unrelated embeddings should give a small cosine gap and
    low (near-chance) accuracy@1.
    """
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 16))
    tgt = rng.normal(size=(50, 16))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)

    metrics = evaluate_pair(src, tgt, seed=1)
    assert abs(metrics["cosine_gap"]) < 0.3
    assert metrics["accuracy_at_1"] < 0.5


def test_sensitivity_specificity_separated_distributions():
    """When gold and random-negative distributions are cleanly separated,
    sensitivity and specificity at the midpoint threshold should both be high.
    """
    gold = np.full(100, 0.9)
    random_neg = np.full(100, 0.1)
    result = sensitivity_specificity(gold, random_neg)
    assert result["sensitivity_midpoint"] == 1.0
    assert result["specificity_midpoint"] == 1.0


def test_evaluate_pair_output_schema():
    """Every expected key must be present so downstream summary/report code
    doesn't silently drop columns.
    """
    rng = np.random.default_rng(2)
    src = rng.normal(size=(10, 8))
    tgt = rng.normal(size=(10, 8))
    metrics = evaluate_pair(src, tgt, seed=2)

    expected_keys = {
        "n", "mean_gold_cosine", "std_gold_cosine", "mean_random_cosine",
        "std_random_cosine", "cosine_gap", "std_cosine_gap",
        "threshold_midpoint", "sensitivity_midpoint", "specificity_midpoint",
        "threshold_optimal", "sensitivity_optimal", "specificity_optimal",
        "best_f1_optimal", "accuracy_at_1", "recall_at_10", "mrr",
    }
    assert expected_keys.issubset(metrics.keys())
