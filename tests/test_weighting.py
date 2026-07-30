"""Unit tests for pair-weight computation (no model/network needed)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.weighting import build_example_weights, compute_pair_weights  # noqa: E402
from labse_research.data import PairExample  # noqa: E402


def _toy_scores() -> pd.DataFrame:
    return pd.DataFrame([
        {"source_language": "hin", "target_language": "tam", "cosine_gap": 0.50, "accuracy_at_1": 0.80, "specificity": 0.92},
        {"source_language": "san", "target_language": "sat", "cosine_gap": 0.10, "accuracy_at_1": 0.20, "specificity": 0.70},
        {"source_language": "ben", "target_language": "mni", "cosine_gap": 0.30, "accuracy_at_1": 0.50, "specificity": 0.85},
    ])


def test_weakest_pair_gets_max_weight():
    scores = _toy_scores()
    weighted = compute_pair_weights(scores, min_pair_weight=1.0, max_pair_weight=2.0, weight_alpha=1.0)
    weakest_row = weighted.loc[weighted["cosine_gap"].idxmin()]
    assert weakest_row["pair_weight"] == weighted["pair_weight"].max()


def test_strongest_pair_gets_min_weight():
    scores = _toy_scores()
    weighted = compute_pair_weights(scores, min_pair_weight=1.0, max_pair_weight=2.0, weight_alpha=1.0)
    strongest_row = weighted.loc[weighted["cosine_gap"].idxmax()]
    assert strongest_row["pair_weight"] == weighted["pair_weight"].min()


def test_weights_within_bounds():
    scores = _toy_scores()
    weighted = compute_pair_weights(scores, min_pair_weight=1.0, max_pair_weight=2.0, weight_alpha=1.0)
    assert (weighted["pair_weight"] >= 1.0).all()
    assert (weighted["pair_weight"] <= 2.0).all()


def test_build_example_weights_matches_pair():
    examples = [
        PairExample("hin", "tam", "s1", "t1"),
        PairExample("san", "sat", "s2", "t2"),
        PairExample("hin", "tam", "s3", "t3"),
    ]
    lookup = {("hin", "tam"): 1.2, ("san", "sat"): 2.0}
    weights = build_example_weights(examples, lookup)
    assert weights == [1.2, 2.0, 1.2]
