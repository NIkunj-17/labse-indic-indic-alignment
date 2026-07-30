"""Weak-pair scoring and weighted-sampling construction for Phase 2.

Every directed pair is scored for how weak it is under the Phase 1 model,
then assigned a training weight in [min_pair_weight, max_pair_weight].
Weak pairs get a higher weight, so a WeightedRandomSampler draws their
training examples more frequently, without duplicating any data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import PairExample, all_directed_pairs
from .metrics import evaluate_pair

logger = logging.getLogger(__name__)

EPS = 1e-4


def score_pairs_with_model(
    model,  # sentence_transformers.SentenceTransformer; typed loosely to avoid a hard import for pure-math callers
    lang_sentences: Dict[str, List[str]],
    examples_per_pair: int,
    n_train: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate a trained model on every directed pair's held-out validation
    slice, returning one row per pair with cosine_gap / accuracy_at_1 / specificity.
    """
    languages = sorted(lang_sentences.keys())
    rows = []
    for src_lang, tgt_lang in all_directed_pairs(languages):
        src_val = lang_sentences[src_lang][:examples_per_pair][n_train:]
        tgt_val = lang_sentences[tgt_lang][:examples_per_pair][n_train:]

        src_emb = model.encode(src_val, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        tgt_emb = model.encode(tgt_val, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        n = min(len(src_emb), len(tgt_emb))

        metrics = evaluate_pair(src_emb[:n], tgt_emb[:n], seed=seed)
        rows.append({
            "source_language": src_lang,
            "target_language": tgt_lang,
            "cosine_gap": metrics["cosine_gap"],
            "accuracy_at_1": metrics["accuracy_at_1"],
            "specificity": metrics["specificity_midpoint"],
        })

    return pd.DataFrame(rows)


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < EPS:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def compute_pair_weights(
    pair_scores: pd.DataFrame,
    min_pair_weight: float,
    max_pair_weight: float,
    weight_alpha: float,
) -> pd.DataFrame:
    """Turn per-pair quality scores into training weights in
    [min_pair_weight, max_pair_weight]. Weaker pairs (lower quality) get
    higher weight.

    Quality score is a weighted blend of normalized cosine_gap (50%),
    accuracy_at_1 (30%), and specificity (20%) -- cosine_gap dominates since
    it is our primary target metric, with the other two as tie-breakers /
    guards against a pair that looks good on gap alone but is weak elsewhere.
    """
    df = pair_scores.copy()
    df["norm_cosine_gap"] = _normalize(df["cosine_gap"])
    df["norm_accuracy_at_1"] = _normalize(df["accuracy_at_1"])
    df["norm_specificity"] = _normalize(df["specificity"])

    df["quality_score"] = (
        df["norm_cosine_gap"] * 0.5
        + df["norm_accuracy_at_1"] * 0.3
        + df["norm_specificity"] * 0.2
    )

    inverse_quality = 1.0 / (df["quality_score"] + EPS) ** weight_alpha
    inverse_quality_norm = _normalize(inverse_quality)
    df["pair_weight"] = min_pair_weight + inverse_quality_norm * (max_pair_weight - min_pair_weight)

    return df


def build_example_weights(
    train_examples: List[PairExample], pair_weight_lookup: Dict[Tuple[str, str], float]
) -> List[float]:
    """Map every individual training example to its pair's weight."""
    return [
        pair_weight_lookup[(ex.source_language, ex.target_language)]
        for ex in train_examples
    ]
