"""Evaluation metrics for cross-lingual sentence embeddings.

For a directed language pair (source -> target) with N aligned sentences,
we compare each source sentence's embedding against:
  - its true translation's embedding ("gold" similarity)
  - one randomly chosen incorrect target embedding ("random negative" similarity)

All metrics below are derived from the full N x N similarity matrix between
source and target embeddings for that pair.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def sensitivity_specificity(
    gold: np.ndarray, random_neg: np.ndarray, n_thresholds: int = 200
) -> Dict[str, float]:
    """Compute sensitivity/specificity at the midpoint threshold and at the
    empirically F1-optimal threshold, scanning n_thresholds candidates
    between the observed similarity range.
    """
    gold = gold.ravel()
    random_neg = random_neg.ravel()

    mean_threshold = (float(np.mean(gold)) + float(np.mean(random_neg))) / 2.0

    def _score_at(threshold: float) -> tuple[float, float]:
        sens = float(np.mean(gold >= threshold))
        spec = float(np.mean(random_neg < threshold))
        return sens, spec

    sens_mid, spec_mid = _score_at(mean_threshold)

    lo = min(gold.min(), random_neg.min())
    hi = max(gold.max(), random_neg.max())
    best_f1, best_threshold = -1.0, mean_threshold
    best_sens_opt, best_spec_opt = sens_mid, spec_mid

    for threshold in np.linspace(lo, hi, n_thresholds):
        tp = np.sum(gold >= threshold)
        fp = np.sum(random_neg >= threshold)
        fn = np.sum(gold < threshold)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_sens_opt, best_spec_opt = _score_at(best_threshold)

    return {
        "threshold_midpoint": mean_threshold,
        "sensitivity_midpoint": sens_mid,
        "specificity_midpoint": spec_mid,
        "threshold_optimal": best_threshold,
        "sensitivity_optimal": best_sens_opt,
        "specificity_optimal": best_spec_opt,
        "best_f1_optimal": best_f1,
    }


def evaluate_pair(
    source_embeddings: np.ndarray, target_embeddings: np.ndarray, seed: int = 42
) -> Dict[str, float]:
    """Full metric suite for one directed language pair.

    Assumes source_embeddings[i] and target_embeddings[i] are a true
    translation pair (aligned by row index).
    """
    similarity = cosine_similarity(source_embeddings, target_embeddings).astype("float32")
    n = similarity.shape[0]

    gold = np.diag(similarity)

    rng = np.random.default_rng(seed)
    negative_idx = rng.permutation(n)
    # Ensure no negative index accidentally equals its own (gold) index.
    for i in range(n):
        if negative_idx[i] == i:
            negative_idx[i] = (negative_idx[i] + 1) % n
    random_negative = similarity[np.arange(n), negative_idx]

    ranked_idx = np.argsort(-similarity, axis=1)
    ranks = np.array([int(np.where(ranked_idx[i] == i)[0][0]) + 1 for i in range(n)])

    ss = sensitivity_specificity(gold, random_negative)

    return {
        "n": int(n),
        "mean_gold_cosine": float(np.mean(gold)),
        "std_gold_cosine": float(np.std(gold)),
        "mean_random_cosine": float(np.mean(random_negative)),
        "std_random_cosine": float(np.std(random_negative)),
        "cosine_gap": float(np.mean(gold) - np.mean(random_negative)),
        "std_cosine_gap": float(np.sqrt(np.var(gold) + np.var(random_negative))),
        **ss,
        "accuracy_at_1": float(np.mean(ranks <= 1)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def validation_cosine_gap(
    model,
    val_records: List,
    seed: int = 42,
    sample_pairs: int | None = None,
    with_specificity: bool = False,
):
    """Cheap validation-time metric: mean cosine gap across a (possibly
    subsampled) set of validation records. Used for checkpoint selection
    during training, where we can't afford the full evaluate_pair() suite
    on every pair at every checkpoint.
    """
    records = val_records
    if sample_pairs is not None and sample_pairs < len(val_records):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(val_records), size=sample_pairs, replace=False)
        records = [val_records[i] for i in idx]

    rows = []
    for rec in records:
        src_emb = model.encode(
            rec.source_sentences, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        tgt_emb = model.encode(
            rec.target_sentences, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        n = min(len(src_emb), len(tgt_emb))
        if n == 0:
            continue
        similarity = src_emb[:n] @ tgt_emb[:n].T
        gold = np.diag(similarity)

        rng = np.random.default_rng(seed)
        negative_idx = rng.permutation(n)
        for i in range(n):
            if negative_idx[i] == i:
                negative_idx[i] = (negative_idx[i] + 1) % n
        random_negative = similarity[np.arange(n), negative_idx]

        row = {
            "source_language": rec.source_language,
            "target_language": rec.target_language,
            "cosine_gap": float(np.mean(gold) - np.mean(random_negative)),
        }
        if with_specificity:
            mean_threshold = (float(np.mean(gold)) + float(np.mean(random_negative))) / 2.0
            row["specificity"] = float(np.mean(random_negative < mean_threshold))
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    if with_specificity:
        return float(df["cosine_gap"].mean()), float(df["specificity"].mean()), df
    return float(df["cosine_gap"].mean()), df


def embed_all_languages(
    model, lang_sentences: Dict[str, List[str]], batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Embed every language's full sentence list once, for reuse across all pairs."""
    embeddings = {}
    for lang, sentences in lang_sentences.items():
        embeddings[lang] = model.encode(
            sentences, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
    return embeddings
