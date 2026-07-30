"""Cross-model evaluation on IN22-Conv: per-pair metrics, summary tables,
per-pair deltas between models, and comparison plots.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .config import EvalConfig
from .data import all_directed_pairs
from .metrics import embed_all_languages, evaluate_pair

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "mean_gold_cosine", "std_gold_cosine", "mean_random_cosine", "std_random_cosine",
    "cosine_gap", "std_cosine_gap", "sensitivity_midpoint", "specificity_midpoint",
    "sensitivity_optimal", "specificity_optimal", "best_f1_optimal",
    "accuracy_at_1", "recall_at_10", "mrr",
]


def evaluate_model_on_conv(
    model_name: str,
    model_path: str,
    lang_sentences: Dict[str, List[str]],
    device: str,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Evaluate one model across every directed pair in the IN22-Conv set."""
    logger.info("Evaluating model: %s", model_name)
    model = SentenceTransformer(str(model_path), device=device)
    embeddings = embed_all_languages(model, lang_sentences, batch_size=batch_size)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    languages = list(lang_sentences.keys())
    rows = []
    for src, tgt in all_directed_pairs(languages):
        n = min(len(lang_sentences[src]), len(lang_sentences[tgt]))
        metrics = evaluate_pair(embeddings[src][:n], embeddings[tgt][:n])
        metrics.update({"model": model_name, "source_language": src, "target_language": tgt})
        rows.append(metrics)

    return pd.DataFrame(rows)


def evaluate_all_models(cfg: EvalConfig, lang_sentences: Dict[str, List[str]], device: str) -> pd.DataFrame:
    """Evaluate every model in cfg.models, concatenate, and save the raw
    per-pair results CSV.
    """
    all_results = []
    for name, path in cfg.models.items():
        if not Path(path).exists() and "/" not in str(path):
            logger.warning("Model path not found and not a HF id, skipping: %s (%s)", name, path)
            continue
        df = evaluate_model_on_conv(name, str(path), lang_sentences, device, cfg.batch_size)
        all_results.append(df)
        logger.info("%s: mean cosine_gap=%.4f", name, df["cosine_gap"].mean())

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(cfg.eval_dir() / "in22conv_eval_all_models_by_pair.csv", index=False)
    return combined


def summarize(combined: pd.DataFrame, eval_dir: Path) -> pd.DataFrame:
    """Per-model mean of every metric, sorted best-to-worst by cosine_gap."""
    summary = (
        combined.groupby("model")[REPORT_COLUMNS]
        .mean()
        .reset_index()
        .sort_values("cosine_gap", ascending=False)
    )
    summary.to_csv(eval_dir / "in22conv_eval_summary.csv", index=False)
    return summary


def plot_comparison(summary: pd.DataFrame, eval_dir: Path, colors: Dict[str, str] | None = None) -> Path:
    colors = colors or {}
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        summary["model"], summary["cosine_gap"],
        color=[colors.get(m, "#888888") for m in summary["model"]],
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel("Cosine Gap")
    ax.set_title("IN22-Conv: Model Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = eval_dir / "comparison_cosine_gap.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def per_pair_delta(combined: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Per-pair metric deltas: model_b minus model_a."""
    a = combined[combined["model"] == model_a].set_index(["source_language", "target_language"])
    b = combined[combined["model"] == model_b].set_index(["source_language", "target_language"])

    delta = pd.DataFrame(index=a.index)
    delta[f"cosine_gap_{model_a}"] = a["cosine_gap"]
    delta[f"cosine_gap_{model_b}"] = b["cosine_gap"]
    delta["delta_cosine_gap"] = b["cosine_gap"] - a["cosine_gap"]
    delta["delta_accuracy_at_1"] = b["accuracy_at_1"] - a["accuracy_at_1"]
    delta["delta_specificity"] = b["specificity_midpoint"] - a["specificity_midpoint"]
    return delta.reset_index()


def summarize_delta(delta: pd.DataFrame, label_a: str, label_b: str) -> dict:
    improved = int((delta["delta_cosine_gap"] > 0).sum())
    regressed = int((delta["delta_cosine_gap"] < 0).sum())
    unchanged = len(delta) - improved - regressed
    return {
        "comparison": f"{label_b} vs {label_a}",
        "total_pairs": len(delta),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "mean_delta_cosine_gap": float(delta["delta_cosine_gap"].mean()),
        "mean_delta_accuracy_at_1": float(delta["delta_accuracy_at_1"].mean()),
        "mean_delta_specificity": float(delta["delta_specificity"].mean()),
    }
