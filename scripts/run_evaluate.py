#!/usr/bin/env python3
"""Evaluate LaBSE baseline, Phase 1, and Phase 2 on unseen IN22-Conv.

Produces:
  - in22conv_eval_all_models_by_pair.csv  (raw per-pair metrics, all models)
  - in22conv_eval_summary.csv             (per-model mean of every metric)
  - comparison_cosine_gap.png             (bar chart)
  - delta_phase2_vs_phase1.csv            (per-pair deltas, phase2 vs phase1)

Usage:
    python scripts/run_evaluate.py \\
        --output-root ./labse_research_output \\
        --phase1-model-dir ./labse_research_output/phase1_balanced_1024/best_model \\
        --phase2-model-dir ./labse_research_output/phase2_weighted_1024/best_model
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.config import EvalConfig  # noqa: E402
from labse_research.data import load_in22_conv  # noqa: E402
from labse_research.evaluation import (  # noqa: E402
    evaluate_all_models,
    per_pair_delta,
    plot_comparison,
    summarize,
    summarize_delta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_evaluate")

MODEL_COLORS = {
    "labse_baseline": "#90A4AE",
    "phase1_balanced_1024": "#1565C0",
    "phase2_weighted_1024": "#21C99A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--phase1-model-dir", type=str, default=None)
    parser.add_argument("--phase2-model-dir", type=str, default=None)
    parser.add_argument("--phase1-name", type=str, default="phase1_balanced_1024")
    parser.add_argument("--phase2-name", type=str, default="phase2_weighted_1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {"labse_baseline": "sentence-transformers/LaBSE"}
    if args.phase1_model_dir:
        models[args.phase1_name] = args.phase1_model_dir
    if args.phase2_model_dir:
        models[args.phase2_name] = args.phase2_model_dir

    cfg = EvalConfig(output_root=Path(args.output_root), models=models)

    lang_sentences = load_in22_conv()
    combined = evaluate_all_models(cfg, lang_sentences, device)

    summary = summarize(combined, cfg.eval_dir())
    logger.info("Summary:\n%s", summary.to_string(index=False))

    plot_comparison(summary, cfg.eval_dir(), colors=MODEL_COLORS)

    if args.phase1_model_dir and args.phase2_model_dir:
        delta = per_pair_delta(combined, args.phase1_name, args.phase2_name)
        delta.to_csv(cfg.eval_dir() / f"delta_{args.phase2_name}_vs_{args.phase1_name}.csv", index=False)
        stats = summarize_delta(delta, args.phase1_name, args.phase2_name)
        logger.info("Per-pair delta summary: %s", stats)

    logger.info("All evaluation outputs saved under: %s", cfg.eval_dir())


if __name__ == "__main__":
    main()
