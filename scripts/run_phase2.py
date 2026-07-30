#!/usr/bin/env python3
"""Phase 2: weak-pair weighted fine-tuning, initialized from Phase 1's best model.

Scores every directed pair's weakness using the Phase 1 model, assigns each
pair a training weight in [min_pair_weight, max_pair_weight], and trains with
a PyTorch WeightedRandomSampler so weak pairs are drawn more frequently.

Usage:
    python scripts/run_phase2.py \\
        --phase1-model-dir ./labse_research_output/phase1_balanced_1024/best_model \\
        --run-name phase2_weighted_1024 \\
        --max-pair-weight 2.0 --learning-rate 2e-6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.config import DataConfig, TrainConfig  # noqa: E402
from labse_research.data import build_training_examples, load_in22_gen  # noqa: E402
from labse_research.metrics import validation_cosine_gap  # noqa: E402
from labse_research.training import run_training  # noqa: E402
from labse_research.weighting import (  # noqa: E402
    build_example_weights,
    compute_pair_weights,
    score_pairs_with_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-model-dir", type=str, required=True,
                         help="Path to Phase 1's best_model directory")
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--examples-per-pair", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-pair-weight", type=float, default=2.0)
    parser.add_argument("--min-pair-weight", type=float, default=1.0)
    parser.add_argument("--specificity-penalty-weight", type=float, default=2.0)
    parser.add_argument("--run-name", type=str, default="phase2_weighted_1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase1_model_dir = Path(args.phase1_model_dir)
    if not (phase1_model_dir / "config.json").exists():
        raise FileNotFoundError(f"Phase 1 model not found at {phase1_model_dir}. Run Phase 1 first.")

    data_cfg = DataConfig(examples_per_directed_pair=args.examples_per_pair)
    train_cfg = TrainConfig(
        run_name=args.run_name,
        init_model=str(phase1_model_dir),
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weighted_sampling=True,
        min_pair_weight=args.min_pair_weight,
        max_pair_weight=args.max_pair_weight,
        specificity_penalty_weight=args.specificity_penalty_weight,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA GPU detected -- this will be extremely slow.")

    logger.info("Data config: %s", data_cfg)
    logger.info("Train config: %s", train_cfg)

    lang_sentences = load_in22_gen()
    train_examples, val_records = build_training_examples(
        lang_sentences,
        examples_per_pair=data_cfg.examples_per_directed_pair,
        train_val_split=data_cfg.train_val_split,
        seed=data_cfg.seed,
    )
    n_train = int(data_cfg.examples_per_directed_pair * data_cfg.train_val_split)

    logger.info("Loading Phase 1 model to score pair-level weaknesses: %s", phase1_model_dir)
    from sentence_transformers import SentenceTransformer
    phase1_model = SentenceTransformer(str(phase1_model_dir), device=device)

    logger.info("Scoring every directed pair's Phase 1 validation performance...")
    pair_scores = score_pairs_with_model(
        phase1_model, lang_sentences, data_cfg.examples_per_directed_pair, n_train, data_cfg.seed
    )

    # Reference metrics for Phase 2's specificity-guarded checkpoint selection,
    # and for later reporting how far Phase 2 is tracking above/below Phase 1.
    reference_val_cosine_gap = float(pair_scores["cosine_gap"].mean())
    reference_val_specificity = float(pair_scores["specificity"].mean())
    logger.info(
        "Phase 1 reference: val_cosine_gap=%.4f, val_specificity=%.4f",
        reference_val_cosine_gap, reference_val_specificity,
    )

    del phase1_model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    weighted_scores = compute_pair_weights(
        pair_scores,
        min_pair_weight=train_cfg.min_pair_weight,
        max_pair_weight=train_cfg.max_pair_weight,
        weight_alpha=train_cfg.weight_alpha,
    )

    output_dir = train_cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    weighted_scores.to_csv(output_dir / "phase1_pair_level_val_scores.csv", index=False)

    logger.info(
        "Top 10 weakest pairs (highest training weight):\n%s",
        weighted_scores.sort_values("pair_weight", ascending=False)
        [["source_language", "target_language", "cosine_gap", "accuracy_at_1", "pair_weight"]]
        .head(10).to_string(index=False),
    )

    weight_lookup = {
        (row["source_language"], row["target_language"]): row["pair_weight"]
        for _, row in weighted_scores.iterrows()
    }
    example_weights = build_example_weights(train_examples, weight_lookup)

    best_model_dir = run_training(
        cfg=train_cfg,
        train_examples=train_examples,
        val_records=val_records,
        device=device,
        example_weights=example_weights,
        reference_val_cosine_gap=reference_val_cosine_gap,
        reference_val_specificity=reference_val_specificity,
    )
    logger.info("Phase 2 best model saved at: %s", best_model_dir)


if __name__ == "__main__":
    main()
