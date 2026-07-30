#!/usr/bin/env python3
"""Phase 1: balanced all-462-pair fine-tuning of LaBSE on IN22-Gen.

Usage:
    python scripts/run_phase1.py --config configs/phase1_1024.json

If no config is given, defaults from TrainConfig / DataConfig are used
(1024 examples/pair, batch 32, lr 2e-6, 5 epochs).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.config import DataConfig, TrainConfig  # noqa: E402
from labse_research.data import build_training_examples, load_in22_gen  # noqa: E402
from labse_research.training import run_training  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file (optional)")
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--examples-per-pair", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="phase1_balanced_1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(examples_per_directed_pair=args.examples_per_pair)
    train_cfg = TrainConfig(
        run_name=args.run_name,
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )

    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for key, value in overrides.get("data", {}).items():
            setattr(data_cfg, key, value)
        for key, value in overrides.get("train", {}).items():
            setattr(train_cfg, key, value)

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

    best_model_dir = run_training(
        cfg=train_cfg,
        train_examples=train_examples,
        val_records=val_records,
        device=device,
    )
    logger.info("Phase 1 best model saved at: %s", best_model_dir)


if __name__ == "__main__":
    main()
