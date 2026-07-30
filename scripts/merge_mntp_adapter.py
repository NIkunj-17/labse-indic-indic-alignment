#!/usr/bin/env python3
"""Approach B, Step 1.5: merge the MNTP LoRA adapter into the base model.

run_mntp.py (from the llm2vec repo) saves only the LoRA adapter weights
(adapter_config.json + adapter_model.safetensors + tokenizer files) into
its output_dir -- not a full model, and NOT a config.json. We merge here so
that later stages can load a single, clean base checkpoint.

IMPORTANT: bidirectionality is not stored in config.json -- it is a
property of which Python class the weights are loaded into (llm2vec's
LlamaBiModel overrides attention to remove the causal mask). This script
therefore loads the base model via LlamaBiModel (or the matching Bi* class
for other architectures), NOT plain AutoModel, and the merged output must
also always be reloaded the same way in every later step.

Usage:
    python scripts/merge_mntp_adapter.py \\
        --base-model sarvamai/sarvam-1 \\
        --mntp-adapter-dir ./labse_research_output/approach_b/sarvam1_bi_mntp \\
        --output-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.bi_model_utils import load_bidirectional_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("merge_mntp_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=str, default="sarvamai/sarvam-1")
    parser.add_argument("--mntp-adapter-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model as bidirectional (BiModel) variant: %s", args.base_model)
    model = load_bidirectional_model(
        args.base_model, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Tokenizer: prefer the copy saved alongside the MNTP adapter (it may
    # have the mask token added during training), fall back to the base model's.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.mntp_adapter_dir)
        logger.info("Loaded tokenizer from MNTP adapter dir (includes any added mask token).")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        logger.info("Loaded tokenizer from base model (no tokenizer found in adapter dir).")

    logger.info("Loading and merging MNTP LoRA adapter from: %s", args.mntp_adapter_dir)
    model = PeftModel.from_pretrained(model, args.mntp_adapter_dir)
    model = model.merge_and_unload()

    logger.info("Saving merged bidirectional model to: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done. Remember: reload this checkpoint via LlamaBiModel (or the matching "
                "Bi* class), never plain AutoModel, or bidirectionality will be silently lost.")


if __name__ == "__main__":
    main()
