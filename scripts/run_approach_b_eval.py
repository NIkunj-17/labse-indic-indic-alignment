#!/usr/bin/env python3
"""Task 2, Approach B: evaluate the LLM2Vec-converted (bidirectional +
MNTP + supervised contrastive trained) decoder model on IN22-Conv.

Run this only after:
  1. python experiments/run_mntp.py train_configs/mntp/sarvam1_mntp.json   (llm2vec repo)
  2. python scripts/merge_mntp_adapter.py --base-model ... --mntp-adapter-dir ... --output-dir ...
  3. python scripts/run_approach_b_supervised.py --merged-mntp-dir ... --output-dir ...

Usage:
    python scripts/run_approach_b_eval.py \\
        --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \\
        --supervised-adapter-dir ./labse_research_output/approach_b/sarvam1_supervised \\
        --output-root ./labse_research_output
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.bi_model_utils import load_bidirectional_model  # noqa: E402
from labse_research.data import load_in22_conv, all_directed_pairs  # noqa: E402
from labse_research.metrics import evaluate_pair  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_approach_b_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-mntp-dir", type=str, required=True,
                         help="Output of merge_mntp_adapter.py -- the bidirectional, "
                              "MNTP-adapted base model")
    parser.add_argument("--supervised-adapter-dir", type=str, default=None,
                         help="Output of run_approach_b_supervised.py (omit to evaluate "
                              "MNTP-only, useful as an intermediate checkpoint)")
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer

    if args.supervised_adapter_dir:
        # Use the actual adapter folder's name (e.g. "sarvam1_supervised_v3_best")
        # rather than a generic label -- otherwise two different checkpoints
        # (e.g. "_v3" vs "_v3_best") silently overwrite the same output path,
        # making it impossible to tell which result came from which checkpoint.
        stage_label = Path(args.supervised_adapter_dir).name
    else:
        stage_label = "mntp_only"
    output_dir = Path(args.output_root) / "decoder_approach_b" / stage_label
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged MNTP model (as bidirectional) from: %s", args.merged_mntp_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.merged_mntp_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_bidirectional_model(
        args.merged_mntp_dir, torch_dtype=torch.bfloat16
    ).to(device)

    if args.supervised_adapter_dir:
        from peft import PeftModel
        logger.info("Loading supervised contrastive LoRA adapter from: %s", args.supervised_adapter_dir)
        model = PeftModel.from_pretrained(model, args.supervised_adapter_dir)
        model = model.merge_and_unload()

    model.eval()

    def mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(sentences, batch_size):
        import numpy as np
        all_embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            ).to(device)
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.float().cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

    logger.info("Loading IN22-Conv (unseen evaluation set)")
    lang_sentences = load_in22_conv()

    logger.info("Embedding all languages...")
    embeddings = {}
    for lang, sentences in tqdm(lang_sentences.items(), desc="Languages", unit="lang"):
        embeddings[lang] = encode(sentences, args.batch_size)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    languages = list(lang_sentences.keys())
    directed_pairs = all_directed_pairs(languages)
    rows = []
    for src, tgt in tqdm(directed_pairs, desc="Evaluating pairs", unit="pair"):
        n = min(len(lang_sentences[src]), len(lang_sentences[tgt]))
        metrics = evaluate_pair(embeddings[src][:n], embeddings[tgt][:n])
        metrics.update({"model": f"sarvam1_{stage_label}", "source_language": src, "target_language": tgt})
        rows.append(metrics)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_pair_results.csv", index=False)

    summary = {
        "model": f"sarvam1_{stage_label}",
        "mean_cosine_gap": float(df["cosine_gap"].mean()),
        "mean_sensitivity": float(df["sensitivity_midpoint"].mean()),
        "mean_specificity": float(df["specificity_midpoint"].mean()),
        "mean_accuracy_at_1": float(df["accuracy_at_1"].mean()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS: sarvam-1 (%s)", stage_label)
    logger.info("  mean_cosine_gap:    %.4f", summary["mean_cosine_gap"])
    logger.info("  mean_sensitivity:   %.4f", summary["mean_sensitivity"])
    logger.info("  mean_specificity:   %.4f", summary["mean_specificity"])
    logger.info("  mean_accuracy_at_1: %.4f", summary["mean_accuracy_at_1"])
    logger.info("  (for reference: LaBSE baseline cosine_gap = 0.3670)")
    logger.info("  (for reference: Approach A last-token cosine_gap = 0.1316)")
    logger.info("=" * 60)
    logger.info("Full results saved to: %s", output_dir)

if __name__ == "__main__":
    main()
