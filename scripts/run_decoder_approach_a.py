#!/usr/bin/env python3
"""Task 2, Approach A: evaluate a decoder-only model's raw (untrained)
sentence embeddings on IN22-Conv, using last-token or mean pooling.

This does NOT train anything -- it's a zero-shot baseline to see how far a
decoder gets on cross-lingual alignment with no adaptation at all, before
deciding whether the heavier LLM2Vec-style conversion (Approach B) is
worth building.

Usage:
    python scripts/run_decoder_approach_a.py \\
        --model-name sarvamai/sarvam-1 \\
        --pooling last_token \\
        --output-root ./labse_research_output

Run twice (once per --pooling value) to compare last_token vs mean pooling.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.data import load_in22_conv, all_directed_pairs  # noqa: E402
from labse_research.decoder_embedding import DecoderEmbedder, embed_all_languages_decoder  # noqa: E402
from labse_research.metrics import evaluate_pair  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_decoder_approach_a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, required=True,
                         help="HF model id or local path, e.g. sarvamai/sarvam-1")
    parser.add_argument("--pooling", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Keep small -- decoder models are much larger than LaBSE")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Load model in 4-bit (bitsandbytes) to reduce GPU memory -- recommended for large models like sarvam-m on a shared GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA GPU detected -- decoder models are large, this will be very slow.")

    output_dir = Path(args.output_root) / "decoder_approach_a" / f"{Path(args.model_name).name}_{args.pooling}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IN22-Conv (unseen evaluation set)")
    lang_sentences = load_in22_conv()

    embedder = DecoderEmbedder(
        model_name_or_path=args.model_name,
        device=device,
        pooling=args.pooling,
        max_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
    )

    logger.info("Embedding all languages with decoder model (pooling=%s)...", args.pooling)
    embeddings = embed_all_languages_decoder(embedder, lang_sentences, batch_size=args.batch_size)

    del embedder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    languages = list(lang_sentences.keys())
    directed_pairs = all_directed_pairs(languages)
    rows = []
    logger.info("Evaluating all %d directed pairs...", len(directed_pairs))
    for src, tgt in tqdm(directed_pairs, desc="Evaluating pairs", unit="pair"):
        n = min(len(lang_sentences[src]), len(lang_sentences[tgt]))
        metrics = evaluate_pair(embeddings[src][:n], embeddings[tgt][:n])
        metrics.update({
            "model": f"{Path(args.model_name).name}_{args.pooling}",
            "source_language": src,
            "target_language": tgt,
        })
        rows.append(metrics)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_pair_results.csv", index=False)

    mean_cosine_gap = df["cosine_gap"].mean()
    mean_sensitivity = df["sensitivity_midpoint"].mean()
    mean_specificity = df["specificity_midpoint"].mean()
    mean_accuracy_at_1 = df["accuracy_at_1"].mean()

    summary = {
        "model": args.model_name,
        "pooling": args.pooling,
        "mean_cosine_gap": float(mean_cosine_gap),
        "mean_sensitivity": float(mean_sensitivity),
        "mean_specificity": float(mean_specificity),
        "mean_accuracy_at_1": float(mean_accuracy_at_1),
    }
    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS: %s (pooling=%s)", args.model_name, args.pooling)
    logger.info("  mean_cosine_gap:    %.4f", mean_cosine_gap)
    logger.info("  mean_sensitivity:   %.4f", mean_sensitivity)
    logger.info("  mean_specificity:   %.4f", mean_specificity)
    logger.info("  mean_accuracy_at_1: %.4f", mean_accuracy_at_1)
    logger.info("  (for reference: LaBSE baseline cosine_gap = 0.3670)")
    logger.info("=" * 60)
    logger.info("Full results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
