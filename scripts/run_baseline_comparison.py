#!/usr/bin/env python3
"""Compare LaBSE against other candidate base embedding models on IN22-Conv,
zero-shot (no fine-tuning) -- same style of check we did for LaBSE vs
Vyakyarth earlier, extended to newer multilingual models.

Models covered by default:
  - LaBSE (sentence-transformers/LaBSE) -- existing baseline
  - BGE-M3 (BAAI/bge-m3) -- strong recent multilingual model, evaluated on
    Indic languages in recent published work
  - E5-Large-Instruct (intfloat/multilingual-e5-large-instruct) -- requires
    a "query: " prefix on inputs for best performance (per its model card);
    handled automatically below

This reuses the existing evaluation.py pipeline unchanged (same
evaluate_pair, summarize, plot_comparison functions used throughout the
project) -- only the input model list and E5's prefixing are new.

Usage:
    python scripts/run_baseline_comparison.py --output-root ./labse_research_output
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.config import EvalConfig  # noqa: E402
from labse_research.data import load_in22_conv  # noqa: E402
from labse_research.evaluation import (  # noqa: E402
    evaluate_model_on_conv,
    per_pair_delta,
    plot_comparison,
    summarize,
    summarize_delta,
)

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_baseline_comparison")

MODEL_COLORS = {
    "labse_baseline": "#90A4AE",
    "bge_m3_baseline": "#7E57C2",
    "e5_large_instruct_baseline": "#EF6C00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=str, default="./labse_research_output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-labse", action="store_true",
                         help="Skip re-evaluating LaBSE if you already have its baseline number")
    return parser.parse_args()


def add_e5_prefix(lang_sentences: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """E5-Instruct models expect a 'query: ' prefix on every input for best
    performance on symmetric similarity tasks (per the model's card) --
    without it, embeddings are noticeably weaker.
    """
    return {lang: [f"query: {s}" for s in sents] for lang, sents in lang_sentences.items()}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_dir = Path(args.output_root) / "baseline_comparison"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IN22-Conv (unseen evaluation set)")
    lang_sentences = load_in22_conv()

    results = []

    if not args.skip_labse:
        df = evaluate_model_on_conv(
            "labse_baseline", "sentence-transformers/LaBSE", lang_sentences, device, args.batch_size
        )
        results.append(df)
        logger.info("labse_baseline: mean cosine_gap=%.4f", df["cosine_gap"].mean())

    df_bge = evaluate_model_on_conv(
        "bge_m3_baseline", "BAAI/bge-m3", lang_sentences, device, args.batch_size
    )
    results.append(df_bge)
    logger.info("bge_m3_baseline: mean cosine_gap=%.4f", df_bge["cosine_gap"].mean())

    e5_sentences = add_e5_prefix(lang_sentences)
    df_e5 = evaluate_model_on_conv(
        "e5_large_instruct_baseline", "intfloat/multilingual-e5-large-instruct",
        e5_sentences, device, args.batch_size,
    )
    results.append(df_e5)
    logger.info("e5_large_instruct_baseline: mean cosine_gap=%.4f", df_e5["cosine_gap"].mean())

    import pandas as pd
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(eval_dir / "baseline_comparison_by_pair.csv", index=False)

    summary = summarize(combined, eval_dir)
    logger.info("\n%s", summary.to_string(index=False))

    plot_comparison(summary, eval_dir, colors=MODEL_COLORS)

    if "labse_baseline" in combined["model"].values:
        for challenger in ["bge_m3_baseline", "e5_large_instruct_baseline"]:
            if challenger in combined["model"].values:
                delta = per_pair_delta(combined, "labse_baseline", challenger)
                delta.to_csv(eval_dir / f"delta_{challenger}_vs_labse.csv", index=False)
                stats = summarize_delta(delta, "labse_baseline", challenger)
                logger.info("%s vs LaBSE: %s", challenger, stats)

    logger.info("Full results saved to: %s", eval_dir)


if __name__ == "__main__":
    main()
