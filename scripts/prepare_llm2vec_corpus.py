#!/usr/bin/env python3
"""Approach B, Step 0: build a plain-text corpus for the MNTP and SimCSE
training stages.

Both stages need plain sentences (not translation pairs) -- MNTP masks
tokens and predicts them, SimCSE builds positive pairs by passing the same
sentence through the model twice with different dropout noise. We reuse
IN22-Gen's sentences across all languages as this corpus, since it's the
same data our LaBSE fine-tuning was built on, and covers all 22 Indic
languages.

Usage:
    python scripts/prepare_llm2vec_corpus.py --output-dir ./llm2vec_corpus
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.data import load_in22_gen  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_llm2vec_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="./llm2vec_corpus")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_sentences = load_in22_gen()

    all_sentences = []
    for lang, sentences in lang_sentences.items():
        all_sentences.extend(sentences)

    rng = random.Random(args.seed)
    rng.shuffle(all_sentences)

    corpus_path = output_dir / "in22gen_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            # One sentence per line -- matches the format run_mntp.py / run_simcse.py expect.
            f.write(sentence.replace("\n", " ").strip() + "\n")

    logger.info(
        "Wrote %d sentences (%d languages) to %s",
        len(all_sentences), len(lang_sentences), corpus_path,
    )


if __name__ == "__main__":
    main()
