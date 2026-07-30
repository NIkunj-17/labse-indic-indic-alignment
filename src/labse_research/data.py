"""Dataset loading for IN22-Gen (training) and IN22-Conv (evaluation).

Both datasets are AI4Bharat's IN22 benchmark releases: parallel sentences
across all 22 scheduled Indic languages, aligned by row index (row i in every
language column is a translation of the same source sentence).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset

from .config import INDIC_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass
class PairExample:
    """One training example: a translation pair with its directed language tag."""

    source_language: str
    target_language: str
    source_text: str
    target_text: str


@dataclass
class ValidationRecord:
    """Held-out sentences for one directed pair, used for validation-time scoring."""

    source_language: str
    target_language: str
    source_sentences: List[str]
    target_sentences: List[str]


def _extract_language_columns(column_names: List[str]) -> Dict[str, str]:
    """Map ISO language code -> dataset column name, restricted to INDIC_LANGUAGES."""
    col_map = {}
    for col in column_names:
        lang_code = col.split("_")[0]
        if lang_code in INDIC_LANGUAGES:
            col_map[lang_code] = col
    return col_map


def load_in22_gen() -> Dict[str, List[str]]:
    """Load IN22-Gen (training source). Returns {lang_code: [sentence, ...]}."""
    logger.info("Loading ai4bharat/IN22-Gen")
    ds = load_dataset("ai4bharat/IN22-Gen")
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    col_map = _extract_language_columns(split.column_names)

    lang_sentences = {}
    for lang, col in col_map.items():
        lang_sentences[lang] = [row[col] for row in split if row[col]]

    n_langs = len(lang_sentences)
    n_sents = len(next(iter(lang_sentences.values()))) if lang_sentences else 0
    logger.info("IN22-Gen: %d languages, %d sentences/language", n_langs, n_sents)
    return lang_sentences


def load_in22_conv() -> Dict[str, List[str]]:
    """Load IN22-Conv (unseen evaluation source). Returns {lang_code: [sentence, ...]}."""
    logger.info("Loading ai4bharat/IN22-Conv")
    ds = load_dataset("ai4bharat/IN22-Conv")
    split = ds["test"]
    col_map = _extract_language_columns(split.column_names)

    lang_sentences = {}
    for lang, col in col_map.items():
        lang_sentences[lang] = [row[col] for row in split if row[col]]

    n_langs = len(lang_sentences)
    n_sents = len(next(iter(lang_sentences.values()))) if lang_sentences else 0
    logger.info("IN22-Conv: %d languages, %d sentences/language", n_langs, n_sents)
    return lang_sentences


def all_directed_pairs(languages: List[str]) -> List[Tuple[str, str]]:
    """All ordered (source, target) pairs, excluding self-pairs. len == n*(n-1)."""
    return [(s, t) for s in languages for t in languages if s != t]


def build_training_examples(
    lang_sentences: Dict[str, List[str]],
    examples_per_pair: int,
    train_val_split: float,
    seed: int,
) -> Tuple[List[PairExample], List[ValidationRecord]]:
    """Build training examples and held-out validation records for every directed pair.

    Sentences are aligned by row index within a language's sentence list, so
    the same slice indices are used for every language when building a pair.
    The train/val split point is identical across pairs, ensuring no leakage
    between the training examples and validation records for any pair.
    """
    languages = sorted(lang_sentences.keys())
    n_available = len(next(iter(lang_sentences.values())))
    if n_available < examples_per_pair:
        raise ValueError(
            f"Requested {examples_per_pair} examples/pair but only "
            f"{n_available} sentences are available per language."
        )

    n_train = int(examples_per_pair * train_val_split)
    n_val = examples_per_pair - n_train
    logger.info(
        "Building directed pairs: %d languages, %d train / %d val per pair",
        len(languages), n_train, n_val,
    )

    train_examples: List[PairExample] = []
    val_records: List[ValidationRecord] = []

    for src_lang, tgt_lang in all_directed_pairs(languages):
        src_sents = lang_sentences[src_lang][:examples_per_pair]
        tgt_sents = lang_sentences[tgt_lang][:examples_per_pair]

        for i in range(n_train):
            train_examples.append(
                PairExample(src_lang, tgt_lang, src_sents[i], tgt_sents[i])
            )

        val_records.append(
            ValidationRecord(
                source_language=src_lang,
                target_language=tgt_lang,
                source_sentences=src_sents[n_train:],
                target_sentences=tgt_sents[n_train:],
            )
        )

    rng = random.Random(seed)
    rng.shuffle(train_examples)

    logger.info("Total training examples: %d", len(train_examples))
    return train_examples, val_records
