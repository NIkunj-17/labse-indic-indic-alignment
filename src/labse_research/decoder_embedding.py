"""Approach A: extract sentence embeddings from a decoder-only (causal) LLM
without any retraining, using either last-token or mean pooling of the final
hidden layer.

Background
----------
Decoder models use causal (unidirectional) attention: token i can only
attend to tokens 1..i, never tokens i+1..n. This means only the LAST token
position has processed the entire input sequence -- every earlier position's
hidden state is built from incomplete (partial-sentence) information.

Two pooling strategies are provided:
  - last_token : use only the final token's hidden state (recommended
                 default for causal models, since it is the only position
                 guaranteed to have seen the whole sentence).
  - mean       : average all token hidden states (simpler, but dilutes the
                 one "complete" vector with several "partial" ones -- see
                 module docstring in README for the full explanation).

This module deliberately mirrors the output shape/behaviour of
`metrics.embed_all_languages` from the encoder pipeline, so downstream
evaluation code (metrics.evaluate_pair, evaluation.py) needs zero changes.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Literal

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["last_token", "mean"]


class DecoderEmbedder:
    """Wraps a HuggingFace causal LM checkpoint and exposes an `.encode()`
    method with the same signature shape as SentenceTransformer.encode(),
    so it's a drop-in replacement in evaluation code.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        pooling: PoolingStrategy = "last_token",
        max_length: int = 128,
        dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
    ):
        logger.info(
            "Loading decoder model: %s (pooling=%s, 4bit=%s)",
            model_name_or_path, pooling, load_in_4bit,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            # Many causal LMs have no pad token by default; reuse EOS for padding.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                quantization_config=quant_config,
                output_hidden_states=True,
                device_map={"": 0},
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, torch_dtype=dtype, output_hidden_states=True
            ).to(device)
        self.model.eval()

        self.device = device
        self.pooling = pooling
        self.max_length = max_length

    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 16,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode a list of sentences into embeddings using the configured
        pooling strategy. Signature intentionally matches
        SentenceTransformer.encode() so it can be swapped into existing
        evaluation code without modification.
        """
        all_embeddings = []

        num_batches = (len(sentences) + batch_size - 1) // batch_size
        starts = range(0, len(sentences), batch_size)
        if show_progress_bar:
            starts = tqdm(starts, total=num_batches, desc="Encoding batches", unit="batch")

        for start in starts:
            batch = sentences[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

            if self.pooling == "last_token":
                embeddings = self._last_token_pool(last_hidden, encoded["attention_mask"])
            elif self.pooling == "mean":
                embeddings = self._mean_pool(last_hidden, encoded["attention_mask"])
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.float().cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the hidden state at each sequence's last non-padding
        token position -- the only position that has attended to the full
        (unpadded) sentence under causal attention.
        """
        # Index of the last real (non-pad) token per sequence.
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average all non-padding token hidden states."""
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


def embed_all_languages_decoder(
    embedder: DecoderEmbedder,
    lang_sentences: Dict[str, List[str]],
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """Decoder-model equivalent of metrics.embed_all_languages -- same
    output shape, so it plugs directly into existing evaluation code.
    """
    embeddings = {}
    lang_progress = tqdm(lang_sentences.items(), desc="Languages", unit="lang")
    for lang, sentences in lang_progress:
        lang_progress.set_postfix({"current": lang, "sentences": len(sentences)})
        embeddings[lang] = embedder.encode(
            sentences, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
    return embeddings
