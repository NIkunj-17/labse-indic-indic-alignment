"""Unit tests for decoder pooling logic. Uses synthetic hidden states --
no model download, no GPU, no network access required.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.decoder_embedding import DecoderEmbedder  # noqa: E402


def test_last_token_pool_picks_correct_position_no_padding():
    """With no padding, last_token pooling should pick index (seq_len - 1)."""
    batch, seq_len, hidden = 2, 4, 8
    hidden_states = torch.arange(batch * seq_len * hidden, dtype=torch.float32).reshape(batch, seq_len, hidden)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    pooled = DecoderEmbedder._last_token_pool(hidden_states, attention_mask)

    assert torch.equal(pooled[0], hidden_states[0, -1])
    assert torch.equal(pooled[1], hidden_states[1, -1])


def test_last_token_pool_skips_padding():
    """With right-padding, last_token pooling must pick the last REAL token,
    not the last position in the tensor (which would be padding).
    """
    batch, seq_len, hidden = 1, 5, 4
    hidden_states = torch.arange(batch * seq_len * hidden, dtype=torch.float32).reshape(batch, seq_len, hidden)
    # Only first 3 positions are real tokens; last 2 are padding.
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    pooled = DecoderEmbedder._last_token_pool(hidden_states, attention_mask)

    # Expect the hidden state at position 2 (the last real token), not position 4.
    assert torch.equal(pooled[0], hidden_states[0, 2])


def test_mean_pool_ignores_padding():
    """Mean pooling should only average over real (non-padding) tokens."""
    batch, seq_len, hidden = 1, 4, 2
    hidden_states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])  # only first 2 tokens are real

    pooled = DecoderEmbedder._mean_pool(hidden_states, attention_mask)

    expected = torch.tensor([[2.0, 2.0]])  # mean of [1,1] and [3,3]
    assert torch.allclose(pooled, expected)


def test_mean_pool_no_padding_matches_simple_average():
    batch, seq_len, hidden = 1, 3, 2
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    pooled = DecoderEmbedder._mean_pool(hidden_states, attention_mask)

    expected = torch.tensor([[3.0, 4.0]])  # mean of the three vectors
    assert torch.allclose(pooled, expected)
