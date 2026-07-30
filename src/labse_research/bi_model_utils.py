"""Helper for loading the correct llm2vec bidirectional model class.

Bidirectionality is not something recorded in a model's config.json -- it's
a property of which Python class the weights are loaded into (llm2vec ships
custom classes like LlamaBiModel that override attention layers to remove
the causal mask). Loading a merged bidirectional checkpoint with plain
AutoModel silently reverts it to standard causal attention, with no error --
so every script in the Approach B pipeline must use this helper (or the
matching class directly) rather than AutoModel, from Step 2 onward.
"""
from __future__ import annotations

from transformers import AutoConfig


def get_bidirectional_model_class(config_class_name: str):
    """Mirror of llm2vec's experiments/run_mntp.py get_model_class dispatch,
    but returning the bare bidirectional backbone (BiModel) rather than the
    BiForMNTP wrapper, since we want raw hidden states for embeddings, not
    a masked-LM head.
    """
    from llm2vec.models import GemmaBiModel, LlamaBiModel, MistralBiModel, Qwen2BiModel

    mapping = {
        "LlamaConfig": LlamaBiModel,
        "MistralConfig": MistralBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel,
    }
    if config_class_name not in mapping:
        raise ValueError(
            f"No bidirectional model class registered for config type {config_class_name}. "
            f"Supported: {sorted(mapping.keys())}"
        )
    return mapping[config_class_name]


def load_bidirectional_model(model_name_or_path: str, **from_pretrained_kwargs):
    """Load a model as its bidirectional variant, auto-detecting the right
    llm2vec Bi* class from the model's own config type.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_class = get_bidirectional_model_class(config.__class__.__name__)
    return model_class.from_pretrained(model_name_or_path, config=config, **from_pretrained_kwargs)
