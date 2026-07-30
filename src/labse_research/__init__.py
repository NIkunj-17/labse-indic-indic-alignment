"""LaBSE Indic-Indic cross-lingual fine-tuning research pipeline.

Modules
-------
config      : experiment configuration dataclasses
data        : IN22-Gen / IN22-Conv loading and directed-pair construction
metrics     : evaluation metrics (cosine gap, sensitivity/specificity, ranking metrics)
weighting   : pair-weakness scoring and weighted-sampler construction (Phase 2)
training    : shared training loop (checkpointing, resume, validation)
evaluation  : cross-model evaluation on IN22-Conv, summary tables, per-pair deltas
"""

__version__ = "1.0.0"
