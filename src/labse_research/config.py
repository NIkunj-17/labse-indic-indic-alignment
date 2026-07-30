"""Experiment configuration.

All hyperparameters live here so that every run is fully described by one
config object and can be serialized alongside its outputs for reproducibility.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


INDIC_LANGUAGES = [
    "asm", "ben", "brx", "doi", "gom", "guj", "hin",
    "kan", "kas", "mai", "mal", "mar", "mni", "npi",
    "ory", "pan", "san", "sat", "snd", "tam", "tel", "urd",
]


@dataclass
class DataConfig:
    """Controls how much data is used and how it is split."""

    examples_per_directed_pair: int = 1024
    train_val_split: float = 0.90
    seed: int = 42
    max_seq_length: int = 128


@dataclass
class TrainConfig:
    """Optimization hyperparameters for a single training run."""

    run_name: str = "phase1_balanced_1024"
    init_model: str = "sentence-transformers/LaBSE"
    output_root: Path = Path("./labse_research_output")
    batch_size: int = 32
    learning_rate: float = 2e-6
    epochs: int = 5
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 500
    checkpoint_keep_last: int = 2
    use_amp: bool = True

    # Phase 2 (weak-pair weighted) specific. Ignored for Phase 1.
    weighted_sampling: bool = False
    min_pair_weight: float = 1.0
    max_pair_weight: float = 2.0
    weight_alpha: float = 1.0
    specificity_penalty_weight: float = 2.0
    phase1_scores_csv: Optional[Path] = None  # required if weighted_sampling=True

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_json(self) -> dict:
        d = asdict(self)
        d["output_root"] = str(self.output_root)
        if self.phase1_scores_csv is not None:
            d["phase1_scores_csv"] = str(self.phase1_scores_csv)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


@dataclass
class EvalConfig:
    """Which models to evaluate, and where to write results."""

    output_root: Path = Path("./labse_research_output")
    eval_dirname: str = "evaluation_results"
    models: dict = field(default_factory=dict)  # {display_name: path_or_hf_id}
    batch_size: int = 128

    def eval_dir(self) -> Path:
        d = Path(self.output_root) / self.eval_dirname
        d.mkdir(parents=True, exist_ok=True)
        return d
