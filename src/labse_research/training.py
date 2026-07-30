"""Shared training loop, used identically by Phase 1 (uniform sampling) and
Phase 2 (weighted sampling). Handles checkpointing, resume-from-checkpoint,
and validation-driven best-model selection.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer, losses
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import PairExample
from .metrics import validation_cosine_gap

logger = logging.getLogger(__name__)


def _identity_collate(batch: List[PairExample]) -> List[PairExample]:
    """Pass raw PairExample objects through unchanged; tokenization happens
    in `_tokenize_batch` so that it runs on the GPU-bound worker rather than
    inside the DataLoader's default (and version-fragile) collate logic.
    """
    return batch


def _tokenize_batch(model: SentenceTransformer, examples: List[PairExample], device: str):
    source_texts = [ex.source_text for ex in examples]
    target_texts = [ex.target_text for ex in examples]
    source_features = model.tokenize(source_texts)
    target_features = model.tokenize(target_texts)
    # Only move tensor-valued entries to device; some sentence-transformers
    # versions include non-tensor metadata keys in the tokenizer output.
    source_features = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in source_features.items()}
    target_features = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in target_features.items()}
    return [source_features, target_features]


def _load_resume_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"global_step": 0, "best_metric": -1e9, "history": []}


def _prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-step-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    for old in checkpoints[:-keep_last] if keep_last > 0 else checkpoints:
        shutil.rmtree(old, ignore_errors=True)


def run_training(
    cfg: TrainConfig,
    train_examples: List[PairExample],
    val_records: list,
    device: str,
    example_weights: Optional[List[float]] = None,
    reference_val_cosine_gap: Optional[float] = None,
    reference_val_specificity: Optional[float] = None,
) -> Path:
    """Run (or resume) a single training phase and return the path to the
    best-model directory.

    If `example_weights` is provided, a WeightedRandomSampler is used
    (Phase 2 style); otherwise plain shuffling is used (Phase 1 style).

    If `reference_val_specificity` is provided, checkpoint selection uses a
    specificity-guarded score (val_cosine_gap minus a penalty for dropping
    below the reference specificity) instead of raw val_cosine_gap alone --
    this discourages Phase 2 from trading away specificity purely for a
    higher gap.
    """
    output_dir = cfg.output_dir()
    checkpoint_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    final_model_dir = output_dir / "final_model"
    logs_dir = output_dir / "logs"
    for d in (checkpoint_dir, best_model_dir, final_model_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    if (final_model_dir / "config.json").exists():
        logger.info("Final model already exists at %s -- skipping training.", final_model_dir)
        return best_model_dir

    total_steps_per_epoch = len(train_examples) // cfg.batch_size
    total_steps = total_steps_per_epoch * cfg.epochs
    logger.info("Total training examples: %d | total steps: %d", len(train_examples), total_steps)

    existing_checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-step-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    start_path = str(existing_checkpoints[-1]) if existing_checkpoints else cfg.init_model
    logger.info("Loading model from: %s", start_path)

    model = SentenceTransformer(start_path, device=device)
    model.max_seq_length = 128

    if example_weights is not None:
        sampler = WeightedRandomSampler(
            weights=example_weights, num_samples=len(train_examples), replacement=True
        )
        dataloader = DataLoader(
            train_examples, sampler=sampler, batch_size=cfg.batch_size, collate_fn=_identity_collate
        )
    else:
        dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=cfg.batch_size, collate_fn=_identity_collate
        )

    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    model.to(device)
    train_loss.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    state_path = checkpoint_dir / "training_state.json"
    state = _load_resume_state(state_path)
    global_step = state["global_step"]
    best_metric = state["best_metric"]
    history = state["history"]
    if global_step > 0:
        logger.info("Resumed from step %d (best_metric=%.4f)", global_step, best_metric)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()

    progress_bar = tqdm(
        total=total_steps, initial=global_step, desc=f"Training [{cfg.run_name}]", unit="step"
    )
    try:
        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            features = _tokenize_batch(model, batch, device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                loss_value = train_loss(features, labels=None)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss_value.item():.4f}"})

            if global_step % cfg.checkpoint_every_steps == 0 or global_step == total_steps:
                checkpoint_path = checkpoint_dir / f"checkpoint-step-{global_step}"
                model.save(str(checkpoint_path))
                model.eval()

                if reference_val_specificity is not None:
                    val_gap, val_specificity, _ = validation_cosine_gap(
                        model, val_records, sample_pairs=80, with_specificity=True
                    )
                    specificity_drop = max(0.0, reference_val_specificity - val_specificity)
                    metric = val_gap - cfg.specificity_penalty_weight * specificity_drop
                else:
                    val_gap, _ = validation_cosine_gap(model, val_records, sample_pairs=80)
                    val_specificity = None
                    metric = val_gap

                model.train()
                elapsed_minutes = (time.time() - start_time) / 60

                record = {
                    "step": global_step,
                    "loss": float(loss_value.item()),
                    "val_cosine_gap": val_gap,
                    "metric": metric,
                    "elapsed_minutes": elapsed_minutes,
                }
                if val_specificity is not None:
                    record["val_specificity"] = val_specificity
                history.append(record)

                progress_bar.write(
                    f"[step {global_step}/{total_steps}] loss={loss_value.item():.4f} "
                    f"val_cosine_gap={val_gap:.4f} metric={metric:.4f} elapsed={elapsed_minutes:.1f}min"
                )

                if metric > best_metric:
                    best_metric = metric
                    model.save(str(best_model_dir))
                    progress_bar.write(f"  New best model (metric={best_metric:.4f})")

                with open(state_path, "w") as f:
                    json.dump({"global_step": global_step, "best_metric": best_metric, "history": history}, f, indent=2)

                _prune_old_checkpoints(checkpoint_dir, cfg.checkpoint_keep_last)
    finally:
        progress_bar.close()

    model.save(str(final_model_dir))

    import pandas as pd
    pd.DataFrame(history).to_csv(logs_dir / "train_metrics.csv", index=False)

    run_config = cfg.to_json()
    run_config["best_metric"] = best_metric
    if reference_val_cosine_gap is not None:
        run_config["reference_val_cosine_gap"] = reference_val_cosine_gap
    if reference_val_specificity is not None:
        run_config["reference_val_specificity"] = reference_val_specificity
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Training complete. Best metric: %.4f", best_metric)
    return best_model_dir
