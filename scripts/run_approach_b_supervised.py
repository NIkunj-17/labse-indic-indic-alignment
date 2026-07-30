#!/usr/bin/env python3
"""Approach B, Step 2 (supervised, replaces unsupervised SimCSE): contrastive
training of the MNTP-adapted bidirectional model on real IN22-Gen
Indic-Indic translation pairs.

Why supervised instead of the original paper's unsupervised SimCSE: SimCSE
builds "positive pairs" artificially, by passing the same sentence through
the model twice with different dropout noise -- a substitute for when real
labeled pairs aren't available. We already have real, human-quality
translation pairs (the same IN22-Gen data used for LaBSE fine-tuning), which
is a strictly stronger training signal. LLM2Vec's own paper shows their
supervised variant outperforms their unsupervised SimCSE variant.

The contrastive objective here mirrors Phase 1's MultipleNegativesRankingLoss
(training.py) -- in-batch negatives, InfoNCE-style loss -- just implemented
directly in PyTorch since the LLM2Vec-wrapped model doesn't share
SentenceTransformer's interface.

Usage:
    python scripts/run_approach_b_supervised.py \\
        --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \\
        --output-dir ./labse_research_output/approach_b/sarvam1_supervised \\
        --examples-per-pair 1024 --batch-size 32 --epochs 1

Improvements over the first version of this script (based on the observed
near-zero train loss and specificity trailing LaBSE):
  - Validation-driven checkpoint selection: holds out the same val split as
    Phase 1/2, evaluates cosine gap + specificity periodically, and only
    keeps a "best" checkpoint when validation improves -- guards against
    silently saving an overfit final-step model.
  - Larger default batch size (32, was 16): more in-batch negatives per
    step gives a harder, more informative contrastive signal -- the same
    effect you already saw drive Phase 1/2's results when comparing
    batch=32 vs batch=512 runs.
  - Specificity-guarded metric for checkpoint selection, mirroring Phase 2's
    approach exactly (see training.py's reference_val_specificity logic).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labse_research.bi_model_utils import load_bidirectional_model  # noqa: E402
from labse_research.data import build_training_examples, load_in22_gen  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_approach_b_supervised")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-mntp-dir", type=str, required=True,
                         help="Output of merge_mntp_adapter.py")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--examples-per-pair", type=int, default=1024)
    parser.add_argument("--train-val-split", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Higher batch = more in-batch negatives = harder contrastive signal; "
                              "reduce if you hit OOM on the shared GPU")
    parser.add_argument("--epochs", type=int, default=1,
                         help="1 epoch over ~425K pairs is already substantial for a LoRA "
                              "adapter on top of an already-adapted model; increase only if "
                              "validation shows clear underfitting")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                         help="Higher than Phase 1/2's LaBSE LR since this is a fresh LoRA "
                              "adapter (small trainable param count), not full fine-tuning")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=32,
                         help="Higher rank = more adapter capacity for sharper decision boundaries; "
                              "16 was the earlier default, 32 gives more room to fit fine distinctions")
    parser.add_argument("--temperature", type=float, default=0.05,
                         help="InfoNCE temperature -- lower = sharper, more confident separation")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                         help="Focal-InfoNCE reweighting strength (Hou & Li, 2023). 0 = plain InfoNCE. "
                              "Downweights easy negatives, upweights hard/confusable ones -- targets "
                              "specificity directly. 2.0 is the paper's typical default.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--val-sample-pairs", type=int, default=80,
                         help="Number of held-out pairs to sample for validation cosine gap/specificity "
                              "at each checkpoint -- mirrors Phase 1/2's validation approach")
    parser.add_argument("--specificity-penalty-weight", type=float, default=2.0,
                         help="Penalty applied to val_cosine_gap when specificity drops below the "
                              "reference (LaBSE baseline) -- discourages the model from trading away "
                              "specificity purely for a higher gap, same idea as Phase 2's checkpoint selection")
    parser.add_argument("--reference-specificity", type=float, default=0.9180,
                         help="Reference specificity to guard against dropping below (default: LaBSE baseline)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_batch(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    outputs = model(**encoded)
    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


def info_nce_loss(
    anchor_emb: torch.Tensor, positive_emb: torch.Tensor, temperature: float,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE, same idea as MultipleNegativesRankingLoss:
    every other example in the batch acts as an automatic negative.

    If focal_gamma > 0, applies focal-style reweighting (Hou & Li, 2023,
    "Focal-InfoNCE"): downweights easy negatives (already far apart, low
    similarity) and upweights hard negatives (confusingly close, high
    similarity), so training spends more effort on the pairs that actually
    risk being confused -- directly targets specificity, since that's
    precisely about not mistaking a hard negative for a positive.
    gamma=0 recovers plain InfoNCE (no reweighting).
    """
    similarity = anchor_emb @ positive_emb.T / temperature  # (batch, batch)
    labels = torch.arange(similarity.size(0), device=similarity.device)

    if focal_gamma <= 0:
        loss_a = F.cross_entropy(similarity, labels)
        loss_b = F.cross_entropy(similarity.T, labels)
        return (loss_a + loss_b) / 2.0

    def focal_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # confidence on the true positive
        focal_weight = (1.0 - pt).pow(focal_gamma)
        ce = F.cross_entropy(logits, labels, reduction="none")
        return (focal_weight * ce).mean()

    loss_a = focal_cross_entropy(similarity)
    loss_b = focal_cross_entropy(similarity.T)
    return (loss_a + loss_b) / 2.0


@torch.no_grad()
def compute_validation_metrics(model, tokenizer, val_records, device, max_length, seed, sample_pairs=80):
    """Validation-time cosine gap and specificity on held-out pairs -- used
    for checkpoint selection so we don't just save whatever comes out after
    the full training run finishes. This is the same idea as Phase 1/2's
    validation_cosine_gap: catches overfitting before it silently degrades
    the final saved model.
    """
    import numpy as np

    model.eval()
    records = val_records
    if sample_pairs is not None and sample_pairs < len(val_records):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(val_records), size=sample_pairs, replace=False)
        records = [val_records[i] for i in idx]

    gaps, specificities = [], []
    for rec in records:
        src_emb = encode_batch(model, tokenizer, rec.source_sentences, device, max_length)
        tgt_emb = encode_batch(model, tokenizer, rec.target_sentences, device, max_length)
        n = min(len(src_emb), len(tgt_emb))
        if n == 0:
            continue
        sim = (src_emb[:n] @ tgt_emb[:n].T).float().cpu().numpy()
        gold = np.diag(sim)
        rng = np.random.default_rng(seed)
        neg_idx = rng.permutation(n)
        for i in range(n):
            if neg_idx[i] == i:
                neg_idx[i] = (neg_idx[i] + 1) % n
        random_neg = sim[np.arange(n), neg_idx]
        gaps.append(float(np.mean(gold) - np.mean(random_neg)))
        mean_t = (float(np.mean(gold)) + float(np.mean(random_neg))) / 2.0
        specificities.append(float(np.mean(random_neg < mean_t)))

    model.train()
    return float(np.mean(gaps)), float(np.mean(specificities))


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged MNTP model from: %s", args.merged_mntp_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.merged_mntp_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_bidirectional_model(
        args.merged_mntp_dir, torch_dtype=torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading IN22-Gen and building training pairs...")
    lang_sentences = load_in22_gen()
    train_examples, val_records = build_training_examples(
        lang_sentences,
        examples_per_pair=args.examples_per_pair,
        train_val_split=args.train_val_split,
        seed=args.seed,
    )
    logger.info("Total training examples: %d | validation pairs: %d", len(train_examples), len(val_records))

    def collate(batch):
        return batch

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, collate_fn=collate)
    total_steps = (len(train_examples) // args.batch_size) * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    model.train()
    global_step = 0
    best_metric = -1e9
    start_time = time.time()
    progress_bar = tqdm(total=total_steps, desc="Supervised contrastive training", unit="step")

    best_dir = Path(str(output_dir) + "_best")
    best_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if global_step >= total_steps:
                break

            anchor_texts = [ex.source_text for ex in batch]
            positive_texts = [ex.target_text for ex in batch]

            optimizer.zero_grad()
            anchor_emb = encode_batch(model, tokenizer, anchor_texts, device, args.max_seq_length)
            positive_emb = encode_batch(model, tokenizer, positive_texts, device, args.max_seq_length)
            loss = info_nce_loss(anchor_emb, positive_emb, args.temperature, focal_gamma=args.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % args.checkpoint_every_steps == 0 or global_step == total_steps:
                elapsed = (time.time() - start_time) / 60
                val_gap, val_specificity = compute_validation_metrics(
                    model, tokenizer, val_records, device, args.max_seq_length,
                    args.seed, sample_pairs=args.val_sample_pairs,
                )
                specificity_drop = max(0.0, args.reference_specificity - val_specificity)
                metric = val_gap - args.specificity_penalty_weight * specificity_drop

                logger.info(
                    "[step %d/%d] loss=%.4f val_cosine_gap=%.4f val_specificity=%.4f metric=%.4f elapsed=%.1fmin",
                    global_step, total_steps, loss.item(), val_gap, val_specificity, metric, elapsed,
                )

                # Always save a "latest" checkpoint (for resuming / inspection)...
                model.save_pretrained(str(output_dir))
                tokenizer.save_pretrained(str(output_dir))

                # ...but only overwrite the BEST checkpoint if validation improved,
                # guarding against the near-zero-train-loss overfitting we saw before.
                if metric > best_metric:
                    best_metric = metric
                    model.save_pretrained(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir))
                    logger.info("  New best checkpoint (metric=%.4f) saved to %s", best_metric, best_dir)

    progress_bar.close()
    logger.info("Supervised contrastive training complete.")
    logger.info("Latest checkpoint: %s", output_dir)
    logger.info("Best checkpoint (by validation metric):  %s", best_dir)
    logger.info(
        "IMPORTANT: use the BEST checkpoint (%s) for evaluation, not the latest one -- "
        "the latest checkpoint reflects the final training step regardless of whether "
        "validation improved, and may be overfit.", best_dir,
    )


if __name__ == "__main__":
    main()
