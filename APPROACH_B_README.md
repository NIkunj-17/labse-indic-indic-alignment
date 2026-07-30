# Task 2, Approach B — LLM2Vec Conversion (sarvam-1 → bidirectional encoder)

## What this does, and why this version of the pipeline

Converts `sarvam-1` (a decoder, Llama-architecture) into a bidirectional
text encoder via a 3-step recipe:

1. **Enable bidirectional attention + MNTP (Masked Next-Token Prediction)**
   — the causal mask is removed so every token attends to the full
   sequence, then a short training stage adapts the model's weights to
   actually use this new context. Uses the official `llm2vec` library's
   training script. Trained on a plain-text corpus built from IN22-Gen.
2. **Merge the MNTP adapter** into a clean base model.
3. **Supervised contrastive training** on real IN22-Gen Indic-Indic
   translation pairs — **not** the original LLM2Vec paper's unsupervised
   SimCSE. SimCSE fabricates positive pairs from dropout noise, which is a
   substitute for when you don't have labeled data. We do have labeled
   data (the same pairs used for LaBSE fine-tuning), and LLM2Vec's own
   paper shows supervised training outperforms unsupervised SimCSE — so we
   use the stronger signal we already have. This stage reuses the same
   in-batch-negative contrastive idea as Phase 1/2's training, just
   implemented directly in PyTorch since the model here doesn't share
   SentenceTransformer's interface.

All training uses LoRA (not full fine-tuning) — cheaper than a full
retrain, but still real training time on the GPU (see estimates below).

## Setup

```bash
cd ~/labse_pipeline
source venv/bin/activate

# Official LLM2Vec repo -- only needed for its MNTP training script
git clone https://github.com/McGill-NLP/llm2vec.git
cd llm2vec
pip install -e .
cd ../labse_research
pip install -e .
python3 -m pytest tests/ -v   # confirm 12 passed
```

## Step 0 — Build the training corpus

```bash
python3 scripts/prepare_llm2vec_corpus.py --output-dir ./llm2vec_corpus
```

## Step 1 — MNTP training (bidirectional attention + adaptation)

**Important: the config file uses paths relative to `~/labse_pipeline/llm2vec/`.**
This only works correctly if `llm2vec` and `labse_research` are cloned as
sibling directories under `~/labse_pipeline/` (as the Setup section above
does), and if you run the command from inside `~/labse_pipeline/llm2vec/`
exactly as shown below. If your directory layout differs, edit
`train_configs/mntp/sarvam1_mntp.json`'s `train_file` and `output_dir`
paths accordingly before running.

Run from inside the cloned `llm2vec` repo:

```bash
cd ~/labse_pipeline/llm2vec
python3 experiments/run_mntp.py \
    ~/labse_pipeline/labse_research/train_configs/mntp/sarvam1_mntp.json
```

**Estimated runtime:** a few hours on the A100 -- real training, budget accordingly.

This saves a LoRA adapter (not a full model) to
`labse_research_output/approach_b/sarvam1_bi_mntp/`.

## Step 2 — Merge the MNTP adapter

```bash
cd ~/labse_pipeline/labse_research
python3 scripts/merge_mntp_adapter.py \
    --base-model sarvamai/sarvam-1 \
    --mntp-adapter-dir ./labse_research_output/approach_b/sarvam1_bi_mntp \
    --output-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged
```

Quick step (no training) -- just loads, merges LoRA into base weights, saves.

## Step 3 — Supervised contrastive training on real Indic-Indic pairs

```bash
python3 scripts/run_approach_b_supervised.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --output-dir ./labse_research_output/approach_b/sarvam1_supervised \
    --examples-per-pair 1024 \
    --batch-size 16 \
    --epochs 1
```

**Estimated runtime:** depends on GPU sharing, but with a small LoRA
adapter and ~425K pairs at 1 epoch, expect this to be comparable to or
shorter than your Phase 1 LaBSE run.

This reuses `data.py`'s `build_training_examples` directly -- same
462-pair, 1024-examples-per-pair data as your LaBSE fine-tuning -- so the
two approaches (encoder vs. this decoder-derived encoder) are trained on
identical data, making the eventual comparison fair.

## Step 4 — Evaluate on IN22-Conv

```bash
python3 scripts/run_approach_b_eval.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --supervised-adapter-dir ./labse_research_output/approach_b/sarvam1_supervised \
    --output-root ./labse_research_output

# Optional: evaluate MNTP-only (before supervised stage) to see each
# stage's individual contribution -- just omit --supervised-adapter-dir:
python3 scripts/run_approach_b_eval.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --output-root ./labse_research_output
```

Results land under `decoder_approach_b/sarvam1_<stage>/summary.json` and
`per_pair_results.csv` -- same format as Approach A and your LaBSE
results, directly comparable.

## Config choices and why (for the paper's methodology section)

- **Supervised contrastive instead of unsupervised SimCSE** — see
  rationale above; this is the main deliberate deviation from the
  original LLM2Vec paper's recipe, and is well-justified given we have
  labeled data the paper's authors didn't.
- **LoRA rank 16** — matches the original paper's published configs for
  comparable model sizes.
- **bf16 + gradient checkpointing (MNTP stage)** — reduces memory
  footprint on the shared A100.
- **`mlm_probability: 0.8`** for MNTP — deliberately high, matching the
  paper's "all_mask" collator recommendation; more aggressive than
  standard BERT-style 15%, since the goal is forcing the model to rely on
  full bidirectional context rather than local cues.
- **Same 1024-examples-per-pair, 462-pair data as LaBSE fine-tuning** —
  ensures Approach B is trained on identical data to your encoder
  baseline, isolating the architecture/method as the variable being
  tested, not the data.
- **Temperature 0.05 for InfoNCE** — standard default for this kind of
  contrastive loss; lower temperature sharpens the separation between
  positive and negative pairs.

## An alternative you could try later: unsupervised SimCSE

The original paper's unsupervised SimCSE config is kept at
`train_configs/simcse/sarvam1_simcse_ALTERNATIVE_not_used.json` for
reference, in case you want to compare supervised vs. unsupervised
contrastive training directly as an ablation for the paper. Not required
for the main pipeline.

## Known follow-ups

- This pipeline currently targets only `sarvam-1` (2B). Approach A showed
  `sarvam-m` (24B) did not clearly outperform it, so we have not built an
  Approach B pipeline for `sarvam-m` -- would need meaningfully more GPU
  time given the trainable footprint even with LoRA at that scale.
- No repeated-seed run yet, same caveat as Task 1.
- The supervised contrastive stage currently runs 1 epoch by default --
  worth checking validation trends before assuming more epochs would help
  (no built-in validation split in this script yet; could be added if
  needed for the paper's tuning story).
