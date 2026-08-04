# LaBSE Indic-Indic Sentence Alignment

Balanced and weak-pair weighted fine-tuning of LaBSE for direct Indic-to-Indic
sentence alignment across all 22 scheduled Indic languages, plus an
exploratory pipeline for converting a decoder-only Indic language model
(`sarvam-1`) into a competitive bidirectional sentence encoder.

WSAI Summer Internship 2026, Project ID: 2026/WSAI/029, IIT Madras.

## Results

All numbers are on the held-out **IN22-Conv** benchmark (462 directed
Indic-Indic pairs, never seen during training).

| Model                              | Cosine Gap | Sensitivity | Specificity |
|-------------------------------------|:----------:|:-----------:|:-----------:|
| LaBSE (zero-shot baseline)          | 0.3670     | 0.8069      | 0.9180      |
| **Stage 1** — balanced fine-tuning  | 0.4594     | 0.9333      | **0.9211**  |
| **Stage 2** — weak-pair reweighting | **0.4758** | 0.9227      | 0.9137      |

Stage 1 improves cosine gap by 25.2% over zero-shot LaBSE. Stage 2 pushes
this to 29.6%, improving 460 of 462 directed pairs relative to Stage 1
alone. A separate experiment (Task 2) converts a decoder-only Indic LLM
(`sarvam-1`) into a bidirectional encoder and reaches a cosine gap of
0.4216 — ahead of zero-shot LaBSE, though still behind Stage 1/2 on
specificity.

## Repository layout

```
.
├── src/labse_research/
│   ├── config.py        # DataConfig / TrainConfig / EvalConfig dataclasses
│   ├── data.py           # IN22-Gen / IN22-Conv loading, directed-pair construction
│   ├── metrics.py        # cosine gap, sensitivity/specificity, accuracy@1, MRR
│   ├── weighting.py      # pair-weakness scoring, weight computation (Stage 2)
│   ├── training.py        # shared training loop: checkpoint/resume, val-driven best-model
│   └── evaluation.py     # cross-model eval, summary table, per-pair deltas, plots
├── scripts/
│   ├── run_phase1.py            # Stage 1 entry point
│   ├── run_phase2.py            # Stage 2 entry point
│   ├── run_evaluate.py          # Evaluation entry point
│   ├── run_full_pipeline.sh     # Runs Stage 1 -> Stage 2 -> evaluation in sequence
│   ├── prepare_llm2vec_corpus.py
│   ├── merge_mntp_adapter.py
│   ├── run_approach_b_supervised.py
│   └── run_approach_b_eval.py
├── configs/
│   └── phase1_1024.json         # Example config: 1024 examples/pair
├── train_configs/
│   └── mntp/sarvam1_mntp.json   # MNTP training config for Task 2, Approach B
├── tests/
│   ├── test_metrics.py          # Unit tests, synthetic data, no GPU/network needed
│   └── test_weighting.py
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .          # editable install so `import labse_research` works anywhere

huggingface-cli login     # or `hf auth login`, if not already authenticated
```

Before trusting the pipeline with a multi-hour run, run the unit tests.
They use only synthetic embeddings — no GPU, no model download, no
dataset download — and finish in under 2 seconds.

```bash
python3 -m pytest tests/ -v
```

---

## Task 1 — Balanced and weak-pair weighted LaBSE fine-tuning

Fine-tunes LaBSE across all 462 directed Indic-Indic pairs in two stages:

1. **Stage 1 (balanced joint fine-tuning):** all 462 pairs, sampled
   uniformly, using in-batch negatives (Multiple Negatives Ranking Loss).
2. **Stage 2 (weak-pair reweighting):** continues from the Stage 1
   checkpoint, oversampling directed pairs that scored worst on Stage 1's
   validation set (blend of cosine gap, Accuracy@1, and specificity),
   with checkpoint selection guarded against specificity regressions.

### Running the full pipeline

Use `tmux` (or any session multiplexer) so a multi-hour run survives an
SSH/network disconnect:

```bash
tmux new -s task1_full_run

source venv/bin/activate
export OUTPUT_ROOT=./labse_research_output
bash scripts/run_full_pipeline.sh
```

Detach with `Ctrl+B` then `D`; reattach anytime with `tmux attach -t task1_full_run`.

**Expected runtime:** with 1024 examples per directed pair (~425,000
training examples total), budget for a multi-hour run per stage on a
single A100-class GPU. Checkpoints save every 500 steps (configurable),
so an interrupted run resumes automatically on re-run — no progress is
lost.

### Running stages individually

```bash
# Stage 1
python3 scripts/run_phase1.py --config configs/phase1_1024.json \
    --output-root "$OUTPUT_ROOT"

# Stage 2 (after Stage 1 finishes)
python3 scripts/run_phase2.py \
    --phase1-model-dir "$OUTPUT_ROOT/phase1_balanced_1024/best_model" \
    --output-root "$OUTPUT_ROOT" \
    --examples-per-pair 1024 --batch-size 32 --learning-rate 2e-6 --max-pair-weight 2.0

# Evaluation
python3 scripts/run_evaluate.py \
    --output-root "$OUTPUT_ROOT" \
    --phase1-model-dir "$OUTPUT_ROOT/phase1_balanced_1024/best_model" \
    --phase2-model-dir "$OUTPUT_ROOT/phase2_weighted_1024/best_model"
```

### Where results land

```
$OUTPUT_ROOT/
├── phase1_balanced_1024/
│   ├── best_model/               # SentenceTransformer checkpoint, load directly
│   ├── final_model/
│   ├── checkpoints/               # intermediate checkpoints, auto-pruned
│   ├── logs/train_metrics.csv     # step, loss, val_cosine_gap over training
│   └── training_config.json       # exact hyperparameters used, for reproducibility
├── phase2_weighted_1024/
│   ├── best_model/  final_model/  checkpoints/  logs/
│   ├── phase1_pair_level_val_scores.csv   # per-pair weakness scores + assigned weights
│   └── training_config.json
└── evaluation_results/
    ├── in22conv_eval_all_models_by_pair.csv
    ├── in22conv_eval_summary.csv
    ├── comparison_cosine_gap.png
    └── delta_phase2_weighted_1024_vs_phase1_balanced_1024.csv
```

### Design notes

- **In-batch negatives (`MultipleNegativesRankingLoss`):** every other
  example in a training batch acts as an automatic negative for a given
  anchor, so batch size directly controls the number of negatives seen
  per step (`batch_size - 1`). Held constant across every model
  comparison in this repo.
- **Stage 2 weighting is continuous, not a hard cutoff:** every pair
  gets a weight in `[min_pair_weight, max_pair_weight]` from a blended
  quality score (50% cosine gap, 30% Accuracy@1, 20% specificity from
  Stage 1), inverted and rescaled — see `weighting.compute_pair_weights`.
- **Specificity-guarded checkpoint selection:** Stage 2's best-model
  selection doesn't just maximize validation cosine gap — it subtracts a
  penalty if specificity drops below the Stage 1 reference, discouraging
  the model from trading away precision purely for a higher gap.
- **Reproducibility:** every run's exact hyperparameters are saved to
  `training_config.json` alongside its outputs.

---

## Task 2 — Converting a decoder LLM into a bidirectional encoder (Approach B)

Converts `sarvam-1` (a decoder, Llama-architecture Indic LLM) into a
bidirectional text encoder via an LLM2Vec-style recipe, then evaluates it
against the LaBSE baselines above.

**Approach A** (not covered by scripts here) is a simple zero-shot
embedding extraction baseline — last-token and mean pooling directly from
the frozen decoder, no training. It substantially underperforms LaBSE and
motivates Approach B below.

### Setup

```bash
source venv/bin/activate

# Official LLM2Vec repo -- only needed for its MNTP training script
git clone https://github.com/McGill-NLP/llm2vec.git
cd llm2vec && pip install -e . && cd ..
```

### Step 0 — Build the training corpus

```bash
python3 scripts/prepare_llm2vec_corpus.py --output-dir ./llm2vec_corpus
```

### Step 1 — MNTP training (bidirectional attention + adaptation)

The causal attention mask is removed so every token attends to the full
sequence, then a short Masked Next-Token Prediction stage adapts the
model's weights to actually use this new bidirectional context. Uses the
official `llm2vec` library's training script, run on a plain-text corpus
built from IN22-Gen.

> The config file uses paths relative to the `llm2vec` clone directory.
> This only works if `llm2vec` and this repository are cloned as sibling
> directories, and if you run the command from inside the `llm2vec`
> directory as shown below. If your layout differs, edit
> `train_configs/mntp/sarvam1_mntp.json`'s `train_file` and `output_dir`
> paths first.

```bash
cd llm2vec
python3 experiments/run_mntp.py \
    ../train_configs/mntp/sarvam1_mntp.json
```

**Estimated runtime:** a few hours on an A100-class GPU.

This saves a LoRA adapter (not a full model) to
`labse_research_output/approach_b/sarvam1_bi_mntp/`.

### Step 2 — Merge the MNTP adapter

```bash
cd ..
python3 scripts/merge_mntp_adapter.py \
    --base-model sarvamai/sarvam-1 \
    --mntp-adapter-dir ./labse_research_output/approach_b/sarvam1_bi_mntp \
    --output-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged
```

### Step 3 — Supervised contrastive training on real Indic-Indic pairs

```bash
python3 scripts/run_approach_b_supervised.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --output-dir ./labse_research_output/approach_b/sarvam1_supervised \
    --examples-per-pair 1024 \
    --batch-size 16 \
    --epochs 1
```

This deliberately uses **supervised contrastive training on real
translation pairs**, not the original LLM2Vec paper's unsupervised
SimCSE — since we already have the same labeled Indic-Indic pairs used
for the LaBSE fine-tuning in Task 1, and LLM2Vec's own paper shows
supervised training outperforms unsupervised SimCSE when labeled data
exists. Reusing `data.py`'s `build_training_examples` directly means both
approaches train on identical data, keeping the comparison fair.

### Step 4 — Evaluate on IN22-Conv

```bash
python3 scripts/run_approach_b_eval.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --supervised-adapter-dir ./labse_research_output/approach_b/sarvam1_supervised \
    --output-root ./labse_research_output

# Optional: evaluate MNTP-only (before the supervised stage) --
# just omit --supervised-adapter-dir:
python3 scripts/run_approach_b_eval.py \
    --merged-mntp-dir ./labse_research_output/approach_b/sarvam1_bi_mntp_merged \
    --output-root ./labse_research_output
```

### Config choices

- **Supervised contrastive instead of unsupervised SimCSE** — see Step 3.
- **LoRA rank 16** — matches the original LLM2Vec paper's published
  configs for comparable model sizes.
- **bf16 + gradient checkpointing (MNTP stage)** — reduces memory
  footprint on a shared/single GPU.
- **`mlm_probability: 0.8` for MNTP** — deliberately high, matching the
  paper's "all_mask" collator recommendation, to force reliance on full
  bidirectional context rather than local cues.
- **Same 1024-examples-per-pair, 462-pair data as the LaBSE fine-tuning**
  — isolates architecture/method as the variable under test, not the data.
- **Temperature 0.05 for InfoNCE** — standard default; lower temperature
  sharpens separation between positive and negative pairs.

An alternative unsupervised-SimCSE config is kept at
`train_configs/simcse/sarvam1_simcse_ALTERNATIVE_not_used.json` for
anyone who wants to run that ablation directly.

---

## Known follow-ups

- No repeated-seed variance estimate — every reported number is from a
  single run per configuration.
- No standalone ablation script for examples-per-pair (500 vs. 1024) —
  supported as a one-line config change (`examples_per_directed_pair`),
  not yet wired up as a separate experiment.
- Distillation-preserving loss (to recover some of Stage 2's specificity
  trade-off) is a proposed follow-up, not implemented here.
- Task 2 currently targets only `sarvam-1` (2B); `sarvam-m` (24B) did not
  clearly outperform it in zero-shot testing, so no Approach B pipeline
  exists for it yet.
- Task 2's supervised contrastive stage runs 1 epoch by default with no
  built-in validation split yet.

## Datasets

This repository does not redistribute any dataset. Training and
evaluation use the AI4Bharat **IN22** benchmark (IN22-Gen for training,
IN22-Conv held out for evaluation), along with **FLORES-200** and
**Samanantar** for the preliminary model benchmark. Obtain these directly
from their original sources:

- IN22 — [AI4Bharat / IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
- FLORES-200 — [Meta AI / flores](https://github.com/facebookresearch/flores)
- Samanantar — [AI4Bharat Samanantar](https://ai4bharat.iitm.ac.in/samanantar)
