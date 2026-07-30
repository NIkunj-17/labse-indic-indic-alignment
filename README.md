# LaBSE Indic-Indic Fine-tuning — Research Pipeline (Task 1: full 1024 examples/pair)

Clean, modular research code for balanced (Phase 1) and weak-pair weighted (Phase 2)
fine-tuning of LaBSE across all 462 directed Indic-Indic pairs, evaluated on unseen
IN22-Conv. Replaces the earlier notebook prototype with proper package structure,
unit tests, and reproducible configs — suitable as the basis for a paper.

## Project layout

```
labse_research/
├── src/labse_research/
│   ├── config.py       # DataConfig / TrainConfig / EvalConfig dataclasses
│   ├── data.py         # IN22-Gen / IN22-Conv loading, directed-pair construction
│   ├── metrics.py       # cosine gap, sensitivity/specificity, accuracy@1, MRR
│   ├── weighting.py     # pair-weakness scoring, weight computation (Phase 2)
│   ├── training.py       # shared training loop: checkpoint/resume, val-driven best-model
│   └── evaluation.py    # cross-model eval, summary table, per-pair deltas, plots
├── scripts/
│   ├── run_phase1.py          # Phase 1 entry point
│   ├── run_phase2.py          # Phase 2 entry point
│   ├── run_evaluate.py         # Evaluation entry point
│   └── run_full_pipeline.sh   # Runs all three in sequence
├── configs/
│   └── phase1_1024.json       # Example config: full 1024 examples/pair
├── tests/
│   ├── test_metrics.py         # Unit tests, synthetic data, no GPU/network needed
│   └── test_weighting.py
├── requirements.txt
└── pyproject.toml
```

## Setup on the GPU machine

You already have a venv at `~/labse_pipeline/venv` from the earlier notebook work —
reuse it.

```bash
ssh -L 8888:localhost:8888 -i ~/.ssh/wsai_iitm_project nikunj@216.48.185.113
cd ~/labse_pipeline
source venv/bin/activate

# Copy/extract this project here (see "Getting the code onto the machine" below)
cd labse_research
pip install -r requirements.txt
pip install -e .          # editable install so `import labse_research` works anywhere

hf auth login              # if not already logged in from before
```

### Getting the code onto the machine

From your **local** PowerShell (a separate window, not the SSH session):

```powershell
scp -i $env:USERPROFILE\.ssh\wsai_iitm_project -r .\labse_research nikunj@216.48.185.113:~/labse_pipeline/
```

(Adjust the local source path to wherever you unzip the project on your laptop.)

## Running the tests (sanity check before a multi-hour run)

```bash
cd ~/labse_pipeline/labse_research
python3 -m pytest tests/ -v
```

These use only synthetic embeddings — no GPU, no model download, no dataset
download. They should finish in under 2 seconds and all pass before you trust
the pipeline with real data.

## Running the full pipeline (Task 1: 1024 examples/pair)

Use tmux so the run survives SSH/WiFi disconnects, exactly like before:

```bash
tmux new -s task1_full_run
cd ~/labse_pipeline/labse_research
source ../venv/bin/activate
export OUTPUT_ROOT=~/labse_pipeline/labse_pipeline_500/labse_research_output
bash scripts/run_full_pipeline.sh
```

Detach with `Ctrl+B` then `D`. Reattach anytime with:
```bash
tmux attach -t task1_full_run
```

### Expected runtime

With 1024 examples/pair (up from 500), total training examples roughly double
to ~425,000 for the combined 462-pair run. At batch size 32 this is a
significantly longer run than your earlier 500-example experiments — budget
for a multi-hour run per phase (Phase 1 + Phase 2), likely spanning many
hours to a day depending on GPU availability on the shared A100. Checkpoints
save every 500 steps (configurable), so an interrupted run resumes
automatically from the last checkpoint on re-run — no progress is lost.

### Running phases individually (if you don't want the combined script)

```bash
# Phase 1
python3 scripts/run_phase1.py --config configs/phase1_1024.json \
    --output-root "$OUTPUT_ROOT"

# Phase 2 (after Phase 1 finishes)
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

## Where results land

```
$OUTPUT_ROOT/
├── phase1_balanced_1024/
│   ├── best_model/              # SentenceTransformer checkpoint, load directly
│   ├── final_model/
│   ├── checkpoints/              # intermediate checkpoints, auto-pruned
│   ├── logs/train_metrics.csv    # step, loss, val_cosine_gap over training
│   └── training_config.json      # exact hyperparameters used, for reproducibility
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

## Design notes (for the paper's methodology section)

- **In-batch negatives (MultipleNegativesRankingLoss)**: every other example in
  a training batch acts as an automatic negative for a given anchor, so batch
  size directly controls the number of negatives seen per step (batch_size - 1).
  This is documented explicitly in `training.py` since it materially affects
  reported metrics and must be held constant across any models you compare.
- **Phase 2 weighting is continuous, not a hard cutoff**: every pair gets a
  weight in `[min_pair_weight, max_pair_weight]` based on a blended quality
  score (50% cosine gap, 30% accuracy@1, 20% specificity from Phase 1),
  inverted and rescaled. See `weighting.compute_pair_weights`.
- **Specificity-guarded checkpoint selection**: Phase 2's best-model selection
  doesn't just maximize val_cosine_gap — it subtracts a penalty if
  specificity drops below the Phase 1 reference, discouraging the model
  from trading away precision purely for a higher gap
  (`training.run_training`, `reference_val_specificity` argument).
- **Reproducibility**: every run's exact hyperparameters are saved to
  `training_config.json` alongside its outputs. No hyperparameter lives only
  in a script argument that isn't logged.

## Known follow-ups (not yet implemented — flag if the paper needs them)

- No repeated-seed variance estimate yet (single run per config). Consider
  running Phase 1/2 with 2-3 seeds if reviewers will ask about significance.
- No ablation yet on examples-per-pair (500 vs 1024) — this codebase makes
  that a one-line config change (`examples_per_directed_pair`) if useful.
- Phase 3 (distillation-preserving loss) is not part of this codebase; it is
  a separate task per the current roadmap.
