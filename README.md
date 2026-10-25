# TI-SAE: Temporal Island Sparse Autoencoders for Clinical Time Series

TI-SAE learns a backbone-specific, patient-shared clinical concept dictionary from a frozen predictive model's representations. Each dictionary entry (latent) is defined by learned temporal Gaussian islands — specifying *when* the concept is active — and a sparse variable-composition vector — specifying *which variables* are involved. A decoupled prediction-aware selector then identifies which dictionary entries drive each patient's prediction, separating representation learning from attribution.

## Installation

```bash
# Recommended: the tested configuration (Python 3.10, CUDA 12.4)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -e .

# For HyMaTE backbone (requires sm>=70 GPU; built against the torch above;
pip install "mamba-ssm==2.2.4" "causal-conv1d==1.5.0.post8" "transformers>=4.40,<4.50" --no-build-isolation

# For baseline attribution methods:
pip install -e ".[baselines]"

# For LLM-based dictionary labeling and verbalization evaluation:
pip install -e ".[llm]"

# Additional baselines (install separately):
pip install tsinterpret
pip install git+https://github.com/mims-harvard/TimeX.git
```

### Exact reproduction environment

`requirements-frozen.txt` pins the exact package versions used for the paper's
results (Linux, Python 3.10, CUDA 12.4). In a fresh virtual environment, in order:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-frozen.txt
pip install tsinterpret==0.4.7 --no-deps
pip install "mamba-ssm==2.2.4" "causal-conv1d==1.5.0.post8" --no-build-isolation  # HyMaTE only
pip install -e .
pip install -e ".[llm]"   # only for the LLM labeling / verbalization steps
```

## Data Preparation

PhysioNet access is required.

```bash
export MIMIC3_DIR=/path/to/mimiciii/1.4
export MIMIC4_DIR=/path/to/mimiciv/3.1
bash scripts/preprocess.sh
```

This produces unified pickles (`data/unified_mimic3_48h.pkl`, etc.) that all training scripts consume.

## Training

The full pipeline has three stages: backbone, TI-SAE, and prediction selector. Each script trains all 30 conditions (5 backbones × 2 datasets × 3 tasks).

```bash
# Stage 1: Train backbones (Discrete, Discrete+PT, STraTS, HyMaTE, BAT)
bash scripts/train_backbone.sh

# Stage 2: Train TI-SAE on each backbone
bash scripts/train_tisae.sh

# Stage 3: Train prediction selector on each TI-SAE
bash scripts/train_pred_selector.sh
```

### Single condition example

```bash
# Discrete backbone, MIMIC-III mortality
python -m tisae.training.train_backbone \
    --dataset_path data/unified_mimic3_48h.pkl \
    --output_dir experiments/discrete/mimic3_mortality/backbone \
    --task mortality \
    --hid_dim 128 --num_layers 4 --num_heads 8 \
    --dropout 0.3 --lr 1e-4 --weight_decay 5e-3 \
    --batch_size 64 --eval_batch_size 64 \
    --epochs 50 --patience 10 --flash_attention --seed 42

# TI-SAE
python -m tisae.training.train_tisae \
    --backbone_checkpoint experiments/discrete/mimic3_mortality/backbone/best_model.pt \
    --dataset_path data/unified_mimic3_48h.pkl \
    --output_dir experiments/discrete/mimic3_mortality/tisae \
    --extract_layer -1 \
    --K 1280 --M 4 --C 10 --n_global 128 \
    --batch_size 32 --lr 1e-4 --epochs 100 --patience 15 --seed 42

# Prediction selector
python -m tisae.training.train_pred_selector \
    --sae_experiment_dir experiments/discrete/mimic3_mortality/tisae \
    --dataset_path data/unified_mimic3_48h.pkl \
    --output_dir experiments/discrete/mimic3_mortality/ps \
    --n_global_pred 64 --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed 42
```

## Evaluation

```bash
# Run faithfulness + plausibility on all 30 conditions
bash scripts/evaluate.sh
```

### Single condition example

```bash
# Faithfulness
python -m tisae.evaluation.faithfulness.run_evaluation \
    --experiment_dir experiments/discrete/mimic3_mortality/tisae \
    --pred_selector_dir experiments/discrete/mimic3_mortality/ps \
    --dataset_path data/unified_mimic3_48h.pkl \
    --output_dir experiments/discrete/mimic3_mortality/eval \
    --n_samples 500 --captum_baseline median --save_attributions \
    --methods attention ig gradient_shap tsr pred_selector_soft dynamask timeshap

# Plausibility (SOFA-derived clinical prior)
python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir experiments/discrete/mimic3_mortality/eval \
    --dataset_path data/unified_mimic3_48h.pkl
```

## Latent Dictionary

Builds the patient-shared concept dictionary from a trained TI-SAE and labels it with an LLM.

```bash
# 1. Per-latent characterization (exemplar episodes, variable composition,
#    temporal profile, outcome odds ratio)
python -m tisae.dictionary.dictionary_builder \
    --experiment_dir experiments/discrete/mimic3_mortality/tisae \
    --pred_selector_dir experiments/discrete/mimic3_mortality/ps \
    --dataset_path data/unified_mimic3_48h.pkl
# -> experiments/.../tisae/latent_dictionary/latent_dictionary.json

# 2. LLM labeling (requires ANTHROPIC_API_KEY or OPENAI_API_KEY; progress is saved
#    per latent, so re-running skips already-labeled latents; --dry_run prints
#    prompts without calling the API)
python -m tisae.dictionary.llm_api \
    --experiment_dir experiments/discrete/mimic3_mortality/tisae \
    --provider claude --model claude-sonnet-4-6
# -> llm_labels.json next to the dictionary, then merged into latent_dictionary.json
```

## Verbalization Evaluation

LLM-based comparison of concept-level vs cell-level explanations. Requires
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, attributions saved by the evaluation step,
and a labeled dictionary.

```bash
# Pairwise verbalization comparison (extract -> verbalize -> judge -> aggregate)
python -m tisae.evaluation.verbalization.run_verbalization \
    --experiment_dir experiments/discrete/mimic3_mortality/tisae \
    --ps_dir experiments/discrete/mimic3_mortality/ps \
    --dataset_path data/unified_mimic3_48h.pkl \
    --attribution_dir experiments/discrete/mimic3_mortality/eval/attributions \
    --dictionary_path experiments/discrete/mimic3_mortality/tisae/latent_dictionary/latent_dictionary.json \
    --output_dir experiments/verbalization
```

## Ablation Studies

```bash
bash scripts/ablation.sh
```

Covers paper appendix experiments: structural mechanism ablation, M/K/n_global sweeps, loss component ablation, layer selection, concept knockout, seed reproducibility (joint, dictionary-only, and selector-only), disjoint-cohort cross-dataset transfer, island stability under event dropout, island window shape (asymmetric Gaussians), and dictionary structure analysis. The structural, sweep, loss-component, layer-selection, transfer, and island-shape sections evaluate faithfulness and plausibility on their trained models; the remaining sections compute their own metrics.

The disjoint-transfer section additionally reads the raw MIMIC-IV tables (set `MIMIC4_DIR`, as in Data Preparation).

## Project Structure

```
src/tisae/
├── data/               # MIMIC preprocessing and dataset classes
├── models/             # TI-SAE, PredSelector, backbone, islands, ontology
│   └── external/       # Ported backbones (STraTS, HyMaTE, BAT)
├── training/           # Training scripts (backbone, TI-SAE, PredSelector)
├── evaluation/
│   ├── faithfulness/   # Attribution methods (16 methods) and AOPC evaluation
│   ├── plausibility/   # SOFA-derived clinical plausibility
│   ├── dictionary/     # Island stability metrics
│   └── verbalization/  # LLM-based explanation evaluation
├── dictionary/         # Latent dictionary builder and LLM labeler
├── ablation/           # architecture/, robustness/, transfer/, knockout/ + structure analyses
├── visualization/      # Paper figure generation
└── utils/              # Grid operations, seeding
scripts/
├── preprocess.sh       # MIMIC-III/IV → unified pickles
├── train_backbone.sh   # All 5 backbones × 2 datasets × 3 tasks
├── train_tisae.sh      # TI-SAE on all backbone checkpoints
├── train_pred_selector.sh  # PredSelector on all TI-SAE models
├── evaluate.sh         # Faithfulness + plausibility evaluation
└── ablation.sh         # All ablation experiments
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yeo2026tisae,
  title     = {Temporal Island Sparse Autoencoders for Interpreting Clinical Time-Series Models},
  author    = {Yeo, Muhan and Kang, Bong Gyun and Kim, HyunGi and Yoon, Sungroh},
  booktitle = {Proceedings of the Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
