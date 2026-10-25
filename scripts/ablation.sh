#!/bin/bash
# Run all ablation experiments reported in the paper appendix.
set -e

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
ABLATION_DIR="${OUTPUT_DIR}/ablation"
SEED=42

# All ablations use Discrete backbone, MIMIC-III mortality as canonical condition
BACKBONE_CKPT="${OUTPUT_DIR}/discrete/mimic3_mortality/backbone/best_model.pt"
TISAE_DIR="${OUTPUT_DIR}/discrete/mimic3_mortality/tisae"
PS_DIR="${OUTPUT_DIR}/discrete/mimic3_mortality/ps"
DATASET="${DATA_DIR}/unified_mimic3_48h.pkl"

# Common TISAE args (same as train_tisae.sh minus extract_layer)
TISAE_COMMON="--K 1280 --M 4 --C 10 --n_global 128 \
    --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
    --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
    --score_mode energy_gated --gumbel_topk_noise 0.05 \
    --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
    --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
    --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
    --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
    --normalize_atoms \
    --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
    --epochs 100 --patience 15 --seed ${SEED} --diag_every 5"

# ===========================================================================
# 1. Structural ablation
# ===========================================================================

# 1a. No temporal (flat mask, no Gaussian islands)
python -m tisae.ablation.architecture.train --model tisae_no_temporal \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_temporal/tisae" \
    --K 1280 --n_global 128 --batch_size 32 --lr 1e-4 --epochs 100 --patience 15 \
    --seed ${SEED}

python -m tisae.ablation.architecture.train_ps \
    --sae_experiment_dir "${ABLATION_DIR}/structural/no_temporal/tisae" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_temporal/ps"

python -m tisae.ablation.architecture.eval \
    --experiment_dir "${ABLATION_DIR}/structural/no_temporal/tisae" \
    --ps_dir "${ABLATION_DIR}/structural/no_temporal/ps" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/eval/no_temporal" \
    --n_samples 500 --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${ABLATION_DIR}/eval/no_temporal/faithfulness" \
    --dataset_path "${DATASET}"

# 1b. No channel organization (lambda_ch_purity=0)
python -m tisae.training.train_tisae \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_channel/tisae" \
    --extract_layer -1 \
    --K 1280 --M 4 --C 10 --n_global 128 \
    --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
    --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
    --score_mode energy_gated --gumbel_topk_noise 0.05 \
    --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
    --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
    --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
    --lambda_mask_coverage 0.005 --lambda_ch_purity 0.0 \
    --normalize_atoms \
    --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
    --epochs 100 --patience 15 --seed ${SEED}

python -m tisae.training.train_pred_selector \
    --sae_experiment_dir "${ABLATION_DIR}/structural/no_channel/tisae" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_channel/ps" \
    --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
    --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
    --soft_selection --soft_tau 1.0 \
    --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

python -m tisae.evaluation.faithfulness.run_evaluation \
    --experiment_dir "${ABLATION_DIR}/structural/no_channel/tisae" \
    --pred_selector_dir "${ABLATION_DIR}/structural/no_channel/ps" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/eval/no_channel" \
    --n_samples 500 --seed ${SEED} \
    --methods sae pred_selector pred_selector_soft \
    --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${ABLATION_DIR}/eval/no_channel" \
    --dataset_path "${DATASET}"

# 1c. No variable routing (W_sparsity_k=135, all vars per latent)
python -m tisae.training.train_tisae \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_routing/tisae" \
    --extract_layer -1 \
    --K 1280 --M 4 --C 10 --n_global 128 \
    --W_sparsity_k 135 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
    --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
    --score_mode energy_gated --gumbel_topk_noise 0.05 \
    --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
    --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
    --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
    --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
    --normalize_atoms \
    --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
    --epochs 100 --patience 15 --seed ${SEED}

python -m tisae.training.train_pred_selector \
    --sae_experiment_dir "${ABLATION_DIR}/structural/no_routing/tisae" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/no_routing/ps" \
    --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
    --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
    --soft_selection --soft_tau 1.0 \
    --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

python -m tisae.evaluation.faithfulness.run_evaluation \
    --experiment_dir "${ABLATION_DIR}/structural/no_routing/tisae" \
    --pred_selector_dir "${ABLATION_DIR}/structural/no_routing/ps" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/eval/no_routing" \
    --n_samples 500 --seed ${SEED} \
    --methods sae pred_selector pred_selector_soft \
    --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${ABLATION_DIR}/eval/no_routing" \
    --dataset_path "${DATASET}"

# 1d. Standard SAE (event-level top-k, no grid)
python -m tisae.ablation.architecture.train --model simple_sae \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/simple_sae/model" \
    --K 1280 --top_k 32 --batch_size 32 --lr 1e-4 --epochs 100 --patience 15 \
    --seed ${SEED}

python -m tisae.ablation.architecture.train_ps \
    --sae_experiment_dir "${ABLATION_DIR}/structural/simple_sae/model" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/simple_sae/ps"

python -m tisae.ablation.architecture.eval \
    --experiment_dir "${ABLATION_DIR}/structural/simple_sae/model" \
    --ps_dir "${ABLATION_DIR}/structural/simple_sae/ps" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/eval/simple_sae" \
    --n_samples 500 --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${ABLATION_DIR}/eval/simple_sae/faithfulness" \
    --dataset_path "${DATASET}"

# 1e. Raw hidden states + prediction selector (no dictionary at all;
#     isolates the dictionary's contribution from the selector's)
python -m tisae.ablation.architecture.raw_hidden_ps \
    --backbone "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/raw_hidden/ps" \
    --seed ${SEED}

python -m tisae.ablation.architecture.raw_hidden_eval \
    --backbone "${BACKBONE_CKPT}" \
    --ps_dir "${ABLATION_DIR}/structural/raw_hidden/ps" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/structural/raw_hidden/eval" \
    --n_samples 500 --seed ${SEED} --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${ABLATION_DIR}/structural/raw_hidden/eval/faithfulness" \
    --dataset_path "${DATASET}"

# ===========================================================================
# 2. Dictionary capacity and sparsity
# ===========================================================================

# 2a. M sweep (M=1,2,8 with proportional n_global=32M)
for M in 1 2 8; do
  NG=$((32 * M))
  python -m tisae.training.train_tisae \
      --backbone_checkpoint "${BACKBONE_CKPT}" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/m_sweep/m${M}/tisae" \
      --extract_layer -1 \
      --K 1280 --M ${M} --C 10 --n_global ${NG} \
      --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
      --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
      --score_mode energy_gated --gumbel_topk_noise 0.05 \
      --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
      --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
      --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
      --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
      --normalize_atoms \
      --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
      --epochs 100 --patience 15 --seed ${SEED}

  python -m tisae.training.train_pred_selector \
      --sae_experiment_dir "${ABLATION_DIR}/m_sweep/m${M}/tisae" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/m_sweep/m${M}/ps" \
      --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
      --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
      --soft_selection --soft_tau 1.0 \
      --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

  python -m tisae.evaluation.faithfulness.run_evaluation \
      --experiment_dir "${ABLATION_DIR}/m_sweep/m${M}/tisae" \
      --pred_selector_dir "${ABLATION_DIR}/m_sweep/m${M}/ps" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/eval/m${M}" \
      --n_samples 500 --seed ${SEED} \
      --methods sae pred_selector pred_selector_soft \
      --save_attributions

  python -m tisae.evaluation.plausibility.run_plausibility \
      --eval_dir "${ABLATION_DIR}/eval/m${M}" \
      --dataset_path "${DATASET}"
done

# 2b. n_global sweep (ng=32, 64, 256, 512)
for NG in 32 64 256 512; do
  python -m tisae.training.train_tisae \
      --backbone_checkpoint "${BACKBONE_CKPT}" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/ng_sweep/ng${NG}/tisae" \
      --extract_layer -1 \
      --K 1280 --M 4 --C 10 --n_global ${NG} \
      --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
      --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
      --score_mode energy_gated --gumbel_topk_noise 0.05 \
      --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
      --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
      --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
      --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
      --normalize_atoms \
      --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
      --epochs 100 --patience 15 --seed ${SEED}

  python -m tisae.training.train_pred_selector \
      --sae_experiment_dir "${ABLATION_DIR}/ng_sweep/ng${NG}/tisae" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/ng_sweep/ng${NG}/ps" \
      --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
      --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
      --soft_selection --soft_tau 1.0 \
      --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

  python -m tisae.evaluation.faithfulness.run_evaluation \
      --experiment_dir "${ABLATION_DIR}/ng_sweep/ng${NG}/tisae" \
      --pred_selector_dir "${ABLATION_DIR}/ng_sweep/ng${NG}/ps" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/eval/ng${NG}" \
      --n_samples 500 --seed ${SEED} \
      --methods sae pred_selector pred_selector_soft \
      --save_attributions

  python -m tisae.evaluation.plausibility.run_plausibility \
      --eval_dir "${ABLATION_DIR}/eval/ng${NG}" \
      --dataset_path "${DATASET}"
done

# 2c. K sweep (K=320, 640, 2560)
for K in 320 640 2560; do
  GC_FLAG=""
  if [ ${K} -ge 2560 ]; then GC_FLAG="--gradient_checkpointing"; fi
  python -m tisae.training.train_tisae \
      --backbone_checkpoint "${BACKBONE_CKPT}" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/k_sweep/k${K}/tisae" \
      --extract_layer -1 \
      --K ${K} --M 4 --C 10 --n_global 128 \
      --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
      --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
      --score_mode energy_gated --gumbel_topk_noise 0.05 \
      --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
      --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
      --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
      --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
      --normalize_atoms \
      --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
      --epochs 100 --patience 15 --seed ${SEED} ${GC_FLAG}

  python -m tisae.training.train_pred_selector \
      --sae_experiment_dir "${ABLATION_DIR}/k_sweep/k${K}/tisae" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/k_sweep/k${K}/ps" \
      --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
      --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
      --soft_selection --soft_tau 1.0 \
      --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

  python -m tisae.evaluation.faithfulness.run_evaluation \
      --experiment_dir "${ABLATION_DIR}/k_sweep/k${K}/tisae" \
      --pred_selector_dir "${ABLATION_DIR}/k_sweep/k${K}/ps" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/eval/k${K}" \
      --n_samples 500 --seed ${SEED} \
      --methods sae pred_selector pred_selector_soft \
      --save_attributions

  python -m tisae.evaluation.plausibility.run_plausibility \
      --eval_dir "${ABLATION_DIR}/eval/k${K}" \
      --dataset_path "${DATASET}"
done

# ===========================================================================
# 3. Loss component ablation
# ===========================================================================

declare -A LOSS_ABLATIONS
LOSS_ABLATIONS[no_div]="--lambda_div_W 0.0"
LOSS_ABLATIONS[no_mu_repulsion]="--lambda_mu_repulsion 0.0"
LOSS_ABLATIONS[no_auxk]="--lambda_auxk 0.0"
LOSS_ABLATIONS[no_mcov]="--lambda_mask_coverage 0.0"

for name in no_div no_mu_repulsion no_auxk no_mcov; do
  override="${LOSS_ABLATIONS[$name]}"
  python -m tisae.training.train_tisae \
      --backbone_checkpoint "${BACKBONE_CKPT}" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/loss_components/${name}/tisae" \
      --extract_layer -1 \
      --K 1280 --M 4 --C 10 --n_global 128 \
      --W_sparsity_k 3 --encoder_sparsity_k 5 --encoder_chunk_size 32 \
      --island_hidden 32 --island_kernel_size 25 --n_sigma_iter 3 \
      --score_mode energy_gated --gumbel_topk_noise 0.05 \
      --dead_boost_scale 0.5 --dead_threshold_batches 5 --auxk_k 256 \
      --lambda_recon 1.0 --lambda_pred 0.0 --lambda_auxk 0.3 \
      --lambda_mu_repulsion 0.01 --lambda_div_W 0.01 \
      --lambda_mask_coverage 0.005 --lambda_ch_purity 0.1 \
      --normalize_atoms \
      --batch_size 32 --lr 1e-4 --weight_decay 0.01 \
      --epochs 100 --patience 15 --seed ${SEED} \
      ${override}

  python -m tisae.training.train_pred_selector \
      --sae_experiment_dir "${ABLATION_DIR}/loss_components/${name}/tisae" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/loss_components/${name}/ps" \
      --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
      --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
      --soft_selection --soft_tau 1.0 \
      --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

  python -m tisae.evaluation.faithfulness.run_evaluation \
      --experiment_dir "${ABLATION_DIR}/loss_components/${name}/tisae" \
      --pred_selector_dir "${ABLATION_DIR}/loss_components/${name}/ps" \
      --dataset_path "${DATASET}" \
      --output_dir "${ABLATION_DIR}/eval/${name}" \
      --n_samples 500 --seed ${SEED} \
      --methods sae pred_selector pred_selector_soft \
      --save_attributions

  python -m tisae.evaluation.plausibility.run_plausibility \
      --eval_dir "${ABLATION_DIR}/eval/${name}" \
      --dataset_path "${DATASET}"
done

# ===========================================================================
# 4. Layer selection
#    All backbones × all available layers (MIMIC-III mortality)
# ===========================================================================

train_layer() {
    local backbone=$1 ckpt=$2 layer=$3
    local out="${ABLATION_DIR}/layer_selection/${backbone}/layer${layer}"
    python -m tisae.training.train_tisae \
        --backbone_checkpoint "${ckpt}" \
        --dataset_path "${DATASET}" \
        --output_dir "${out}/tisae" \
        --extract_layer ${layer} \
        ${TISAE_COMMON}
    python -m tisae.training.train_pred_selector \
        --sae_experiment_dir "${out}/tisae" \
        --dataset_path "${DATASET}" \
        --output_dir "${out}/ps" \
        --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
        --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
        --soft_selection --soft_tau 1.0 \
        --batch_size 52 --lr 1e-3 --epochs 100 --patience 20 --seed ${SEED}

    python -m tisae.evaluation.faithfulness.run_evaluation \
        --experiment_dir "${out}/tisae" \
        --pred_selector_dir "${out}/ps" \
        --dataset_path "${DATASET}" \
        --output_dir "${out}/eval" \
        --n_samples 500 --seed ${SEED} \
        --methods sae pred_selector pred_selector_soft \
        --save_attributions

    python -m tisae.evaluation.plausibility.run_plausibility \
        --eval_dir "${out}/eval" \
        --dataset_path "${DATASET}"
}

# Discrete: 4 layers
for LAYER in -1 -2 -3 -4; do
  train_layer "discrete" "${BACKBONE_CKPT}" ${LAYER}
done

# Disc+PT: 4 layers
for LAYER in -1 -2 -3 -4; do
  train_layer "discrete_pt" "${OUTPUT_DIR}/discrete_pt/mimic3_mortality/backbone/best_model.pt" ${LAYER}
done

# HyMaTE: 3 layers (2 mamba blocks + input)
for LAYER in -1 -2 -3; do
  train_layer "hymate" "${OUTPUT_DIR}/hymate/mimic3_mortality/backbone/best_model.pt" ${LAYER}
done

# STraTS: 4 layers
for LAYER in -1 -2 -3 -4; do
  train_layer "strats" "${OUTPUT_DIR}/strats/mimic3_mortality/backbone/best_model.pt" ${LAYER}
done

# BAT: 2 layers (1 layer per track)
for LAYER in -1 -2; do
  train_layer "bat" "${OUTPUT_DIR}/bat/mimic3_mortality/backbone/best_model.pt" ${LAYER}
done

# ===========================================================================
# 5. Concept knockout
# ===========================================================================

python -m tisae.ablation.knockout.concept_knockout \
    --experiment_dir "${TISAE_DIR}" \
    --pred_selector_dir "${PS_DIR}" \
    --dataset_path "${DATASET}" \
    --n_samples 500 --seed ${SEED} \
    --output_dir "${ABLATION_DIR}/concept_knockout"

# ===========================================================================
# 6. Seed reproducibility
#    6a: joint retraining of both stages per seed (3 seeds)
#    6b: dictionary-only, de-confounded (5 seeds, selector excluded)
#    6c: selector-only on the fixed seed-42 dictionary (5 seeds)
# ===========================================================================

# 6a. Joint retraining (dictionary + selector per seed) + two-stage matching
python -m tisae.ablation.robustness.seed_reproducibility train \
    --reference_dir "${TISAE_DIR}" \
    --ps_reference_dir "${PS_DIR}" \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/seed_stability" \
    --seeds 43 44

python -m tisae.ablation.robustness.seed_reproducibility compare \
    --reference_dir "${TISAE_DIR}" \
    --ps_reference_dir "${PS_DIR}" \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/seed_stability" \
    --seeds 43 44

# 6b. Dictionary-only reproducibility across five seeds (selector excluded).
#     Trains the two additional seed dictionaries, then matches all five
#     (42 = the canonical run) with the same two-stage alignment.
python -m tisae.ablation.robustness.seed_reproducibility train \
    --reference_dir "${TISAE_DIR}" \
    --ps_reference_dir "${PS_DIR}" \
    --backbone_checkpoint "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --output_dir "${ABLATION_DIR}/seed_stability" \
    --seeds 45 46

python -m tisae.ablation.robustness.dict_activation_metrics \
    --backbone "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --ref 42="${TISAE_DIR}" \
    --seeds 43="${ABLATION_DIR}/seed_stability/seed_43/tisae" \
            44="${ABLATION_DIR}/seed_stability/seed_44/tisae" \
            45="${ABLATION_DIR}/seed_stability/seed_45/tisae" \
            46="${ABLATION_DIR}/seed_stability/seed_46/tisae" \
    --n_samples 500 \
    --output "${ABLATION_DIR}/seed_stability/dict_full_metrics.json"

# 6c. Selector-only stability: retrain the prediction selector over five seeds
#     on the FIXED canonical dictionary, then measure per-patient consistency
#     of the selected concepts (top-k Jaccard, Spearman) vs a random null.
SEL_DIR="${ABLATION_DIR}/selector_stability"
for S in 42 43 44 45 46; do
  out="${SEL_DIR}/seed_${S}/ps"
  [ -f "${out}/best_model.pt" ] && echo "SKIP ${out}" && continue
  python -m tisae.training.train_pred_selector \
      --sae_experiment_dir "${TISAE_DIR}" \
      --dataset_path "${DATASET}" \
      --output_dir "${out}" \
      --n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
      --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
      --lambda_prior 0.0 --prior_tau 1.0 \
      --gumbel_topk_noise 0.0 \
      --soft_selection --soft_tau 1.0 \
      --batch_size 52 --lr 1e-3 --weight_decay 0.0 \
      --epochs 100 --patience 20 --seed ${S}
done

python -m tisae.ablation.robustness.selector_stability \
    --sae_dir "${TISAE_DIR}" \
    --backbone "${BACKBONE_CKPT}" \
    --dataset_path "${DATASET}" \
    --ps original="${PS_DIR}" \
         42="${SEL_DIR}/seed_42/ps" \
         43="${SEL_DIR}/seed_43/ps" \
         44="${SEL_DIR}/seed_44/ps" \
         45="${SEL_DIR}/seed_45/ps" \
         46="${SEL_DIR}/seed_46/ps" \
    --n_samples 500 \
    --output "${SEL_DIR}/stability_results.json"

# ===========================================================================
# 7. Cross-dataset transfer to a strictly disjoint MIMIC-IV cohort
#    (frozen MIMIC-III model on MIMIC-IV test stays with anchor_year_group
#     >= 2014, entirely after MIMIC-III's 2012 cutoff -> no patient or
#     temporal overlap)
# ===========================================================================

OOD_DIR="${ABLATION_DIR}/disjoint_transfer"

# 7a. Remap MIMIC-IV into the MIMIC-III variable space
python -m tisae.ablation.transfer.remap_dataset \
    --source "${DATA_DIR}/unified_mimic3_48h.pkl" \
    --target "${DATA_DIR}/unified_mimic4_48h.pkl" \
    --output "${DATA_DIR}/unified_mimic4_48h_m3space.pkl"

# 7b. Reconstruct sample->stay mapping and join anchor_year_group
#     (reads icustays/patients from MIMIC4_DIR; emits disjoint/overlap indices)
python -m tisae.ablation.transfer.nonoverlap_mimic --stage reconstruct \
    --raw "${DATA_DIR}/raw_mimic4/mimic_iv_48h.pkl" \
    --output_dir "${OOD_DIR}"

# 7c. Frozen MIMIC-III model: prediction AUROC on full / disjoint / overlap
#     (full index sets, no subsampling)
python -m tisae.ablation.transfer.nonoverlap_mimic --stage predict \
    --experiment_dir "${TISAE_DIR}" \
    --source_dataset "${DATASET}" \
    --target_dataset "${DATA_DIR}/unified_mimic4_48h.pkl" \
    --output_dir "${OOD_DIR}"

# 7d. Derive m3-space pickles whose test split is the disjoint / overlap subset
python -m tisae.ablation.transfer.nonoverlap_mimic --stage derive_pickles \
    --m3space "${DATA_DIR}/unified_mimic4_48h_m3space.pkl" \
    --output_dir "${OOD_DIR}"

# 7e. Faithfulness + plausibility on the disjoint cohort (stock evaluation;
#     the frozen MIMIC-III model consumes the m3-space pickle directly)
python -m tisae.evaluation.faithfulness.run_evaluation \
    --experiment_dir "${TISAE_DIR}" \
    --pred_selector_dir "${PS_DIR}" \
    --dataset_path "${DATA_DIR}/unified_mimic4_48h_m3space_disjoint.pkl" \
    --output_dir "${OOD_DIR}/eval_disjoint" \
    --n_samples 500 --seed ${SEED} \
    --methods random attention ig tsr dynamask timeshap winit_removal pred_selector pred_selector_soft \
    --captum_baseline median \
    --save_attributions

python -m tisae.evaluation.plausibility.run_plausibility \
    --eval_dir "${OOD_DIR}/eval_disjoint" \
    --dataset_path "${DATA_DIR}/unified_mimic4_48h_m3space_disjoint.pkl"

# 7f. Concept knockout on the disjoint cohort + figure
python -m tisae.ablation.knockout.concept_knockout \
    --experiment_dir "${TISAE_DIR}" \
    --pred_selector_dir "${PS_DIR}" \
    --dataset_path "${DATA_DIR}/unified_mimic4_48h_m3space_disjoint.pkl" \
    --n_samples 500 --seed ${SEED} \
    --output_dir "${OOD_DIR}/knockout_disjoint"

python -m tisae.visualization.concept_knockout_fig \
    --results "${OOD_DIR}/knockout_disjoint/knockout_results.json" \
    --output "${OOD_DIR}/knockout_disjoint/knockout_ood.png"

# ===========================================================================
# 8. Island stability (reconstruction and stability under event dropout)
# ===========================================================================

python -m tisae.evaluation.dictionary.stability \
    --experiment_dir "${TISAE_DIR}" \
    --dataset_path "${DATASET}" \
    --n_samples 500 --seed ${SEED}

# ===========================================================================
# 9. Island window shape (asymmetric vs symmetric Gaussian)
#    Learned-skew and fixed-skew (a=0.5) variants over seeds 42-44. The
#    symmetric row is the canonical pipeline at the same seeds: seed 42 =
#    the main experiment's eval; seeds 43/44 are evaluated below from the
#    seed-stability retrains.
# ===========================================================================

ISL_DIR="${ABLATION_DIR}/island_shape"
for S in 42 43 44; do
  for SHAPE in learned a05; do
    out="${ISL_DIR}/seed_${S}/${SHAPE}"
    EXTRA=""
    [ "${SHAPE}" = "a05" ] && EXTRA="--fix_a 0.5"
    python -m tisae.ablation.architecture.train_asym \
        --backbone_checkpoint "${BACKBONE_CKPT}" \
        --dataset_path "${DATASET}" \
        --canonical_config "${TISAE_DIR}/config.yaml" \
        --output_dir "${out}/tisae" \
        --seed ${S} ${EXTRA}

    python -m tisae.ablation.architecture.ps_asym \
        --sae_experiment_dir "${out}/tisae" \
        --dataset_path "${DATASET}" \
        --ps_config "${PS_DIR}/config.yaml" \
        --output_dir "${out}/ps" \
        --seed ${S}

    python -m tisae.ablation.architecture.eval_asym \
        --sae_experiment_dir "${out}/tisae" \
        --ps_dir "${out}/ps" \
        --dataset_path "${DATASET}" \
        --output_dir "${out}/eval" \
        --n_samples 500 --seed ${SEED}

    python -m tisae.evaluation.plausibility.run_plausibility \
        --eval_dir "${out}/eval" \
        --dataset_path "${DATASET}"
  done
done

for S in 43 44; do
  python -m tisae.evaluation.faithfulness.run_evaluation \
      --experiment_dir "${ABLATION_DIR}/seed_stability/seed_${S}/tisae" \
      --pred_selector_dir "${ABLATION_DIR}/seed_stability/seed_${S}/ps" \
      --dataset_path "${DATASET}" \
      --output_dir "${ISL_DIR}/seed_${S}/symmetric_eval" \
      --n_samples 500 --seed ${SEED} \
      --methods pred_selector_soft \
      --save_attributions

  python -m tisae.evaluation.plausibility.run_plausibility \
      --eval_dir "${ISL_DIR}/seed_${S}/symmetric_eval" \
      --dataset_path "${DATASET}"
done

# Windows-per-concept distribution + single-Gaussian fit of realized masks
python -m tisae.ablation.mask_shapes \
    --experiment_dir "${TISAE_DIR}" \
    --ps_dir "${PS_DIR}" \
    --dataset_path "${DATASET}" \
    --n_episodes 300 \
    --out "${ISL_DIR}/mask_shapes.json"

# ===========================================================================
# 10. Dictionary structure (separability / groupings / outcome significance)
#     Requires the latent dictionary built by tisae.dictionary.dictionary_builder
#     on the canonical condition.
# ===========================================================================

python -m tisae.ablation.dict_structure_analysis \
    --experiment_dir "${TISAE_DIR}" \
    --dataset "${DATASET}" \
    --dictionary "${OUTPUT_DIR}/discrete/mimic3_mortality/dictionary/latent_dictionary.json" \
    | tee "${ABLATION_DIR}/dict_structure.txt"

echo "All ablation experiments complete."
