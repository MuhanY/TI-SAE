#!/bin/bash
# Train TISAE on all 30 backbone conditions.
# Extract layer per backbone: Discrete=-1, Discrete+PT=-3, HyMaTE=-2, STraTS=-4, BAT=-2
set -e

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
SEED="${SEED:-42}"

TASKS=(mortality los_gt7d readmission)
DATASETS=(mimic3 mimic4)

# Common TISAE hyperparameters (shared across all backbones)
TISAE_ARGS="--K 1280 --M 4 --C 10 --n_global 128 \
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

train_tisae() {
    local backbone=$1 extract_layer=$2
    for ds in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do
        local backbone_ckpt="${OUTPUT_DIR}/${backbone}/${ds}_${task}/backbone/best_model.pt"
        local out="${OUTPUT_DIR}/${backbone}/${ds}_${task}/tisae"
        [ -f "${out}/best_model.pt" ] && echo "SKIP ${out}" && continue
        python -m tisae.training.train_tisae \
            --backbone_checkpoint "${backbone_ckpt}" \
            --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
            --output_dir "${out}" \
            --extract_layer ${extract_layer} \
            ${TISAE_ARGS}
      done
    done
}

# Train per backbone with correct extract_layer (from paper's layer selection)
train_tisae "discrete"    -1
train_tisae "discrete_pt" -3
train_tisae "hymate"      -2
train_tisae "strats"      -4
train_tisae "bat"         -2

echo "All TISAE training complete."
