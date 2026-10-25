#!/bin/bash
# Train PredSelector on all 30 TISAE experiments.
set -e

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
SEED="${SEED:-42}"

BACKBONES=(discrete discrete_pt strats hymate bat)
TASKS=(mortality los_gt7d readmission)
DATASETS=(mimic3 mimic4)

PS_ARGS="--n_global_pred 64 --d_sel 64 --mlp_hidden 64 \
    --lambda_logit 1.0 --lambda_comp 0.1 --delta_comp 0.5 \
    --lambda_prior 0.0 --prior_tau 1.0 \
    --gumbel_topk_noise 0.0 \
    --soft_selection --soft_tau 1.0 \
    --batch_size 52 --lr 1e-3 --weight_decay 0.0 \
    --epochs 100 --patience 20 --seed ${SEED}"

for backbone in "${BACKBONES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
      sae_dir="${OUTPUT_DIR}/${backbone}/${ds}_${task}/tisae"
      out="${OUTPUT_DIR}/${backbone}/${ds}_${task}/ps"
      [ -f "${out}/best_model.pt" ] && echo "SKIP ${out}" && continue
      python -m tisae.training.train_pred_selector \
          --sae_experiment_dir "${sae_dir}" \
          --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
          --output_dir "${out}" \
          ${PS_ARGS}
    done
  done
done

echo "All PredSelector training complete."
