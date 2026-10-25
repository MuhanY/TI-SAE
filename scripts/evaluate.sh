#!/bin/bash
# Run faithfulness + plausibility evaluation on all 30 conditions.
set -e

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
SEED="${SEED:-42}"
N_SAMPLES=500

BACKBONES=(discrete discrete_pt strats hymate bat)
TASKS=(mortality los_gt7d readmission)
DATASETS=(mimic3 mimic4)

METHODS="random attention ig gradient_shap tsr sae pred_selector pred_selector_soft dynamask timeshap fit_removal winit_removal concept"
FRACTIONS="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"

for backbone in "${BACKBONES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
      exp_dir="${OUTPUT_DIR}/${backbone}/${ds}_${task}/tisae"
      ps_dir="${OUTPUT_DIR}/${backbone}/${ds}_${task}/ps"
      eval_dir="${OUTPUT_DIR}/${backbone}/${ds}_${task}/eval"

      # Skip if already evaluated
      [ -f "${eval_dir}/evaluation_results.json" ] && echo "SKIP ${eval_dir}" && continue

      # Faithfulness evaluation
      python -m tisae.evaluation.faithfulness.run_evaluation \
          --experiment_dir "${exp_dir}" \
          --pred_selector_dir "${ps_dir}" \
          --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
          --output_dir "${eval_dir}" \
          --n_samples ${N_SAMPLES} --seed ${SEED} \
          --methods ${METHODS} \
          --fractions ${FRACTIONS} \
          --captum_baseline median \
          --save_attributions

      # Plausibility evaluation (on saved attributions)
      python -m tisae.evaluation.plausibility.run_plausibility \
          --eval_dir "${eval_dir}" \
          --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl"
    done
  done
done

echo "All evaluations complete."
