#!/bin/bash
# Train all backbone models (5 backbones × 2 datasets × 3 tasks = 30 conditions).
set -e

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
SEED=42

MIMIC3="${DATA_DIR}/unified_mimic3_48h.pkl"
MIMIC4="${DATA_DIR}/unified_mimic4_48h.pkl"
MIMIC3_240="${DATA_DIR}/unified_mimic3_240h.pkl"
MIMIC4_240="${DATA_DIR}/unified_mimic4_240h.pkl"

TASKS=(mortality los_gt7d readmission)

# ===========================================================================
# 1. Discrete backbone (supervised, no pretrain)
# ===========================================================================
for task in "${TASKS[@]}"; do
  for ds in mimic3 mimic4; do
    dataset_var="${ds^^}"  # MIMIC3 or MIMIC4
    dataset_path="${!dataset_var:-${DATA_DIR}/unified_${ds}_48h.pkl}"
    OUT="${OUTPUT_DIR}/discrete/${ds}_${task}/backbone"
    [ -f "${OUT}/best_model.pt" ] && echo "SKIP ${OUT}" && continue
    python -m tisae.training.train_backbone \
        --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
        --output_dir "${OUT}" \
        --task "${task}" \
        --hid_dim 128 --num_layers 4 --num_heads 8 \
        --dropout 0.3 \
        --lr 1e-4 --weight_decay 5e-3 \
        --batch_size 64 --eval_batch_size 64 \
        --epochs 50 --patience 10 \
        --max_grad_norm 1.0 --flash_attention \
        --seed ${SEED}
  done
done

# ===========================================================================
# 2. Discrete+Pretrain (pretrain then finetune)
# ===========================================================================
for ds in mimic3 mimic4; do
  # Pretrain (shared across tasks within a dataset)
  PT_OUT="${OUTPUT_DIR}/discrete_pt/${ds}_pretrain/backbone"
  if [ ! -f "${PT_OUT}/best_model.pt" ]; then
    python -m tisae.training.train_backbone \
        --dataset_path "${DATA_DIR}/unified_${ds}_240h.pkl" \
        --output_dir "${PT_OUT}" \
        --stage pretrain \
        --hid_dim 128 --num_layers 4 --num_heads 8 \
        --dropout 0.3 \
        --lr 1e-4 --weight_decay 5e-3 \
        --batch_size 64 --eval_batch_size 64 \
        --epochs 50 --patience 10 \
        --max_grad_norm 1.0 --flash_attention \
        --seed ${SEED}
  fi
  # Finetune per task
  for task in "${TASKS[@]}"; do
    OUT="${OUTPUT_DIR}/discrete_pt/${ds}_${task}/backbone"
    [ -f "${OUT}/best_model.pt" ] && echo "SKIP ${OUT}" && continue
    python -m tisae.training.train_backbone \
        --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
        --output_dir "${OUT}" \
        --task "${task}" --stage finetune \
        --load_ckpt_path "${PT_OUT}/best_model.pt" \
        --hid_dim 128 --num_layers 4 --num_heads 8 \
        --dropout 0.3 \
        --lr 1e-5 --weight_decay 5e-3 \
        --batch_size 64 --eval_batch_size 64 \
        --epochs 50 --patience 10 \
        --max_grad_norm 1.0 --flash_attention \
        --seed ${SEED}
  done
done

# ===========================================================================
# 3. STraTS (pretrain then finetune)
# ===========================================================================
for ds in mimic3 mimic4; do
  # Pretrain
  PT_OUT="${OUTPUT_DIR}/strats/${ds}_pretrain/backbone"
  if [ ! -f "${PT_OUT}/best_model.pt" ]; then
    python -m tisae.training.train_external_backbone \
        --backbone strats --stage pretrain \
        --dataset_path "${DATA_DIR}/unified_${ds}_240h.pkl" \
        --output_dir "${PT_OUT}" \
        --hid_dim 64 --num_layers 4 --num_heads 16 --num_blocks 3 \
        --dropout 0.2 --attention_dropout 0.2 --residual average \
        --train_batch_size 16 --eval_batch_size 32 \
        --lr 5e-4 --max_grad_norm 0.3 \
        --epochs 50 --patience 10 \
        --seed ${SEED}
  fi
  # Finetune per task
  for task in "${TASKS[@]}"; do
    OUT="${OUTPUT_DIR}/strats/${ds}_${task}/backbone"
    [ -f "${OUT}/best_model.pt" ] && echo "SKIP ${OUT}" && continue
    python -m tisae.training.train_external_backbone \
        --backbone strats --stage finetune \
        --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
        --output_dir "${OUT}" --task "${task}" \
        --load_ckpt_path "${PT_OUT}/best_model.pt" \
        --hid_dim 64 --num_layers 4 --num_heads 16 --num_blocks 3 \
        --dropout 0.2 --attention_dropout 0.2 --residual average \
        --train_batch_size 16 --eval_batch_size 32 \
        --lr 5e-5 --max_grad_norm 0.3 \
        --epochs 50 --patience 10 \
        --seed ${SEED}
  done
done

# ===========================================================================
# 4. HyMaTE (pretrain then finetune)
# ===========================================================================
for ds in mimic3 mimic4; do
  # Pretrain
  PT_OUT="${OUTPUT_DIR}/hymate/${ds}_pretrain/backbone"
  if [ ! -f "${PT_OUT}/best_model.pt" ]; then
    python -m tisae.training.train_external_backbone \
        --backbone hymate --stage pretrain \
        --dataset_path "${DATA_DIR}/unified_${ds}_240h.pkl" \
        --output_dir "${PT_OUT}" \
        --hid_dim 64 --num_layers 2 --num_heads 4 --num_blocks 3 \
        --dropout 0.2 --attention_dropout 0.2 --residual average \
        --train_batch_size 16 --eval_batch_size 32 \
        --lr 5e-4 --max_grad_norm 0.3 \
        --epochs 50 --patience 10 \
        --seed ${SEED}
  fi
  # Finetune per task
  for task in "${TASKS[@]}"; do
    OUT="${OUTPUT_DIR}/hymate/${ds}_${task}/backbone"
    [ -f "${OUT}/best_model.pt" ] && echo "SKIP ${OUT}" && continue
    python -m tisae.training.train_external_backbone \
        --backbone hymate --stage finetune \
        --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
        --output_dir "${OUT}" --task "${task}" \
        --load_ckpt_path "${PT_OUT}/best_model.pt" \
        --hid_dim 64 --num_layers 2 --num_heads 4 --num_blocks 3 \
        --dropout 0.2 --attention_dropout 0.2 --residual average \
        --train_batch_size 16 --eval_batch_size 32 \
        --lr 5e-5 --max_grad_norm 0.3 \
        --epochs 50 --patience 10 \
        --seed ${SEED}
  done
done

# ===========================================================================
# 5. BAT (supervised only, no pretrain)
# ===========================================================================
for task in "${TASKS[@]}"; do
  for ds in mimic3 mimic4; do
    OUT="${OUTPUT_DIR}/bat/${ds}_${task}/backbone"
    [ -f "${OUT}/best_model.pt" ] && echo "SKIP ${OUT}" && continue
    python -m tisae.training.train_bat \
        --dataset_path "${DATA_DIR}/unified_${ds}_48h.pkl" \
        --output_dir "${OUT}" \
        --task "${task}" --stage finetune \
        --embed_out 44 --layers 1 --heads 1 --pooling mean \
        --dropout 0.2 --attn_dropout 0.2 --use_mask \
        --train_batch_size 16 --eval_batch_size 32 \
        --lr 5e-4 --max_grad_norm 0.3 \
        --epochs 50 --patience 10 \
        --seed ${SEED}
  done
done

echo "All backbone training complete."
