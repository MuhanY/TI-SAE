#!/bin/bash
# Preprocess MIMIC-III and MIMIC-IV into unified pickles.
set -e

MIMIC3_DIR="${MIMIC3_DIR:-/path/to/mimiciii/1.4}"
MIMIC4_DIR="${MIMIC4_DIR:-/path/to/mimiciv/3.1}"
DATA_DIR="${DATA_DIR:-data}"

mkdir -p "${DATA_DIR}"

# ===========================================================================
# Step 1: Extract raw event tables from MIMIC CSVs
# ===========================================================================

# MIMIC-III
python -m tisae.data.preprocess \
    --raw_data_path "${MIMIC3_DIR}" \
    --output_dir "${DATA_DIR}/raw_mimic3" \
    --T 48

# MIMIC-IV
python -m tisae.data.preprocess_mimiciv \
    --raw_data_path "${MIMIC4_DIR}" \
    --output_dir "${DATA_DIR}/raw_mimic4" \
    --T 48

# ===========================================================================
# Step 2: Build unified pickles (binned + continuous representations)
# ===========================================================================

# 48h pickles (main training/eval)
python -m tisae.data.unified_dataset \
    --raw "${DATA_DIR}/raw_mimic3/mimic_iii_48h.pkl" \
    --output "${DATA_DIR}/unified_mimic3_48h.pkl" \
    --max_hours 48 --max_obs 1024

python -m tisae.data.unified_dataset \
    --raw "${DATA_DIR}/raw_mimic4/mimic_iv_48h.pkl" \
    --output "${DATA_DIR}/unified_mimic4_48h.pkl" \
    --max_hours 48 --max_obs 1024

# 240h pickles (for pretraining only)
python -m tisae.data.unified_dataset \
    --raw "${DATA_DIR}/raw_mimic3/mimic_iii_48h.pkl" \
    --output "${DATA_DIR}/unified_mimic3_240h.pkl" \
    --max_hours 240 --max_obs 1024 \
    --var_order_from "${DATA_DIR}/unified_mimic3_48h.pkl"

python -m tisae.data.unified_dataset \
    --raw "${DATA_DIR}/raw_mimic4/mimic_iv_48h.pkl" \
    --output "${DATA_DIR}/unified_mimic4_240h.pkl" \
    --max_hours 240 --max_obs 1024 \
    --var_order_from "${DATA_DIR}/unified_mimic4_48h.pkl"

echo "Preprocessing complete. Unified pickles at ${DATA_DIR}/"
