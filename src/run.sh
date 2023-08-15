#!/bin/bash
GPU="$1" # leave empty for cpu.
CHECKPOINT_DIR="EleutherAI/gpt-j-6b"
DATASET_NAME="gsm8k"
DATASET_CONFIG_NAME="main"
CONVERTER="math"
OUTPUT_PATH="$2"
ITER="${3:-3}"
SRC_KEY="${4:-question}"
TGT_KEY="${5:-answer}"
BATCH_SIZE=5
MAX_LEN="${6:-100}"
#
echo "CUDA_VISIBLE_DEVICES=${GPU} python src/plugin_algorithm.py --max_length ${MAX_LEN} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --n_iter ${ITER} \
    --model_path ${CHECKPOINT_DIR} \
    --output_path ${OUTPUT_PATH} \
    --src_key ${SRC_KEY} \
    --tgt_key ${TGT_KEY} \
    --batch_size ${BATCH_SIZE} \
    --converter ${CONVERTER}
"

python plugin_filter.py --max_length ${MAX_LEN} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --n_iter ${ITER} \
    --model_path ${CHECKPOINT_DIR} \
    --output_path ${OUTPUT_PATH} \
    --src_key ${SRC_KEY} \
    --tgt_key ${TGT_KEY} \
    --batch_size ${BATCH_SIZE} \
    --converter ${CONVERTER}