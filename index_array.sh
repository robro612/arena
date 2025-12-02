#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH -t 12:00:00
#SBATCH -J index

MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
NUM_SHARDS=400
NUM_NODES=4
BATCH_SIZE=128
SHARDS_PER_NODE=$((NUM_SHARDS / NUM_NODES))

# Use SLURM_ARRAY_TASK_ID instead of loop variable
i=${SLURM_ARRAY_TASK_ID}

# Create output filename with descriptive information
MODEL_NAME_SAFE=${MODEL_NAME//\//_}
OUTPUT_FILE="logs/index-${MODEL_NAME_SAFE}-numshards_${NUM_SHARDS}-idx_${i}-job_${SLURM_JOB_ID}.out"

# Redirect all output to the file
exec > ${OUTPUT_FILE} 2>&1

echo "Redirecting output to ${OUTPUT_FILE}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "SHARDS_PER_NODE: ${SHARDS_PER_NODE}"

echo "Running script..."
python test_model_manager_local.py \
    --model_meta_path model_meta.yml \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --new_script \
    --force_batch_loop \
    --num_shards ${NUM_SHARDS} \
    --shards_start $((${SHARDS_PER_NODE} * i)) \
    --shards_end $((${SHARDS_PER_NODE} * (i + 1) - 1))

