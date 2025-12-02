#!/bin/bash

MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
NUM_SHARDS=400
NUM_NODES=4
BATCH_SIZE=1024
SHARDS_PER_NODE=$((NUM_SHARDS / NUM_NODES))
TIMESTAMP=$(date +%Y%m%d%H%M%S)

for ((i=0; i<NUM_NODES; i++)); do
    
    command="srun --gres=gpu:a100:1 -u -t 12:00:00 -J '${i} index' -o logs/index_${MODEL_NAME}_${i}_${NUM_NODES}_${TIMESTAMP}.out python test_model_manager_local.py --model_meta_path model_meta.yml --model_name ${MODEL_NAME} --batch_size ${BATCH_SIZE} --new_script --force_batch_loop --num_shards ${NUM_SHARDS} --shards_start $((${SHARDS_PER_NODE} * i)) --shards_end $((${SHARDS_PER_NODE} * (i + 1) - 1))"
    echo $command
    eval $command &
done