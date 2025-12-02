#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH -t 12:00:00
#SBATCH -J index_vllm

MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
NUM_SHARDS=400
NUM_NODES=4
BATCH_SIZE=128
SHARDS_PER_NODE=$((NUM_SHARDS / NUM_NODES))

# Use SLURM_ARRAY_TASK_ID to get unique port per job
i=${SLURM_ARRAY_TASK_ID}
VLLM_PORT=$((6644 + i))

# Create output filename with descriptive information
MODEL_NAME_SAFE=${MODEL_NAME//\//_}
OUTPUT_FILE="logs/index-vllm-${MODEL_NAME_SAFE}-numshards_${NUM_SHARDS}-idx_${i}-job_${SLURM_JOB_ID}.out"

# Create logs directory if it doesn't exist
mkdir -p logs

# Redirect all output to the file
exec > ${OUTPUT_FILE} 2>&1

echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Redirecting output to ${OUTPUT_FILE}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "SHARDS_PER_NODE: ${SHARDS_PER_NODE}"
echo "VLLM_PORT: ${VLLM_PORT}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Activate virtual environment
echo "=========================================="
echo "Activating Virtual Environment"
echo "=========================================="
source .venv/bin/activate
echo "Python: $(which python)"
echo "vLLM location: $(which vllm)"
echo ""

# Launch vLLM server in background
echo "=========================================="
echo "Launching vLLM Server"
echo "=========================================="
echo "Starting vLLM server on port ${VLLM_PORT}..."

# Launch vLLM with appropriate settings for embedding model
vllm serve ${MODEL_NAME} \
    --port ${VLLM_PORT} \
    --host 0.0.0.0 \
    --dtype float16 \
    > "logs/vllm-server-${i}-${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: ${VLLM_PID}"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes max
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready! (waited ${WAIT_TIME}s)"
        break
    fi
    echo "Still waiting... (${WAIT_TIME}s elapsed)"
    sleep ${SLEEP_INTERVAL}
    WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

# Check if server is actually ready
if ! curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    echo "Check logs at: logs/vllm-server-${i}-${SLURM_JOB_ID}.log"
    kill ${VLLM_PID} 2>/dev/null
    exit 1
fi

# Test the server with a quick embedding request
echo "Testing vLLM server with sample request..."
curl -s http://localhost:${VLLM_PORT}/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "input": "test"
    }' | head -n 5
echo ""
echo ""

# Run indexing script
echo "=========================================="
echo "Starting Indexing"
echo "=========================================="
echo "Shards to process: $((${SHARDS_PER_NODE} * i)) to $((${SHARDS_PER_NODE} * (i + 1) - 1))"
echo ""

python test_model_manager_local.py \
    --model_meta_path model_meta.yml \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --new_script \
    --num_shards ${NUM_SHARDS} \
    --shards_start $((${SHARDS_PER_NODE} * i)) \
    --shards_end $((${SHARDS_PER_NODE} * (i + 1) - 1)) \
    --use_vllm_endpoint \
    --vllm_host localhost \
    --vllm_port ${VLLM_PORT}

INDEXING_EXIT_CODE=$?

# Cleanup: kill vLLM server
echo ""
echo "=========================================="
echo "Cleanup"
echo "=========================================="
echo "Shutting down vLLM server (PID: ${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done!"
echo "Indexing exit code: ${INDEXING_EXIT_CODE}"

exit ${INDEXING_EXIT_CODE}

