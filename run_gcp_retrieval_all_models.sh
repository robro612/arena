#!/bin/bash
#
# Run GCP retrieval for all models that don't have existing files
# Excludes proprietary models (voyage, text-embedding, embed-english) and BM25
#
# Usage: ./run_gcp_retrieval_all_models.sh
#

set -e  # Exit on error

# Default parameters
FILTERED_QUERIES="query_filtering/arena-query-filter.wikipedia.judge-gpt-5-nano.seed-42.numqueries-1828.jsonl"
TOPK=10
CORPUS="wikipedia"
BATCH_SIZE=128
REQUESTS_PER_MINUTE=3

# Array of model names from model_meta.yml
# Only includes models that:
# - Don't have existing files in arena_retrieval/
# - Aren't proprietary (voyage, text-embedding, embed-english)
# - Aren't BM25
MODELS=(
    # "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # "Salesforce/SFR-Embedding-2_R"
    # text-embedding-004
    voyage-multilingual-2
    # embed-english-v3.0
    # text-embedding-3-large
)

echo "Starting GCP retrieval for ${#MODELS[@]} models"
echo "Corpus: $CORPUS"
echo "TopK: $TOPK"
echo "Filtered queries: $FILTERED_QUERIES"
echo "======================================"

# Run retrieval for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Processing model: $MODEL"
    echo "--------------------------------------"
    
    python retrieve_filtered_queries_from_gcp_index.py \
        --filtered-queries-jsonl "$FILTERED_QUERIES" \
        --model-name "$MODEL" \
        --corpus "$CORPUS" \
        --topk "$TOPK" \
        --batch-size "$BATCH_SIZE" \
        --requests-per-minute "$REQUESTS_PER_MINUTE"

    echo "--------------------------------------"
done

echo ""
echo "======================================"
echo "All models processed!"
echo "Results saved in: gcp_retrieval.model=*.jsonl"

