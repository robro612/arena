#!/usr/bin/env python3
"""
Script to retrieve filtered queries from GCP index.

Loads queries from a JSONL file, filters for include==True, 
submits them to the GCP index, and stores topk results.
"""
import argparse
import json
import os
import tempfile
import hashlib
import time
from itertools import batched
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import yaml
from tqdm import tqdm
import pandas as pd
import torch
import mteb
import random
from models import ModelManager, MODEL_TO_CUDA_DEVICE, CORPUS_TO_FORMAT

load_dotenv()

def get_credentials():
    """Load GCP credentials from environment variable."""
    creds_json_str = os.getenv("GCP_CREDENTIALS")
    if creds_json_str is None:
        raise ValueError("GCP_CREDENTIALS not found in environment")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        temp.write(creds_json_str)
        temp_filename = temp.name 
    
    return temp_filename

def load_model_meta_yaml(file_path: str | Path) -> dict:
    """Load model metadata from YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_filtered_queries(input_file: str, limit: int = None) -> List[Dict]:
    """Load queries from JSONL file and filter for include==True."""
    
    df = pd.read_json(input_file, lines=True)
    filtered_df = df.query("include == True")
    if limit is not None:
        filtered_df = filtered_df.head(limit)
    filtered_queries = filtered_df.to_dict('records')
    
    print(f"Loaded {len(filtered_queries)} kept queries out of {len(df)} total queries from {input_file}")
    
    return filtered_queries

def retrieve_and_save(
    queries: List[Dict],
    model_manager: ModelManager,
    model_name: str,
    corpus: str,
    topk: int,
    output_file: str,
    device: str = "cuda",
    batch_size: int = 32,
    requests_per_minute: int = None
):
    """
    Retrieve documents for each query and save to JSONL using batch encoding.
    
    Args:
        queries: List of query dictionaries
        model_manager: ModelManager instance
        model_name: Name of the model to use
        corpus: Corpus to search
        topk: Number of top documents to retrieve
        output_file: Path to output JSONL file
        device: Device to use for model
        batch_size: Batch size for query embedding
        requests_per_minute: Rate limit for requests (None for no limit)
    """
    results_written = 0
    
    # Get corpus format for document formatting
    corpus_format = CORPUS_TO_FORMAT[corpus]
    
    # Calculate sleep time between batches if rate limiting is enabled
    sleep_time = None
    if requests_per_minute is not None:
        sleep_time = (60.0 / requests_per_minute) + random.uniform(0.5, 2.5)
    
    model = model_manager.load_model(model_name)
    print(f"Model: {model}")

    if "voyage" in model_name:
        # need to load the model in the manager for index stuff, but it doesn't work for some reason
        print(f"Loading model {model_name} from mteb directly")
        model = mteb.get_model(model_name)
        print(f"Model: {model}")
    
    # Load the GCP index once
    index = model_manager.load_gcp_index(model_name, corpus)
    
    # Get encoding kwargs (instruction if needed)
    kwargs = {}
    if f"instruction_query_{corpus}" in model_manager.model_meta[model_name]:
        kwargs["instruction"] = model_manager.model_meta[model_name][f"instruction_query_{corpus}"]
        print(f"Using instruction: {kwargs['instruction']}")
    
    with open(output_file, 'w') as out_f:
        for batch in tqdm(list(batched(queries, batch_size)), desc=f"Retrieving with {model_name}"):
            batch_start_time = time.time()
            
            # Extract query texts from batch
            query_texts = [q["query"] for q in batch]

            if hasattr(model, "encode_queries"):
                query_embeds = model.encode_queries(query_texts, **kwargs)
            else:
                query_embeds = model.encode(query_texts, **kwargs)
            
            # Convert to list for individual lookups
            query_embeds_list = query_embeds.tolist()
            
            # Do individual lookups for each query
            for i, (query_record, query_text) in enumerate(zip(batch, query_texts)):
                tstamp = query_record["tstamp"]
                
                # Single lookup with the pre-encoded embedding
                docs = index.search(query_embeds=[query_embeds_list[i]], topk=topk)
                
                # Format all topk documents
                if corpus == "stackexchange":
                    # stackexchange docs only have "text" field
                    doc_strings = [corpus_format.format(text=doc["text"]) for doc in docs]
                else:
                    # wikipedia and arxiv have both "title" and "text"
                    doc_strings = [corpus_format.format(title=doc.get("title", ""), text=doc["text"]) for doc in docs]
                
                # Format result
                result = {
                    "tstamp": tstamp,
                    "model_name": model_name,
                    "query": query_text,
                    "docs": doc_strings  # List of topk formatted document strings
                }
                
                # Write to output file
                out_f.write(json.dumps(result) + '\n')
                results_written += 1
            
            # Rate limiting: sleep if needed
            if sleep_time is not None:
                elapsed = time.time() - batch_start_time
                if elapsed < sleep_time:
                    print(f"Sleeping for {sleep_time - elapsed} seconds to meet rate limit of {requests_per_minute} requests per minute")
                    time.sleep(sleep_time - elapsed)
    
    return results_written

def generate_output_path(filtered_queries_path: str, model_name: str, topk: int) -> str:
    """Generate output path with model name, hash of input filename, and topk."""
    # Get the filename (without path) for hashing
    input_filename = Path(filtered_queries_path).name
    
    # Generate 8-character hash of the filename
    hash_obj = hashlib.sha256(input_filename.encode())
    hash_short = hash_obj.hexdigest()[:8]
    
    # Clean model name for use in filename (replace / with _)
    model_name_clean = model_name.replace("/", "_")
    
    # Generate output filename
    output_filename = f"gcp_retrieval.model={model_name_clean}.filter_file_hash={hash_short}.topk={topk}.jsonl"
    
    return output_filename

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve filtered queries from GCP index"
    )
    parser.add_argument(
        "--filtered-queries-jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with filtered queries"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arena_retrieval",
        help="Output directory for results (default: arena_retrieval)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Model name to use for retrieval"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="wikipedia",
        choices=["wikipedia", "arxiv", "stackexchange"],
        help="Corpus to search"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top documents to retrieve"
    )
    parser.add_argument(
        "--model-meta",
        type=str,
        default="/exp/rjha/arena/model_meta.yml",
        help="Path to model metadata YAML file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of queries to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for query embedding (default: 32)"
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Rate limit for requests per minute (default: None, no limit)"
    )
    args = parser.parse_args()
    
    # Generate output path if not provided
    output_path = generate_output_path(
        args.filtered_queries_jsonl,
        args.model_name,
        args.topk
    )
    output_path = os.path.join(args.output_dir, output_path)
    print(f"Output path: {output_path}")
    
    # Set CUDA device for the model
    MODEL_TO_CUDA_DEVICE[args.model_name] = "0"
    
    # Set up GCP credentials
    print("Setting up GCP credentials...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_credentials()
    
    # Load model metadata
    print(f"Loading model metadata from {args.model_meta}...")
    model_meta = load_model_meta_yaml(args.model_meta)
    
    # Load and filter queries
    print(f"Loading queries from {args.filtered_queries_jsonl}...")
    filtered_queries = load_filtered_queries(args.filtered_queries_jsonl, args.limit)
    print(f"Found {len(filtered_queries)} queries with include==True")
    
    if len(filtered_queries) == 0:
        print("No queries to process. Exiting.")
        return
    
    # Initialize model manager with GCP index
    print("Initializing ModelManager with GCP index...")
    model_manager = ModelManager(model_meta, use_gcp_index=True, load_all=False)
    
    # Retrieve and save results
    print(f"Starting retrieval with {args.model_name} on {args.corpus} corpus...")
    print(f"Batch size: {args.batch_size}")
    if args.requests_per_minute:
        print(f"Rate limit: {args.requests_per_minute} requests per minute")
    else:
        print("Rate limit: None (no limit)")
    results_written = retrieve_and_save(
        queries=filtered_queries,
        model_manager=model_manager,
        model_name=args.model_name,
        corpus=args.corpus,
        topk=args.topk,
        output_file=output_path,
        device="cuda:0",
        batch_size=args.batch_size,
        requests_per_minute=args.requests_per_minute
    )
    
    print(f"\nCompleted! Wrote {results_written} results to {output_path}")

if __name__ == "__main__":
    main()

