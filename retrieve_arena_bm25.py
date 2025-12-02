import argparse
from datetime import datetime
import glob
import json
import os
from typing import Iterable, List, Tuple, Dict, Any, Optional
import time

from tqdm.rich import tqdm
from datasets import load_dataset
from retrieval.bm25_index import BM25Index
import bm25s


def load_queries(source: str, split: str, field: str, max_query_length: Optional[int] = None, max_num_queries: Optional[int] = None, filter_field: Optional[str] = None, filter_value: Optional[str] = None) -> List[dict]:
    """
    Load queries from a datasets source. Default mirrors the notebook:
    dataset: mteb/arena-results, config: retrieval_battle, split: data, field: "0_prompt".
    """
    # Allow passing config via colon, e.g. "mteb/arena-results:retrieval_battle"
    if ":" in source:
        base, config = source.split(":", 1)
        ds = load_dataset(base, config, split=split)
    else:
        ds = load_dataset(source, split=split)
    if filter_field and filter_value:
        ds = ds.filter(lambda x: x[filter_field] == filter_value)

    # Build records with dataset-provided tstamp and query string
    records = []
    seen = set()
    
    # Create initial records
    initial_records = [
        {"tstamp": ds["tstamp"][i], "query": ds[field][i][:max_query_length] if max_query_length else ds[field][i]}
        for i in range(len(ds["tstamp"]))
    ]
    
    # deduplicate records by query, keep the first one according to tstamp
    for record in sorted(initial_records, key=lambda x: x["tstamp"]):
        if record["query"] in seen:
            continue
        seen.add(record["query"])
        records.append(record)

    if max_num_queries is not None:
        records = records[:max_num_queries]
    return records


def search_bm25_index(bm25_index: BM25Index, queries: List[dict], topk: int, batch_size: int, progress: bool) -> Iterable[dict]:
    """
    Run BM25 retrieval and yield JSONL-serializable dicts per query.
    """
    total = len(queries)
    
    # Process queries in batches
    for i in tqdm(range(0, total, batch_size), desc=f"Retrieving queries with BM25 [bs={batch_size}]", disable=not progress):
        batch_end = min(i + batch_size, total)
        qbatch = queries[i:batch_end]
        qtexts = [qrec["query"] for qrec in qbatch]
        
        # BM25 search returns results and scores
        # Tokenize queries using bm25s
        queries_tokenized = bm25s.tokenize(qtexts, stemmer=bm25_index.stemmer)
        results, scores = bm25_index.index.retrieve(queries_tokenized, k=topk)
        
        for j, q in enumerate(qbatch):
            # Extract document information from results
            batch_docs = []
            batch_scores = []
            batch_indices = []
            
            for doc_idx, doc in enumerate(results[j]):
                if doc is not None:
                    # Format document as text (title + text if available)
                    if isinstance(doc, dict):
                        if "title" in doc and "text" in doc:
                            doc_text = f"{doc['title']}\n\n{doc['text']}"
                        elif "text" in doc:
                            doc_text = doc["text"]
                        else:
                            doc_text = str(doc)
                        doc_id = doc.get("_id", doc.get("id", str(doc_idx)))
                    else:
                        doc_text = str(doc)
                        doc_id = str(doc_idx)
                    
                    batch_docs.append(doc_text)
                    batch_scores.append(float(scores[j][doc_idx]) if scores is not None else 0.0)
                    batch_indices.append(doc_id)
            
            yield {
                "tstamp": q["tstamp"],
                "model_name": "BM25",
                "query": q["query"],
                "docs": batch_docs,
                "scores": batch_scores,
                "indices": batch_indices,
            }


def write_jsonl(records: Iterable[dict], out_path: str) -> Tuple[int, int]:
    """
    Append records to a JSONL file. Returns (#written, #errors)
    """
    written = 0
    errors = 0
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in records:
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                errors += 1
                print(f"Error writing record: {e}")
    return written, errors


def default_output_path(prefix: str, topk: int, output_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(output_dir, f"{prefix}.BM25.top_{topk}.{ts}.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-k BM25 retrieval for arena retrieval battle queries and write JSONL results.")
    parser.add_argument(
        "--corpus",
        type=str,
        default="wikipedia",
        help="Corpus to use for BM25 index (default: wikipedia)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of documents to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for query processing (default: 32)",
    )
    parser.add_argument(
        "--queries_source",
        type=str,
        default="mteb/arena-results:retrieval_battle",
        help="Dataset source for queries. Use format dataset[:config] (default: mteb/arena-results:retrieval_battle)",
    )
    parser.add_argument(
        "--queries_split",
        type=str,
        default="data",
        help="Dataset split for queries (default: data)",
    )
    parser.add_argument(
        "--queries_field",
        type=str,
        default="0_prompt",
        help="Field name containing queries (default: 0_prompt)",
    )
    parser.add_argument(
        "--max_query_length",
        type=int,
        default=1000,
        help="Optional limit on query length (default: 1000)",
    )
    parser.add_argument(
        "--max_num_queries",
        type=int,
        default=None,
        help="Optional limit on number of queries to process, if None, all queries are processed (default: None)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arena_retrieval",
        help="Output directory (default: arena_retrieval)",
    )
    parser.add_argument(
        "--output_prefix",
        dest="output_prefix",
        type=str,
        default="arena_retrieval",
        help="Output filename prefix; output will be '<prefix>.BM25.top<k>.<utc>.jsonl'",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars",
    )
    parser.add_argument(
        "--limit_corpus",
        type=int,
        default=None,
        help="Optional limit on corpus size for testing (default: None)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load queries (hardcoded to arena retrieval battle dataset, Wikipedia only)
    print(f"Loading queries from {args.queries_source}...")
    queries = load_queries(
        source=args.queries_source,
        split=args.queries_split,
        field=args.queries_field,
        max_query_length=args.max_query_length,
        max_num_queries=args.max_num_queries,
        filter_field="0_corpus",
        filter_value=args.corpus,
    )
    print(f"Loaded {len(queries)} queries")

    # Check if output already exists
    existing_files = glob.glob(os.path.join(args.output_dir, f"{args.output_prefix}.BM25.top_{args.topk}.*.jsonl"))
    if existing_files:
        existing_file = existing_files[0]
        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for line in f if line.strip())
            expected_queries = len(queries)
            if num_lines == expected_queries:
                print(f"Output already exists with {num_lines} queries (found {len(existing_files)} existing file(s)). Exiting.")
                return
            else:
                print(f"Re-running: existing file has {num_lines} queries but expected {expected_queries}")
        except Exception as e:
            print(f"Re-running: error checking existing file ({e})")

    # Initialize BM25 index
    print(f"Initializing BM25 index for corpus: {args.corpus}")
    bm25_index = BM25Index("BM25", corpus=args.corpus, limit=args.limit_corpus)
    
    print("Loading BM25 index...")
    bm25_index.load_index()
    
    # Run retrieval and stream to JSONL
    print(f"Running retrieval for {len(queries)} queries with topk={args.topk}")
    records = search_bm25_index(
        bm25_index=bm25_index,
        queries=queries,
        topk=args.topk,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    
    out_path = default_output_path(prefix=args.output_prefix, topk=args.topk, output_dir=args.output_dir)
    written, errors = write_jsonl(records, out_path)
    print(f"BM25 retrieval complete: wrote {written} records to {out_path}, errors={errors}")


if __name__ == "__main__":
    main()

"""
Example usage:
python retrieve_arena_bm25.py --corpus wikipedia --topk 100 --batch_size 32 --progress
python retrieve_arena_bm25.py --corpus wikipedia --topk 100 --batch_size 32 --progress --limit_corpus 1000 --max_num_queries 10
"""

