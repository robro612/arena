import argparse
from datetime import datetime
import glob
import json
import os
from typing import Iterable, List, Tuple, Dict, Any, Optional
import time
import yaml

import torch
from tqdm.rich import tqdm, trange
from datasets import load_dataset
from mteb import get_model
from retrieval.index import DistributedIndex, DTYPE_TO_TORCH_DTYPE
from itertools import batched

def list_index_dirs(index_glob_pattern: str) -> List[str]:
    """
    Return sorted list of index directories matching a glob pattern.
    Only returns paths that are directories and have at least one file inside.
    """
    candidates = sorted(glob.glob(index_glob_pattern))
    index_dirs = [p for p in candidates if os.path.isdir(p) and len(os.listdir(p)) > 0]
    return index_dirs


def model_name_from_index_dir(index_dir: str) -> str:
    """
    Parse model name from an index directory name.
    Expects names like: index_<corpus>_<org>_<model> (underscores in model path replaced with '_').
    For example: index_wikipedia_nomic-ai_nomic-embed-text-v1.5 -> nomic-ai/nomic-embed-text-v1.5
    """
    parts = os.path.basename(index_dir).split("_")
    if len(parts) < 3:
        raise ValueError(f"Index dir name not in expected format: {index_dir}")
    # Join everything after the first two tokens with '/' between provider and model root
    # Original notebook used '/'.join(parts[2:])
    return "/".join(parts[2:])


def build_index_dir(corpus: str, model_name: str, index_root: str | None = None) -> str:
    """
    Construct index directory from corpus and model name.
    Example: corpus='wikipedia', model='nomic-ai/nomic-embed-text-v1.5'
    -> index_wikipedia_nomic-ai_nomic-embed-text-v1.5
    If index_root is provided, prefix it.
    """
    org_model = model_name.replace("/", "_")
    dir_name = f"index_{corpus}_{org_model}"
    return os.path.join(index_root, dir_name) if index_root else dir_name


def dir_has_files(path: str) -> bool:
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def load_model_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expect top-level key 'model_meta'
    if not isinstance(data, dict) or "model_meta" not in data or not isinstance(data["model_meta"], dict):
        raise ValueError("model_meta.yml missing top-level 'model_meta' mapping")
    return data["model_meta"]


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
    records = [
        {"tstamp": ds["tstamp"][i], "query": ds[field][i][:max_query_length]}
        for i in range(len(ds["tstamp"]))
    ]
    seen = set()
    
    # deduplicate records by query, keep the first one according to tstamp
    for record in sorted(records, key=lambda x: x["tstamp"]):
        if record["query"] in seen:
            continue
        seen.add(record["query"])
        records.append(record)

    if max_num_queries is not None:
        records = records[:max_num_queries]
    return records


def encode_queries_with_model(model, queries: List[str], batch_size: int, dtype: Optional[str] = None, show_progress: bool = True, extra_kwargs: Dict[str, Any] | None = None) -> torch.Tensor:
    # Some models use encode_query, others use encode_queries
    encode_fn = getattr(model, 'encode_queries', None) or getattr(model, 'encode_query', None)
    if encode_fn is None:
        raise AttributeError(f"Model {type(model).__name__} has neither encode_query nor encode_queries method")
    
    call_kwargs: Dict[str, Any] = {
        "convert_to_tensor": True,
        "batch_size": batch_size,
        "show_progress_bar": show_progress,
    }
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)

    encoded = encode_fn(
        queries,
        **call_kwargs,
    )
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded)
    if dtype is not None:
        encoded = encoded.to(dtype=DTYPE_TO_TORCH_DTYPE[dtype])
    return encoded


def search_index(index_dir: str, model_name: str, queries: List[dict], topk: int, batch_size: int, progress: bool, model_revision: str | None = None, instruction_prefix: str | None = None, index_dtype: Optional[str] = None, truncate_queries: int = 10000, verbose: bool = False) -> Iterable[dict]:
    """
    Load index and model, run retrieval, and yield JSONL-serializable dicts per query.
    Loads only one index and one model at a time to limit memory.
    """
    # Load index and ensure embeddings are on GPU if available
    index = DistributedIndex()
    index.load_index(index_dir, dtype=index_dtype, verbose=verbose)
    index.to_gpu()

    # Load model (optionally with a specific revision if supported)
    print(f"Loading model {model_name} with revision {model_revision}")
    try:
        model = get_model(model_name, revision=model_revision)  # type: ignore[arg-type]
    except TypeError:
        # Older mteb versions may not support revision kwarg
        model = get_model(model_name)

    try:
        extra_kwargs = {"instruction": instruction_prefix} if instruction_prefix else None
        # Process queries in chunks to limit memory and avoid large matmul spikes
        
        total = len(queries)
        for i, qbatch in tqdm(enumerate(batched(queries, batch_size)), total=(total + batch_size - 1) // batch_size, desc=f"Retrieving queries for {model_name} [bs={batch_size}]"):
            qbatch = list(qbatch)
            # Truncate queries to 10000 characters, most models will do this internally due to their max context length
            # but others like Jina can handle 8912 and will run OOM with the same batch size as the rest
            qtexts = [qrec["query"][:truncate_queries] for qrec in qbatch]
            qs = encode_queries_with_model(
                model,
                qtexts,
                batch_size=batch_size,
                show_progress=False,
                extra_kwargs=extra_kwargs,
                dtype=index_dtype,
            ).to(device=index.embeddings.device, dtype=index.embeddings.dtype)
            docs, scores, indices = index.search_knn(qs, topk=topk)
            del qs
            torch.cuda.empty_cache()

            for i, q in enumerate(qbatch):
                yield {
                    "tstamp": q["tstamp"],
                    "model_name": model_name,
                    "query": q["query"],
                    "docs": docs[i],
                    "scores": scores[i],
                    "indices": indices[i],
                }
    finally:
        # Free memory
        del model
        del index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
            except Exception:
                errors += 1
    return written, errors


def sanitize_model_name(model_name: str) -> str:
    """Make model name filesystem-friendly (e.g., replace '/')."""
    return model_name.replace("/", "_")


def default_output_path(prefix: str, model_name: str, topk: int, output_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_safe = sanitize_model_name(model_name)
    return os.path.join(output_dir, f"{prefix}.{model_safe}.top_{topk}.{ts}.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-k retrieval for selected models (from model_meta.yml) and write JSONL results.")
    parser.add_argument(
        "--model_meta",
        type=str,
        default="model_meta.yml",
        help="Path to model_meta.yml (default: model_meta.yml)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model ids to run (default: all from model_meta.yml)",
    )
    # Instructions are fixed to Wikipedia retrieval context to match arena retrieval battle
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
        help="Batch size for query encoding (default: 32)",
    )
    parser.add_argument(
        "--index_dtype",
        type=str,
        default=None,
        help="Dtype for index (default: None - use model dtype)",
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
        default=10000,
        help="Optional limit on query length, if None, all queries are processed (default: 10000)",
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
        help="Output filename prefix; per-model files will be created as '<prefix>.<model>.top<k>.<utc>.jsonl'",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars for encoding/search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model metadata
    meta: Dict[str, Any] = load_model_meta(args.model_meta)

    # Resolve model list
    if args.models.strip().lower() == "all":
        selected_models = list(meta.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in selected_models if m not in meta]
        if unknown:
            raise SystemExit(f"Models not found in meta: {unknown}")

    # Load queries once (hardcoded to arena retrieval battle dataset)
    queries = load_queries(
        source="mteb/arena-results:retrieval_battle",
        split="data",
        field="0_prompt",
        max_query_length=args.max_query_length,
        max_num_queries=args.max_num_queries,
        filter_field="0_corpus",
        filter_value="wikipedia",
    )

    # Iterate models one at a time
    for model_name in tqdm(selected_models, desc="Models", disable=not args.progress):
        model_info: Dict[str, Any] = meta.get(model_name, {})
        # Hardcode index naming scheme to index_wikipedia_*
        index_dir = build_index_dir("wikipedia", model_name, None)

        # Always require index existence; skip models without it
        if not dir_has_files(index_dir):
            tqdm.write(f"Skipping {model_name}: missing or empty index at {index_dir}")
            continue

        # Skip if index is already processed - check for any output file matching the pattern (ignore timestamp)
        model_safe = sanitize_model_name(model_name)
        existing_files = glob.glob(os.path.join(args.output_dir, f"{args.output_prefix}.{model_safe}.top_{args.topk}.*.jsonl"))
        if existing_files:
            # Check if existing file has the correct number of queries
            existing_file = existing_files[0]  # Use the first matching file
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    num_lines = sum(1 for line in f if line.strip())  # Count non-empty lines
                expected_queries = len(queries)
                if num_lines == expected_queries:
                    tqdm.write(f"Skipping {model_name}: already processed with {num_lines} queries (found {len(existing_files)} existing file(s))")
                    continue
                else:
                    tqdm.write(f"Re-running {model_name}: existing file has {num_lines} queries but expected {expected_queries}")
            except Exception as e:
                tqdm.write(f"Re-running {model_name}: error checking existing file ({e})")

        model_revision = model_info.get("revision")

        # Default instruction to Wikipedia retrieval if available
        instruction_prefix = model_info.get("instruction_query_wikipedia")

        # Run retrieval and stream to JSONL (per-model file including model and topk)
        records = search_index(
            index_dir=index_dir,
            model_name=model_name,
            queries=queries,
            topk=args.topk,
            batch_size=args.batch_size,
            progress=args.progress,
            model_revision=model_revision,
            instruction_prefix=instruction_prefix,
            index_dtype=args.index_dtype,
            verbose=args.progress,
        )
        out_path = default_output_path(prefix=args.output_prefix, model_name=model_name, topk=args.topk, output_dir=args.output_dir)
        written, errors = write_jsonl(records, out_path)
        if args.progress:
            tqdm.write(f"{index_dir} -> {model_name}@{model_revision or 'latest'}: wrote {written} records to {out_path}, errors={errors}")


if __name__ == "__main__":
    main()

"""
python retrieve_arena.py --model_meta model_meta.yml --models GritLM/GritLM-7B --batch_size 32 --index_dtype bfloat16 --max_query_length 2048 --progress
"""
