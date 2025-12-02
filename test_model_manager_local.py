# Save original stdout BEFORE any imports that might redirect it
import sys

original_stdout = sys.stdout
original_stderr = sys.stderr

# Disable loguru logger before importing modules that use it
from loguru import logger

logger.disable("")

import yaml
from pathlib import Path
import torch
from models import ModelManager
import argparse
from mteb import get_model
from datasets import load_dataset
import os
from tqdm.auto import tqdm
from itertools import batched
from openai import OpenAI


# After all imports, restore original stdout to bypass logger redirection
sys.stdout = original_stdout
sys.stderr = original_stderr


def load_model_meta_yaml(file_path: str | Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def encode_with_vllm(
    texts: list[str],
    host: str,
    port: int,
    model_name: str,
    batch_size: int = None,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    """
    Encode texts using a vLLM encoder endpoint via OpenAI client.

    Args:
        texts: List of texts to encode
        host: vLLM server host
        port: vLLM server port
        model_name: Model name to pass to the API
        batch_size: Optional batch size for processing (will batch requests if provided)
        show_progress_bar: Whether to show progress bar

    Returns:
        Tensor of embeddings with shape (num_texts, embedding_dim)
    """
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="not-needed")
    print(f"Using vLLM endpoint at {base_url}")
    all_embeddings = []

    if batch_size is not None and batch_size < len(texts):
        # Process in batches
        batches = list(batched(texts, batch_size))
        iterator = (
            tqdm(batches, desc="Encoding batches") if show_progress_bar else batches
        )

        for batch in iterator:
            response = client.embeddings.create(model=model_name, input=list(batch))
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)
    else:
        # Process all at once
        response = client.embeddings.create(model=model_name, input=texts)
        all_embeddings = [d.embedding for d in response.data]

    # Convert to tensor
    embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)

    return embeddings_tensor


model_meta_path = Path("model_meta.yml")
model_meta = load_model_meta_yaml(model_meta_path)
model_manager = ModelManager(model_meta=model_meta)


def index_small_models(model_name: str, batch_size: int, limit: int = None):
    model_manager.model_meta[model_name].update({"index_shards": 1})
    print(f"Model: {model_name}")
    print(f"Model meta: {model_manager.model_meta[model_name]}")
    model = model_manager.load_model(model_name, device="cuda")
    model_manager.model_meta[model_name].update({"limit": limit})
    model_manager.load_local_index(
        model_name=model_name,
        corpus="wikipedia",
        embedbs=batch_size,
    )
    index = model_manager.loaded_indices[model_name]["wikipedia"]
    print(f"Model name: {model_name}")
    print(
        f"Index embeddings - shape: {index.embeddings.shape}, dtype: {index.embeddings.dtype}, device: {index.embeddings.device}"
    )

    print("Done")


def index_new(
    model_name: str,
    model_meta_path: str,
    batch_size: int,
    limit: int = None,
    num_shards: int = 1,
    force_batch_loop: bool = False,
    shards_start: int | None = None,
    shards_end: int | None = None,
    shards_list: list[int] | None = None,
    use_vllm_endpoint: bool = False,
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
):
    model = None

    if use_vllm_endpoint:
        print(f"Using vLLM endpoint at {vllm_host}:{vllm_port}/v1")
        # No model loading needed, we'll use the endpoint
    else:
        model_meta = load_model_meta_yaml(model_meta_path)["model_meta"]
        print("Loading model...")
        model = get_model(
            model_name,
            revision=model_meta[model_name].get("revision", None),
            device="cuda",
        )
        # set max_seq_length to 4096 to avoid OOM
        print(f"Setting max_seq_length to 4096 for {model_name}")
        try:
            model.max_seq_length = 4096
        except AttributeError:
            print(f"Model {model_name} does not support setting max_seq_length")
            print(f"Model: {model}")
            print(f"Model type: {type(model)}")
            print(f"Model attributes: {dir(model)}")

        print(f"Model max_seq_length: {model.max_seq_length}")
    print("Loading dataset...")
    wiki = load_dataset("mteb/arena-wikipedia-7-15-24", split="train")[:limit]
    print("Formatting text...")
    formatted_text = [
        f"{title}\n\n{text}" for title, text in tqdm(list(zip(wiki["title"], wiki["text"])), desc="Formatting text")
    ]
    # split formatted_text into num_shards shards in order
    per_shard = len(formatted_text) // num_shards
    shards = [
        formatted_text[i * per_shard : (i + 1) * per_shard]
        for i in range(num_shards - 1)
    ]
    # Add the last shard with any remainder
    shards.append(formatted_text[(num_shards - 1) * per_shard :])

    # Determine which shards to process
    if shards_list is not None:
        # Use ad-hoc list of shards
        shards_to_process = set(shards_list)
        # Validate shard indices
        for shard_idx in shards_to_process:
            if not (0 <= shard_idx < num_shards):
                raise ValueError(
                    f"Invalid shard index {shard_idx}: must be in range [0, {num_shards - 1}]"
                )
    else:
        # Use range-based approach
        if shards_start is None and shards_end is None:
            shards_start, shards_end = 0, num_shards - 1
        elif shards_start is None or shards_end is None:
            raise ValueError(
                "Both shards_start and shards_end must be provided together or omitted together"
            )
        if not (0 <= shards_start <= shards_end < num_shards):
            raise ValueError(
                f"Invalid shard range: shards_start={shards_start}, shards_end={shards_end}, num_shards={num_shards}"
            )
        shards_to_process = set(range(shards_start, shards_end + 1))

    for i, shard in enumerate(shards):
        if i not in shards_to_process:
            print(f"Skipping shard {i}")
            continue
        print(f"Encoding shard {i}")

        if use_vllm_endpoint:
            # Use vLLM endpoint for encoding
            embeddings = encode_with_vllm(
                shard,
                host=vllm_host,
                port=vllm_port,
                model_name=model_name,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        elif force_batch_loop:
            print(
                f"Using force_batch_loop, batches per shard {len(shard) // batch_size}"
            )
            batch_embeddings = []
            for batch in tqdm(
                batched(shard, batch_size),
                desc=f"Encoding batches for shard {i}",
                total=len(shard) // batch_size,
            ):
                embeddings = model.encode(
                    list(batch), convert_to_tensor=True, show_progress_bar=False
                )
                if not isinstance(embeddings, torch.Tensor):
                    print(
                        f"Embeddings is not a tensor despite explicit convert_to_tensor=True, converting to tensor"
                    )
                    embeddings = torch.tensor(embeddings)
                batch_embeddings.append(embeddings)
            embeddings = torch.cat(batch_embeddings, dim=0)
        else:
            embeddings = model.encode(
                shard,
                convert_to_tensor=True,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        print("Done")
        if not isinstance(embeddings, torch.Tensor):
            print(
                f"Embeddings is not a tensor despite explicit convert_to_tensor=True, converting to tensor"
            )
            embeddings = torch.tensor(embeddings)

        print("Transposing embeddings")
        embeddings = embeddings.T
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        print(f"Embeddings device: {embeddings.device}")
        index_dir = "index_wikipedia_" + model_name.replace("/", "_")
        os.makedirs(index_dir, exist_ok=True)
        torch.save(
            embeddings, os.path.join(index_dir, f"embeddings.{i}.{num_shards}.pt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_meta_path", type=str, required=False, default="model_meta.yml"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name to index"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for encoding"
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=None,
        help="Limit on number of documents to index",
    )
    parser.add_argument(
        "--new_script", action="store_true", help="Use new script for indexing"
    )
    parser.add_argument(
        "--force_batch_loop",
        action="store_true",
        help="Force outer batch loop for encoding for models that don't properly support batching",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        required=False,
        default=1,
        help="Number of shards to index",
    )
    # vLLM endpoint options
    parser.add_argument(
        "--use_vllm_endpoint",
        action="store_true",
        help="Use vLLM encoder endpoint instead of loading model locally",
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--vllm_port", type=int, default=8000, help="vLLM server port (default: 8000)"
    )
    # Mutually exclusive group for shard selection
    shard_group = parser.add_mutually_exclusive_group()
    shard_group.add_argument(
        "--shards",
        type=int,
        nargs="+",
        help="Ad-hoc list of shard indices to encode (e.g., --shards 0 2 5)",
    )
    # Add a subgroup for range-based arguments within the mutually exclusive group
    range_subgroup = shard_group.add_argument_group("range-based shard selection")
    range_subgroup.add_argument(
        "--shards_start",
        type=int,
        help="Inclusive start shard index to encode when running multiple instances (use with --shards_end)",
    )
    range_subgroup.add_argument(
        "--shards_end",
        type=int,
        help="Inclusive end shard index to encode when running multiple instances (use with --shards_start)",
    )

    args = parser.parse_args()

    # Validate mutual exclusivity between --shards and (--shards_start, --shards_end)
    if args.shards is not None and (
        args.shards_start is not None or args.shards_end is not None
    ):
        parser.error("--shards cannot be used with --shards_start or --shards_end")

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if args.new_script:
        index_new(
            args.model_name,
            args.model_meta_path,
            args.batch_size,
            args.limit,
            args.num_shards,
            args.force_batch_loop,
            args.shards_start,
            args.shards_end,
            args.shards,
            args.use_vllm_endpoint,
            args.vllm_host,
            args.vllm_port,
        )
    else:
        index_small_models(args.model_name, args.batch_size, args.limit)

"""
Usage:
# Range-based shard processing with local model:
python test_model_manager_local.py --model_meta_path model_meta.yml --model_name intfloat/e5-mistral-7b-instruct --batch_size 512 --num_shards 100 --new_script --force_batch_loop --shards_start 0 --shards_end 3

# Ad-hoc list of shards with local model:
python test_model_manager_local.py --model_meta_path model_meta.yml --model_name intfloat/e5-mistral-7b-instruct --batch_size 512 --num_shards 8 --new_script --force_batch_loop --shards 0 2 5 7

# Using vLLM endpoint:
python test_model_manager_local.py --model_name Alibaba-NLP/gte-Qwen2-7B-instruct --batch_size 512 --num_shards 400 --new_script --use_vllm_endpoint --vllm_host http://rack7n05 --vllm_port 6644 --shards_start 0 --shards_end 3

Notes:
- Use either --shards (ad-hoc list) OR --shards_start/--shards_end (range), not both.
- Run multiple instances with disjoint [--shards_start, --shards_end] ranges that cover [0, --num_shards-1].
- Output files are saved as embeddings.{i}.{num_shards}.pt (backwards compatible with embeddings.{i}.pt).
- When using --use_vllm_endpoint, the script will query the vLLM server instead of loading the model locally.
- vLLM endpoint defaults to localhost:8000, customize with --vllm_host and --vllm_port.
"""
