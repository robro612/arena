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


# After all imports, restore original stdout to bypass logger redirection
sys.stdout = original_stdout
sys.stderr = original_stderr


def load_model_meta_yaml(file_path: str | Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


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
):
    model_meta = load_model_meta_yaml(model_meta_path)["model_meta"]
    print("Loading model...")
    model = get_model(
        model_name, 
        # revision=model_meta[model_name].get("revision", None), 
        device="cuda",
    )
    print("Loading dataset...")
    wiki = load_dataset("mteb/arena-wikipedia-7-15-24", split="train")[:limit]
    print("Formatting text...")
    formatted_text = [
        f"{title}\n\n{text}" for title, text in zip(wiki["title"], wiki["text"])
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
            raise ValueError("Both shards_start and shards_end must be provided together or omitted together")
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
        if force_batch_loop:
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
        torch.save(embeddings, os.path.join(index_dir, f"embeddings.{i}.{num_shards}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_meta_path", type=str, required=False, default="model_meta.yml")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to index")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for encoding")
    parser.add_argument("--limit", type=int, required=False, default=None, help="Limit on number of documents to index")
    parser.add_argument("--new_script", action="store_true", help="Use new script for indexing")
    parser.add_argument("--force_batch_loop", action="store_true", help="Force outer batch loop for encoding for models that don't properly support batching")
    parser.add_argument("--num_shards", type=int, required=False, default=1, help="Number of shards to index")
    # Mutually exclusive group for shard selection
    shard_group = parser.add_mutually_exclusive_group()
    shard_group.add_argument("--shards", type=int, nargs='+', help="Ad-hoc list of shard indices to encode (e.g., --shards 0 2 5)")
    # Add a subgroup for range-based arguments within the mutually exclusive group
    range_subgroup = shard_group.add_argument_group('range-based shard selection')
    range_subgroup.add_argument("--shards_start", type=int, help="Inclusive start shard index to encode when running multiple instances (use with --shards_end)")
    range_subgroup.add_argument("--shards_end", type=int, help="Inclusive end shard index to encode when running multiple instances (use with --shards_start)")
    
    args = parser.parse_args()
    
    # Validate mutual exclusivity between --shards and (--shards_start, --shards_end)
    if args.shards is not None and (args.shards_start is not None or args.shards_end is not None):
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
        )
    else:
        index_small_models(args.model_name, args.batch_size, args.limit)

"""
Usage:
# Range-based shard processing:
python test_model_manager_local.py --model_meta_path model_meta.yml --model_name intfloat/e5-mistral-7b-instruct --batch_size 512 --num_shards 100 --new_script --force_batch_loop --shards_start 0 --shards_end 3

# Ad-hoc list of shards:
python test_model_manager_local.py --model_meta_path model_meta.yml --model_name intfloat/e5-mistral-7b-instruct --batch_size 512 --num_shards 8 --new_script --force_batch_loop --shards 0 2 5 7

Notes:
- Use either --shards (ad-hoc list) OR --shards_start/--shards_end (range), not both.
- Run multiple instances with disjoint [--shards_start, --shards_end] ranges that cover [0, --num_shards-1].
- Output files are saved as embeddings.{i}.{num_shards}.pt (backwards compatible with embeddings.{i}.pt).
"""