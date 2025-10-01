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


model_meta_path = Path("/home/hltcoe/rjha/arena/model_meta.yml")
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
    batch_size: int,
    limit: int = None,
    num_shards: int = 1,
    force_batch_loop: bool = False,
):
    print("Loading model...")
    model = get_model(model_name, device="cuda")
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
    for i, shard in enumerate(shards):
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
        torch.save(embeddings, os.path.join(index_dir, f"embeddings.{i}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--limit", type=int, required=False, default=None)
    parser.add_argument("--shards", type=int, required=False, default=1)
    parser.add_argument("--new_script", action="store_true")
    parser.add_argument("--force_batch_loop", action="store_true")
    args = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if args.new_script:
        index_new(
            args.model_name,
            args.batch_size,
            args.limit,
            args.shards,
            args.force_batch_loop,
        )
    else:
        index_small_models(args.model_name, args.batch_size, args.limit)
