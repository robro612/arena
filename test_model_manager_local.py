# Save original stdout BEFORE any imports that might redirect it
import sys

original_stdout = sys.stdout
original_stderr = sys.stderr

# Disable loguru logger before importing modules that use it
from loguru import logger

logger.disable("")

import yaml
import os
from pathlib import Path
import torch
import json
import ir_datasets
import numpy as np
import mteb
from retrieval.index import DistributedIndex, load_or_initialize_index, build_index
from models import ModelManager
from rich.progress import track
from retrieval.common import force_to_tensor
import argparse

# After all imports, restore original stdout to bypass logger redirection
sys.stdout = original_stdout
sys.stderr = original_stderr


def load_model_meta_yaml(file_path: str | Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


model_meta_path = Path("/home/rjha5/603-nvme2/arena/model_meta.yml")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--limit", type=int, required=False, default=None)
    args = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    index_small_models(args.model_name, args.batch_size, args.limit)
