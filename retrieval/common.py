from datasets import load_dataset
import json
from . import dist_utils
import torch
import numpy as np
from typing import Optional


CORPORA = {
    "wikipedia": {"name": "mteb/arena-wikipedia-7-15-24", "columns": {"id": "_id", "text": "text", "title": "title"}, "format" : "{title}\n\n{text}"},
    "arxiv": {"name": "mteb/arena-arxiv-7-2-24", "columns": {"id": "_id", "abstract": "text", "title": "title"}, "format" : "Title: {title}\n\nAbstract: {text}"},
    "stackexchange": {"name": "mteb/arena-stackexchange", "columns": {"id": "_id", "text": "text"}, "format" : "{text}"},
}

def force_to_tensor(x : torch.Tensor | np.ndarray | list, device: Optional[str] = None, dtype: Optional[torch.dtype] = None, verbose: bool = False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if verbose:
            print(f"x is already a torch.Tensor with shape {x.shape}, dtype {x.dtype}")
        return x
    elif isinstance(x, np.ndarray):
        if verbose:
            print(f"x is a numpy array with shape {x.shape}, dtype {x.dtype}")
        return torch.from_numpy(x).to(device, dtype)
    elif isinstance(x, list):
        if verbose:
            print(f"x is a list with outer length {len(x)}")
        return torch.tensor(x, device=device, dtype=dtype)
    else:
        if verbose:
            print(f"Unsupported type: {type(x)}")
        raise ValueError(f"Unsupported type: {type(x)}")

def load_passages(origin: str | list[str], limit: int = None) -> list[dict[str, str]]:
    if isinstance(origin, str) and origin in CORPORA:
        print(f"loading {origin} corpus from HF")
        return load_passages_from_hf(corpus=origin, limit=limit)
    else:
        print(f"loading corpus locally from {origin}")
        return load_passages_from_local_files(filenames=origin, limit=limit)

def load_passages_from_hf(corpus: str, limit: int = None) -> list[dict[str, str]]:
    """Returns a list of passages. Each passage is a dict with keys defined in CORPORA"""
    if CORPORA.get(corpus) is None:
        raise NotImplementedError(f"Corpus={corpus} is not found. Currently supported: {list(CORPORA.keys())}.")
    corpus_dict = CORPORA[corpus]
    ds = load_dataset(corpus_dict['name'], split="train")
    # Rename & remove cols
    ds = ds.rename_columns(corpus_dict['columns'])
    ds = ds.remove_columns([col for col in ds.column_names if col not in corpus_dict['columns'].values()])
    if limit and limit > 1:
        ds = ds.take(limit)
    return ds.to_list()

def load_passages_from_local_files(filenames : list[str], limit : int = None) -> list[dict[str, str]]:
    """ 
    Returns a list of passages. Each passage is a dict with the following keys:
    {
        "_id:" doc0,
        "title": "Title 1",
        "text": "Body text 1",
    }
    """
    def process_jsonl(
        fname,
        counter,
        corpus,
        world_size,
        global_rank,
        limit,
    ):
        def load_item(line):
            if line.strip() != "":
                item = json.loads(line)
                if "title" in item and "section" in item and len(item["section"]) > 0:
                    item["title"] = f"{item['title']}: {item['section']}"
                return item
            else:
                print("empty line")

        for line in open(fname):
            if limit and counter >= limit:
                break

            ex = None
            if (counter % world_size) == global_rank:
                ex = load_item(line)
                corpus.append(ex)
            counter += 1
        return corpus, counter

    counter = 0
    passages = []
    global_rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    for filename in filenames:

        passages, counter = process_jsonl(
            filename,
            counter,
            passages,
            world_size,
            global_rank,
            limit,
        )

    return passages