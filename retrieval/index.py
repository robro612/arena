# https://github.com/ContextualAI/gritlm/blob/9883da1e77812e6ba2c107dc7b65d8c5ddc7396b/rag/index.py
import json
import math
import os
import pickle
from typing import Optional, Set, Tuple, Union, Any
from tqdm.auto import tqdm, trange
import numpy as np
import torch
import sys
try:
    # Use rich tqdm only in interactive mode
    if sys.stdout.isatty():
        from tqdm.rich import tqdm, trange
    else:
        from tqdm import tqdm, trange
except ImportError:
    from tqdm import tqdm, trange
from retrieval import dist_utils

DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}

class DistributedIndex(object):
    def __init__(self, dtype=torch.float32):
        self.embeddings = None
        self.doc_map = dict()
        self.dtype = dtype

    def init_embeddings(self, passages, dim: Optional[int]):
        self.doc_map = {i: doc for i, doc in enumerate(passages)}
        self.embeddings = torch.zeros(dim, (len(passages)), dtype=self.dtype)
        # Embeddings are created on CPU by default, use to_gpu() for explicit GPU movement

    def _get_saved_embedding_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"embeddings.{shard}.pt")

    def _get_saved_passages_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"passages.{shard}.pt")

    def save_index(self, path: str, total_saved_shards: int = 1, overwrite_saved_passages: bool = False) -> None:
        """
        Saves index state to disk, which can later be loaded by the load_index method.
        Specifically, it saves the embeddings and passages into total_saved_shards separate file shards.
        This option enables loading the index in another session with a different number of workers, as long as the number of workers is divisible by total_saved_shards.
        Note that the embeddings will always be saved to disk (it will overwrite any embeddings previously saved there).
        The passages will only be saved to disk if they have not already been written to the save directory before, unless the option --overwrite_saved_passages is passed.
        """
        assert self.embeddings is not None
        rank = dist_utils.get_rank()
        ws = dist_utils.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        n_embeddings = self.embeddings.shape[1]
        embeddings_per_shard = math.ceil(n_embeddings / shards_per_worker)
        assert n_embeddings == len(self.doc_map), len(self.doc_map)
        for shard_ind, (shard_start) in enumerate(range(0, n_embeddings, embeddings_per_shard)):
            shard_end = min(shard_start + embeddings_per_shard, n_embeddings)
            shard_id = shard_ind + rank * shards_per_worker  # get global shard number
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            if not os.path.exists(passage_shard_path) or overwrite_saved_passages:
                passage_shard = [self.doc_map[i] for i in range(shard_start, shard_end)]
                with open(passage_shard_path, "wb") as fobj:
                    pickle.dump(passage_shard, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            embeddings_shard = self.embeddings[:, shard_start:shard_end]#.clone()
            embedding_shard_path = self._get_saved_embedding_path(path, shard_id)
            torch.save(embeddings_shard, embedding_shard_path)

    def load_index(self, path: str, dtype : Optional[str] = None, verbose: bool = False):
        """
        Loads sharded embeddings and passages files (no index is loaded).
        Automatically discovers all embedding and passage shards in the directory.
        """
        # Discover all embedding shards (support old and new naming)
        embedding_entries = []  # list[(shard_id:int, filename:str)]
        invalid_embedding_filenames = []
        for f in os.listdir(path):
            if not f.endswith('.pt') or not f.startswith('embeddings.'):
                continue
            parts = f.split('.')
            # We only care that parts[0] == 'embeddings' and parts[1] is an int
            if len(parts) >= 3 and parts[0] == 'embeddings' and parts[-1] == 'pt':
                try:
                    shard_id = int(parts[1])
                    embedding_entries.append((shard_id, f))
                except ValueError:
                    invalid_embedding_filenames.append(f)
        if invalid_embedding_filenames:
            raise ValueError(
                "Invalid embedding shard filename(s) found (expected 'embeddings.{i}.pt' or 'embeddings.{i}.{total}.pt'):\n"
                + "\n".join(invalid_embedding_filenames)
            )
        embedding_entries.sort(key=lambda x: x[0])
        total_embedding_shards = len(embedding_entries)

        # Discover all passage shards (old style only for now)
        passage_entries = []  # list[(shard_id:int, filename:str)]
        invalid_passage_filenames = []
        for f in os.listdir(path):
            if not f.endswith('.pt') or not f.startswith('passages.'):
                continue
            parts = f.split('.')
            # Expect at minimum ['passages', '{i}', 'pt']
            if len(parts) >= 3 and parts[0] == 'passages' and parts[-1] == 'pt':
                try:
                    shard_id = int(parts[1])
                    passage_entries.append((shard_id, f))
                except ValueError:
                    invalid_passage_filenames.append(f)
        if invalid_passage_filenames:
            raise ValueError(
                "Invalid passage shard filename(s) found (expected 'passages.{i}.pt'):\n"
                + "\n".join(invalid_passage_filenames)
            )
        passage_entries.sort(key=lambda x: x[0])
        total_passage_shards = len(passage_entries)
        
        # Load all passage shards
        passages = []
        for shard_id, fname in tqdm(passage_entries, desc="Loading passage shards", disable=not verbose):
            passage_shard_path = os.path.join(path, fname)
            with open(passage_shard_path, "rb") as fobj:
                passages.append(pickle.load(fobj))
        
        # Load all embedding shards
        embeddings = []
        for shard_id, fname in tqdm(embedding_entries, desc="Loading embedding shards", disable=not verbose):
            embeddings_shard_path = os.path.join(path, fname)
            # Always load to CPU initially, use to_gpu() for explicit GPU movement
            shard_embeddings = torch.load(embeddings_shard_path, map_location="cpu", weights_only=True)
            if dtype is not None:
                shard_embeddings = shard_embeddings.to(dtype=DTYPE_TO_TORCH_DTYPE[dtype])
            embeddings.append(shard_embeddings)
            torch.cuda.empty_cache()
        
        # Build doc_map from all passage shards
        self.doc_map = {}
        n_passages = 0
        for chunk in passages:
            for p in chunk:
                self.doc_map[n_passages] = p
                n_passages += 1
        
        # Concatenate embeddings
        if len(embeddings) > 1:
            self.embeddings = torch.concat(embeddings, dim=1)
        else:
            self.embeddings = embeddings[0]
        self.dtype = self.embeddings.dtype

    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        # TODO: Switch to cosine sim?
        # (nqueries, dim) x (dim, npassages) -> (nqueries, npassages)
        scores = torch.matmul(allqueries.to(self.embeddings.device), self.embeddings)
        scores, indices = torch.topk(scores, topk, dim=1)
        return scores, indices

    @torch.no_grad()
    def search_knn(self, queries, topk) -> tuple[list[list[str]], list[list[float]], list[list[int]]]:
        """
        Conducts exhaustive search of the k-nearest neighbours using the inner product metric.
        returns list (query) of list (topk) of docs, list (query) of list (topk) of scores, and list (query) of list (topk) of doc indices
        """
        allqueries = dist_utils.varsize_all_gather(queries)
        allsizes = dist_utils.get_varsize(queries)
        allsizes = np.cumsum([0] + allsizes.cpu().tolist())
        # compute scores for the part of the index located on each process
        scores, indices = self._compute_scores_and_indices(allqueries, topk)
        indices_list = indices.tolist()
        docs = [[self.doc_map[x] for x in sample_indices] for sample_indices in indices_list]
        if torch.distributed.is_initialized():
            docs = [docs[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            docs = [serialize_listdocs(x) for x in docs]
            scores = [scores[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            indices_chunks = [indices[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            gather_docs = [dist_utils.varsize_gather(docs[k], dst=k, dim=0) for k in range(dist_utils.get_world_size())]
            gather_scores = [
                dist_utils.varsize_gather(scores[k], dst=k, dim=1) for k in range(dist_utils.get_world_size())
            ]
            gather_indices = [
                dist_utils.varsize_gather(indices_chunks[k], dst=k, dim=1) for k in range(dist_utils.get_world_size())
            ]
            rank_scores = gather_scores[dist_utils.get_rank()]
            rank_docs = gather_docs[dist_utils.get_rank()]
            rank_indices = gather_indices[dist_utils.get_rank()]
            scores = torch.cat(rank_scores, dim=1)
            indices = torch.cat(rank_indices, dim=1)
            rank_docs = deserialize_listdocs(rank_docs)
            merge_docs = [[] for _ in range(queries.size(0))]
            for docs in rank_docs:
                for k, x in enumerate(docs):
                    merge_docs[k].extend(x)
            docs = merge_docs
            indices_list = indices.tolist()
        _, subindices = torch.topk(scores, topk, dim=1)
        scores = scores.tolist()
        subindices = subindices.tolist()
        # Extract topk scores and associated ids
        scores = [[scores[k][j] for j in idx] for k, idx in enumerate(subindices)]
        docs = [[docs[k][j] for j in idx] for k, idx in enumerate(subindices)]
        doc_indices = [[indices_list[k][j] for j in idx] for k, idx in enumerate(subindices)]
        return docs, scores, doc_indices

    def is_index_trained(self) -> bool:
        return True
    
    def to_gpu(self):
        """
        Explicitly move embeddings to GPU if available.
        """
        if torch.cuda.is_available() and self.embeddings is not None:
            if self.embeddings.device != "cuda":
                self.embeddings = self.embeddings.cuda()
                print("Moved embeddings to GPU")
            else:
                print("Embeddings already on GPU")
        else:
            if not torch.cuda.is_available():
                print("CUDA not available, cannot move to GPU")
            else:
                print("No embeddings to move to GPU")
    
    def to_cpu(self):
        """
        Explicitly move embeddings to CPU.
        """
        if self.embeddings is not None:
            if self.embeddings.device != "cpu":
                self.embeddings = self.embeddings.cpu()
                print("Moved embeddings to CPU")
            else:
                print("Embeddings already on CPU")
        else:
            print("No embeddings to move to CPU")

def load_or_initialize_index(load_index_path=None, dim=None, index_dtype='bfloat16', save_index_n_shards=1, passages=None, limit=None, customd=None):
    """
    load_index_path:
        path for loading the index, passage embeddings and passages
    save_index_n_shards:
        how many shards to save an index to file with. Must be an integer multiple of the number of workers
    passages:
        list of paths to jsonl files containing passages to index and retrieve from. Unused if load_index_path is set.
    """
    index = DistributedIndex(dtype=DTYPE_TO_TORCH_DTYPE[index_dtype])

    if load_index_path is not None:
        print(f"Loading index from: {load_index_path}")
        index.load_index(load_index_path, save_index_n_shards)
        passages = [index.doc_map[i] for i in tqdm(range(len(index.doc_map)))]
    else:
        if limit is not None:            
            passages = passages[:limit]
            print(f"Limiting to {len(passages)} passages")
        if customd:
            if os.path.exists(customd):
                with open(customd, "r") as f:
                    passages = [{"text": f.read(), "title": ""}]
            else: # Is number
                passages = [{"text": "<s>" * int(customd), "title": ""}]
        # print(f"Example passage: {passages[0]}")
        index.init_embeddings(passages, dim)

    return index, passages

@torch.no_grad()
def build_index(model, index, passages, gpu_embedder_batch_size=512, accumulation_batches=20):
    """
    Build index with batch accumulation to reduce transfer overhead.
    
    Args:
        accumulation_batches: Number of batches to accumulate before transferring to index
    """
    n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
    total = 0
    encode_kwargs = {"show_progress_bar" : False}
    
    # Check dtype/device compatibility once at the start
    index_device = index.embeddings.device
    index_dtype = index.dtype
    
    # Always use accumulation strategy
    embeddings_list = []
    batch_sizes = []
    
    for i in trange(n_batch, desc=f"Encoding passages [bs={gpu_embedder_batch_size} acc={accumulation_batches}]"):
        batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
        #, instruction=gritlm_instruction_format())
        embeddings = model.encode(batch, convert_to_tensor=True, batch_size=gpu_embedder_batch_size, **encode_kwargs)
        
        if not isinstance(embeddings, torch.Tensor):
            if i == 0: 
                print(f"model.encode returned non-tensor of type {type(embeddings)} when convert_to_tensor=True")
            embeddings = torch.tensor(embeddings, dtype=index_dtype, device=index_device)
        else:
            # Warn about dtype mismatch only once
            if i == 0 and embeddings.dtype != index_dtype:
                print(f"embeddings.dtype: {embeddings.dtype}, embeddings.device: {embeddings.device}")
                print(f"index.dtype: {index_dtype}, index.embeddings.device: {index_device}")
                print(f"WARNING: {embeddings.dtype=} != {index_dtype=}, converting embeddings to index.dtype at every batch is slow")
            
            # Keep original dtype for accumulation, convert at transfer time
            embeddings_list.append(embeddings.T)  # Transpose on current device
            batch_sizes.append(len(embeddings))
            
            # Transfer accumulated batches when we hit the accumulation limit or at the end
            if len(embeddings_list) >= accumulation_batches or i == n_batch - 1:
                # Concatenate on current device, then convert dtype + device in one operation
                accumulated_embeddings = torch.cat(embeddings_list, dim=1)
                accumulated_size = sum(batch_sizes)
                
                # Single conversion: dtype + device transfer
                index.embeddings[:, total : total + accumulated_size] = accumulated_embeddings.to(dtype=index_dtype, device=index_device)
                
                total += accumulated_size
                embeddings_list.clear()
                batch_sizes.clear()
                
    dist_utils.barrier()
    print(f"{len(passages)} passages encoded on process: {dist_utils.get_rank()}")