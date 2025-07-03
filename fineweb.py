"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard

# tokenizer and end-of-text token
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    text = doc.get("text", "")
    if not text:
        return np.array([eot], dtype=np.uint16)
    tokens = [eot] + enc.encode_ordinary(text)
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    # setup local save directory
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # stream dataset (requires login if gated)
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        remote_name,
        split="train",
        streaming=True,
    )

    # set number of workers
    nprocs = max(1, os.cpu_count() // 2)
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Final shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
