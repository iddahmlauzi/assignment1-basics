import argparse
import pathlib
import os
import json
import cProfile
import pstats
import psutil
import time
from cs336_basics.tokenizer import pretokenize, train_bpe_tokenizer
from tests.common import gpt2_bytes_to_unicode

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
BYTES_TO_UNICODE = gpt2_bytes_to_unicode() 
DATASETS = {
        "tinystories": {
            True: "TinyStoriesV2-GPT4-valid.txt",
            False: "TinyStoriesV2-GPT4-train.txt"
        },
        "owt": {
            True: "owt_valid.txt",
            False: "owt_train.txt"
        }
}

def render(token_bytes: bytes) -> str:
    """
    Helps with visualizing the bytes (esp for unprintabel characters)
    """
    return "".join(BYTES_TO_UNICODE[b] for b in token_bytes)


def save_tokenizer(vocab, merges, name):
    """
    Serialize the resulting vocabulary and merges to disk for further inspection
    """
    # 1. Save Vocab as JSON
    vocab_path = OUTPUT_DIR / f"{name}-vocab.json"
    serialized_vocab = {render(t_bytes): t_id for t_id, t_bytes in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, indent=4, ensure_ascii=False)
        
    # 2. Save Merges as TXT
    merges_path = OUTPUT_DIR / f"{name}-merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{render(p1)} {render(p2)}\n")
            
    print(f"Successfully saved to {OUTPUT_DIR}")


def main(dataset: str, vocab_size: int, debug: bool):
    """
    Args:
        dataset (str): Which dataset to train BPE tokenizer on (either TinyStories or OpenWebText)
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        debug (bool): In debug mode, we downscale the training (i.e. we use the smaller validation set)
    """
    filename = DATASETS[dataset][debug]
    input_path = PROJECT_ROOT/ "data" / filename
    
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find the dataset at {input_path}")
    
    print(f"Training on {dataset}, (Debug: {debug})")
    print(f"Target vocab size: {vocab_size}")

    special_tokens = ["<|endoftext|>"]
    pretokens = pretokenize(input_path, special_tokens)
    vocab, merges = train_bpe_tokenizer(pretokens, vocab_size, special_tokens)
    save_name = f"train-bpe-{dataset}"
    if debug:
        save_name = f"train-bpe-{dataset}-debug"
    
    save_tokenizer(vocab, merges, save_name)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BPE Training on either TinyStories or OpenWebText")
    
    parser.add_argument("--dataset", type=str, choices=["tinystories", "owt"], help="Select the dataset to use")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum size of the vocabulary")
    parser.add_argument("--debug", action="store_true", help="Run on a tiny subset of the data for quick testing")
    
    args = parser.parse_args()
    profiler = cProfile.Profile()
    
    
    # Keep track of memory
    process = psutil.Process(os.getpid())
    mem_in_bytes = process.memory_info().rss
    mem_before = mem_in_bytes / (1024 ** 2)
    
    # Keep track of time
    start_time = time.time()
    
    
    # I'm gonna turn off the profiler for now
    profiler.run('main(**vars(args))')
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(50)
    
    # main(**vars(args))
    
    
    mem_in_bytes = process.memory_info().rss
    mem_after = mem_in_bytes / (1024 ** 2)
    end_time = time.time()
    
    mem_used = mem_after - mem_before
    run_time = end_time - start_time
    
    print(f"Memory used: {mem_used:.2f} MiB")
    print(f"Total Run time: {run_time} seconds")
    
