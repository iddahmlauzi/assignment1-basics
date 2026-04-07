import regex as re 
from cs336_basics.tokenization import Tokenizer
from typing import Literal
import numpy as np
from itertools import islice
import time
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
# Create a dedicated directory on the actual LFS storage
OUTPUT_DIR = pathlib.Path("/lfs/skampere2/0/iddah/encoded_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"


SPECIAL_TOKEN = "<|endoftext|>"

TOKENIZER_PATHS = {
    "tinystories": {
        "vocab": "train-bpe-tinystories-vocab.json",
        "merges": "train-bpe-tinystories-merges.txt"
    },
    "openwebtext": {
        "vocab": "train-bpe-owt-vocab.json",
        "merges": "train-bpe-owt-merges.txt"
    },
}
DATASET_PATHS = {
        "tinystories": {
            "valid": "TinyStoriesV2-GPT4-valid.txt",
            "train": "TinyStoriesV2-GPT4-train.txt"
        },
        "openwebtext": {
            "valid": "owt_valid.txt",
            "train": "owt_train.txt"
        }
}

def stream_document(dataset: Literal["tinystories", "openwebtext"],
                     split: Literal["train", "valid"]):
    """Given a dataset, creates a generstor to lazily sample documents from it"""
    dataset_path = DATA_DIR / DATASET_PATHS[dataset][split]
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        buffer = ""
        for line in f:
            buffer += line
            # Check if the delimiter exists in our current accumulated text
            while SPECIAL_TOKEN in buffer:
                # Split at the first occurrence of the token
                doc, buffer = buffer.split(SPECIAL_TOKEN, 1)
                if doc.strip():
                    yield doc.strip()
            
            
# def encode_dataset(dataset: Literal["tinystories", "openwebtext"], 
#                    split: Literal["train", "valid"], 
#                    tokenizer: Tokenizer):
#     """Given the name of the dataset to encode, encodes the dataset and writes the output"""
#     dataset_path = DATA_DIR / DATASET_PATHS[dataset][split]
#     output_path = OUTPUT_DIR / f"{dataset}-{split}-encoded.bin"
#     with open(dataset_path, 'r') as infile, open(output_path, 'ab') as outfile:
#         # Lazy token generator
#         token_stream = tokenizer.encode_iterable(infile)
#         chunk = []
#         # Process in chunks to avoid calling .tofile() millions of times
#         chunk_size = 1000
#         for token_id in token_stream:
#             chunk.append(token_id)
#             if len(chunk) >= chunk_size:
#                 np.array(chunk, dtype='uint16').tofile(outfile)
#                 chunk = []
#         # Last chunk : ) 
#         if chunk:
#             np.array(chunk, dtype='uint16').tofile(outfile)

def encode_dataset(dataset: Literal["tinystories", "openwebtext"], 
                   split: Literal["train", "valid"], 
                   tokenizer: Tokenizer):
    dataset_path = DATA_DIR / DATASET_PATHS[dataset][split]
    output_path = OUTPUT_DIR / f"{dataset}-{split}-encoded.bin"
    
    # Read everything into memory
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Encode the full string at once: doing this for speed (not a good idea though)
    token_ids = tokenizer.encode(text)
    
    # Convert to numpy and save
    np.array(token_ids, dtype='uint16').tofile(output_path)
            
def get_stats(samples, tokenizer):
    """Given a list of sample documents and a tokenizer, calculates the compression ratio"""
    total_bytes = 0
    total_tokens = 0
    
    for doc in samples:
        # Original size in bytes
        total_bytes += len(doc.encode("utf-8"))
        # Token count
        tokens = list(tokenizer.encode_iterable([doc])) 
        # tokens = tokenizer.encode(doc)
        total_tokens += len(tokens)
        
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    return ratio, total_bytes, total_tokens
            
            

if __name__ == "__main__":
    # Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained
    # TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
    # encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio
    # (bytes/token)?
    print("--- Section (a) & (b): Compression Ratios ---")
    tinystories_tokenizer = Tokenizer.from_files(vocab_filepath=OUTPUT_DIR / TOKENIZER_PATHS["tinystories"]["vocab"],
                                                 merges_filepath=OUTPUT_DIR / TOKENIZER_PATHS["tinystories"]["merges"],
                                                 special_tokens=[SPECIAL_TOKEN])
    
    owt_tokenizer = Tokenizer.from_files(vocab_filepath=OUTPUT_DIR / TOKENIZER_PATHS["openwebtext"]["vocab"],
                                                merges_filepath=OUTPUT_DIR / TOKENIZER_PATHS["openwebtext"]["merges"],
                                                special_tokens=[SPECIAL_TOKEN])
    
    tokenizers = {"tinystories": tinystories_tokenizer, "openwebtext": owt_tokenizer}
    tinystories_samples = list(islice(stream_document("tinystories", "train"), 10))
    owt_samples = list(islice(stream_document("openwebtext", "train"), 10))

    ratio_a1, _, _ = get_stats(tinystories_samples, tinystories_tokenizer)
    ratio_a2, _, _ = get_stats(owt_samples, owt_tokenizer)
    ratio_b, _, _ = get_stats(owt_samples, tinystories_tokenizer)

    print(f"TinyStories Tokenizer: {ratio_a1:.3f} bytes/token")
    print(f"OpenWebText Tokenizer: {ratio_a2:.3f} bytes/token")
    print(f"TinyStories Tokenizer on OWT (Cross-test): {ratio_b:.3f} bytes/token")

    # --- Part (c) ---
    print("\n--- Section (c): Throughput & The Pile ---")
    # We use the valid split for a quick throughput estimate
    dataset_to_test = "tinystories"
    split_to_test = "valid"
    
    start_time = time.perf_counter()
    encode_dataset(dataset_to_test, split_to_test, tokenizers[dataset_to_test])
    elapsed = time.perf_counter() - start_time
    
    file_size = (DATA_DIR / DATASET_PATHS[dataset_to_test][split_to_test]).stat().st_size
    throughput = file_size / elapsed # bytes per second
    
    pile_size = 825 * 10**9 # 825 GB
    pile_time_hours = (pile_size / throughput) / 3600

    print(f"Estimated Throughput: {throughput / 1e6:.2f} MB/s")
    print(f"Deliverable (c): At this rate, tokenizing the 825GB Pile dataset would take approximately {pile_time_hours:.2f} hours.")

    # --- Part (d) ---
    print("\n--- Section (d): Full Dataset Encoding ---")
    for dataset in DATASET_PATHS:
        tokenizer = tokenizers[dataset]
        for split in DATASET_PATHS[dataset]:
            print(f"Encoding {dataset}-{split}.........")
            encode_dataset(dataset, split, tokenizer)
            
    print("Done encoding all datasets.")