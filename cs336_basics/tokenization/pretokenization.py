import regex as re
import os
from collections import Counter
from functools import partial
from typing import BinaryIO
from multiprocessing import Pool
import mmap

# Better to use re.ignorecase for contractions
# Precompile the pattern
RE_PAT = re.compile(br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(bounds: tuple[int, int],
                      input_path: str | os.PathLike,
                      special_tokens: list[bytes],
                      split_pat) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenizes the given chunk from the input corpus
    Args:
        bounds: tuple[int, int]: the start, end boundaries of the chunk to pretokenize
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be kept as a single token.
        split_pat: precompiled regex to split
    Returns:
        pretokens (dict[tuple[bytes, ...], int]):
            A Counter mapping each pre-token to the number of times it appears in the corpus. 
    """
    start, end = bounds
    raw_counts = Counter()
    with open(input_path, "rb") as f:
        # Trying to make code faster haha
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        chunk = memoryview(mm)[start: end]
        # Strip the special tokens from the corpus so no merging occurs across special token boundaries
        for text in  split_pat.splititer(chunk):
            if not text:
                continue
            for match in RE_PAT.finditer(text):
                # Use the bytes representation of the token
                # A 3-byte character such as こ should be separated into 3 separate bytes
                # This should be faster than a python comprehension cause it uses C
                raw_counts[match.group()] += 1
    
    # Final transformation: Convert unique bytes to tuples of individual bytes.
    # We do this only once per unique token to save massive overhead.
    return {
        tuple(memoryview(token).cast('c')): count 
        for token, count in raw_counts.items()
    }


def pretokenize(input_path: str | os.PathLike,
                special_tokens: list[str]
    ) -> dict[tuple[bytes, ...], int]:
    """
    Given the path to an input corpus and a list of special tokens, 
    pretokenizes the corpus and returns a frequency table for the pretokens
    
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be kept as a single token.
        
    Returns:
        pretokens (dict[tuple[bytes, ...], int]):
            A dict mapping each pre-token to the number of times it appears in the corpus
            The tokens are stored as tuples of individual bytes e.g.
            {
            (b'h', b'e', b'l', b'l', b'o'): 5,
            (b'w', b'o', b'r', b'l', b'd'): 3,
            }
    """
    from time import time
    start_time = time()
    
    # Automatically set the number of processes
    # num_processes = os.cpu_count() - 2
    num_processes = 4
    
    # Pre-compile the special token splitter once
    split_pat = re.compile(b"|".join([re.escape(special_token.encode('utf-8')) for special_token in special_tokens]))
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        bounds =  zip(boundaries[:-1], boundaries[1:])
        
    # Pretokenize in parallel
    worker_fn = partial(pretokenize_chunk, input_path=input_path, special_tokens=special_tokens, split_pat=split_pat)
    with Pool(num_processes) as p:
        pretoken_dicts = p.map(worker_fn, bounds)
    
    pretokens = Counter()
    for pretoken_dict in pretoken_dicts:
        pretokens.update(pretoken_dict)
        
    time_elapsed = time() - start_time
    print(f"Pretokenization took {time_elapsed:.2f} s")
    return pretokens
 


        
        
    



