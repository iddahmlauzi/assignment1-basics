import regex as re
import os
from typing import BinaryIO

# Better to use re.ignorecase for contractions
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def pretokenize_chunk(corpus: str ,special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenizes one chunk: Used in Multi-processing
    """
    
    # Strip the special tokens from the corpus so no merging occurs across special token boundaries
    splits = re.split("|".join([re.escape(special_token) for special_token in special_tokens]), corpus)
    pretokens = {}
    for text in splits:
        matches = re.finditer(PAT, text)
        for match in matches:
            # Use the bytes representation of the token
            # A 3-byte character such as こ should be separated into 3 separate butes
            token_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            pretokens[token_bytes] = pretokens.get(token_bytes, 0) + 1
            
    return pretokens


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
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # So basically, I need to do pool.map(pretokenize_chunk, )



        
        
    



