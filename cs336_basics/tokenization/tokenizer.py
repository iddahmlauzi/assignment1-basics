from collections.abc import Iterable, Iterator
from functools import partial
import regex as re
import os 
from multiprocessing import Pool
import itertools
from .utils import gpt2_bytes_to_unicode
import json

# TODO: Think about how we can possibly use a heap or a doubly linked list instead

# define a function to pretokenize 
RE_PAT = re.compile(br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _find_text_boundaries(data: memoryview, 
                          num_chunks: int, 
                          split_special_token: bytes = b" ") -> list[tuple[int, int]]:
    """
    Chunk the data into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    We use the space as the split token because no mutli-byte characters can contain the space
    """
    total_size = len(data)
    chunk_size = total_size // num_chunks
    
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = total_size
    
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        while True:
            # Start at boundary guess and read a mini chunk
            mini_chunk = data[initial_position: initial_position + mini_chunk_size]

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = total_size
                break

            # Find the special token in the mini chunk
            found_at = bytes(mini_chunk).find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    boundaries = sorted(set(chunk_boundaries))
    return zip(boundaries[:-1], boundaries[1:])
    
    
    
def _merge_pretoken(pretoken: bytes,
                    rank_map: dict[tuple[bytes, bytes], int],
                    vocab_map: dict[bytes, int]) -> list[int]:
    """
    Given a pretoken, continuously applies merges to get the token IDs
    Args:
        pretoken (bytes): byte string of the pretoken
        rank_map (dict[tuple[bytes, bytes], int]): maps each byte pair to it's index in the merges list
        vocab_map: dict[bytes, int]: Map from inverting the vocab dict so we can map bytes to their encoding int
    Returns:
        ids (list[int]): The sequence of token IDs after encoding
    """
    current_pretoken = list(memoryview(pretoken).cast("c"))
    while True:
        pairs = []
        # 1. Find pair to merge
        for pair in zip(current_pretoken, current_pretoken[1:]):
            pairs.append(pair)
        if not pairs:
            break
        merge_pair = min(pairs, key= lambda pair: rank_map.get(pair, float("inf")))
        # 2. No more pairs --> we end
        if merge_pair not in rank_map:
            break
        # 3. Otherwise --> merge
        new_pretoken = []
        i = 0 
        while i < len(current_pretoken):
            # Merge the pair
            if i < len(current_pretoken) - 1 and current_pretoken[i] == merge_pair[0] and current_pretoken[i + 1] == merge_pair[1]:
                new_bytes = b''.join(merge_pair)
                new_pretoken.append(new_bytes)                 
                i += 2
            else:
                new_pretoken.append(current_pretoken[i])
                i += 1
        current_pretoken = new_pretoken
        
    return [vocab_map[t] for t in current_pretoken]
    
    

def _encode_chunk(bounds: tuple[int, int],
                  data: bytes,
                  split_pat: re.Pattern[bytes],
                  special_tokens: set[bytes],
                  rank_map: dict[tuple[bytes, bytes], int],
                  vocab_map: dict[bytes, int]) -> list[int]:
    """
    Encodes a chunk of the data and returns the IDs
    Args:
        bounds: tuple[int, int]: the start, end boundaries of the chunk to pretokenize
        data (bytes): The text we are encoding
        split_pat: precompiled regex to split the chunk on every special token (don't want to merge across special tokens)
        special_tokens (set[bytes]): A list of string special tokens to be added to the tokenizer vocabulary.
        rank_map (dict[tuple[bytes, bytes], int]): maps each byte pair to it's index in the merges list
        vocab_map: dict[bytes, int]: Map from inverting the vocab dict so we can map bytes to their encoding int
    Returns:
        ids (list[int]): The sequence of token IDs after encoding
    """
    
    start, end = bounds
    chunk = data[start:end]
    ids = []
    
    # 1. Split by special tokens --> Get a list of texts (includes the special tokens)
    texts = split_pat.splititer(chunk) if split_pat else [chunk]
    for text in texts:
        if not text:
            continue
        # Special tokens can be found in vocab and added to ids directly
        if text in special_tokens:
            ids.append(vocab_map[text])
            continue
        # Otherwise, we need to split the text into its pretokens
        for match in RE_PAT.finditer(text):
            pretoken_bytes = match.group()
            ids.extend(_merge_pretoken(pretoken_bytes, rank_map, vocab_map))
            
    return ids


class Tokenizer:
    """Construct a tokenizer from a given vocabulary, 
    list of merges, and (optionally) a list of special tokens. """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        
        self.vocab = vocab
        # Make an inverted vocabulary dictionary so we can get the token IDs more efficiently
        self.vocab_map = {v: k for k, v in self.vocab.items()}
        
        self.special_tokens = set()
        if special_tokens:
            self.special_tokens = {s.encode("utf-8") for s in special_tokens}
            # Support user provided special tokens (appending them to the vocabulary if they aren’t already there)
            for token in self.special_tokens:
                if token not in self.vocab_map:
                    token_id = len(self.vocab)
                    self.vocab[token_id] = token
                    self.vocab_map[token] = token_id
                    
        self.merges = merges # dont think I need to store this anymore
        self.num_processes = 4
        
        self.rank_map = {}
        # Use an index to map each merged byte pair to the index it occurs in merges
        for i, merge in enumerate(merges):
            self.rank_map[merge] = i

        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges
        # This utility is taken from the test_tokenizer file (though it seems we don't test it)
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
            
        vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }

        gpt2_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_merges.append(tuple(cleaned_line.split(" ")))
                    
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_merges
        ]
            
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        # Find the chunk boundaries
        data = text.encode("utf-8")
        bounds = _find_text_boundaries(memoryview(data), self.num_processes)
        
        # Precompile the pattern to split text at each special token
        split_pat = None
        if self.special_tokens:
            # Sort by length DESCENDING to avoid prefix matching issues
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            tokens = (re.escape(t) for t in sorted_tokens)
            # We add the parenthesis to make a capturing group --> So we capture the speicial
            # tokens as well when matching
            pattern = b"(" + b"|".join(tokens) + b")"
            split_pat = re.compile(pattern)
            
            
        # Step 2: Dispatch (start, end) tuples to a worker function
        worker_fn = partial(_encode_chunk, 
                            data=data, 
                            split_pat=split_pat, 
                            special_tokens=self.special_tokens, 
                            rank_map=self.rank_map,
                            vocab_map = self.vocab_map)
        with Pool(self.num_processes) as p:
            token_ids = p.map(worker_fn, bounds)
            
        return [id for ids in token_ids for id in ids]
            
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that 
        lazily yields token IDs. This is required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        
        # Precompile the pattern to split text at each special token
        split_pat = None
        if self.special_tokens:
            # Sort by length DESCENDING to avoid prefix matching issues
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            tokens = (re.escape(t) for t in sorted_tokens)
            # We add the parenthesis to make a capturing group --> So we capture the speicial
            # tokens as well when matching
            pattern = b"(" + b"|".join(tokens) + b")"
            split_pat = re.compile(pattern)
            
        
        for text in iterable:
            # Find the chunk boundaries
            data = text.encode("utf-8")
            bounds = (0, len(data))
            ids = _encode_chunk(bounds=bounds, 
                                data=data, 
                                split_pat=split_pat, 
                                special_tokens=self.special_tokens, 
                                rank_map=self.rank_map, 
                                vocab_map=self.vocab_map)
            yield from ids 
    
        
    
    def decode(self, ids: list[int]) -> str:
        """ Decode a sequence of token IDs into text.""" 
        # Should I check to see if it is in vocab (id might not be in vocab)
        tokens = b''.join(self.vocab[id] for id in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
        
        
    