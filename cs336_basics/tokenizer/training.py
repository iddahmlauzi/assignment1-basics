import heapq
from .utils import BytePair

# TODO: Figure out how to not use the custom Bytes object. Need to figure out how to do lexicographically greater sorting 
# TODO: Store a list of (Pretoken, count) tuples. Index should map pair to a set of the indices of the relevant pretokens (can use list instead of set perhaps cause if we store the indices then havng stale entries shouldnt have as much memory overhead...)


def merge(
    pair: tuple[bytes, bytes],
    pretokens: dict[tuple[bytes, ...], int],
    index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair_counts: dict[tuple[bytes, bytes], int],
    heap: list[BytePair]
) -> None:
    """
    Given a byte pair, merges it into the pretokens and updates the index and pair counts
    """
    
    pair_count_updates = {}
    for pretoken in index[pair]:
        # Skip stale entries --> intentionally left stale entries in indes[pair] to avoid set size change while looping
        if pretoken not in pretokens:
            continue
        count = pretokens[pretoken]
        new_pretoken = []
        i = 0 
        while i < len(pretoken):
            # Merge the pair
            if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
                new_bytes = b''.join(pair)
                new_pretoken.append(new_bytes)                 
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1
                
        # Update index with new pretoken information
        new_pretoken = tuple(new_pretoken)
        for p in zip(new_pretoken, new_pretoken[1:]):
            if p not in index:
                index[p] = set()
            index[p].add(new_pretoken)
            pair_count_updates[p] = pair_count_updates.get(p, 0) + count 
            
            
        # Update index by deleting references to old pretoken
        # We could keep stale references but this might lead to memory bloat
        for p in zip(pretoken, pretoken[1:]):
            # We want to skip over current pair cause we are looping over index[pair]
            # So we don't want to change size of set we are looping ober
            if p != pair:
                index[p].discard(pretoken)
            pair_count_updates[p] = pair_count_updates.get(p, 0) - count
    
                
        # Update the pretokens dict
        pretokens[new_pretoken] = count
        del pretokens[pretoken]
    
    # Now we can remove the pair from the index safely after iterating through it all
    del index[pair]
    
    # Update all the pair counts         
    for p in pair_count_updates:
        pair_counts[p] = pair_counts.get(p, 0) + pair_count_updates[p]
        # This might mean pushing the same pair multiple times onto the heap
        # But we will skip over stale ones when we pick the next pair
        # That is, if the next pair's count is not up to date with what is in the pair counts,
        # We consider it to be stale
        heapq.heappush(heap, BytePair(p, pair_counts[p]))
        if pair_counts[p] <= 0:
            del pair_counts[p]
        
    

def train_bpe_tokenizer(
    pretokens: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the counts obtained from pretokenization, trains a BPE tokenizer and
    output its vocabulary and merges.
    Args:
        pretokens: (dict[tuple[bytes, ...], int]): Maps each pretoken to it's frequency in the corpus
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
    
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    # Initialize vocab to incluce initial byte vocabulary and any special tokens
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(len(special_tokens)):
        vocab[256 + i] = bytes(special_tokens[i].encode("utf-8"))
        
    num_merges = vocab_size - len(vocab)
    pair_counts: dict[tuple[bytes, bytes], int] = {} # pair frequencies
    index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {} # which pretokens contain each pair
    
    # We use an index to keep track of which pretokens contain each pair
    # This will help when merging to avoid looping over the entire pretoken dict
    for pretoken in pretokens:
        count = pretokens[pretoken]
        for pair in zip(pretoken, pretoken[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            if pair not in index:
                index[pair] = set()
            index[pair].add(pretoken)

            
    # Make the priority queue --> Will help with finding best pair quicker
    heap = [BytePair(pair, pair_counts[pair]) for pair in pair_counts]
    heapq.heapify(heap)      
    merges = []
    
    for i in range(num_merges):
        
        # Nothing left to merge
        if not heap:
            break
        
        # Find a non-stale pair to merge
        while heap:
            byte_pair = heapq.heappop(heap)
            if byte_pair.count == pair_counts.get(byte_pair.pair, 0):
                break
        
        # Edge case: Last pair could be stale
        if byte_pair.count != pair_counts.get(byte_pair.pair, 0):
            break
        
        # Need to change what's below:
        merge(byte_pair.pair, pretokens, index, pair_counts, heap)
        merges.append(byte_pair.pair)
        idx = len(vocab)
        vocab[idx] = b''.join(byte_pair.pair)
        
    return vocab, merges

                    
            
        
    
    
    
    
    
