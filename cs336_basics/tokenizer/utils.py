
class BytePair:
    """
    A wrapper around a byte pair and its count for use in a min-heap.
    Implements __lt__ so that heapq behaves as a max-heap, with ties
    broken by lexicographically greater pair.
    """
    def __init__(self, pair, count):
        self.pair = pair
        self.count = count
        
        # Use ORD
    
    # Override the "less than" operator for heapq
    def __lt__(self, other):
        # 1. Compare the count values (Simulate max heap beahvior)
        if self.count != other.count:
            # We return True when 'self' is GREATER so heapq treats the larger number as "smaller"
            return self.count > other.count
        
        # 2. Tie-breaker (Lexicographically greater string wins)
        return self.pair > other.pair