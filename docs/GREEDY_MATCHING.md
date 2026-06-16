# Greedy Distributed Matching Algorithm

## Overview

The Greedy Matching algorithm is a distributed algorithm for computing a weighted matching in a graph. Each node acts autonomously, bidding for its highest-weight neighbor and accepting the best incoming bid.

## Algorithm Properties

- **Type**: Distributed, asynchronous
- **Produces Maximal**: Often, but not guaranteed
- **Produces Maximum**: No (greedy heuristic)
- **Deterministic**: No (uses randomness in neighbor selection with same weight)
- **Round Complexity**: O(log n) in typical cases
- **Message Complexity**: O(m) where m = edges
- **Convergence**: Usually fast (5-20 rounds for small graphs)

## Protocol

The algorithm uses a **4-message protocol** for symmetric matching:

### Message Types

1. **BID**: Node sends bid for a neighbor
   - Payload: `{type: "BID", weight: float, bidder_id: int}`
   - Initiates negotiation for a match

2. **ACCEPT**: Node accepts a bid
   - Payload: `{type: "ACCEPT"}`
   - Agrees to match with bidder

3. **CONFIRM**: Bidder confirms the match
   - Payload: `{type: "CONFIRM"}`
   - Ensures symmetric matching (both nodes match simultaneously)

4. **REJECT**: Node rejects a bid
   - Payload: `{type: "REJECT"}`
   - Allows node to choose better partners

### Message Flow

```
Round 1: Node A sends BID to Node B (highest weight neighbor)
Round 2: Node B receives BID, decides to ACCEPT or REJECT
Round 3: Node A receives ACCEPT, sends CONFIRM and matches
         Node B receives CONFIRM and matches
```

## Algorithm Behavior

### Per-Round Execution

Each node follows these steps in order:

1. **Process CONFIRM messages**: If current partner confirms, match!
2. **Process ACCEPT messages**: If current partner accepts, match and confirm
3. **Process REJECT messages**: Clear current bid if rejected
4. **Send new BID**: If no active bid, bid to highest-weight unmatched neighbor
5. **Process incoming BIDs**: Accept best, reject others

### Node State

Each node maintains:
- `matched_to`: Current match partner (null if unmatched)
- `current_bid`: Weight of current bid
- `current_bid_partner`: Node we're currently negotiating with
- `neighbors`: List of adjacent nodes
- `active`: Whether node is still trying to match

### Termination Condition

Algorithm terminates when:
- All nodes are inactive (matched or have no neighbors)
- No messages sent in a round (convergence)
- Maximum rounds exceeded

## Characteristics

### Strengths

✓ **Fast convergence**: Typically converges in O(log n) rounds  
✓ **Weight-aware**: Nodes prefer higher-weight edges  
✓ **Distributed**: No central coordinator needed  
✓ **Symmetric**: Uses CONFIRM to ensure both nodes agree  

### Limitations

✗ **Non-optimal**: Greedy decisions may miss global optimum  
✗ **Non-maximal**: May not always produce maximal matchings  
✗ **Chatty**: Uses 4 message types, can be message-intensive  
✗ **Unstable**: Can oscillate if edges have equal weights  

## Comparison with Other Algorithms

### vs. Itai-Israeli

| Aspect | Greedy | Itai-Israeli |
|--------|--------|--------------|
| Convergence | Fast (empirical) | O(log n) guaranteed |
| Matching Quality | Weight-greedy | Maximal guaranteed |
| Messages | High (4 types) | Medium (3 types) |
| Implementation | Simple | Complex |

### vs. Randomized (Luby-style)

| Aspect | Greedy | Randomized |
|--------|--------|-----------|
| Determinism | Mostly deterministic | Probabilistic |
| Quality | Weight-biased | More uniform |
| Speed | Fast | Variable |
| Analysis | Empirical | Probabilistic |

## Usage Example

```python
from src.graph import GraphManager
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.simulation import Scheduler

# Create graph
graph = GraphManager.create_empty_graph()
for i in range(1, 5):
    graph.add_vertex(i)
graph.add_edge(1, 2, 10.0)
graph.add_edge(2, 3, 8.0)
graph.add_edge(3, 4, 6.0)

# Run algorithm
algo = GreedyMatching(seed=42)
scheduler = Scheduler(graph, algo)
rounds = scheduler.run_until_termination()

# Get results
matching = scheduler.final_matching
print(f"Rounds: {rounds}")
print(f"Matching: {matching}")
```

## Implementation Details

### Node Behavior Loop

```python
def node_behavior(self, node_id, node_state, messages, context):
    # 1. Process CONFIRM -> match if from current partner
    # 2. Process ACCEPT -> match and send CONFIRM
    # 3. Process REJECT -> clear bid
    # 4. If no bid, bid to best neighbor
    # 5. Process BIDs -> ACCEPT best, REJECT others
    # Return updated state and outgoing messages
```

### Tie-Breaking

When weights are equal, tie-breaking uses node IDs (higher ID wins):

```python
should_accept = (
    current_partner is None or
    new_weight > old_weight or
    (new_weight == old_weight and bidder_id > current_partner)
)
```

## Performance Analysis

### Time Complexity

- **Per-round**: O(degree) per node
- **Total rounds**: O(log n) average case, O(n) worst case
- **Overall**: O(n log n) to O(n²)

### Space Complexity

- **Per-node**: O(degree) for state
- **Total**: O(n + m) for graph + state

### Message Complexity

- **Per negotiation**: ~4 messages (BID, ACCEPT, CONFIRM, possibly REJECT)
- **Per node**: O(degree) negotiations typical
- **Total**: O(m) to O(degree²) in worst case

## Convergence Guarantees

The algorithm is **guaranteed to terminate** because:

1. Nodes only become inactive when matched or isolated
2. Once matched, a node never becomes active again
3. Each matched pair consumes at least 2 nodes
4. Eventually all nodes are matched or isolated

However, the **quality** of matching is not guaranteed to be:
- Maximal (all unmatched nodes have no common neighbor)
- Maximum (largest possible matching)

## Variants and Extensions

### Possible Improvements

1. **Weighted Preference**: Adjust bid weight by node degree
2. **Exponential Backoff**: Increase timeout for repeated rejections
3. **Hybrid**: Combine with Itai-Israeli for post-processing
4. **Randomized**: Add randomness to break symmetries

### Configuration Parameters

```python
algo = GreedyMatching(seed=42)  # Seed for randomness
```

No major configuration parameters currently; the algorithm is deterministic aside from random seed.

## Testing

### Unit Tests

- Metadata validation
- State initialization
- Node behavior correctness
- Termination conditions
- Matching validation

### Integration Tests

- Simple path graphs
- Complete graphs
- Weighted graphs
- Isolated nodes
- Star topologies
- Determinism with seeds

See `tests/unit/test_greedy_matching.py` and `tests/integration/test_greedy_matching_integration.py`.

## References

- **Related Work**: Greedy matching is a classical approach in distributed algorithms
- **Distributed Matching**: See Awerbuch et al. and Israeli & Itai
- **Weight-Maximizing Matching**: Related to auction algorithms

## Future Improvements

1. Add exponential backoff for rejections
2. Implement weighted degree adjustments
3. Add randomization for tie-breaking
4. Combine with post-processing phases
5. Profile message complexity on large graphs
