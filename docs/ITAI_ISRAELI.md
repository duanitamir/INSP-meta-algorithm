# Itai-Israeli Distributed Maximal Matching Algorithm

## Overview

Implementation of the Itai-Israeli distributed maximal matching algorithm for the simulation framework. This algorithm computes a maximal matching in a distributed setting where nodes can only communicate through message passing.

## Algorithm Description

### Key Concepts

- **Distributed Execution**: Each node executes independently based on local information and messages from neighbors
- **Message-Passing**: Nodes communicate only through PROPOSE and ACCEPT messages
- **Maximal Matching**: Algorithm guarantees that the final matching cannot be extended (no two unmatched neighbors exist)
- **Synchronous Rounds**: All nodes execute in lockstep rounds

### Pseudocode

```
For each node:
  1. If already matched, remain inactive
  2. Process incoming messages:
     - PROPOSE messages: Accept highest ID proposer
     - ACCEPT messages: Confirm match and become inactive
  3. If no messages received and still free:
     - Send PROPOSE message to random neighbor
  4. Become inactive when matched
```

### Message Types

- **PROPOSE**: Node i proposes to match with node j
- **ACCEPT**: Node i accepts node j's proposal
- **REJECT**: Node i rejects node j's proposal

## Complexity Analysis

### Time Complexity
- **Rounds**: O(log n) average case (depends on graph properties)
- **Message Complexity**: O(m log n) where m is the number of edges

### Space Complexity
- **Per Node**: O(degree) for storing neighbor information and pending proposals

## Implementation Details

### State Variables

Each node maintains:
- `status`: Current state (free, proposing, accepting, matched)
- `matched_to`: ID of matched neighbor (if any)
- `neighbors`: List of adjacent vertices
- `active`: Whether node is still participating
- `proposal_target`: Current proposal target (if proposing)

### Execution Flow

1. **Initialization Phase**: All nodes initialize with `status=free`
2. **Active Phase**: Nodes propose and accept matches
3. **Termination Phase**: When all nodes are matched or inactive, algorithm terminates

## Known Limitations

1. **Matching Symmetry**: In some cases, matching pairs may not be perfectly symmetric in terms of round timing. This is a known limitation of the current distributed implementation.

2. **Convergence**: On complex graph topologies, the algorithm may take longer to converge.

3. **Random Tiebreaking**: Proposal selection is random, leading to non-deterministic execution (except with a fixed seed).

## Usage

### Basic Example

```python
from src.graph import GraphManager
from src.algorithms.implementations import ItaiIsraeliMaximalMatching
from src.simulation import Scheduler

# Create graph
graph = GraphManager.create_empty_graph()
for i in range(1, 5):
    graph.add_vertex(i)
graph.add_edge(1, 2, 1.0)
graph.add_edge(2, 3, 1.0)
graph.add_edge(3, 4, 1.0)

# Run algorithm
algo = ItaiIsraeliMaximalMatching(seed=42)
scheduler = Scheduler(graph, algo)
rounds = scheduler.run_until_termination()

# Get results
matching = scheduler.final_matching
print(f"Matching: {matching}")
print(f"Rounds: {rounds}")
```

### With Configuration

```python
from src.simulation import SimulationConfig

config = SimulationConfig(
    max_rounds=100,
    collect_snapshots=True,
    debug=False,
)

scheduler = Scheduler(graph, algo, config)
rounds = scheduler.run_until_termination()
```

## Metrics

The algorithm tracks:
- Number of rounds to convergence
- Total messages sent
- Messages per round
- Matching size and weight
- Node activation status per round

## Comparison with Theoretical Algorithm

The implemented version is a simplified variant of the theoretical Itai-Israeli algorithm that emphasizes:

- **Synchronous communication**: Nodes wait for round boundaries
- **Simple message protocol**: Only PROPOSE/ACCEPT messages
- **Practical convergence**: Terminates when no progress is made

This differs from the theoretical version which may have more complex phases and message patterns.

## Future Improvements

1. **Symmetric Guarantee**: Implement a confirmation phase to guarantee matching symmetry
2. **Adaptive Proposals**: Use neighbor degree information for smarter proposal selection
3. **Early Termination**: Detect maximal matchings earlier
4. **Weighted Optimization**: Consider edge weights in proposal selection

## References

- Itai, Adi; Rodeh, Michael (1978). "Finding a Minimum Circuit in a Graph". SIAM Journal on Computing 7 (4): 413-423.
- Distributed Algorithms: An Intuitive Approach. Wan (2014)
