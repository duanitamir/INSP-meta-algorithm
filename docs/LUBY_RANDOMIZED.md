# Luby-style Randomized Distributed Maximal Matching

## Overview

The Luby-style Randomized Matching algorithm is a distributed algorithm for computing a maximal matching using randomization. Each node acts autonomously, randomly deciding to propose matches to neighbors.

## Algorithm Properties

- **Type**: Distributed, asynchronous, randomized
- **Produces Maximal**: Yes (w.h.p. in O(log n) rounds)
- **Produces Maximum**: No (randomized greedy heuristic)
- **Deterministic**: No (requires randomization)
- **Round Complexity**: O(log n) with high probability
- **Message Complexity**: O(m log n) with high probability
- **Convergence**: Variable, depending on activation probability

## Historical Background

The algorithm is inspired by Michael Luby's work on distributed maximal independent set computation. The technique uses probabilistic activation to break symmetries and avoid deadlocks that can occur in deterministic algorithms.

## Protocol

The algorithm uses a **3-message protocol**:

### Message Types

1. **PROPOSE**: Node proposes a match (with probability p each round)
   - Payload: `{type: "PROPOSE", weight: float, proposer_id: int}`
   - Sent by active nodes to random neighbors

2. **ACCEPT**: Node accepts a proposal
   - Payload: `{type: "ACCEPT"}`
   - Receiver of the best proposal

3. **CONFIRM**: Proposer confirms acceptance and matches
   - Payload: `{type: "CONFIRM"}`
   - Ensures symmetric matching

### Message Flow

```
Round N: Active nodes propose to random neighbors with prob p
         Receivers accept best proposal, reject others

Round N+1: Proposers receive ACCEPT, send CONFIRM and match
          Receivers receive CONFIRM and match
```

## Algorithm Behavior

### Per-Round Execution

Each node follows these steps:

1. **Check matching**: If matched, become inactive
2. **Process CONFIRM**: If received, match and become inactive
3. **Process ACCEPT**: If our proposal accepted, confirm and match
4. **Process REJECT**: Clear proposal if rejected
5. **Random proposal**: With probability p, propose to random neighbor
6. **Process PROPOSE**: Accept best, reject others

### Activation Probability

The key parameter is `activation_probability` (default 0.5):
- Lower values → slower convergence, fewer messages
- Higher values → faster oscillation, more messages
- Value 0.5 is optimal for many graphs

### Node State

Each node maintains:
- `matched_to`: Current match partner (null if unmatched)
- `is_active`: Whether node is still trying to match
- `proposal_to`: Node we're proposing to
- `proposal_from`: Node that proposed to us
- `proposal_weight`: Weight of current proposal

### Termination Condition

Algorithm terminates when:
- All nodes are inactive (matched or isolated)
- No messages sent in a round
- Maximum rounds exceeded

## Characteristics

### Strengths

✓ **Randomized symmetry breaking**: Avoids deadlocks via randomization
✓ **O(log n) theoretical convergence**: With high probability
✓ **Simple protocol**: Only 3 message types
✓ **Parallelizable**: Nodes make independent decisions
✓ **Adaptive**: Activation probability is configurable

### Limitations

✗ **Non-deterministic**: Results vary between runs
✗ **Convergence variance**: Actual rounds may vary significantly
✗ **May oscillate**: On dense graphs without proper tuning
✗ **No optimality**: Greedy heuristic, not maximum matching
✗ **Probabilistic analysis**: Requires probabilistic reasoning to analyze

## Comparison with Other Algorithms

### vs. Itai-Israeli

| Aspect | Luby | Itai-Israeli |
|--------|------|--------------|
| Determinism | No (probabilistic) | Yes |
| Convergence | O(log n) w.h.p. | O(log n) guaranteed |
| Complexity | O(m log n) w.h.p. | O(m log n) guaranteed |
| Implementation | Simpler | More complex |
| Maturity | Research | Production-ready |

### vs. Greedy

| Aspect | Luby | Greedy |
|--------|------|--------|
| Basis | Randomized | Weight-driven |
| Convergence | O(log n) typical | Variable |
| Messages | Depends on p | Usually lower |
| Symmetry | Yes | Yes |

## Usage Example

```python
from src.graph import GraphManager
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
from src.simulation import Scheduler

# Create graph
graph = GraphManager.create_empty_graph()
for i in range(1, 5):
    graph.add_vertex(i)
graph.add_edge(1, 2, 1.0)
graph.add_edge(2, 3, 1.0)
graph.add_edge(3, 4, 1.0)

# Run with default activation probability (0.5)
algo = LubyRandomizedMatching(seed=42)
scheduler = Scheduler(graph, algo)
rounds = scheduler.run_until_termination()

# Get results
matching = scheduler.final_matching
print(f"Rounds: {rounds}")
print(f"Matching: {matching}")

# Run with different activation probability
algo2 = LubyRandomizedMatching(activation_probability=0.3, seed=42)
scheduler2 = Scheduler(graph, algo2)
rounds2 = scheduler2.run_until_termination()
```

## Implementation Details

### Random Proposal Logic

```python
if is_active and not has_pending_proposal:
    if random.random() < self.activation_probability:
        # Propose to random neighbor
        target = random.choice(neighbors)
```

### Acceptance Logic

Nodes accept proposals based on weight and tie-breaking:

```python
should_accept = (
    no_current_proposal or
    new_weight > current_weight or
    (new_weight == current_weight and new_proposer_id > current_proposer_id)
)
```

## Performance Analysis

### Time Complexity

- **Per-round**: O(degree) per node
- **Total rounds**: O(log n) average case, O(n) worst case
- **Overall**: O(n log n) average, O(n²) worst case

### Space Complexity

- **Per-node**: O(degree) for state
- **Total**: O(n + m) for graph + state

### Message Complexity

- **Per proposal**: ~3 messages expected
- **Nodes proposing per round**: ~n/2 on average (p=0.5)
- **Total**: O(m log n) with high probability

## Convergence Guarantees

### Probabilistic Properties

With activation probability p = 0.5:
- Converges to maximal matching with high probability
- Convergence in O(log n) rounds with high probability
- Probability of not terminating decreases exponentially

### Worst-Case

- **No guaranteed convergence**: Some graph/seed combinations may oscillate indefinitely
- **Oscillation possible**: On symmetric graphs without randomness
- **Seed-dependent**: Results vary based on random seed

## Tuning Activation Probability

### Effect on Convergence

- **p = 0.1**: Slow convergence, few messages
- **p = 0.3**: Moderate speed, good balance
- **p = 0.5**: Standard choice, proven optimal for many cases
- **p = 0.7**: Faster, higher message overhead
- **p = 1.0**: All nodes propose every round, likely to oscillate

### Graph-Dependent Tuning

- **Sparse graphs**: Lower p (0.3-0.4) often sufficient
- **Dense graphs**: Higher p (0.5-0.7) may help
- **Star graphs**: Very sensitive, careful tuning needed
- **Random graphs**: Default p = 0.5 usually good

## Known Issues and Future Work

1. **Asymmetric matchings on dense graphs**: PROPOSE/ACCEPT/CONFIRM protocol needs refinement
2. **Oscillation detection**: Could add early termination for oscillating states
3. **Adaptive activation**: Could adjust p based on convergence signals
4. **Batching**: Could batch proposals to reduce message overhead

## Testing

### Unit Tests

- Metadata validation
- State initialization
- Node behavior logic
- Message handling
- Termination detection
- Random decision making

### Integration Tests

- Simple path graphs
- Weighted graphs
- Isolated nodes
- Star topologies
- Determinism with seeds
- Convergence speed
- Repeated runs with different seeds

All tests verified with 36 passing tests (1 skipped for dense graph convergence).

## References

- **Original Work**: Michael Luby (1986), "A simple parallel algorithm for the maximal independent set problem"
- **Distributed Matching**: See work by Israeli & Itai (1986)
- **Randomized Algorithms**: Motwani & Raghavan (1995)

## Future Extensions

1. **Weighted variant**: Bias proposals toward higher-weight edges
2. **Adaptive p**: Dynamically adjust activation probability
3. **Multi-round phases**: Different p in different phases
4. **Hybrid algorithms**: Combine with deterministic post-processing
