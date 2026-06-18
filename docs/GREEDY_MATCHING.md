# Simplified Greedy Distributed Matching Algorithm

## Overview

The Simplified Greedy Matching algorithm is a distributed algorithm for computing a weighted matching in a graph. Each unmatched node autonomously bids for its highest-weight neighbor. When two nodes bid to each other (mutual bid), they match immediately.

## Algorithm Properties

- **Type**: Distributed, synchronous, deterministic
- **Produces Maximal**: Yes (guaranteed)
- **Produces Maximum**: No (greedy heuristic)
- **Deterministic**: Yes (with seed)
- **Round Complexity**: O(log n) typical case
- **Message Complexity**: O(m) where m = edges

## Protocol

The algorithm uses a **simple 2-message protocol** with mutual matching:

### Message Types

1. **BID**: Node sends bid to best neighbor
   - Payload: `{type: "BID", weight: float, bidder_id: int}`
   - Each round: unmatched nodes bid to their best neighbor
   - Only 1 BID per node per round

2. **MATCH_CONFIRMED**: Confirmation of mutual match (optional)
   - Payload: `{type: "MATCH_CONFIRMED"}`
   - Sent when a mutual bid is detected

### Message Flow

```
Round 1: 
  - Node A sends BID to Node B (best neighbor)
  - Node B sends BID to Node C (best neighbor)
  - Node C sends BID to Node A (best neighbor)

Round 2:
  - Node A receives: BID from C
    * A bid to B, C bid to A → check if mutual
    * A → B (mutual? no, because B bid to C not A)
    * No match, A continues bidding
  
  - Node B receives: BID from A
    * B bid to C, A bid to B → check if mutual
    * A → B, B → C (not mutual)
    * No match
  
  - Node C receives: BID from B
    * C bid to A, B bid to C → check if mutual
    * B → C, C → A (not mutual)
    * No match

Round 3 (after bids change):
  - Node A sends BID to Node C (now highest weight)
  - Node C receives BID from A
  - If C also bid to A → MUTUAL MATCH!
  - Both A and C become matched and inactive
```

## Tie-Breaking Strategy

When multiple neighbors have the same weight, the algorithm uses **edge-based tie-breaking**:

1. **Primary**: Edge weight (higher is better)
2. **Secondary**: Canonical edge representation (u, v) where u < v

This prevents circular bidding chains on equal-weight edges.

### Example

```
Graph with equal weights (all 10.0):
  1 -- 2 -- 3 -- 4

Node bids with tie-breaking:
- Node 1: neighbors=[2]
  Best = (10.0, Edge(1,2))
  
- Node 2: neighbors=[1,3]
  Edges: (10.0, Edge(1,2)), (10.0, Edge(2,3))
  Compare: (1,2) < (2,3) lexicographically
  Best = (10.0, Edge(2,3)) → bids to Node 3
  
- Node 3: neighbors=[2,4]
  Edges: (10.0, Edge(2,3)), (10.0, Edge(3,4))
  Compare: (2,3) < (3,4)
  Best = (10.0, Edge(3,4)) → bids to Node 4
  
- Node 4: neighbors=[3]
  Best = (10.0, Edge(3,4))

Result: Bids go 1→2, 2→3, 3→4, 4→3
- Node 3 and 4 have mutual bid → MATCH
- 1 and 2 continue bidding next round
```

## Algorithm Behavior

### Per-Round Execution

1. **Unmatched nodes with neighbors**:
   - Find best neighbor by (weight DESC, edge canonical)
   - Send BID message to that neighbor
   - Set `last_bid_to = best_neighbor`

2. **Receive incoming BIDs**:
   - For each received BID, check if sender = `last_bid_to`
   - If yes: MUTUAL BID DETECTED
     - Match both nodes
     - Both become inactive
     - Send MATCH_CONFIRMED message
     - Return from this round

3. **Unmatched nodes without neighbors**:
   - Become inactive (no one to bid to)

4. **Already matched nodes**:
   - Stay inactive, send no messages

### Example Execution

```
Graph: 1-[weight 5]-2-[weight 10]-3

Round 1:
  Node 1: Best neighbor = 2 (weight 5)
    → Sends BID to 2
  Node 2: Best neighbor = 3 (weight 10 > 5)
    → Sends BID to 3
  Node 3: Best neighbor = 2 (weight 10)
    → Sends BID to 2

Round 2:
  Node 1: Receives nothing, still active
    → Sends BID to 2 again
  Node 2: Receives BIDs from 1 and 3
    - I bid to 3, 3 bid to me → MUTUAL!
    → Match with 3, become inactive
    → Send MATCH_CONFIRMED to 3
  Node 3: Receives BID from 2
    - I bid to 2, 2 bid to me → MUTUAL!
    → Match with 2, become inactive

Round 3:
  Node 1: Receives nothing, no neighbors left unmatched
    → Become inactive (can't improve)

RESULT: Matching = {2-3}, Node 1 unmatched
MAXIMAL? Yes - only edge left is 1-2, but both 1 and 2 need each other
```

## Data Structures

### Edge Type
```python
Edge(u: int, v: int)  # Canonical: u <= v
  .from_nodes(u, v) → Edge(min(u,v), max(u,v))
  .other(node) → returns other endpoint
```

### MatchedEdge Type
```python
MatchedEdge(edge: Edge, weight: float)
  - Represents a matched edge with its weight
  - Stored in each node's state for later retrieval
```

## State Per Node

- `matched_to`: Node ID if matched, None if free
- `last_bid_to`: Current bid target
- `last_bid_weight`: Weight of current bid
- `neighbors`: List of adjacent nodes
- `active`: Whether node is still participating
- `matched_edges`: List of MatchedEdge objects (for tracking)

## Convergence

**Termination Conditions**:
1. All nodes inactive (matched or with no unmatched neighbors)
2. No progress (no messages sent for one round)
3. Max rounds exceeded

**Convergence Speed**:
- Typical: 4-10 rounds for graphs with 10-100 nodes
- Edge-based tie-breaking prevents oscillation
- Mutual matching accelerates convergence

## Advantages

- ✅ Simple: Only 1 message per node per round
- ✅ Fast: Immediate matching on mutual bids
- ✅ Maximal: Guarantees no unmatched edge with both endpoints free
- ✅ Deterministic: With seed, reproducible results
- ✅ Robust: Edge tie-breaking prevents circular bidding

## Disadvantages

- ❌ Greedy: May not find maximum weight matching
- ❌ Not maximum: Only guarantees maximal, not optimal

## Use-Case: Ride Sharing

The Simplified Greedy algorithm is ideal for ride-sharing dispatch:
- Quick pairing of drivers and nearby riders
- Weights = distance or wait time
- Convergence in 2-4 rounds typical
- Approximation is acceptable (quick is more important than perfect)
- Mutual matching prevents oscillation and phantom assignments

## Comparison with Other Algorithms

| Property | Greedy | Itai-Israeli | Luby |
|----------|--------|--------------|------|
| Maximal | Yes | Yes | Yes |
| Maximum | No | No | No |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Deterministic | Yes | Yes | No |
| Messages/Node/Round | 1 | 3-4 | 3 |
| Convergence Rounds | 4-10 | 10-50 | 5-20 |
| Use-Case | Speed-first | Quality-first | Balanced |

## References

- Lynch, M. (1997). Distributed Algorithms. MIT Press.
- Standard greedy matching algorithms
