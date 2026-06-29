# Algorithm Flow Overview

## High-Level Process

This document provides a bird's-eye view of how the entire system works. For detailed explanations of each step, refer to the links provided.

---

## The Complete Flow

### 1. **Input: Graph**
Start with a graph: nodes and weighted edges.
- See: [GRAPH.md](GRAPH.md) for graph representation

### 2. **Calculate Metrics**
Compute graph properties: degree, weight distribution, clustering, centrality.
- Purpose: Understand graph structure for node selection
- See: [META_ALGORITHM.md](META_ALGORITHM.md#metrics-calculation) for details

### 3. **Select Nodes**
Choose subsets of nodes for each algorithm based on percentile criteria.
- Each algorithm specializes on different node types
- See: [META_ALGORITHM.md](META_ALGORITHM.md#node-selection) for selection logic

### 4. **Execute Three Algorithms**
Run matching algorithms independently on selected nodes:
- **Greedy**: Fast heuristic-based approach
- **Itai-Israeli**: Synchronous deterministic algorithm
- **Luby**: Probabilistic randomized algorithm
- See: [ITAI_ISRAELI.md](ITAI_ISRAELI.md), [GREEDY_MATCHING.md](GREEDY_MATCHING.md), [LUBY_RANDOMIZED.md](LUBY_RANDOMIZED.md)

### 5. **Resolve Conflicts**
Merge the three matchings by keeping highest-weight edges:
- When multiple algorithms propose different matches, use edge weight to decide
- Filter to symmetric pairs (if u matched to v, then v matched to u)
- See: [META_ALGORITHM.md](META_ALGORITHM.md#conflict-resolution) for details

### 6. **Check Convergence**
Did we find new matches in this iteration?
- **Yes** → Go back to step 2 (iterate on remaining unmatched nodes)
- **No** → Go to step 7 (done)
- See: [META_ALGORITHM.md](META_ALGORITHM.md#cascading-loop) for convergence logic

### 7. **Output: Final Matching**
Return the combined matching from all iterations.
- Valid: No node matched twice
- Symmetric: u↔v both present
- Maximal: No unmatched edge has both endpoints unmatched

---

## Optional: Optimize with Genetic Algorithm

If you want to tune the system for a specific graph type:

### 8. **Define Parameter Vector**
Create a chromosome with 10 parameters controlling the algorithms.
- See: [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) for parameters

### 9. **Run Genetic Algorithm**
Evolve the population to maximize matching weight:
1. Create random population
2. Evaluate fitness (run steps 1-7 for each vector)
3. Select best performers
4. Create offspring via crossover + mutation
5. Repeat for N generations
- See: [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md) for GA details

### 10. **Output: Optimized Parameters**
Get the best parameter vector for your graph type.
- Can be reused on similar graphs
- See: [GAConfig](../src/meta/ga_config.py) for preset configurations

---

## Architecture Variants

### Centralized (Phase 2 - Reference)
All orchestration happens in one place: `CascadingLoop` → `ConflictResolver` → repeat
- See: [META_ALGORITHM.md](META_ALGORITHM.md) for full details
- Implementation: `src/meta/meta_algorithm_ga.py`

### Distributed (Phase 3 - Primary)
Each node runs its own GA, votes on conflicts, decides termination:
- **DistributedParameterEvolver**: Local GA with gossip
- **DistributedConflictResolver**: Edge voting consensus
- **DistributedConvergenceDetector**: Quorum-based termination
- Implementation: `src/meta/distributed_*.py`

---

## Key Concepts Summary

| Concept | Purpose | Details |
|---------|---------|---------|
| **Metrics** | Understand graph structure | Degree, weight, clustering, centrality |
| **Node Selection** | Specialize each algorithm | Percentile-based filtering (AND logic) |
| **Conflict Resolution** | Combine algorithm results | Highest weight wins, keep symmetric pairs |
| **Cascading Loop** | Iterate until convergence | Repeat on unmatched nodes |
| **Genetic Algorithm** | Auto-tune parameters | Population evolution over generations |
| **Canonical Vector** | GA chromosome | 10 parameters controlling the system |

---

## Which Implementation?

**Use Centralized (Phase 2)** if:
- You want a simple, clear reference implementation
- Running on a single machine
- You want to understand the algorithm

**Use Distributed (Phase 3)** if:
- You need real distributed deployment
- You want no central bottleneck
- You want fault tolerance
- You're running on a network

---

## Where to Start

1. **Want to understand the system?** → Read this doc (you're here) + [META_ALGORITHM.md](META_ALGORITHM.md)
2. **Want to understand parameters?** → [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) + [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md)
3. **Want to understand algorithms?** → [ITAI_ISRAELI.md](ITAI_ISRAELI.md), [GREEDY_MATCHING.md](GREEDY_MATCHING.md), [LUBY_RANDOMIZED.md](LUBY_RANDOMIZED.md)
4. **Want to run code?** → Check `notebooks/` for examples or `src/meta/` for implementations

---

## Testing

- **Unit Tests**: 214 tests covering all phases (see `tests/unit/`)
- **Coverage**: 95% on Phase 2-3, 94% overall
- **Verification**: All critical properties tested (validity, symmetry, maximality)

See [INDEX.md](INDEX.md) for complete testing breakdown.
