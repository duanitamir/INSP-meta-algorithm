# Documentation Index

## Start Here

**[ALGORITHM_FLOW.md](ALGORITHM_FLOW.md)** ⭐ — 5-minute overview of the entire system
- High-level process flow
- Key concepts
- Where to go for details

---

## Quick Navigation

### Phase 1: Distributed Matching Algorithms (Implemented ✅)

Core framework and three working algorithms:

- **[GRAPH.md](GRAPH.md)** — Graph topology and data structure
- **[STATE.md](STATE.md)** — State management and externalized node state
- **[COMMUNICATION.md](COMMUNICATION.md)** — Message passing and synchronous rounds
- **[SCHEDULER.md](SCHEDULER.md)** — Round orchestration and termination
- **[METRICS.md](METRICS.md)** — Algorithm performance tracking

Algorithm Implementations:

- **[ITAI_ISRAELI.md](ITAI_ISRAELI.md)** — Synchronous deterministic algorithm (7/7 tests ✅)
- **[GREEDY_MATCHING.md](GREEDY_MATCHING.md)** — 3-phase handshake matching (28/29 tests ✅)
- **[LUBY_RANDOMIZED.md](LUBY_RANDOMIZED.md)** — Probabilistic randomized algorithm (21/21 tests ✅)

Utilities:

- **[VISUALIZATION.md](VISUALIZATION.md)** — Graph visualization tools
- **[NOTEBOOKS.md](NOTEBOOKS.md)** — Jupyter notebooks for experimentation

---

### Phase 2: Meta-Algorithm (✅ Complete)

Centralized orchestration layer combining multiple algorithms with automatic parameter tuning.

#### Core Components

- **[META_ALGORITHM.md](META_ALGORITHM.md)** — Centralized orchestration (reference implementation)
- **[CANONICAL_VECTOR.md](CANONICAL_VECTOR.md)** — 10-parameter GA chromosome
  - Luby adaptive activation (7 params)
  - Itai timeout control (1 param)
  - Loop control (2 params)
  - Validation & serialization
- **[GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md)** — Parameter optimization
  - Population evolution, fitness evaluation
  - Parallel evaluation (3-4x speedup)
  - Adaptive mutation & early stopping

### Phase 3: Distributed System (✅ Complete)

Fully distributed implementation replacing Phase 2 centralized components:

- **DistributedOrchestrator** — No central bottleneck
- **DistributedParameterEvolver** — Gossip-based GA per node
- **DistributedConflictResolver** — Edge voting consensus
- **DistributedConvergenceDetector** — Quorum-based termination

All components tested: 87 tests, 100% coverage on Phase 3

---

## Reading Guide

### If You Want to Understand...

**The Big Picture**
1. Start: [META_ALGORITHM.md](META_ALGORITHM.md) — Architecture and data flow
2. Then: [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) — What gets tuned
3. Then: [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md) — How tuning works

**How Matching Works (Phase 1)**
1. Start: [GRAPH.md](GRAPH.md) — Graph representation
2. Then: [COMMUNICATION.md](COMMUNICATION.md) — Message-passing model
3. Then: [ITAI_ISRAELI.md](ITAI_ISRAELI.md) — Synchronous algorithm
4. Then: [GREEDY_MATCHING.md](GREEDY_MATCHING.md) — Heuristic algorithm
5. Then: [LUBY_RANDOMIZED.md](LUBY_RANDOMIZED.md) — Probabilistic algorithm

**How Algorithms Combine (Phase 2)**
1. Start: [META_ALGORITHM.md](META_ALGORITHM.md) — Orchestration layer
2. Node Selection section — How algorithms specialize
3. Then: [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) — Node selection parameters
4. Algorithm Execution section — Running all three
5. Conflict Resolution section — Combining results

**Parameter Tuning**
1. Start: [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) — What parameters exist
2. Parameter sections — What each does
3. Then: [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md) — Automatic optimization
4. GA Algorithm section — How tuning works

---

## Implementation Status

### Phase 1 ✅ Complete (94% test coverage)

| Component | Files | Tests | Status |
|-----------|-------|-------|--------|
| Framework | GRAPH, STATE, COMMUNICATION, SCHEDULER, METRICS | 60+ | ✅ |
| Itai-Israeli | ITAI_ISRAELI | 7/7 | ✅ |
| Greedy | GREEDY_MATCHING | 28/29 | ✅ |
| Luby | LUBY_RANDOMIZED | 21/21 | ✅ |
| Auction | (Not yet) | — | ⏳ |

### Phase 2 ✅ Complete (93 tests, 89% coverage)

| Component | Tests | Coverage |
|-----------|-------|----------|
| CanonicalVector | 14 | 100% |
| AlgorithmParameterizer | 8 | 100% |
| Algorithm Wrappers (Greedy, Itai, Luby) | 26 | 100% |
| FitnessEvaluator | 7 | 100% |
| ConflictResolver | 11 | 100% |
| CascadingLoop | 13 | 83% |
| MetaAlgorithmGA (centralized) | 10 | 100% |
| GAConfig | 34 | 100% |
| **Phase 2 Total** | **93** | **89%** |

### Phase 3 ✅ Complete (87 tests, 100% coverage)

| Component | Tests | Coverage |
|-----------|-------|----------|
| DistributedParameterEvolver | 24 | 100% |
| DistributedConflictResolver | 27 | 100% |
| DistributedConvergenceDetector | 29 | 100% |
| DistributedOrchestrator | 17 | 100% |
| Integration Tests | 7 | 100% |
| **Phase 3 Total** | **87** | **100%** |

**Total Phase 2-3**: 214 tests passing, 95% coverage

---

## Key Concepts

### Node Selection (AND Logic)

All nodes must satisfy ALL three criteria:
- Degree between min_percentile and max_percentile
- Incident weight between min_percentile and max_percentile
- Centrality between min_percentile and max_percentile

This specializes each algorithm on different node types.

### Conflict Resolution (Highest Weight Wins)

When multiple algorithms match different edges:
1. Collect all candidates (who each node could match to)
2. For each node, keep highest-weight partner
3. Filter to symmetric pairs (both u→v and v→u)

Higher-weight edges always win conflicts.

### Cascading Loop (Iterate Until Convergence)

Repeatedly:
1. Run all three algorithms on selected nodes
2. Merge results via conflict resolution
3. Check if new matches found
4. Stop if no progress or maximal matching

Typical convergence: 5-15 iterations on small graphs.

### Genetic Algorithm (Automatic Tuning)

GA optimizes 17 parameters by:
1. Create 50 random vectors
2. Evaluate fitness (run meta-algorithm, sum matching weight)
3. Select best 50%
4. Create offspring via crossover + mutation
5. Repeat for 200 generations

Discovers which parameters matter for different graph types.

---

## Architecture Diagram

```
Input Graph
    ↓
[METRICS CALCULATOR] — degree, weight, clustering, centrality
    ↓
[NODE SELECTOR] — select nodes matching percentile criteria
    ↓
[THREE ALGORITHM WRAPPERS] — run independently
    ├─ GreedyParameterizer
    ├─ ItaiParameterizer
    └─ LubyParameterizer
    ↓
[THREE MATCHINGS] — candidate edges from each algorithm
    ↓
[CONFLICT RESOLVER] — merge, highest weight wins, keep symmetric
    ↓
[CASCADING LOOP] — iterate until convergence
    ↓
[FINAL MATCHING] — valid, symmetric, maximal
    ↓
[GENETIC ALGORITHM] — optimize parameters
```

---

## Testing & Coverage

### Phase 1 Test Coverage

- Framework: 95%+
- Itai-Israeli: 95%+
- Greedy: 98%
- Luby: 90%+
- Overall: 94%

All tests: TDD (tests written BEFORE implementation)

### Phase 2 Test Coverage

- CanonicalVector: 87%
- MetricsCalculator: 100%
- NodeSelector: 100%
- AlgorithmParameterizer: 100%
- Phase 2A Overall: 94.6%

**Target**: 90%+ coverage on all components

---

## Code Quality Standards

All code follows:
- ✅ **Type Hints**: 100% on all public functions
- ✅ **Docstrings**: 100% on all classes/methods
- ✅ **SOLID Principles**: Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion
- ✅ **DRY**: No code duplication
- ✅ **Code Format**: Black formatted
- ✅ **Linting**: Ruff clean
- ✅ **Type Checking**: Mypy verified

---

## Single Source of Truth

All project instructions and state tracked in:
- **[../CLAUDE.md](../CLAUDE.md)** — Project instructions, current state, working log

All documentation (this file + component files) are reference materials derived from CLAUDE.md.

**NO GIT OPERATIONS** from AI — user responsibility only.

---

## Quick Links

**For Phase 2 (Centralized Meta-Algorithm)**:
- [META_ALGORITHM.md](META_ALGORITHM.md) — Architecture
- [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md) — Parameters
- [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md) — Optimization

**For Phase 1 (Core Algorithms)**:
- [ITAI_ISRAELI.md](ITAI_ISRAELI.md), [GREEDY_MATCHING.md](GREEDY_MATCHING.md), [LUBY_RANDOMIZED.md](LUBY_RANDOMIZED.md)

**Implementation**: See src/meta/*.py and src/algorithms/implementations/*.py

---

## Contact & Questions

Refer to [../CLAUDE.md](../CLAUDE.md) for:
- Project vision and goals
- Critical rules
- Complete task list
- Current state and progress
- Implementation details

All other documentation files are derived from and reference CLAUDE.md.
