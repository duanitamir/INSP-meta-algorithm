# Documentation Index

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

### Phase 2: Meta-Algorithm with Genetic Algorithm Optimization (In Progress 🔄)

Orchestration layer combining multiple algorithms with automatic parameter tuning:

#### Core Meta-Algorithm

- **[META_ALGORITHM.md](META_ALGORITHM.md)** ⭐ — Start here
  - Overview of meta-algorithm orchestration
  - How three algorithms work together
  - Conflict resolution strategy
  - Cascading loop iteration
  - Node selection using percentiles

#### Parameter Representation

- **[CANONICAL_VECTOR.md](CANONICAL_VECTOR.md)** ⭐ — GA chromosome
  - 17-parameter vector encoding
  - Node selection parameters (6)
  - Luby adaptive activation (7)
  - Algorithm-specific parameters (1)
  - Meta control parameters (2)
  - Validation and serialization
  - How GA uses these parameters

#### Genetic Algorithm Optimization

- **[GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md)** ⭐ — Parameter optimization
  - GA algorithm explanation
  - Selection, crossover, mutation operators
  - Fitness landscape and convergence
  - Why GA for this problem
  - Performance on different graph types
  - Research questions GA answers

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

### Phase 2 🔄 In Progress

| Component | Files | Tests | Status |
|-----------|-------|-------|--------|
| CanonicalVector | CANONICAL_VECTOR.md | 21 | ✅ |
| MetricsCalculator | — | 21 | ✅ |
| NodeSelector | — | 16 | ✅ |
| AlgorithmParameterizer | — | 19 | ✅ |
| Algorithm Wrappers | — | — | ⏳ |
| CascadingLoop | — | — | ⏳ |
| ConflictResolver | — | — | ⏳ |
| MetaAlgorithmGA | — | — | ⏳ |

**Total Phase 2 Tests So Far**: 77 passing ✅

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

## How to Use These Docs

1. **For Understanding Architecture**: Read [META_ALGORITHM.md](META_ALGORITHM.md)
2. **For Understanding Parameters**: Read [CANONICAL_VECTOR.md](CANONICAL_VECTOR.md)
3. **For Understanding Tuning**: Read [GENETIC_ALGORITHM.md](GENETIC_ALGORITHM.md)
4. **For Implementation Details**: Check code (src/meta/*.py)
5. **For Tests**: Check tests/unit/test_*.py

Each doc includes:
- Overview/motivation
- Detailed explanation
- Architecture/design
- Code structure
- Usage examples
- Performance characteristics
- File locations

---

## Contact & Questions

Refer to [../CLAUDE.md](../CLAUDE.md) for:
- Project vision and goals
- Critical rules
- Complete task list
- Current state and progress
- Implementation details

All other documentation files are derived from and reference CLAUDE.md.
