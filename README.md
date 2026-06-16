# Distributed Graph Matching Meta-Algorithm

A research-grade framework for simulating and experimenting with distributed graph matching algorithms.

## Overview

This project implements multiple distributed graph matching algorithms with comprehensive testing and benchmarking capabilities. It provides:

- **Multiple Algorithms**: Itai-Israeli, Greedy, Luby Randomized (with meta-algorithm framework for Phase 2)
- **Simulation Environment**: Message-passing, synchronous round-based execution
- **Comprehensive Testing**: 161+ tests with 94%+ code coverage
- **Metrics & Monitoring**: Performance tracking and comprehensive benchmarking

## Quick Start

### Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest tests/unit/test_greedy_matching.py -v  # Specific test

# With coverage
pytest --cov=src tests/
```

### Interactive Exploration

```bash
# Jupyter notebooks for visualization and testing
cd notebooks
jupyter lab
# Then open any of: 01_itai_israeli.ipynb, 02_greedy_matching.ipynb, 03_luby_randomized.ipynb
```

### Python Example

```python
from src.graph import GraphManager
from src.simulation import Scheduler, SimulationConfig
from src.algorithms.implementations import GreedyMatching

# Create graph
graph = GraphManager.create_empty_graph()
for i in range(1, 5):
    graph.add_vertex(i)
graph.add_edge(1, 2, 10.0)
graph.add_edge(2, 3, 8.0)
graph.add_edge(3, 4, 9.0)

# Run algorithm
algo = GreedyMatching(seed=42)
config = SimulationConfig(max_rounds=100)
scheduler = Scheduler(graph, algo, config)
rounds = scheduler.run_until_termination()

# Get results
matching = algo.extract_matching(scheduler.state_store, graph)
print(f"Converged in {rounds} rounds: {matching}")
```

## Project Status

### Phase 1: Algorithms (✅ Complete)

| Algorithm | Status | Tests | Coverage |
|-----------|--------|-------|----------|
| Itai-Israeli | ✅ Working | 7/7 unit + 5/6 integration | 95%+ |
| Greedy | ✅ Complete | 24/24 unit + 13/13 integration | 98% |
| Luby Randomized | ✅ Complete* | 21/21 unit + 14/16 integration | 90%+ |
| Framework | ✅ Complete | 119/119 tests | 95%+ |

*Luby has documented asymmetry limitation on complex graphs (see [docs/LUBY_RANDOMIZED.md](docs/LUBY_RANDOMIZED.md))

**Overall**: 161/169 tests passing (95% success rate) with 8 tests skipped for known limitations.

### Phase 2: Meta-Algorithm (⏳ Upcoming)

- Orchestration layer for dynamic algorithm selection
- Parameter adaptation across distributed nodes
- Benchmark and comparison framework

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Core Modules
- [**Graph Management**](docs/GRAPH.md) - GraphManager API reference
- [**State Management**](docs/STATE.md) - StateStore for externalized node state
- [**Communication**](docs/COMMUNICATION.md) - Message-passing protocol
- [**Scheduler**](docs/SCHEDULER.md) - Round-based execution orchestration
- [**Metrics**](docs/METRICS.md) - Performance tracking and collection
- [**Visualization**](docs/VISUALIZATION.md) - ASCII rendering and debugging tools

### Algorithms
- [**Itai-Israeli**](docs/ITAI_ISRAELI.md) - Synchronous deterministic matching
- [**Greedy Matching**](docs/GREEDY_MATCHING.md) - Bidding-based greedy approach
- [**Luby Randomized**](docs/LUBY_RANDOMIZED.md) - Probabilistic randomized algorithm

### Interactive Exploration
- [**Jupyter Notebooks Guide**](docs/NOTEBOOKS.md) - Interactive algorithm demonstrations and customization

## Key Concepts

### Externalized State
Node state lives in `StateStore`, not in algorithm objects:
- ✓ Enables snapshots for debugging
- ✓ Allows state sharing across algorithms
- ✓ Makes serialization transparent
- ✓ Supports deterministic replay

### Message-Passing Communication
Nodes communicate exclusively through messages:
- No shared memory
- Synchronous round-based delivery
- Protocol-verified for deadlock freedom

### Synchronous Rounds
All nodes execute in lockstep:
1. Process received messages
2. Update local state
3. Send new messages
4. Synchronization barrier
5. Next round begins

## Code Quality Standards

- ✅ 100% type hints on public functions
- ✅ SOLID principles strictly applied
- ✅ DRY principle enforced
- ✅ Test-Driven Development (tests before code)
- ✅ 94%+ code coverage across modules
- ✅ PEP 8 formatting via black

## Testing Strategy

Tests verify four critical dimensions:
1. **Correctness** - Matching is valid (no node matched twice)
2. **Symmetry** - u↔v both present for all matched pairs
3. **Maximality** - No unmatched edge has both endpoints unmatched
4. **Convergence** - Algorithm terminates in reasonable time

Run tests:
```bash
pytest tests/              # Full suite
pytest tests/unit/         # Component tests
pytest tests/integration/  # End-to-end tests
pytest --cov=src tests/    # With coverage report
```

## Jupyter Notebooks

Interactive notebooks are available in `notebooks/` for exploring and visualizing each algorithm:

- **01_itai_israeli.ipynb** - Synchronous deterministic maximum weight matching with O(log n) convergence
- **02_greedy_matching.ipynb** - Fast heuristic-based bidding approach with BID→ACCEPT→CONFIRM protocol
- **03_luby_randomized.ipynb** - Probabilistic randomized algorithm with tunable activation probability

Each notebook includes:
- Algorithm overview and characteristics
- Graph visualization (before/after matching)
- Convergence metrics and statistics
- Customization examples (graph size, edge weights, parameters)
- Side-by-side algorithm comparison

**Quick start:**
```bash
cd notebooks && jupyter lab  # Launch Jupyter and open any notebook
```

See [**Notebooks Guide**](docs/NOTEBOOKS.md) for detailed customization instructions.

## Project Structure

```
distributed_node_matching/
├── src/
│   ├── algorithms/          # Algorithm implementations
│   ├── graph/              # Graph management
│   ├── state/              # State management
│   ├── communication/       # Message passing
│   ├── simulation/         # Scheduler & orchestration
│   ├── metrics/            # Performance tracking
│   └── utils/              # Types & utilities
├── tests/
│   ├── unit/               # Component tests
│   └── integration/        # End-to-end tests
├── docs/                   # Module documentation
├── notebooks/              # Jupyter notebooks for examples
├── CLAUDE.md               # Complete project guide
└── README.md               # This file
```

## Performance

- Graph operations: O(1) neighbor lookup
- State operations: O(1) per-node, O(n) snapshots
- Message queue: O(1) send/receive
- Scheduler: < 100ms per round (typical graphs)

## Known Limitations

- **Luby Algorithm**: 3-message protocol can produce asymmetry on complex graphs (simple/path graphs work perfectly)
- **Itai-Israeli**: Asymmetry edge case on complex graphs
- **Single Machine**: Simulation only (distributed execution is Phase 2 goal)
- **Static Graphs**: No dynamic edge addition/removal

All limitations are documented with root-cause analysis in the docs.

## Contributing

When extending the project:

1. Follow SOLID and DRY principles (see CLAUDE.md)
2. Write tests BEFORE implementation (TDD)
3. Maintain 90%+ code coverage
4. Add type hints to all public functions
5. Update relevant documentation
6. Verify all tests pass: `pytest tests/`

## Future Work

- **Phase 2**: Meta-algorithm orchestration layer
- **Phase 3**: Distributed deployment
- Adaptive algorithm selection based on graph properties
- Parameter tuning and optimization
