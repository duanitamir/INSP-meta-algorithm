# Distributed Graph Matching Meta-Algorithm

A research-grade framework for simulating and experimenting with distributed graph matching algorithms.

## Overview

This project implements multiple distributed graph matching algorithms with comprehensive testing and benchmarking capabilities. It provides:

- **Multiple Algorithms**: Itai-Israeli, Greedy, Luby Randomized
- **Genetic Algorithm Optimizer**: Auto-tunes 10 parameters across all 3 algorithms
- **Simulation Environment**: Message-passing, synchronous round-based execution
- **Comprehensive Testing**: 147 tests with 100% passing rate
- **Interactive Analytics**: Web-based dashboard with component documentation and flow visualization

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

### Genetic Algorithm Optimization & Dashboard

```bash
# 1. Run the GA optimization notebook
cd notebooks
jupyter lab test_meta_algorithm.ipynb

# 2. In Cell 12, run to export experiment results
# Creates: results/jsons/experiment_YYYYMMDD_HHMMSS.json

# 3. View results in interactive dashboard
# Open in browser: results/index.html
# Dashboard auto-discovers all experiments and displays them
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

## Features

### Multiple Graph Matching Algorithms

| Algorithm | Approach | Complexity | Best For |
|-----------|----------|-----------|----------|
| **Greedy** | Fast heuristic with bidding protocol | O(d²) per round | Speed-priority, real-time |
| **Itai-Israeli** | Synchronous deterministic matching | O(log n) rounds | Quality-priority, optimal bounds |
| **Luby Randomized** | Probabilistic parallel algorithm | O(log² n) rounds | Large graphs, distributed settings |

All algorithms guarantee valid matchings with configurable tuning parameters.

### Genetic Algorithm Meta-Optimizer

Automatically tunes algorithm parameters across all 3 algorithms:
- **Evolutionary Search**: 20-population × 10-generation GA (configurable)
- **Adaptive Parameters**: 10 tunable parameters across algorithms
- **Multi-Algorithm Merging**: Combines outputs from all 3 algorithms intelligently
- **Quality Metrics**: Tracks fitness improvement across generations
- **Cascading Evaluation**: Re-runs optimized parameters on dynamic graph updates

### Performance-Optimized Execution

- **Node Parallelization**: 21.65x speedup with ThreadPoolExecutor (4 workers)
- **Batch State Updates**: Reduced lock contention via batching
- **Lazy Computation**: Skip expensive operations when unnecessary
- **Early Termination**: Saturation detection for algorithm convergence
- **Total Speedup**: 24.4x from baseline (2460ms → 101.2ms per evaluation)

### Distributed Simulation Environment

- **Message-Passing Communication**: Async message delivery with per-node queues
- **Synchronous Rounds**: Lock-step execution with barrier synchronization
- **State Externalization**: Centralized StateStore for snapshot/replay debugging
- **Protocol Verification**: Deadlock-free message delivery guarantee
- **Cascade Support**: Run multiple algorithm passes with state preservation

### Comprehensive Analytics & Visualization

- **Interactive Dashboard**: Web-based analytics hub with component descriptions and execution flows
- **Experiment Tracking**: Auto-discovery of experiment results with live updates
- **Component Documentation**: 14-component system with detailed descriptions and interactions
- **Flow Visualization**: Visual pipeline of 6-phase distributed execution
- **GA Analysis**: Per-seed fitness progression, algorithm contributions, and parameter evolution
- **Comparative Analysis**: Side-by-side seed comparison with improvement metrics

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
- Enables snapshots for debugging
- Allows state sharing across algorithms
- Makes serialization transparent
- Supports deterministic replay

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

## Key Capabilities

### Algorithm Features
- **10 Tunable Parameters**: Configure algorithm behavior (probabilities, thresholds, iteration limits)
- **Real-Time Tuning**: Modify parameters mid-run via genetic algorithm optimization
- **Symmetric Matching Guarantee**: All algorithms produce valid A↔B pairs only
- **Conflict Resolution**: Intelligent merging of algorithm outputs using endpoint voting
- **Convergence Detection**: Network-wide quorum-based consensus detection

### Performance Features
- Optimized execution pipeline with parallel node processing
- **Scalable Architecture**: Design verified on 50-1000 node graphs
- **Early Exit Detection**: Saturation detection for on-large graphs
- **Executor-Based Parallelism**: 4-worker ThreadPoolExecutor with dynamic scaling

## Analytics Dashboard

The interactive dashboard (`results/index.html`) provides comprehensive analysis of GA optimization runs:

### Dashboard Features

- **Experiment Browser**: View all GA runs with auto-discovery of experiment JSONs
- **Component Documentation**: 14-component system with detailed descriptions
  - State Store, Message Queue, Transport Interface
  - Greedy Matching, Itai-Israeli, Luby Randomized
  - Conflict Resolver, Convergence Detector
  - And 6 more core system components
- **Flow Visualization**: Interactive pipeline showing 6-phase distributed execution
  - Message Processing → Proposals → Accumulation → Conflict Resolution → Confirmation → Convergence
  - Per-node execution details with algorithm proposals
  - State management and message queue visualization
- **Per-Experiment Analysis**:
  - Multi-seed comparison table (baseline, GA, optimal)
  - Generation-by-generation fitness progression
  - Algorithm contribution analysis
  - Parameter evolution across seeds
  - Gap to optimal and improvement metrics

### Using the Dashboard

1. **Run optimization**: Execute `notebooks/test_meta_algorithm.ipynb` Cell 12
2. **Export results**: Experiment JSON saved to `results/jsons/`
3. **View dashboard**: Open `results/index.html` in a web browser
4. **Explore**: Dashboard auto-discovers experiments, sorted by timestamp (newest first)
5. **Details**: Click "View Details" on any experiment card for full analysis

Dashboard updates automatically when new experiment JSONs are added to `results/jsons/`.

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
│   │   └── implementations/ # Greedy, Itai-Israeli, Luby
│   ├── graph/              # Graph management (GraphManager)
│   ├── state/              # State management (StateStore, NodeState)
│   ├── communication/       # Message passing (semantic messages, queues)
│   ├── simulation/         # Scheduler & orchestration
│   ├── meta/               # Meta-algorithm (GA, fitness evaluator, parameterizers)
│   ├── metrics/            # Performance tracking
│   └── utils/              # Types & utilities
├── tests/
│   ├── unit/               # Component tests (147 tests, 100% passing)
│   └── integration/        # End-to-end tests
├── results/
│   ├── index.html          # Interactive analytics dashboard
│   ├── jsons/              # Experiment data (auto-discovered)
│   └── serve_dashboard.py  # Optional Flask server
├── notebooks/
│   ├── test_meta_algorithm.ipynb     # GA optimization & analysis (main entry point)
│   ├── 01_itai_israeli.ipynb         # Algorithm exploration
│   ├── 02_greedy_matching.ipynb      # Algorithm exploration
│   ├── 03_luby_randomized.ipynb      # Algorithm exploration
│   └── helpers/                      # Reusable utilities for notebooks
├── CLAUDE.md               # Complete project guide & development rules
└── README.md               # This file
```


## Known Limitations

- **Luby Algorithm**: 3-message protocol can produce asymmetry on complex graphs (simple/path graphs work perfectly)
- **Itai-Israeli**: Asymmetry edge case on complex graphs
- **Single Machine**: Simulation only (distributed execution is Phase 2 goal)
- **Static Graphs**: No dynamic edge addition/removal

All limitations are documented with root-cause analysis in the docs.