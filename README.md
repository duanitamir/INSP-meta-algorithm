# Distributed Graph Matching Meta-Algorithm

This project searches offline for a configuration vector and applies it online
through autonomous distributed graph-matching nodes. A node sees only its
direct neighbours, incident edge weights, local state, and messages addressed
to it. The simulator may schedule and observe nodes, but it never chooses a
match or decides convergence for them.

The architecture is defined by
[Distributed Execution: Source of Truth](DISTRIBUTED_EXECUTION_SOURCE_OF_TRUTH.md).

## Workflow

1. Register a self-contained local proposal policy.
2. Select the policies for an experiment with `SELECTED_ALGORITHMS` in
   `notebooks/test_meta_algorithm.ipynb`.
3. The registry derives the vector schema, baselines, policy weights, plots,
   and exports from that selection.
4. The GA evaluates each candidate only with `DistributedRuntimeEvaluator`.
5. A distributed run completes normally only when every node is matched or
   terminal-unmatched (`outcome == "quiescent"`).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -v
ruff check src tests
```

To run the experiment notebook:

```bash
cd notebooks
jupyter lab test_meta_algorithm.ipynb
```

## Minimal runtime example

```python
from src.graph import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.distributed.orchestrator import DistributedOrchestrator

graph = GraphManager.create_from_edges(
    [1, 2, 3, 4],
    [(1, 2, 10.0), (2, 3, 8.0), (3, 4, 9.0)],
)

matching, report = DistributedOrchestrator(max_workers=1).execute(
    graph, CanonicalVector()
)
assert report["outcome"] == "quiescent"
print(matching, report["final_weight"])
```

## Project map

- `src/algorithms/` — registered local proposal policies
- `src/graph/` — graph storage and local topology views
- `src/communication/` — recipient-scoped in-memory transport
- `src/simulation/` — endpoint protocol and neutral scheduling
- `src/meta/` — canonical vectors, offline GA, evaluation, and benchmarks
- `notebooks/test_meta_algorithm.ipynb` — registry-driven experiment workflow
- `tests/` — unit and integration coverage
- `docs/` — current references and dated historical records

## Known scope

The project simulates a distributed system in one process. Graph topology and
the vector are fixed during one run; dynamic topology changes and live
configuration replacement are outside the current model.
