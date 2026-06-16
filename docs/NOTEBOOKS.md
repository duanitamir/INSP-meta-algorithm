# Interactive Jupyter Notebooks

This directory contains interactive Jupyter notebooks for exploring and visualizing each distributed matching algorithm implementation.

## Available Notebooks

### 1. `01_itai_israeli.ipynb`
**Itai-Israeli Algorithm - Maximum Weight Matching**

Demonstrates the Itai-Israeli algorithm, which finds maximum weight matchings in a distributed manner.

**Key Features:**
- Synchronous, deterministic execution
- Guaranteed O(log n) convergence
- Visualization of before/after matching states
- Convergence metrics (messages, rounds, nodes)
- Algorithm statistics and performance

---

### 2. `02_greedy_matching.ipynb`
**Greedy Matching Algorithm**

Demonstrates the greedy matching algorithm where each node bids for the highest-weight available neighbor.

**Key Features:**
- Fast heuristic-based approach
- BID → ACCEPT → CONFIRM protocol
- Visualization of matching evolution
- Message and convergence analysis
- Comparison with other approaches

---

### 3. `03_luby_randomized.ipynb`
**Luby-Style Randomized Matching**

Demonstrates the Luby-style randomized matching algorithm using probabilistic node activation.

**Key Features:**
- Randomized algorithm with O(log n) expected convergence
- Tunable activation probability (default 0.5)
- PROPOSE → ACCEPT → CONFIRM 3-message protocol
- Visualization of matching evolution
- Probabilistic analysis across multiple runs with different seeds
- Statistics averaged over multiple executions

---

## How to Use

### Quick Start

1. **Navigate to notebooks directory:**
   ```bash
   cd notebooks
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

3. **Open any notebook and run all cells**
   - Click "Run" → "Run All Cells" or use `Shift+Enter` on each cell

### Notebook Structure

Each notebook includes:

1. **Algorithm Overview** - Brief description and characteristics
2. **Graph Creation** - Generates a test graph with weighted edges
3. **Graph Visualization** - Visual display of the initial graph
4. **Algorithm Execution** - Runs the algorithm and captures metrics
5. **Matching Validation** - Verifies correctness and maximality
6. **Before/After Visualization** - Side-by-side comparison of matched/unmatched graphs
7. **Statistics** - Convergence metrics and performance data
8. **Convergence Analysis** - Line plots showing algorithm behavior over rounds

### Customization

Modify each notebook to experiment:

- **Change graph size/density:**
  ```python
  vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Add more vertices
  ```

- **Adjust edge weights:**
  ```python
  edges = [
      (1, 2, 20.0),  # Higher weights
      (1, 3, 15.0),
      ...
  ]
  ```

- **Change algorithm parameters:**
  ```python
  # For Luby randomized:
  algo = LubyRandomizedMatching(seed=42, activation_probability=0.3)  # Lower activation
  ```

- **Run on random graphs:**
  ```python
  graph = GraphManager.create_random_graph(num_vertices=20, density=0.4)
  ```

---

## Algorithm Comparison

All three notebooks use the **same test graph** for easy comparison:
- **6 vertices**
- **8 weighted edges**
- Uses `seed=42` for reproducibility

To compare algorithms on the same input:
1. Open all three notebooks side-by-side
2. Run them sequentially
3. Compare the convergence plots and matching quality

---

## Expected Outputs

### Before/After Matching Visualization
- **Left panel:** Original graph with all edges
- **Right panel:** Graph with matched edges highlighted in bold

### Convergence Plots
- **Messages per round:** Shows communication overhead
- **Matched nodes over time:** Shows matching progress
- **Active nodes over time:** Shows node participation
- **Summary statistics:** Algorithm performance metrics

---

## Troubleshooting

**ImportError when running cells:**
- Make sure you're running from the `notebooks/` directory
- The `sys.path.insert(0, '..')` line adds the parent directory to the path

**No plots showing:**
- Ensure you have `matplotlib` and `plotly` installed
- Run `%matplotlib inline` in a cell if needed

**Algorithm takes too long:**
- Reduce `max_rounds` in the `SimulationConfig`
- Use a smaller graph (fewer vertices, lower density)

---

## Next Steps

- **Modify these notebooks** for your own experiments
- **Compare algorithms** on different graph topologies
- **Run benchmarks** using the `algorithm_comparison.py` script in `examples/`
- **Extend the framework** by adding new algorithms

See `CLAUDE.md` for instructions on implementing new algorithms.
