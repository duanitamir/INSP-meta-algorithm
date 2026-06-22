# Canonical Vector: GA Chromosome for Meta-Algorithm

## Overview

The **Canonical Vector** is the chromosome representation used by the genetic algorithm to optimize distributed graph matching. It encodes 10 parameters that control how multiple matching algorithms execute and combine their results.

## Purpose

The canonical vector serves as:
1. **GA Chromosome**: Individuals in the genetic algorithm population
2. **Parameter Container**: All 10 tuning parameters in one structure
3. **Serializable State**: Supports to_list/from_list for GA operations
4. **Validated Container**: Enforces parameter ranges and constraints

## The 10 Parameters

### Luby Adaptive Activation (Parameters 0-6)
Coefficients for adaptive activation probability in Luby algorithm.

| Param | Name | Type | Range | Meaning |
|-------|------|------|-------|---------|
| 0 | luby_base_probability | float | [0, 1] | Base activation probability |
| 1 | luby_coeff_degree | float | [-1, 1] | Degree adjustment coefficient |
| 2 | luby_coeff_neighbors_unmatched | float | [-1, 1] | Unmatched neighbors weight |
| 3 | luby_coeff_clustering | float | [-1, 1] | Clustering coefficient weight |
| 4 | luby_coeff_matched | float | [-1, 1] | Matched neighbors weight |
| 5 | luby_coeff_round | float | [-1, 1] | Round number adjustment |
| 6 | luby_coeff_weight | float | [-1, 1] | Edge weight adjustment |

**How It Works** (Hybrid Fixed + Tuned):
```
activation_probability = clamp(
    base_probability
    + (coeff_degree * normalized_degree)
    + (coeff_unmatched * unmatched_ratio)
    + (coeff_clustering * clustering_coeff)
    + (coeff_matched * matched_ratio)
    + (coeff_round * round_progress)
    + (coeff_weight * normalized_weight),
    0.0, 1.0
)
```

You (the researcher) decide WHICH factors matter (the structure). The GA discovers optimal WEIGHTS for those factors.

**GA Insight**: This reveals which node properties actually matter for probabilistic activation. Some coefficients may converge to near-zero (unimportant), others to ±1 (critical).

### Itai-Israeli Timeout (Parameter 7)
Algorithm-specific parameter for Itai-Israeli matching.

| Param | Name | Type | Range | Meaning |
|-------|------|------|-------|---------|
| 7 | itai_timeout_rounds | int | [1, 20] | Max rounds per iteration before reset |

**Purpose**: Prevent infinite loops if algorithm gets stuck. If no progress after N rounds, restart.

### Cascading Loop Control (Parameters 8-9)
Meta-algorithm convergence control.

| Param | Name | Type | Range | Meaning |
|-------|------|------|-------|---------|
| 8 | max_iterations | int | [5, 100] | Maximum cascading loop iterations |
| 9 | convergence_threshold | float | [0, 0.1] | Min improvement to continue (fraction of edges) |

**How It Works**:
- Loop continues while all true:
  1. New matches found this iteration
  2. Total improvement >= convergence_threshold
  3. iterations < max_iterations
- Loop stops when any becomes false

## Design Rationale

### Why Only 10 Parameters?

The original design had 17 parameters including:
- 6 node selection percentile thresholds (degree, weight, centrality)
- 1 processing order parameter

**Problem Discovered**: Metrics don't change between cascading loop iterations (graph structure is fixed). Only the matching grows. This meant:
- Static filtering selected same nodes every iteration
- Nodes filtering was solving a non-existent problem
- Overly complex design

**Solution**: Removed node selection layer entirely:
- Let each algorithm's internal heuristics decide which nodes to focus on
- GA optimizes only algorithm-specific tuning + loop control
- 10 parameters is optimal (meaningful tuning without over-parameterization)

## Implementation

### Class: CanonicalVector

```python
class CanonicalVector:
    """10-parameter GA chromosome for meta-algorithm."""
    
    def __init__(self,
                 # Luby Adaptive (8)
                 luby_base_probability: float = 0.5,
                 luby_coeff_degree: float = 0.0,
                 luby_coeff_neighbors_unmatched: float = 0.0,
                 luby_coeff_clustering: float = 0.0,
                 luby_coeff_matched: float = 0.0,
                 luby_coeff_round: float = 0.0,
                 luby_coeff_weight: float = 0.0,
                 # Itai-Israeli (1)
                 itai_timeout_rounds: int = 5,
                 # Meta-algorithm (2)
                 max_iterations: int = 50,
                 convergence_threshold: float = 0.01):
        """Initialize with parameter values."""
    
    def validate(self) -> Tuple[bool, str | None]:
        """Check all constraints. Returns (is_valid, error_message)."""
    
    def to_list(self) -> List[float | int]:
        """Convert to list [p0, p1, ..., p9] for GA."""
    
    @classmethod
    def from_list(params: List[float | int]) -> 'CanonicalVector':
        """Create from list [p0, p1, ..., p9]."""
    
    @classmethod
    def random() -> 'CanonicalVector':
        """Create random valid vector."""
    
    def __str__(self) -> str:
        """Pretty-print all 10 parameters."""
```

### Key Methods

**validate()**
- Checks all percentiles in [0, 1]
- Checks Luby coefficients in [-1, 1]
- Checks base probability in [0, 1]
- Checks timeout in [1, 20]
- Checks max_iterations in [5, 100]
- Checks convergence_threshold in [0, 0.1]
- Returns (is_valid, error_message) tuple

**to_list() / from_list()**
- Used by genetic algorithm during crossover/mutation
- Ensures consistent parameter order: Luby[0:7], Itai[7], Loop[8:9]
- Enables round-trip serialization for GA operations

**random()**
- Creates valid random vector
- All parameters within valid ranges
- Respects parameter constraints
- Used to initialize GA population

## Usage in Meta-Algorithm

### Algorithm Execution

```python
# Each algorithm wrapper inherits from AlgorithmParameterizer
vector = canonical_vector

# Greedy: uses vector but has no algorithm-specific params
greedy_matching = greedy_parameterizer.execute(graph, vector)

# Itai: uses timeout parameter [7]
itai_matching = itai_parameterizer.execute(graph, vector)
# Internally uses: vector.itai_timeout_rounds

# Luby: uses adaptive activation coefficients [0:7]
luby_matching = luby_parameterizer.execute(graph, vector)
# Internally uses: vector.luby_* coefficients + vector.max_iterations + vector.convergence_threshold
```

### Convergence Control

```python
iteration = 0
while iteration < vector.max_iterations and has_improvement:
    new_matches = len(symmetric_matches)
    improvement_ratio = new_matches / total_edges
    has_improvement = improvement_ratio >= vector.convergence_threshold
    iteration += 1
```

## Genetic Algorithm Search Space

The GA searches over a 10-dimensional continuous space:
- **Population**: 50 vectors
- **Generations**: 200 (or early stop)
- **Fitness**: Total matching weight from cascading loop
- **Selection**: Tournament (best 50% survive)
- **Crossover**: Uniform (each parameter 50/50 from parent)
- **Mutation**: Gaussian (small perturbation per parameter)

### What the GA Discovers

1. **Luby Activation Tuning**
   - Base probability: Should activation be conservative (0.2) or aggressive (0.8)?
   - Which factors drive good probabilistic activation?
   - Should high-degree nodes activate more? (yes/no)
   - Is round progress important? (yes/no)
   - Which factor has biggest impact on convergence?

2. **Algorithm Timeout Strategy**
   - Should Itai-Israeli timeout early (5 rounds) or late (15 rounds)?
   - How does timeout affect convergence speed vs quality?

3. **Cascading Loop Efficiency**
   - How many iterations until convergence? (typically 5-20)
   - What improvement threshold balances speed vs quality?
   - Different thresholds for different graph types?

4. **Algorithm Adaptation**
   - Best parameter vector for dense graphs vs sparse graphs?
   - Different coefficients for different graph structures?

## Testing

Tests verify:
1. ✅ Initialization with defaults and custom values
2. ✅ Validation catches all constraint violations
3. ✅ Serialization round-trips correctly
4. ✅ Random generation always produces valid vectors
5. ✅ String representation includes all 10 parameters
6. ✅ Parameter ranges respect genetic algorithm needs

**Test Coverage**: 100% (14 tests, all passing)

## Files

- **Implementation**: `src/meta/canonical_vector.py`
- **Tests**: `tests/unit/test_canonical_vector.py`