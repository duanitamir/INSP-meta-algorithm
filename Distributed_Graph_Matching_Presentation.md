# Distributed Graph Matching Meta-Algorithm

---

## Slide 1: Title & Core Objective

# Distributed Graph Matching Meta-Algorithm

**Objective**: Compute maximum weight matchings in weighted graphs using multiple complementary algorithms optimized by genetic algorithm parameter tuning.

**Core Insight**: Combining diverse algorithms with systematic parameter optimization outperforms single-algorithm approaches.

---

## Slide 2: The Problem (High Level)

### What We're Solving

Given a weighted graph, find the best way to match vertices (pair them up) such that:
- No vertex appears twice
- Cannot add more edges without conflicts
- Maximize total weight of all pairs

### Why It Matters

This is fundamental to many applications:
- Network resource allocation
- Job scheduling
- Market clearing
- Social network analysis

### The Challenge: Which Algorithm to Use?

```
Different algorithms find different matchings:

Greedy:       Fast, finds obvious high-weight edges
Itai-Israeli: Exhaustive, explores all possibilities
Luby:         Parallel, randomized, tunable

Question: How do we combine them? Which parameters work best?
Answer: Try all combinations via genetic algorithm
```

---

## Slide 3: Our Solution Strategy

### Three-Step Approach

```
1. Run 3 different algorithms
   Each produces a different matching

2. Merge results intelligently
   Take best non-conflicting edges from all three
   Conflict resolution via endpoint voting

3. Optimize parameters
   Use genetic algorithm to find parameter combinations
   that maximize merged matching quality
```

### Why This Works

- No single algorithm is best on all graphs
- Algorithms find different high-quality edges
- Merging captures benefits of all three
- GA finds which parameters maximize the merge

---

## Slide 4: System Architecture (Top-Down View)

### Five Layers

```
Layer 5: Genetic Algorithm
         - Evolves 10-parameter vectors
         - Fitness = matching quality from running all 3 algorithms
         
Layer 4: Matching Algorithms
         - Greedy, Itai-Israeli, Luby Randomized
         - Each runs independently on same graph
         
Layer 3: Merging & Coordination
         - Combines results from 3 algorithms
         - Conflict resolution via voting
         
Layer 2: Communication & State
         - Algorithms communicate via messages
         - Central state store during GA fitness evaluation
         
Layer 1: Graph Representation
         - Vertices and edges
         - Weights on edges
```

Each layer is independent and can be tested/modified separately.

---

## Slide 5: The 10 Parameters

### What Gets Optimized?

GA evolves 10 parameters that control algorithm behavior:

```
Luby Algorithm (7 parameters):
  - Base activation probability: How likely each node proposes
  - 6 adaptive coefficients: Weight different node properties
    (degree, unmatched neighbors, clustering, matched status, round, edge weight)

Itai Algorithm (1 parameter):
  - Timeout rounds: When to stop voting

Meta Control (2 parameters):
  - Max iterations: How many cascades to run
  - Convergence threshold: Stop when improvement drops below this
```

Each parameter affects algorithm behavior in the cascading loop.

---

## Slide 6: How It Works - End-to-End Execution

### Cascading Execution Model

The system runs multiple cascades. Each cascade:
1. All 3 algorithms execute independently
2. Produce 3 different matchings
3. Merge results (highest-weight edges win, conflicts resolved)
4. Mark matched nodes as inactive
5. Next cascade runs on remaining graph
6. Stop when improvement drops below threshold

```
Cascade 0: Match high-weight edges
           Graph shrinks (matched nodes inactive)
           
Cascade 1: Match remaining edges
           Graph shrinks more
           
Cascade 2: Match what's left
           Stop when no improvement
           
Total fitness = sum of all matched edge weights
```

### Why Cascading?

Real matching finds edges iteratively. Cascading simulates this:
- First cascade gets obvious high-weight matches
- Later cascades find secondary matches
- Early termination when no progress (convergence threshold)

---

## Slide 7: Concrete Example - Algorithm Walkthrough

### Setup: 4-node graph
```
Edges: (0,1):2.5, (1,3):3.0, (0,2):1.0, (2,3):2.0, (0,3):0.5
All nodes start unmatched
```

### Cascade 0, Round 1: Three Algorithms Run

**Greedy** (tries highest-weight edges):
```
Node 0 proposes (0,1): 2.5 to Node 1
Node 1 votes YES (unmatched)
Node 1 proposes (1,3): 3.0 to Node 3
Node 3 votes YES
Result: Greedy matches (0,1) and (1,3)
```

**Itai-Israeli** (voting on all proposals):
```
All nodes send proposals simultaneously
Each votes for max-weight proposal they receive
Result: Itai matches (0,1) and (1,3)
```

**Luby** (adaptive random activation):
```
Each node calculates activation probability
based on degree, edge weights, coefficients
Activated nodes propose to best neighbors
Result: Luby matches (0,1) and (1,3)
```

### Merge Phase

```
Greedy:  [(0,1):2.5, (1,3):3.0]
Itai:    [(0,1):2.5, (1,3):3.0]
Luby:    [(0,1):2.5, (1,3):3.0]

Endpoint voting on conflicts:
  (0,1): nodes 0,1 both vote YES → ACCEPT
  (1,3): nodes 1,3 both vote YES → ACCEPT
  
Final Matching: {(0,1):2.5, (1,3):3.0}
Weight: 5.5

Convergence check:
  Matched nodes: 0,1,3
  Unmatched: 2
  Continue? Yes (improvement significant)
```

### Cascade 1

```
Only Node 2 unmatched, can't match (no unmatched neighbors)
Stop cascading

Total fitness: 5.5
```

---

## Slide 8: GA Evolution Loop

### How GA Optimizes Parameters

```
Generation 0: Create 20 random parameter vectors
              Evaluate each via cascading loop
              Fitness scores: [5.2, 5.4, 5.1, 5.3, ...]

Generation 1: Select top 50% (10 best vectors)
              Crossover: Blend parents to create 10 offspring
                For each parameter: randomly pick from one parent
              Mutation: 10% chance to perturb each parameter
              Evaluate all 20

Generation 2-10: Repeat selection → crossover → mutation → evaluation
                Best fitness improves each generation
                Converges when no more improvement
```

### Example: Parameter Blending

```
Parent A: [0.7, 0.1, 0.8, ...]  (fitness: 5.5)
Parent B: [0.6, 0.3, 0.7, ...]  (fitness: 5.4)

Offspring: [0.7, 0.3, 0.8, ...]  (different blend)
           Each parameter randomly from one parent
           
Evaluate offspring → fitness: 5.6 (improvement!)
This offspring survives to next generation
```

### Why Parameters Matter

Different parameter vectors lead to different cascade behaviors:
- Vector A: Luby favors high-degree nodes → finds different edges
- Vector B: Luby favors weight-heavy edges → finds other edges
- GA discovers which combination maximizes total matching weight

---

## Slide 9: Meta Layer - System Coordination

### What is the Meta Layer?

The meta layer orchestrates how algorithms work together and how parameters get tuned. It has four key responsibilities:

### 1. Parameter Management

**Responsibility**: Define and validate the search space for GA

```
CanonicalVector (10 parameters):
  - Controls how aggressive each algorithm is
  - Defines which node properties matter in activation
  - Sets convergence tolerance
  - Immutable across entire execution (no mid-flight changes)

GA Population:
  Generation 0: 20 random vectors
  Generation 1-10: Evolve via selection, crossover, mutation
  Result: Best vector found for this graph
```

### 2. Fitness Evaluation

**Responsibility**: Score how good a parameter vector is

```
Evaluator takes: CanonicalVector + Graph
         runs: Three algorithms with these parameters
         does: Merge results intelligently
      returns: Total matching weight (fitness score)

Two strategies:
  Single-Pass: Run once, merge, done (fast)
  Cascading: Run on full graph, then on remaining graph,
             repeat until convergence (realistic)

Both strategies answer: "How good is this parameter set?"
```

### 3. Parameter Binding

**Responsibility**: Connect abstract parameters to concrete algorithm behavior

```
The problem: How do we make parameters actually matter?

Solution: Parameter binding
  Greedy algorithm:
    - Gets: max_iterations parameter
    - Does: Runs that many rounds
    - Effect: More iterations → finds more edges
    
  Itai algorithm:
    - Gets: itai_timeout_rounds parameter
    - Does: Stops voting after this many rounds
    - Effect: Higher timeout → more exploration
    
  Luby algorithm:
    - Gets: base_probability + 6 coefficients
    - Does: Adjusts activation based on node properties
    - Effect: Different coefficients prefer different node types

Without binding: Parameters exist but don't change behavior
With binding: GA can actually search for improvements
```

### 4. Coordination Protocol

**Responsibility**: Define how nodes communicate during distributed execution

```
Message types exchanged between nodes:

Edge Conflict Protocol:
  Node proposes: "I want to match (0,1) with weight 2.5"
  Node responds: "I vote YES" or "I vote NO"
  Rule: Both endpoints must agree → conflict-free matching

Convergence Protocol:
  Each node measures: "Am I making progress?"
  Node votes: "Should we stop?" YES or NO
  Rule: If >50% vote YES → entire network stops

Parameter Sharing Protocol (Future):
  Node: "These parameters worked well for me"
  Neighbors: Receive and try these parameters
  Result: Network learns good parameters without central trainer
```

### How These Pieces Work Together

```
GA Side (Centralized):                Distributed Side (Network):
  
  CanonicalVector                      Nodes running algorithms
        ↓                                      ↑
  FitnessEvaluator                    Coordination Protocol
        ↓                                      ↓
  Parameter Binding                   Message Exchange
        ↓                                      ↓
  Run Algorithms                       Reach Agreement
        ↓                                      ↓
  Merge Results                        Final Matching
        ↓                                      ↑
  Return Weight ←────────────────────────────┘
  
Score tells GA which parameters work best.
Network uses best parameters to execute efficiently.
```

---

## Slide 10: Current Implementation Status

### What's Working

- Three algorithms running independently with full parameter support
- Cascading loop with intelligent convergence detection (2 cascades on 1K graph)
- Genetic algorithm successfully evolving parameter vectors across generations
- Merging strategy combining best edges from all algorithms
- Distributed message protocols defined and ready

### System Performance (1000-Node Clustered Graph)

```
Individual Algorithm Results:
  Greedy:              1897 weight (57.4% gap to optimal)
  Itai-Israeli:        3484 weight (21.7% gap to optimal)
  Luby Randomized:     3636 weight (18.3% gap to optimal)

After Merging:
  Merged Baseline:     3871 weight (13.1% gap to optimal)

After GA + Cascading:
  GA Best Found:       4183 weight (6.04% gap to optimal)
  NetworkX Optimal:    4452 weight

GA Improvement:        +8.1% over merged baseline
Cascading Benefit:     +2.1% from multi-round execution
```

### Foundation Enables

1. Centralized GA optimization on large graphs (1K+ nodes)
2. Distributed execution infrastructure (ready for deployment)
3. Parameter adaptation framework (gossip-based learning vision)
4. Easy algorithm composition (add new algorithms without rebuilding)

---

## Slide 11: Experimental Results (1000-Node Graph)

### Test Graph: Clustered Graph with Communities

```
Vertices: 1000
Edges: ~5000
Structure: Multiple dense clusters with weak inter-cluster bridges
Purpose: Large-scale evaluation of system performance
```

### Individual Algorithm Performance

```
Algorithm Results (1000-node clustered):
  Greedy:              1897 weight
  Itai-Israeli:        3484 weight
  Luby Randomized:     3636 weight
  
Quality gaps to NetworkX optimal (4452):
  Greedy:              -57.4%
  Itai-Israeli:        -21.7%
  Luby Randomized:     -18.3%
```

### Cascading & Merging Impact

```
Merged Baseline:       3871 weight
  (Best non-conflicting edges from all 3)
  
Cascading Baseline:    3952 weight
  (Runs 2 cascades: 3804 + 3952)
  
Cascading benefit:     +2.1% improvement
```

### GA Optimization Results

```
GA Evolution (10 generations, population 20):
  Generation 0:        4049 weight
  Generation 5:        4117 weight
  Generation 10:       4183 weight
  
  Total Improvement:   +3.31%
  Gap to Optimal:      6.04%
```

### Fitness Progression Visualization

[CHART from cell 27: Fitness vs Generation]
- Shows GA curve reaching 4183
- Cascading baseline at 3952
- Individual algorithm baselines (Greedy 1897, Itai 3484, Luby 3636)
- NetworkX optimal at 4452

### System Performance Summary

- Merging benefit:     +8.1% (3871 vs best single 3636)
- Cascading benefit:   +2.1% (3952 vs 3871)
- GA benefit:          +3.31% (4183 vs 4049)
- Execution per gen:   1.7 seconds (20 population)

---

## Slide 12: Conclusions & Next Steps

### What We Demonstrated

- Multiple algorithms can be combined via intelligent merging
- Cascading loop effectively simulates iterative matching
- Genetic algorithm tunes parameters to improve quality
- System scales to 1000+ nodes
- Architecture enables distributed deployment

### Key Results

- Merging algorithms beats any individual approach
- Cascading adds realistic multi-round behavior
- GA discovers parameter combinations that optimize results
- Foundation ready for real network deployment

### Immediate Next Steps

**Expand parameter space**: Add more parameters per algorithm
**Add 4th algorithm**: Increase diversity for GA to optimize
**Scale testing**: Run on larger graphs (5K, 10K nodes)
**Distributed deployment**: Move from centralized GA to gossip-based learning

### What This Enables

Decentralized graph matching at scale without central coordinator. Nodes autonomously execute, coordinate via message passing, and converge on optimal matchings through parameter-optimized algorithms.

---
