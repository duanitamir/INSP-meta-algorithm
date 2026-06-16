# Visualization Module

## Overview

The Visualization module renders graph structure, state, and matching results for analysis and debugging.

## Responsibilities

- Render graphs to multiple formats (ASCII, DOT, JSON)
- Display node state and matching
- Provide graph summaries and analysis
- Support debugging visualization

## Key Classes

### GraphVisualizer

Visualizes graph structure and overlays state/matching.

```python
viz = GraphVisualizer(graph)
print(viz.summary())
print(viz.render_to_ascii(state_store=store))
print(viz.render_matching_to_ascii(matching))
```

## Public API

### GraphVisualizer

**Construction:**
- `__init__(graph)` - Create visualizer for graph

**Rendering:**
- `render_to_ascii(state_store=None, matching=None) -> str` - ASCII graph
- `render_matching_to_ascii(matching) -> str` - Matching analysis
- `summary() -> str` - Graph summary statistics

## Examples

### Simple Graph Summary

```python
viz = GraphVisualizer(graph)
print(viz.summary())
```

Output:
```
==================================================
GRAPH SUMMARY
==================================================
Vertices: 4
Edges: 3
Max degree: 2
Average degree: 1.50
Connected: True
```

### Render Graph with State

```python
print(viz.render_to_ascii(state_store=store))
```

Output:
```
Graph: 4 vertices, 3 edges

Vertex Summary:
  1: degree=1 → matched to 2
  2: degree=2 → matched to 1
  3: degree=2
  4: degree=1

Edge Summary:
  1 -- 2: weight=1.00 [MATCHED]
  2 -- 3: weight=1.00
  3 -- 4: weight=1.00
```

### Render Matching Results

```python
matching = {1: 2, 2: 1, 3: 4, 4: 3}
print(viz.render_matching_to_ascii(matching))
```

Output:
```
MATCHING ANALYSIS
==================================================
Total vertices: 4
Matched vertices: 4
Unmatched vertices: 0
Matching size (edges): 2
Total weight: 2.00
Average edge weight: 1.00

Matched edges:
  1 -- 2: weight=1.00
  3 -- 4: weight=1.00
```

## Output Formats

### ASCII Format (Terminal)

Pros:
- Human-readable
- No external dependencies
- Works everywhere

Example:
```
Graph: 10 vertices, 15 edges

Vertex Summary:
  1: degree=3 → matched to 5
  2: degree=2
  ...
```

### For Future: DOT Format

For Graphviz visualization:
```
graph G {
  1 -- 2 [weight=1.0, color=black];
  1 -- 5 [weight=1.0, color=red, style=bold];
  ...
}
```

### For Future: JSON Format

For web/interactive visualization:
```json
{
  "vertices": [...],
  "edges": [...],
  "matching": {...}
}
```

## Design Notes

- ASCII format for immediate feedback
- State overlays show matching status
- Edge weights displayed with weights
- Unmatched vertices listed
- Graph connectivity information included

## Performance

- O(v + e) rendering (v = vertices, e = edges)
- ~100ms for typical graph
- No caching needed

## Limitations

- ASCII format limited to ~50 vertices (readability)
- No interactive visualization (yet)
- No automatic layout (yet)

## Future Extensions

- DOT file generation for Graphviz
- JSON export for web visualization
- Interactive HTML visualization
- Animated execution playback
- Matching quality metrics in rendering
