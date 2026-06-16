# Graph Module

## Overview

The Graph module manages graph structure using NetworkX. It provides an interface for creating, querying, and analyzing undirected weighted graphs.

## Responsibilities

- Create and manage graph topology
- Add vertices and edges
- Query neighbors and degrees
- Check connectivity
- Find connected components

## Key Classes

### GraphManager

Main class for graph operations.

```python
graph = GraphManager.create_empty_graph()
graph.add_vertex(1)
graph.add_vertex(2)
graph.add_edge(1, 2, weight=1.5)

neighbors = graph.neighbors(1)  # frozenset([2])
degree = graph.degree(1)         # 1
num_edges = graph.num_edges()    # 1
```

## Public API

### Construction

- `create_empty_graph()` - Create empty graph
- `create_from_edges(vertices, edges)` - Create graph from lists

### Vertices

- `add_vertex(vertex_id, properties=None)` - Add vertex
- `num_vertices()` - Get vertex count
- `vertices()` - Get all vertices as frozenset

### Edges

- `add_edge(u, v, weight)` - Add weighted edge
- `num_edges()` - Get edge count
- `has_edge(u, v)` - Check if edge exists
- `get_edge_weight(u, v)` - Get edge weight

### Queries

- `neighbors(vertex_id)` - Get adjacent vertices
- `degree(vertex_id)` - Get vertex degree
- `max_degree()` - Get maximum degree
- `get_vertex_property(vertex_id, key)` - Get vertex metadata

### Connectivity

- `is_connected()` - Check if graph is connected
- `get_connected_components()` - Get all components

## Examples

### Create a simple path graph

```python
graph = GraphManager.create_empty_graph()
for i in range(1, 5):
    graph.add_vertex(i)

edges = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
for u, v, w in edges:
    graph.add_edge(u, v, w)
```

### Find neighbors of a vertex

```python
neighbors = graph.neighbors(2)  # frozenset([1, 3])
```

### Check connectivity

```python
if graph.is_connected():
    print("Graph is connected")
else:
    components = graph.get_connected_components()
    print(f"Found {len(components)} components")
```

## Design Notes

- Uses NetworkX for efficiency
- All queries return frozensets (immutable)
- Graphs are undirected (edges work both ways)
- Edge weights default to 1.0
- Vertices stored as integers

## Limitations

- No dynamic edge removal (would need design extension)
- Multigraphs not supported (one edge per pair)
- No directed graphs in current version
