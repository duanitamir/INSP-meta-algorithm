"""Simple graph definitions for notebook testing.

Each graph is a dictionary with:
- vertices: list of vertex IDs
- edges: list of (u, v, weight) tuples
- optimal_weight: estimated optimal total weight
- best_matches: list of expected optimal matching pairs (for comparison)
"""

# Graph 1: 4x4 Grid
GRID_4x4 = {
    "name": "4x4 Grid Graph",
    "vertices": list(range(1, 17)),
    "edges": [
        # Horizontal edges (weight 10)
        (1, 2, 10), (2, 3, 10), (3, 4, 10),
        (5, 6, 10), (6, 7, 10), (7, 8, 10),
        (9, 10, 10), (10, 11, 10), (11, 12, 10),
        (13, 14, 10), (14, 15, 10), (15, 16, 10),
        # Vertical edges (weight 8)
        (1, 5, 8), (2, 6, 8), (3, 7, 8), (4, 8, 8),
        (5, 9, 8), (6, 10, 8), (7, 11, 8), (8, 12, 8),
        (9, 13, 8), (10, 14, 8), (11, 15, 8), (12, 16, 8),
    ],
    "optimal_weight": 96,
    "best_matches": [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    ],
}

# Graph 2: Two K5 Clusters
K5_CLUSTERS = {
    "name": "Two K5 Clusters Merged",
    "vertices": list(range(1, 11)),
    "edges": [
        # Cluster 1: K5 on nodes 1-5 (weight 15)
        (1, 2, 15), (1, 3, 15), (1, 4, 15), (1, 5, 15),
        (2, 3, 15), (2, 4, 15), (2, 5, 15),
        (3, 4, 15), (3, 5, 15),
        (4, 5, 15),
        # Cluster 2: K5 on nodes 6-10 (weight 15)
        (6, 7, 15), (6, 8, 15), (6, 9, 15), (6, 10, 15),
        (7, 8, 15), (7, 9, 15), (7, 10, 15),
        (8, 9, 15), (8, 10, 15),
        (9, 10, 15),
        # Bridge (weight 1) - should be avoided
        (5, 6, 1),
    ],
    "optimal_weight": 60,
    "best_matches": [
        (1, 2), (3, 4), (6, 7), (8, 9),  # 4 pairs within clusters
    ],
}

# Graph 3: Star with Tail
STAR_WITH_TAIL = {
    "name": "Weighted Star with Tail",
    "vertices": list(range(1, 12)),
    "edges": [
        # Star edges (center is 7, weight 12)
        (7, 1, 12), (7, 2, 12), (7, 3, 12), (7, 4, 12), (7, 5, 12), (7, 6, 12),
        # Tail edges (weight 5)
        (1, 8, 5), (8, 9, 5), (9, 10, 5), (10, 11, 5),
    ],
    "optimal_weight": 22,
    "best_matches": [
        (7, 2), (1, 8), (9, 10),  # 3 pairs: weight 12 + 5 + 5 = 22 (maximal)
    ],
}

# Graph 4: Random Dense Graph (1000 nodes)
def _create_random_dense_graph():
    """Generate random dense graph with ~1000 nodes and ~5000 edges."""
    import random
    random.seed(42)

    vertices = list(range(1, 1001))
    edges = []

    # Create connected backbone first (ensures connectivity)
    for i in range(len(vertices) - 1):
        u, v = vertices[i], vertices[i + 1]
        weight = random.randint(1, 100)
        edges.append((u, v, weight))

    # Add random edges to increase density
    target_edges = 5000
    while len(edges) < target_edges:
        u = random.choice(vertices)
        v = random.choice(vertices)
        if u != v and (min(u, v), max(u, v)) not in {(min(a, b), max(a, b)) for a, b, _ in edges}:
            weight = random.randint(1, 100)
            edges.append((u, v, weight))

    # Calculate approximate optimal weight (rough estimate)
    # For random graphs, maximal matching ≈ n/2 * avg_weight
    avg_weight = sum(w for _, _, w in edges) / len(edges)
    optimal_weight = int(500 * avg_weight)

    return {
        "name": "Random Dense Graph (1000 nodes)",
        "vertices": vertices,
        "edges": edges,
        "optimal_weight": optimal_weight,
        "best_matches": [],  # Unknown for random graph
    }

RANDOM_DENSE_GRAPH_1K = _create_random_dense_graph()

# Graph 5: Clustered Graph with Communities (1000 nodes)
def _create_clustered_graph():
    """Generate clustered graph with 10 communities of ~100 nodes each."""
    import random
    random.seed(43)

    vertices = list(range(1, 1001))
    edges = []
    num_communities = 10
    community_size = 100

    # Create dense communities
    for c in range(num_communities):
        start = c * community_size + 1
        end = (c + 1) * community_size + 1
        community_nodes = list(range(start, end))

        # Add edges within community (high weight)
        for i in range(len(community_nodes)):
            for j in range(i + 1, min(i + 5, len(community_nodes))):  # Each node connects to ~5 others
                u, v = community_nodes[i], community_nodes[j]
                weight = random.randint(50, 100)
                edges.append((u, v, weight))

    # Add sparse inter-community edges (low weight)
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            u = random.randint(c1 * community_size + 1, (c1 + 1) * community_size)
            v = random.randint(c2 * community_size + 1, (c2 + 1) * community_size)
            weight = random.randint(1, 20)
            edges.append((u, v, weight))

    # Approximate optimal: ~50 nodes per community match internally at high weight
    optimal_weight = int(num_communities * 25 * 75)  # Rough estimate

    return {
        "name": "Clustered Graph with Communities (1000 nodes)",
        "vertices": vertices,
        "edges": list(set(edges)),  # Remove duplicates
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

CLUSTERED_GRAPH_1K = _create_clustered_graph()

# Graph 6: Scale-Free Graph (1000 nodes, power-law degree distribution)
def _create_scale_free_graph():
    """Generate scale-free graph using preferential attachment."""
    import random
    random.seed(44)

    vertices = list(range(1, 1001))
    edges = []
    degrees = {v: 0 for v in vertices}

    # Start with a small connected core
    core = vertices[:10]
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            u, v = core[i], core[j]
            weight = random.randint(10, 100)
            edges.append((u, v, weight))
            degrees[u] += 1
            degrees[v] += 1

    # Preferential attachment: new nodes connect to high-degree nodes
    for new_node in vertices[10:]:
        # Select k nodes based on current degrees (preferential attachment)
        k = min(5, len(vertices) - 1)
        candidates = random.choices(vertices[:vertices.index(new_node)],
                                   weights=[degrees[v] + 1 for v in vertices[:vertices.index(new_node)]],
                                   k=k)

        for target in set(candidates):
            if target != new_node:
                weight = random.randint(1, 100)
                edges.append((min(new_node, target), max(new_node, target), weight))
                degrees[new_node] += 1
                degrees[target] += 1

    # Remove duplicate edges
    edges = list(set(edges))

    # Hubs in scale-free networks have degree ~sqrt(n), expect ~20-30 high-degree nodes
    # Optimal matching exploits hubs: ~5-10 hub matches at high weight
    optimal_weight = int(10 * 50)  # Rough: 10 hub pairs at avg 50 weight

    return {
        "name": "Scale-Free Graph (1000 nodes, power-law)",
        "vertices": vertices,
        "edges": edges,
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

SCALE_FREE_GRAPH_1K = _create_scale_free_graph()

# Graph 7: Bipartite Graph (1000 nodes, two sides of 500 each)
def _create_bipartite_graph():
    """Generate complete bipartite graph K(500,500) with random weights."""
    import random
    random.seed(45)

    left_nodes = list(range(1, 501))
    right_nodes = list(range(501, 1001))
    vertices = left_nodes + right_nodes
    edges = []

    # Connect each left node to ~10 random right nodes
    for u in left_nodes:
        targets = random.sample(right_nodes, k=min(10, len(right_nodes)))
        for v in targets:
            weight = random.randint(1, 100)
            edges.append((u, v, weight))

    # Perfect matching possible: 500 pairs at various weights
    avg_weight = sum(w for _, _, w in edges) / len(edges)
    optimal_weight = int(500 * avg_weight)

    return {
        "name": "Bipartite Graph (500+500 nodes)",
        "vertices": vertices,
        "edges": edges,
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

BIPARTITE_GRAPH_1K = _create_bipartite_graph()

# Easy access
GRAPHS = [GRID_4x4, K5_CLUSTERS, STAR_WITH_TAIL, RANDOM_DENSE_GRAPH_1K, CLUSTERED_GRAPH_1K, SCALE_FREE_GRAPH_1K, BIPARTITE_GRAPH_1K]


def get_graph(name):
    """Get graph by name."""
    for g in GRAPHS:
        if g["name"].lower() == name.lower():
            return g
    raise ValueError(f"Graph '{name}' not found. Available: {[g['name'] for g in GRAPHS]}")
