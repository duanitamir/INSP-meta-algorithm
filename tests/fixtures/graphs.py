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

# Easy access
GRAPHS = [GRID_4x4, K5_CLUSTERS, STAR_WITH_TAIL]


def get_graph(name):
    """Get graph by name."""
    for g in GRAPHS:
        if g["name"].lower() == name.lower():
            return g
    raise ValueError(f"Graph '{name}' not found. Available: {[g['name'] for g in GRAPHS]}")
