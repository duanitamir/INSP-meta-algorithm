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
def _create_random_dense_graph(seed=None):
    """Generate random dense graph with ~1000 nodes and ~5000 edges.

    Args:
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=42 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, 1001))
    edges = []

    # Create connected backbone first (ensures connectivity)
    for i in range(len(vertices) - 1):
        u, v = vertices[i], vertices[i + 1]
        weight = rng.randint(1, 100)
        edges.append((u, v, weight))

    # Add random edges to increase density
    target_edges = 5000
    while len(edges) < target_edges:
        u = rng.choice(vertices)
        v = rng.choice(vertices)
        if u != v and (min(u, v), max(u, v)) not in {(min(a, b), max(a, b)) for a, b, _ in edges}:
            weight = rng.randint(1, 100)
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

RANDOM_DENSE_GRAPH_1K = _create_random_dense_graph(seed=42)  # seed=42 for reproducibility

# Graph 5: Clustered Graph with Communities (1000 nodes)
def _create_clustered_graph(nr_of_nudes: int = 1000, nr_communities: int = 10, seed=None):
    """Generate clustered graph with communities.

    Args:
        nr_of_nudes: Number of vertices
        nr_communities: Number of communities
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=43 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, nr_of_nudes+1))
    edges = []
    num_communities = nr_communities
    community_size = nr_of_nudes // nr_communities  # Dynamic sizing

    # Create dense communities
    for c in range(num_communities):
        start = c * community_size + 1
        end = (c + 1) * community_size + 1
        community_nodes = list(range(start, end))

        # Add edges within community (high weight)
        for i in range(len(community_nodes)):
            for j in range(i + 1, min(i + 5, len(community_nodes))):  # Each node connects to ~5 others
                u, v = community_nodes[i], community_nodes[j]
                weight = rng.randint(50, 100)
                edges.append((u, v, weight))

    # Add sparse inter-community edges (low weight)
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            u = rng.randint(c1 * community_size + 1, (c1 + 1) * community_size)
            v = rng.randint(c2 * community_size + 1, (c2 + 1) * community_size)
            weight = rng.randint(1, 20)
            edges.append((u, v, weight))

    # Approximate optimal: ~50 nodes per community match internally at high weight
    optimal_weight = int(num_communities * 25 * 75)  # Rough estimate

    return {
        "name": f"Clustered Graph with Communities ({nr_of_nudes} nodes)",
        "vertices": vertices,
        "edges": list(set(edges)),  # Remove duplicates
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

CLUSTERED_GRAPH_1K = _create_clustered_graph(seed=43)  # seed=43 for reproducibility
CLUSTERED_GRAPH_500 = _create_clustered_graph(nr_of_nudes=500, nr_communities=10, seed=43)
CLUSTERED_GRAPH_100 = _create_clustered_graph(nr_of_nudes=100, nr_communities=5, seed=43)

# Graph 6: Scale-Free Graph (1000 nodes, power-law degree distribution)
def _create_scale_free_graph(seed=None):
    """Generate scale-free graph using preferential attachment.

    Args:
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=44 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, 1001))
    edges = []
    degrees = {v: 0 for v in vertices}

    # Start with a small connected core
    core = vertices[:10]
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            u, v = core[i], core[j]
            weight = rng.randint(10, 100)
            edges.append((u, v, weight))
            degrees[u] += 1
            degrees[v] += 1

    # Preferential attachment: new nodes connect to high-degree nodes
    for new_node in vertices[10:]:
        # Select k nodes based on current degrees (preferential attachment)
        k = min(5, len(vertices) - 1)
        candidates = rng.choices(vertices[:vertices.index(new_node)],
                                   weights=[degrees[v] + 1 for v in vertices[:vertices.index(new_node)]],
                                   k=k)

        for target in set(candidates):
            if target != new_node:
                weight = rng.randint(1, 100)
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

SCALE_FREE_GRAPH_1K = _create_scale_free_graph(seed=44)  # seed=44 for reproducibility

# Graph 7: Bipartite Graph (1000 nodes, two sides of 500 each)
def _create_bipartite_graph(seed=None):
    """Generate complete bipartite graph K(500,500) with random weights.

    Args:
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=45 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    left_nodes = list(range(1, 501))
    right_nodes = list(range(501, 1001))
    vertices = left_nodes + right_nodes
    edges = []

    # Connect each left node to ~10 random right nodes
    for u in left_nodes:
        targets = rng.sample(right_nodes, k=min(10, len(right_nodes)))
        for v in targets:
            weight = rng.randint(1, 100)
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

BIPARTITE_GRAPH_1K = _create_bipartite_graph(seed=45)  # seed=45 for reproducibility

# Graph 8: Conflict Graph - designed for GA parameter tuning
def _create_conflict_graph(num_hubs: int = 5, nodes_per_hub: int = 20, seed=None):
    """Generate conflict graph where Luby parameter tuning makes a difference.

    Structure:
    - Multiple hub nodes with high connectivity
    - Hub-to-hub edges have HIGH weight (100) → conflicts!
    - Hub-to-peripheral edges have MEDIUM weight (20)
    - Peripheral-to-peripheral edges have LOW weight (5)

    Why this matters for GA:
    - Greedy greedily takes hub edges, blocks other hubs
    - Itai has timeout mechanism, works OK
    - Luby adaptive activation can strategically activate/deactivate hubs
    - Different coefficients → different final matching weights
    - GA can optimize: find coefficients that avoid conflicts

    Args:
        num_hubs: Number of hub nodes
        nodes_per_hub: Peripheral nodes per hub
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=45 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    num_hubs = num_hubs
    nodes_per_hub = nodes_per_hub
    total_nodes = num_hubs + (num_hubs * nodes_per_hub)

    vertices = list(range(1, total_nodes + 1))
    edges = []

    hub_ids = list(range(1, num_hubs + 1))
    peripheral_start = num_hubs + 1
    peripheral_ids = list(range(peripheral_start, total_nodes + 1))

    # Hub-to-hub edges (weight 100 - high weight conflicts)
    for i, hub1 in enumerate(hub_ids):
        for hub2 in hub_ids[i+1:]:
            edges.append((hub1, hub2, 100))

    # Hub-to-peripheral edges (weight 20)
    for hub_id in hub_ids:
        for _ in range(nodes_per_hub):
            peripheral = peripheral_ids.pop(0)
            edges.append((hub_id, peripheral, 20))

    # Peripheral edges (weight 5)
    for i in range(len(peripheral_ids)):
        for j in range(i+1, min(i+3, len(peripheral_ids))):
            edges.append((peripheral_ids[i], peripheral_ids[j], 5))

    # Expected optimal: num_hubs hub matches (at 100 each) + extras
    optimal_weight = num_hubs * 100 + num_hubs * (nodes_per_hub // 2) * 5

    return {
        "name": f"Conflict Graph ({num_hubs} hubs, {nodes_per_hub} nodes/hub)",
        "vertices": vertices,
        "edges": list(set(edges)),  # Remove duplicates
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

CONFLICT_GRAPH_100 = _create_conflict_graph(num_hubs=3, nodes_per_hub=30, seed=45)  # 3 hubs, 90 periph
CONFLICT_GRAPH_500 = _create_conflict_graph(num_hubs=5, nodes_per_hub=100, seed=45)  # 5 hubs, 500 periph
CONFLICT_GRAPH_1K = _create_conflict_graph(num_hubs=10, nodes_per_hub=100, seed=45)  # 10 hubs, 1000 periph

# Graph 9: Greedy-Trap Graph - forces algorithms to navigate conflicting choices
def _create_greedy_trap_graph(size: int = 100, seed=None):
    """Generate graph where greedy fails but adaptive strategies succeed.

    Structure:
    - Start: obvious high-weight edges (greedy takes them)
    - Middle: medium-weight edges that block better matches
    - End: hidden high-weight matches only accessible without greedy's first choices

    Example: Two complete subgraphs connected by low edges.
    If greedy matches within subgraph 1, it can't match within subgraph 2 well.
    But balanced activation (Luby adaptive) can distribute matches better.

    Args:
        size: Number of vertices
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=46 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, size + 1))
    edges = []

    # Split into 2 dense subgraphs
    mid = size // 2
    group1 = list(range(1, mid + 1))
    group2 = list(range(mid + 1, size + 1))

    # Intra-group edges (weight 100 - obvious matches)
    for group in [group1, group2]:
        for i in range(len(group)):
            for j in range(i+1, min(i+8, len(group))):
                edges.append((group[i], group[j], 100))

    # Inter-group edges (weight 1 - low priority)
    for u in group1[:5]:
        for v in group2[:5]:
            edges.append((u, v, 1))

    optimal_weight = (mid // 2) * 100 + (len(group2) // 2) * 100

    return {
        "name": f"Greedy-Trap Graph ({size} nodes)",
        "vertices": vertices,
        "edges": list(set(edges)),
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

GREEDY_TRAP_100 = _create_greedy_trap_graph(100, seed=46)
GREEDY_TRAP_500 = _create_greedy_trap_graph(500, seed=46)

# Graph 10: Adversarial Graph - designed specifically to enable GA optimization
def _create_adversarial_graph(size: int = 100, seed=None):
    """Generate graph where different Luby activation strategies produce different final matchings.

    Key insight: Create a graph where:
    1. High-degree hub nodes have MEDIUM weight edges (tempts Greedy)
    2. Low-degree peripheral nodes have HIGH weight edges (hidden gems)
    3. Hub-to-hub edges create conflicts that Greedy/Itai can't resolve optimally
    4. Different Luby coefficients → different hub activation → different final weights

    Structure:
    - N/5 hub nodes (high degree, medium weight edges)
    - 4N/5 peripheral nodes (low degree, high weight edges)
    - Hubs connected to each other (weight 50 - medium, creates conflicts)
    - Hubs connected to peripherals (weight 30 - lower, but high-degree)
    - Peripherals connected locally (weight 100 - high, but hard to reach)

    Why this matters:
    - Greedy: Sees hubs first (degree is obvious), matches within hubs (weight 50)
    - Itai: Similar to Greedy, gets stuck in hub matching
    - Luby with low coeff_degree: Avoids hubs, finds peripheral matches (weight 100) ✅ Better!
    - Luby with high coeff_degree: Activates hubs, matches hubs (weight 50) ❌ Worse!
    - GA learns: Low coeff_degree is better → optimizes

    Args:
        size: Number of vertices
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=47 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, size + 1))
    edges = []

    # Split into hubs and peripherals
    num_hubs = max(2, size // 5)  # ~20% hubs
    num_peripherals = size - num_hubs

    hub_ids = list(range(1, num_hubs + 1))
    peripheral_ids = list(range(num_hubs + 1, size + 1))

    # Hub-to-hub edges (weight 50 - medium, creates conflicts)
    for i, hub1 in enumerate(hub_ids):
        for hub2 in hub_ids[i+1:]:
            edges.append((hub1, hub2, 50))

    # Hub-to-peripheral edges (weight 30 - low attractiveness)
    for hub_id in hub_ids:
        # Each hub connects to ~half the peripherals
        targets = rng.sample(peripheral_ids, k=min(num_peripherals // 2, len(peripheral_ids)))
        for peripheral_id in targets:
            edges.append((hub_id, peripheral_id, 30))

    # Peripheral-to-peripheral edges (weight 100 - high, but isolated from hubs)
    # Create local clusters of peripherals with high-weight edges
    for i in range(0, len(peripheral_ids) - 1, 5):
        # Local cluster: connect nearby peripherals with high weight
        cluster = peripheral_ids[i:min(i+5, len(peripheral_ids))]
        for j in range(len(cluster)):
            for k in range(j+1, len(cluster)):
                edges.append((cluster[j], cluster[k], 100))

    # Expected: GA finds that avoiding hubs → accessing peripherals → weight ~(num_peripherals/2 * 100)
    # vs Greedy: matches hubs → weight ~(num_hubs/2 * 50) + low peripheral matches
    expected_peripheral_matches = num_peripherals // 10  # Conservative estimate
    optimal_weight = expected_peripheral_matches * 100 + num_hubs // 2 * 50

    return {
        "name": f"Adversarial Graph ({size} nodes, {num_hubs} hubs)",
        "vertices": vertices,
        "edges": list(set(edges)),
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

ADVERSARIAL_100 = _create_adversarial_graph(100, seed=47)
ADVERSARIAL_200 = _create_adversarial_graph(200, seed=47)

# Graph 11: Extreme Matching Disparity - hubs vs isolated pairs
def _create_extreme_disparity_graph(size: int = 100, seed=None):
    """Generate graph to force GA to learn parameter tuning.

    Design philosophy: Make two completely different strategies yield different results.

    Structure:
    - Central hub cluster: 5 hubs all connected to each other (weight 10 - low!)
    - Peripheral isolated pairs: N-5 nodes form 47 isolated high-weight pairs (weight 1000!)
    - NO connections between hubs and peripherals
    - NO overlap between hub and peripheral matching

    Strategy A (Greedy/Itai favor hubs): Match hubs at weight 10 each
    - Greedy matches hub edges: 2-3 pairs × 10 = 20-30 weight

    Strategy B (Luby avoids hubs, uses peripherals): Find isolated pairs at weight 1000 each
    - Luby avoids low-weight hubs, never reaches them
    - Finds peripheral pairs: 47 pairs × 1000 = 47000 weight

    Why Luby can succeed: Each peripheral pair is ONLY connected to itself (weight 1000)
    - No conflicting edges
    - No greedy temptation
    - Pure connectivity matters
    - Different activation patterns → different discovery of peripheral clusters

    Args:
        size: Number of vertices
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=48 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, size + 1))
    edges = []

    # 5 central hub nodes
    hub_ids = [1, 2, 3, 4, 5]

    # Remaining are peripheral
    peripheral_ids = list(range(6, size + 1))

    # Hub-to-hub edges (weight 10 - low temptation for greedy)
    for i, hub1 in enumerate(hub_ids):
        for hub2 in hub_ids[i+1:]:
            edges.append((hub1, hub2, 10))

    # Create isolated pairs from remaining nodes
    # Each pair only connected to itself
    for i in range(0, len(peripheral_ids) - 1, 2):
        u, v = peripheral_ids[i], peripheral_ids[i + 1]
        edges.append((u, v, 1000))  # Massive weight!

    return {
        "name": f"Extreme Disparity Graph ({size} nodes)",
        "vertices": vertices,
        "edges": list(set(edges)),
        "optimal_weight": (len(peripheral_ids) // 2) * 1000 + 15,  # All peripheral pairs + maybe 1-2 hub pairs
        "best_matches": [],
    }

EXTREME_DISPARITY_100 = _create_extreme_disparity_graph(100, seed=48)

# Graph 12: Competing Matchings - forces choice between competing strategies
def _create_competing_matchings_graph(size: int = 100, seed=None):
    """Generate graph with two competing complete matchings of different weights.

    Key insight: Create a scenario where different algorithms get stuck in different local maxima.

    Structure:
    - Two complete matchings: M1 (total weight 5000) and M2 (total weight 4000)
    - M1 edges are high-degree nodes (easy for Greedy to find)
    - M2 edges are low-degree nodes (hard for Greedy, better for probabilistic Luby)
    - Connections between M1 and M2 nodes (weight 1 - distractions)
    - M1 edges: weight 100 each
    - M2 edges: weight 60 each
    - M1-M2 cross edges: weight 1 each

    Expected behavior:
    - Greedy/Itai: Match M1 (obvious high-degree nodes), get stuck at 5000
    - Luby with right params: Avoid M1 hubs, explore M2, get 4000 (then M1 leftovers?)
    - GA learns: Sometimes avoiding greed is better!

    This forces a REAL tradeoff: explore M1 (high reward, medium exploration) vs M2 (lower reward, better exploration)

    Args:
        size: Number of vertices
        seed: Random seed for reproducibility. If None, graph is randomized.
              Pass seed=49 to get the original deterministic graph.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed()  # Randomize with current time/entropy

    vertices = list(range(1, size + 1))
    edges = []

    # Split vertices into two groups: M1 nodes and M2 nodes
    m1_size = size // 2
    m2_size = size - m1_size

    m1_nodes = list(range(1, m1_size + 1))
    m2_nodes = list(range(m1_size + 1, size + 1))

    # M1: Complete matching among first half (high weight, attracts Greedy)
    for i in range(0, len(m1_nodes) - 1, 2):
        u, v = m1_nodes[i], m1_nodes[i + 1]
        edges.append((u, v, 100))

    # M2: Complete matching among second half (lower weight, but real alternative)
    for i in range(0, len(m2_nodes) - 1, 2):
        u, v = m2_nodes[i], m2_nodes[i + 1]
        edges.append((u, v, 60))

    # Cross edges (weight 1 - temptations that don't help)
    for u in m1_nodes[:5]:
        for v in m2_nodes[:5]:
            edges.append((u, v, 1))

    # M1 complete graph (within group)
    for i in range(len(m1_nodes)):
        for j in range(i+1, min(i+4, len(m1_nodes))):
            if (m1_nodes[i], m1_nodes[j]) not in edges and (m1_nodes[j], m1_nodes[i]) not in edges:
                edges.append((m1_nodes[i], m1_nodes[j], 100))

    # M2 internal edges (within group)
    for i in range(len(m2_nodes)):
        for j in range(i+1, min(i+4, len(m2_nodes))):
            if (m2_nodes[i], m2_nodes[j]) not in edges and (m2_nodes[j], m2_nodes[i]) not in edges:
                edges.append((m2_nodes[i], m2_nodes[j], 60))

    optimal_weight = (m1_size // 2) * 100 + (m2_size // 2) * 60

    return {
        "name": f"Competing Matchings Graph ({size} nodes)",
        "vertices": vertices,
        "edges": list(set(edges)),
        "optimal_weight": optimal_weight,
        "best_matches": [],
    }

COMPETING_100 = _create_competing_matchings_graph(100, seed=49)

# Easy access
GRAPHS = [_create_clustered_graph, GRID_4x4, K5_CLUSTERS, STAR_WITH_TAIL, RANDOM_DENSE_GRAPH_1K, CLUSTERED_GRAPH_1K, SCALE_FREE_GRAPH_1K, BIPARTITE_GRAPH_1K, CONFLICT_GRAPH_1K, GREEDY_TRAP_100, ADVERSARIAL_100, EXTREME_DISPARITY_100, COMPETING_100]


def get_graph(name):
    """Get graph by name."""
    for g in GRAPHS:
        if g["name"].lower() == name.lower():
            return g
    raise ValueError(f"Graph '{name}' not found. Available: {[g['name'] for g in GRAPHS]}")


def get_graph_seed_info():
    """Return information about default seeds for each graph type.

    Returns:
        dict: Mapping of graph creation function to its default seed
    """
    return {
        "random_dense_graph": 42,
        "clustered_graph": 43,
        "scale_free_graph": 44,
        "bipartite_graph": 45,
        "conflict_graph": 45,
        "greedy_trap_graph": 46,
        "adversarial_graph": 47,
        "extreme_disparity_graph": 48,
        "competing_matchings_graph": 49,
    }
