"""Shared matching merge logic for fitness evaluators."""

from src.graph.graph_manager import GraphManager


def merge_matchings(matchings: list, graph: GraphManager) -> dict:
    """Merge multiple matchings via conflict resolution.

    Prioritizes edges agreed upon by multiple algorithms, breaks ties by weight.
    Ensures symmetric matching.

    Args:
        matchings: List of matching dicts
        graph: GraphManager for edge weights

    Returns:
        Merged matching dict
    """
    if not matchings:
        return {}

    # OPTIMIZATION: Phase 2A - Skip merge overhead if only 1-2 algorithms have output
    # Filter out empty matchings first for faster processing
    non_empty_matchings = [m for m in matchings if m]

    # Quick win: if only 1 algorithm has output, skip merge logic entirely
    if len(non_empty_matchings) == 1:
        return non_empty_matchings[0]

    if not non_empty_matchings:
        return {}

    edge_proposals = {}
    for matching in non_empty_matchings:
        for u, v in matching.items():
            if u is None or v is None:
                continue
            edge = tuple(sorted([u, v]))
            if edge not in edge_proposals:
                edge_proposals[edge] = {"weights": [], "count": 0}
            weight = graph.get_edge_weight(u, v)
            edge_proposals[edge]["weights"].append(weight)
            edge_proposals[edge]["count"] += 1

    # Sort by: (proposal_count DESC, max_weight DESC)
    # Prefer edges found by multiple algorithms, break ties by weight
    final_matching = {}
    used_nodes = set()
    for edge in sorted(
        edge_proposals.keys(),
        key=lambda e: (
            edge_proposals[e]["count"],  # Algorithms agreeing (desc)
            max(edge_proposals[e]["weights"])  # Highest weight (desc)
        ),
        reverse=True,
    ):
        u, v = edge
        if u not in used_nodes and v not in used_nodes:
            final_matching[u] = v
            final_matching[v] = u
            used_nodes.add(u)
            used_nodes.add(v)

    return final_matching
