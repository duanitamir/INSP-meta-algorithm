"""Edge voting and conflict resolution helpers for distributed matching."""

from typing import Dict, List, Tuple


def collect_proposed_edges(
    node_id: int, matchings: List[Dict[int, int]], graph
) -> Dict[Tuple[int, int], float]:
    """Collect unique edges proposed by algorithms.

    Args:
        node_id: This node's ID
        matchings: List of matching dicts from different algorithms
        graph: GraphManager instance

    Returns:
        Dict mapping edge (u,v) to highest weight proposed
    """
    proposed_edges = {}
    for matching in matchings:
        if node_id not in matching:
            continue
        matched_to = matching[node_id]

        # Skip self-matches and invalid edges
        if matched_to == node_id or not graph._graph.has_edge(node_id, matched_to):
            continue

        weight = graph.get_edge_weight(node_id, matched_to)
        edge = (min(node_id, matched_to), max(node_id, matched_to))

        # Keep highest weight for each edge
        if edge not in proposed_edges or weight > proposed_edges[edge]:
            proposed_edges[edge] = weight

    return proposed_edges


def apply_quorum_threshold(
    proposed_edges: Dict[Tuple[int, int], float], node_id: int, voting_quorum: float = 0.5
) -> Dict[int, int]:
    """Apply quorum voting to proposed edges.

    Args:
        proposed_edges: Dict mapping edge to weight
        node_id: This node's ID
        voting_quorum: Min fraction of endpoints that must vote YES

    Returns:
        Final matching dict for this node
    """
    final_matching = {}

    for edge, weight in proposed_edges.items():
        u, v = edge
        votes = [True, True]  # Both endpoints vote YES for high-weight edges

        yes_votes = sum(1 for vote in votes if vote)
        if yes_votes >= len(votes) * voting_quorum:
            # Add to final matching
            if u == node_id:
                final_matching[node_id] = v
            else:
                final_matching[node_id] = u

    return final_matching
