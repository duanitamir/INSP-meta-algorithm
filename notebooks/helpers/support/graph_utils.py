"""Graph loading and transformation utilities."""

from src.graph.graph_manager import GraphManager


def fixture_to_graph(fixture_dict) -> GraphManager:
    """Convert fixture dictionary to GraphManager.

    Args:
        fixture_dict: Dictionary with 'vertices' and 'edges' lists

    Returns:
        GraphManager instance
    """
    graph = GraphManager.create_empty_graph()
    for v in fixture_dict['vertices']:
        graph.add_vertex(v)
    for u, v, w in fixture_dict['edges']:
        graph.add_edge(u, v, float(w))
    return graph


def format_time(seconds: float) -> str:
    """Format time in human-readable form.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted time string (e.g., "5.2s" or "1.3m")
    """
    if seconds < 60:
        return f'{seconds:.1f}s'
    else:
        return f'{seconds/60:.1f}m'
