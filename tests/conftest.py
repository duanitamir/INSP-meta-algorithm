import pytest
from src.graph import GraphManager
from src.simulation import SimulationConfig
from src.state.state_store import StateStore


@pytest.fixture
def simple_graph():
    """Create a simple 4-vertex path graph: 1-2-3-4."""
    graph = GraphManager.create_empty_graph()
    for i in range(1, 5):
        graph.add_vertex(i)
    edges = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    return graph


@pytest.fixture
def medium_graph():
    """Create a medium-sized test graph."""
    graph = GraphManager.create_empty_graph()
    vertices = list(range(1, 21))
    for v in vertices:
        graph.add_vertex(v)

    edges = [
        (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0),
        (1, 5, 2.0), (2, 6, 1.5), (3, 7, 1.5), (4, 8, 1.5),
        (6, 7, 1.0), (7, 8, 1.0), (8, 9, 1.0), (9, 10, 1.0),
        (5, 10, 2.0), (11, 12, 1.0), (12, 13, 1.0), (13, 14, 1.0),
        (14, 15, 1.0), (15, 16, 1.0), (16, 17, 1.0), (17, 18, 1.0),
    ]
    for u, v, w in edges:
        graph.add_edge(u, v, w)

    return graph


@pytest.fixture
def state_store_simple(simple_graph):
    """Create a state store for the simple graph."""
    state_store = StateStore(simple_graph)
    return state_store


@pytest.fixture
def simulation_config():
    """Create a simulation config for testing."""
    return SimulationConfig(max_rounds=100, collect_snapshots=True)
