import pytest
from src.config import ExperimentConfig
from src.state.store import StateStore

# Import all graph fixtures
from tests.fixtures.conftest_graphs import (
    simple_path_graph,
    simple_grid_graph,
    simple_complete_graph,
    simple_star_graph,
    simple_bipartite_graph,
    medium_path_graph,
    medium_grid_graph,
    medium_clustered_graph,
    medium_random_graph,
    large_clustered_graph,
    large_scale_free_graph,
    large_random_graph,
    parametrized_path_graph,
    parametrized_grid_graph,
    parametrized_complete_graph,
    parametrized_bipartite_graph,
    parametrized_clustered_graph,
    test_case_simple,
    test_case_conflicting,
    test_case_sparse,
    simple_graph,
    medium_graph,
)


@pytest.fixture
def state_store_simple(simple_graph):
    """Create a state store for the simple graph."""
    state_store = StateStore(simple_graph)
    return state_store


@pytest.fixture
def simulation_config():
    """Create a simulation config for testing."""
    return ExperimentConfig(max_rounds=100, collect_snapshots=True)
