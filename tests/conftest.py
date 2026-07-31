import pytest
from src.config import ExperimentConfig
from src.state.store import StateStore

pytest_plugins = ["tests.fixtures.conftest_graphs"]


@pytest.fixture
def state_store_simple(simple_graph):
    """Create a state store for the simple graph."""
    state_store = StateStore(simple_graph)
    return state_store


@pytest.fixture
def simulation_config():
    """Create a simulation config for testing."""
    return ExperimentConfig(max_rounds=100, collect_snapshots=True)
