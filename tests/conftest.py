import pytest
from src.config import ExperimentConfig

pytest_plugins = ["tests.fixtures.conftest_graphs"]


@pytest.fixture
def simulation_config():
    """Create a simulation config for testing."""
    return ExperimentConfig(max_rounds=100, collect_snapshots=True)
