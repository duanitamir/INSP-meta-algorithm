"""Pytest fixtures for graph fixtures with optimal matching computation.

Import these into tests/conftest.py or use directly in tests.
Each fixture returns a GraphFixtureBase subclass with:
  - .graph: The actual GraphManager instance
  - .get_optimal_weight(): Optimal matching weight
  - .get_optimal_matching(): Optimal matching dict
  - .quality_ratio(matching): How close to optimal
  - .summary(): Human-readable summary
"""

import pytest
from tests.fixtures.graph_fixture_base import (
    PathGraphFixture,
    GridGraphFixture,
    CompleteGraphFixture,
    StarGraphFixture,
    BipartiteGraphFixture,
    CustomGraphFixture,
    ClusteredGraphFixture,
    ScaleFreeGraphFixture,
    RandomGraphFixture,
)


# ============================================================================
# SIMPLE FIXTURES (Small graphs for unit tests)
# ============================================================================


@pytest.fixture
def simple_path_graph():
    """Simple 4-vertex path: 1-2-3-4."""
    return PathGraphFixture().build(num_vertices=4, edge_weight=1.0)


@pytest.fixture
def simple_grid_graph():
    """Simple 3x3 grid graph."""
    return GridGraphFixture().build(rows=3, cols=3, edge_weight=1.0)


@pytest.fixture
def simple_complete_graph():
    """Simple complete graph K_5."""
    return CompleteGraphFixture().build(num_vertices=5, edge_weight=1.0)


@pytest.fixture
def simple_star_graph():
    """Simple star graph: hub + 5 leaves."""
    return StarGraphFixture().build(num_leaves=5, hub_weight=1.0)


@pytest.fixture
def simple_bipartite_graph():
    """Simple bipartite graph: 3x3."""
    return BipartiteGraphFixture().build(left_size=3, right_size=3, edge_weight=1.0)


# ============================================================================
# MEDIUM FIXTURES (Moderate size for integration tests)
# ============================================================================


@pytest.fixture
def medium_path_graph():
    """Medium path: 20 vertices."""
    return PathGraphFixture().build(num_vertices=20, edge_weight=1.0)


@pytest.fixture
def medium_grid_graph():
    """Medium 6x6 grid."""
    return GridGraphFixture().build(rows=6, cols=6, edge_weight=1.0)


@pytest.fixture
def medium_clustered_graph():
    """Medium clustered graph: 3 clusters of 10 vertices."""
    return ClusteredGraphFixture().build(
        num_clusters=3, cluster_size=10, intra_weight=10.0, inter_weight=1.0
    )


@pytest.fixture
def medium_random_graph():
    """Medium random graph: 30 vertices, 30% edge probability."""
    return RandomGraphFixture().build(num_vertices=30, edge_probability=0.3)


# ============================================================================
# LARGE FIXTURES (For GA and performance testing)
# ============================================================================


@pytest.fixture
def large_clustered_graph():
    """Large clustered graph: 5 clusters of 20 vertices."""
    return ClusteredGraphFixture().build(
        num_clusters=5, cluster_size=20, intra_weight=10.0, inter_weight=1.0
    )


@pytest.fixture
def large_scale_free_graph():
    """Large scale-free graph: 100 vertices."""
    return ScaleFreeGraphFixture().build(num_vertices=100, attachment_edges=3)


@pytest.fixture
def large_random_graph():
    """Large random graph: 100 vertices, 20% edge probability."""
    return RandomGraphFixture().build(num_vertices=100, edge_probability=0.2)


# ============================================================================
# PARAMETRIZED FIXTURES (For flexible testing)
# ============================================================================


@pytest.fixture(params=[4, 8, 16])
def parametrized_path_graph(request):
    """Path graph with parametrized sizes."""
    return PathGraphFixture().build(num_vertices=request.param, edge_weight=1.0)


@pytest.fixture(params=[(3, 3), (4, 4), (5, 5)])
def parametrized_grid_graph(request):
    """Grid graph with parametrized dimensions."""
    rows, cols = request.param
    return GridGraphFixture().build(rows=rows, cols=cols, edge_weight=1.0)


@pytest.fixture(params=[5, 10, 20])
def parametrized_complete_graph(request):
    """Complete graph with parametrized sizes."""
    return CompleteGraphFixture().build(num_vertices=request.param, edge_weight=1.0)


@pytest.fixture(params=[(2, 3), (3, 5), (5, 5)])
def parametrized_bipartite_graph(request):
    """Bipartite graph with parametrized sizes."""
    left, right = request.param
    return BipartiteGraphFixture().build(left_size=left, right_size=right, edge_weight=1.0)


@pytest.fixture(params=[(2, 5), (3, 10), (5, 20)])
def parametrized_clustered_graph(request):
    """Clustered graph with parametrized cluster count and size."""
    num_clusters, cluster_size = request.param
    return ClusteredGraphFixture().build(
        num_clusters=num_clusters,
        cluster_size=cluster_size,
        intra_weight=10.0,
        inter_weight=1.0,
    )


# ============================================================================
# CUSTOM FIXTURES (Pre-defined test cases)
# ============================================================================


@pytest.fixture
def test_case_simple():
    """Test case from algorithm paper: simple 4-node example.

    Optimal matching should be {(1,2), (3,4)} with weight 2.0
    """
    return CustomGraphFixture().build(
        vertices=[1, 2, 3, 4],
        edges=[(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
    )


@pytest.fixture
def test_case_conflicting():
    """Test case with algorithm conflicts.

    Different algorithms should prefer different matchings.
    Greedy: {(1,2), (3,4)} weight=15
    Optimal: {(1,4), (2,3)} weight=21
    """
    return CustomGraphFixture().build(
        vertices=[1, 2, 3, 4],
        edges=[(1, 2, 5.0), (1, 4, 10.0), (2, 3, 11.0), (3, 4, 1.0)],
    )


@pytest.fixture
def test_case_sparse():
    """Test case with sparse structure (star-like)."""
    return CustomGraphFixture().build(
        vertices=list(range(1, 11)),
        edges=[
            (1, 2, 5.0),
            (1, 3, 5.0),
            (1, 4, 5.0),
            (5, 6, 8.0),
            (5, 7, 8.0),
            (5, 8, 8.0),
            (9, 10, 3.0),
        ],
    )


# ============================================================================
# BACKWARD COMPATIBILITY FIXTURES (Legacy names)
# ============================================================================


@pytest.fixture
def simple_graph():
    """Backward compatible: simple path graph."""
    return simple_path_graph()


@pytest.fixture
def medium_graph():
    """Backward compatible: medium clustered graph."""
    return medium_clustered_graph()
