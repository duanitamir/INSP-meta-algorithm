"""Test graph fixtures with optimal matching computation.

This module provides a comprehensive system for creating test graphs with
automatic optimal matching computation via NetworkX.

Usage:
    from tests.fixtures.graph_fixture_base import PathGraphFixture

    # Create a path graph with 10 vertices
    graph_fixture = PathGraphFixture().build(num_vertices=10, edge_weight=1.5)

    # Access the graph
    graph = graph_fixture.graph

    # Get optimal matching information
    optimal_weight = graph_fixture.get_optimal_weight()
    optimal_matching = graph_fixture.get_optimal_matching()

    # Evaluate quality of your matching
    your_matching = {"your": "matching_dict"}
    quality = graph_fixture.quality_ratio(your_matching)
    gap = graph_fixture.get_gap_percent(your_matching)

    # Get summary
    print(graph_fixture.summary())

Available Fixtures:
    - PathGraphFixture: Linear path 1-2-3-...
    - GridGraphFixture: 2D rectangular lattice
    - CompleteGraphFixture: Every vertex connected to every other
    - StarGraphFixture: Hub with leaves
    - BipartiteGraphFixture: Two groups with cross-group edges only
    - CustomGraphFixture: Build from vertices/edges lists
    - ClusteredGraphFixture: Multiple dense clusters
    - ScaleFreeGraphFixture: Power-law degree distribution
    - RandomGraphFixture: Erdős-Rényi random graph

Pytest Integration:
    Import from tests/conftest.py for pytest fixtures with all graph types.
"""

from .graph_fixture_base import (
    GraphFixtureBase,
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

__all__ = [
    "GraphFixtureBase",
    "PathGraphFixture",
    "GridGraphFixture",
    "CompleteGraphFixture",
    "StarGraphFixture",
    "BipartiteGraphFixture",
    "CustomGraphFixture",
    "ClusteredGraphFixture",
    "ScaleFreeGraphFixture",
    "RandomGraphFixture",
]
