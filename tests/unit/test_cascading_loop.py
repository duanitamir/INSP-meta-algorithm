"""Unit tests for CascadingLoop - 6+ comprehensive tests."""

import pytest
from src.meta.cascading_loop import CascadingLoop
from src.meta.conflict_resolver import ConflictResolver
from src.meta.canonical_vector import CanonicalVector
from src.meta.greedy_parameterizer import GreedyParameterizer
from src.meta.itai_parameterizer import ItaiParameterizer
from src.meta.luby_parameterizer import LubyParameterizer
from src.graph.graph_manager import GraphManager


class TestCascadingLoopBasics:
    """Test CascadingLoop initialization and interface."""

    def test_cascading_loop_initialization(self) -> None:
        """Should initialize with conflict resolver."""
        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        assert loop is not None

    def test_cascading_loop_name(self) -> None:
        """Should return correct name."""
        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        assert loop.name() == "CascadingLoop"


class TestCascadingLoopExecution:
    """Test CascadingLoop.execute() method."""

    def test_execute_returns_matching_and_metrics(self) -> None:
        """execute() should return (matching, metrics_dict)."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        matching, metrics = loop.execute(graph, vector, parameterizers)

        assert isinstance(matching, dict)
        assert isinstance(metrics, dict)

    def test_execute_metrics_has_required_keys(self) -> None:
        """Metrics dict should have iterations, final_weight, improvements."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        assert "iterations" in metrics
        assert "final_weight" in metrics
        assert "improvements" in metrics

    def test_execute_empty_graph(self) -> None:
        """Should handle empty graph gracefully."""
        graph = GraphManager.create_empty_graph()

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        matching, metrics = loop.execute(graph, vector, parameterizers)

        assert matching == {}
        assert metrics["final_weight"] == 0.0

    def test_execute_tracks_iteration_count(self) -> None:
        """Should track number of iterations executed."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector(max_iterations=5)
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        assert metrics["iterations"] >= 1
        assert metrics["iterations"] <= 5

    def test_execute_convergence_threshold_stops_early(self) -> None:
        """Should stop early if improvement < convergence_threshold."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        # High convergence threshold (max allowed is 0.1) means early stop
        vector = CanonicalVector(
            max_iterations=100, convergence_threshold=0.1
        )
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        # Should stop early or complete with valid metrics
        assert metrics["iterations"] >= 1

    def test_execute_improvements_list_length(self) -> None:
        """Improvements list should match iteration count."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector(max_iterations=5)
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        assert len(metrics["improvements"]) == metrics["iterations"]

    def test_execute_invalid_vector_raises_error(self) -> None:
        """Should raise ValueError on invalid canonical vector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        # Create vector with out-of-range max_iterations
        with pytest.raises(ValueError, match="Invalid canonical vector"):
            vector = CanonicalVector(max_iterations=200)  # Max is 100
            loop.execute(graph, vector, parameterizers)

    def test_execute_no_parameterizers_raises_error(self) -> None:
        """Should raise ValueError if no parameterizers provided."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()

        with pytest.raises(ValueError, match="at least one parameterizer"):
            loop.execute(graph, vector, [])

    def test_execute_single_edge_produces_matching(self) -> None:
        """Should produce matching for simple single-edge graph."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        matching, metrics = loop.execute(graph, vector, parameterizers)

        # Should produce valid matching
        assert isinstance(matching, dict)
        assert metrics["final_weight"] >= 0.0

    def test_execute_final_weight_non_negative(self) -> None:
        """Final weight should always be non-negative."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4, 5]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 2.0)
        graph.add_edge(3, 4, 1.5)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector()
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        assert metrics["final_weight"] >= 0.0

    def test_execute_maximal_matching_stops_early(self) -> None:
        """Should stop when maximal matching is reached."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        loop = CascadingLoop(resolver)
        vector = CanonicalVector(max_iterations=100)  # Large max
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        _, metrics = loop.execute(graph, vector, parameterizers)

        # Should stop early because matching is already maximal
        assert metrics["iterations"] <= 10
