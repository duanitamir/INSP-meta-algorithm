"""Unit tests for ConflictResolver - 6+ comprehensive tests."""

import pytest
from src.meta.conflict_resolver import ConflictResolver
from src.graph.graph_manager import GraphManager


class TestConflictResolverBasics:
    """Test ConflictResolver initialization and interface."""

    def test_conflict_resolver_initialization(self) -> None:
        """Should initialize without errors."""
        resolver = ConflictResolver()
        assert resolver is not None

    def test_conflict_resolver_name(self) -> None:
        """Should return correct name."""
        resolver = ConflictResolver()
        assert resolver.name() == "ConflictResolver"


class TestConflictResolverResolution:
    """Test ConflictResolver.resolve() method."""

    def test_resolve_identical_matchings(self) -> None:
        """Should handle case where all 3 matchings are identical."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        matching = {1: 2, 2: 1, 3: 4, 4: 3}

        resolver = ConflictResolver()
        result = resolver.resolve(matching, matching, matching, graph)

        assert isinstance(result, dict)
        assert result == matching

    def test_resolve_with_conflicts_highest_weight_wins(self) -> None:
        """Should keep highest-weight edge when conflicts occur."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 5.0)  # High weight
        graph.add_edge(1, 3, 2.0)  # Low weight

        greedy = {1: 2, 2: 1}
        itai = {1: 3, 3: 1}  # Conflict: 1 matched to different partners
        luby = {1: 2, 2: 1}

        resolver = ConflictResolver()
        result = resolver.resolve(greedy, itai, luby, graph)

        # Should choose (1,2) because it has higher weight (5.0 vs 2.0)
        assert result.get(1) == 2
        assert result.get(2) == 1
        assert 3 not in result

    def test_resolve_enforces_symmetry(self) -> None:
        """Should enforce symmetry: if u->v then v->u must exist."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        greedy = {1: 2, 2: 1}
        itai = {3: 4, 4: 3}
        luby = {}

        resolver = ConflictResolver()
        result = resolver.resolve(greedy, itai, luby, graph)

        # Check symmetry
        for u, v in result.items():
            assert v in result, f"Asymmetry: {u} -> {v} but {v} not in result"
            assert result[v] == u, f"Non-symmetric: {u} -> {v} but {v} -> {result[v]}"

    def test_resolve_empty_matchings(self) -> None:
        """Should handle empty matchings gracefully."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        resolver = ConflictResolver()
        result = resolver.resolve({}, {}, {}, graph)

        assert result == {}

    def test_resolve_single_edge(self) -> None:
        """Should handle case with single edge."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        matching = {1: 2, 2: 1}

        resolver = ConflictResolver()
        result = resolver.resolve(matching, {}, {}, graph)

        assert result == {1: 2, 2: 1}

    def test_resolve_none_graph(self) -> None:
        """Should handle None graph gracefully."""
        resolver = ConflictResolver()
        result = resolver.resolve({}, {}, {}, None)

        assert result == {}

    def test_resolve_empty_graph(self) -> None:
        """Should handle empty graph gracefully."""
        graph = GraphManager.create_empty_graph()

        resolver = ConflictResolver()
        result = resolver.resolve({}, {}, {}, graph)

        assert result == {}

    def test_resolve_partial_overlap(self) -> None:
        """Should handle matchings with partial overlap."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4, 5, 6]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)
        graph.add_edge(5, 6, 1.0)

        greedy = {1: 2, 2: 1, 3: 4, 4: 3}
        itai = {5: 6, 6: 5}
        luby = {}

        resolver = ConflictResolver()
        result = resolver.resolve(greedy, itai, luby, graph)

        # All edges should be present with symmetry
        assert result.get(1) == 2 and result.get(2) == 1
        assert result.get(3) == 4 and result.get(4) == 3
        assert result.get(5) == 6 and result.get(6) == 5

    def test_resolve_multiple_conflicts_greedy(self) -> None:
        """Should handle multiple conflicts using greedy highest weight."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(1, 3, 5.0)
        graph.add_edge(2, 4, 3.0)
        graph.add_edge(3, 4, 2.0)

        greedy = {1: 2, 2: 1}
        itai = {1: 3, 3: 1}
        luby = {2: 4, 4: 2}

        resolver = ConflictResolver()
        result = resolver.resolve(greedy, itai, luby, graph)

        # (1,2) highest weight (10.0) should be chosen
        # (2,4) conflicts with (1,2) because 2 already matched
        # (3,4) conflicts with (1,3) because 3 already matched
        assert result.get(1) == 2 and result.get(2) == 1
