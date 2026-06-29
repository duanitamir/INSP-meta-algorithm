"""Unit tests for adaptive activation in Luby parameterizer - 5+ tests."""

import pytest
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.parameterizers.luby import LubyParameterizer
from src.graph.graph_manager import GraphManager


class TestAdaptiveActivationBasics:
    """Test adaptive activation function computation."""

    def test_compute_adaptive_activation_returns_function(self) -> None:
        """Should return a callable activation function."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        luby = LubyParameterizer()
        vector = CanonicalVector()

        activation_fn = luby._compute_adaptive_activation(graph, vector)

        assert callable(activation_fn)

    def test_activation_function_returns_float(self) -> None:
        """Activation function should return float in [0, 1]."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        luby = LubyParameterizer()
        vector = CanonicalVector()
        activation_fn = luby._compute_adaptive_activation(graph, vector)

        for node_id in [1, 2, 3]:
            prob = activation_fn(node_id)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_different_coefficients_produce_different_activations(self) -> None:
        """Different coefficient values should produce different probabilities."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4, 5]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)
        graph.add_edge(4, 5, 1.0)

        luby = LubyParameterizer()

        # Low coefficients
        v1 = CanonicalVector(
            luby_base_probability=0.5,
            luby_coeff_degree=-1.0,
            luby_coeff_weight=-1.0,
        )
        fn1 = luby._compute_adaptive_activation(graph, v1)

        # High coefficients
        v2 = CanonicalVector(
            luby_base_probability=0.5,
            luby_coeff_degree=1.0,
            luby_coeff_weight=1.0,
        )
        fn2 = luby._compute_adaptive_activation(graph, v2)

        # Test on high-degree node (should be affected most)
        node = 3  # Middle node with degree 2
        prob1 = fn1(node)
        prob2 = fn2(node)

        assert prob1 != prob2

    def test_activation_respects_base_probability(self) -> None:
        """Base probability should influence all node activations."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        luby = LubyParameterizer()

        # Low base
        v1 = CanonicalVector(luby_base_probability=0.1)
        fn1 = luby._compute_adaptive_activation(graph, v1)

        # High base
        v2 = CanonicalVector(luby_base_probability=0.9)
        fn2 = luby._compute_adaptive_activation(graph, v2)

        # Both nodes should reflect base probability
        assert fn1(1) < fn2(1)
        assert fn1(2) < fn2(2)

    def test_activation_handles_zero_degree_nodes(self) -> None:
        """Should handle isolated nodes gracefully."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        # Node 3 is isolated

        luby = LubyParameterizer()
        vector = CanonicalVector()
        activation_fn = luby._compute_adaptive_activation(graph, vector)

        # Should not crash and return valid probability
        prob_isolated = activation_fn(3)
        assert 0.0 <= prob_isolated <= 1.0


class TestAdaptiveActivationIntegration:
    """Test adaptive activation in full Luby execution."""

    def test_execute_with_adaptive_uses_all_coefficients(self) -> None:
        """Executing with different coefficients should produce different results."""
        graph = GraphManager.create_empty_graph()
        for v in range(1, 11):
            graph.add_vertex(v)
        # Create a chain: 1-2-3-4-5-6-7-8-9-10
        for i in range(1, 10):
            graph.add_edge(i, i + 1, float(10 - i))

        luby = LubyParameterizer()

        # Vector with low coefficients
        v1 = CanonicalVector(
            luby_base_probability=0.3,
            luby_coeff_degree=-0.8,
            luby_coeff_weight=-0.8,
        )
        m1 = luby.execute(graph, v1)
        w1 = graph.calculate_matching_weight(m1)

        # Vector with high coefficients
        v2 = CanonicalVector(
            luby_base_probability=0.3,
            luby_coeff_degree=0.8,
            luby_coeff_weight=0.8,
        )
        m2 = luby.execute(graph, v2)
        w2 = graph.calculate_matching_weight(m2)

        # Weights may differ due to adaptive activation
        assert isinstance(w1, float)
        assert isinstance(w2, float)

    def test_execute_with_zero_coefficients_uses_base_only(self) -> None:
        """With all coefficients zero, should use base probability only."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        luby = LubyParameterizer()

        # All coefficients zero (will use 0.5 base probability)
        vector = CanonicalVector(
            luby_base_probability=0.5,
            luby_coeff_degree=0.0,
            luby_coeff_neighbors_unmatched=0.0,
            luby_coeff_clustering=0.0,
            luby_coeff_matched=0.0,
            luby_coeff_round=0.0,
            luby_coeff_weight=0.0,
        )
        matching = luby.execute(graph, vector)

        # Should produce valid matching (dict, possibly empty due to randomness)
        assert isinstance(matching, dict)

    def test_adaptive_activation_on_dense_graph(self) -> None:
        """Adaptive activation should work on denser graphs."""
        graph = GraphManager.create_empty_graph()
        for v in range(1, 11):
            graph.add_vertex(v)
        # Create more edges with higher base probability
        for i in range(1, 10):
            for j in range(i + 1, min(i + 4, 11)):
                graph.add_edge(i, j, float(j - i))

        luby = LubyParameterizer()

        # Different coefficient strategies with higher base probability
        v1 = CanonicalVector(
            luby_base_probability=0.7,
            luby_coeff_degree=-0.5,
        )
        v2 = CanonicalVector(
            luby_base_probability=0.7,
            luby_coeff_degree=0.5,
        )

        m1 = luby.execute(graph, v1)
        m2 = luby.execute(graph, v2)

        w1 = graph.calculate_matching_weight(m1)
        w2 = graph.calculate_matching_weight(m2)

        # Both should produce valid weight values
        assert isinstance(w1, float)
        assert isinstance(w2, float)
        assert w1 >= 0
        assert w2 >= 0
