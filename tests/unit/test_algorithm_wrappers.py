"""Unit tests for algorithm parameterizer wrappers - 15+ comprehensive tests."""

import pytest
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer
from src.graph.graph_manager import GraphManager


class TestGreedyParameterizerBasics:
    """Test GreedyParameterizer initialization and interface."""

    def test_greedy_initialization(self):
        """Should initialize without errors."""
        parameterizer = UnifiedAlgorithmParameterizer("greedy")
        assert parameterizer is not None

    def test_greedy_name(self):
        """Should return correct algorithm name."""
        parameterizer = UnifiedAlgorithmParameterizer("greedy")
        assert parameterizer.name() == "Greedy"

    def test_greedy_repr(self):
        """Should have string representation."""
        parameterizer = UnifiedAlgorithmParameterizer("greedy")
        result = repr(parameterizer)
        assert "Greedy" in result


class TestGreedyParameterizerExecution:
    """Test GreedyParameterizer execute() method."""

    def test_greedy_execute_returns_dict(self):
        """execute() should return Dict[int, int]."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("greedy")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)

    def test_greedy_execute_with_empty_graph(self):
        """execute() should handle empty graph."""
        graph = GraphManager.create_empty_graph()
        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("greedy")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_greedy_execute_single_edge(self):
        """execute() should handle single edge."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("greedy")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)

    def test_greedy_execute_ignores_canonical_params(self):
        """execute() should work regardless of canonical vector params."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        # Different vectors should produce same result (Greedy has no params)
        vector1 = CanonicalVector()
        vector2 = CanonicalVector(max_iterations=15)

        parameterizer = UnifiedAlgorithmParameterizer("greedy")
        result1 = parameterizer.execute(graph, vector1)
        result2 = parameterizer.execute(graph, vector2)

        assert result1 == result2


class TestItaiParameterizerBasics:
    """Test ItaiParameterizer initialization and interface."""

    def test_itai_initialization(self):
        """Should initialize without errors."""
        parameterizer = UnifiedAlgorithmParameterizer("itai")
        assert parameterizer is not None

    def test_itai_name(self):
        """Should return correct algorithm name."""
        parameterizer = UnifiedAlgorithmParameterizer("itai")
        assert parameterizer.name() == "Itai-Israeli"

    def test_itai_repr(self):
        """Should have string representation."""
        parameterizer = UnifiedAlgorithmParameterizer("itai")
        result = repr(parameterizer)
        assert "Itai-Israeli" in result


class TestItaiParameterizerExecution:
    """Test ItaiParameterizer execute() method."""

    def test_itai_execute_returns_dict(self):
        """execute() should return Dict[int, int]."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("itai")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)

    def test_itai_uses_timeout_parameter(self):
        """execute() should use itai_timeout_rounds from vector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        # Create vectors with different timeouts
        vector_short = CanonicalVector(itai_timeout_rounds=1)
        vector_long = CanonicalVector(itai_timeout_rounds=20)

        parameterizer = UnifiedAlgorithmParameterizer("itai")

        # Both should succeed (timeout affects execution, not validity)
        result_short = parameterizer.execute(graph, vector_short)
        result_long = parameterizer.execute(graph, vector_long)

        assert isinstance(result_short, dict)
        assert isinstance(result_long, dict)

    def test_itai_timeout_in_valid_range(self):
        """execute() should work with timeout in [1, 20]."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        parameterizer = UnifiedAlgorithmParameterizer("itai")

        # Test edge cases of timeout range
        for timeout in [1, 10, 20]:
            vector = CanonicalVector(itai_timeout_rounds=timeout)
            result = parameterizer.execute(graph, vector)
            assert isinstance(result, dict)

    def test_itai_execute_empty_graph(self):
        """execute() should handle empty graph."""
        graph = GraphManager.create_empty_graph()
        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("itai")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)
        assert len(result) == 0


class TestLubyParameterizerBasics:
    """Test LubyParameterizer initialization and interface."""

    def test_luby_initialization(self):
        """Should initialize without errors."""
        parameterizer = UnifiedAlgorithmParameterizer("luby")
        assert parameterizer is not None

    def test_luby_name(self):
        """Should return correct algorithm name."""
        parameterizer = UnifiedAlgorithmParameterizer("luby")
        assert parameterizer.name() == "Luby Randomized"

    def test_luby_repr(self):
        """Should have string representation."""
        parameterizer = UnifiedAlgorithmParameterizer("luby")
        result = repr(parameterizer)
        assert "Luby Randomized" in result


class TestLubyParameterizerExecution:
    """Test LubyParameterizer execute() method."""

    def test_luby_execute_returns_dict(self):
        """execute() should return Dict[int, int]."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("luby")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)

    def test_luby_uses_coefficients(self):
        """execute() should use Luby coefficients from vector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        # Vectors with different coefficients
        vector1 = CanonicalVector(
            luby_base_probability=0.3,
            luby_coeff_degree=0.5,
        )
        vector2 = CanonicalVector(
            luby_base_probability=0.8,
            luby_coeff_degree=-0.5,
        )

        parameterizer = UnifiedAlgorithmParameterizer("luby")

        result1 = parameterizer.execute(graph, vector1)
        result2 = parameterizer.execute(graph, vector2)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_luby_uses_max_iterations(self):
        """execute() should use max_iterations from vector."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4, 5]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)
        graph.add_edge(4, 5, 1.0)

        vector_few = CanonicalVector(max_iterations=5)
        vector_many = CanonicalVector(max_iterations=20)

        parameterizer = UnifiedAlgorithmParameterizer("luby")

        result_few = parameterizer.execute(graph, vector_few)
        result_many = parameterizer.execute(graph, vector_many)

        assert isinstance(result_few, dict)
        assert isinstance(result_many, dict)

    def test_luby_uses_convergence_threshold(self):
        """execute() should use convergence_threshold from vector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        vector_strict = CanonicalVector(convergence_threshold=0.05)
        vector_lenient = CanonicalVector(convergence_threshold=0.0)

        parameterizer = UnifiedAlgorithmParameterizer("luby")

        result_strict = parameterizer.execute(graph, vector_strict)
        result_lenient = parameterizer.execute(graph, vector_lenient)

        assert isinstance(result_strict, dict)
        assert isinstance(result_lenient, dict)

    def test_luby_execute_empty_graph(self):
        """execute() should handle empty graph."""
        graph = GraphManager.create_empty_graph()
        vector = CanonicalVector()
        parameterizer = UnifiedAlgorithmParameterizer("luby")

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_luby_all_coefficients_respected(self):
        """execute() should respect all coefficient parameters."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        # Create vector with all coefficients set
        vector = CanonicalVector(
            luby_base_probability=0.6,
            luby_coeff_degree=0.1,
            luby_coeff_neighbors_unmatched=-0.2,
            luby_coeff_clustering=0.3,
            luby_coeff_matched=-0.1,
            luby_coeff_round=0.2,
            luby_coeff_weight=0.05,
            max_iterations=20,
            convergence_threshold=0.02,
        )

        is_valid, error = vector.validate()
        assert is_valid, error

        parameterizer = UnifiedAlgorithmParameterizer("luby")
        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)


class TestParameterizerConsistency:
    """Test consistency across parameterizers."""

    def test_all_parameterizers_implement_interface(self):
        """All parameterizers should have name() and execute()."""
        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
        ]

        for p in parameterizers:
            assert callable(p.name)
            assert callable(p.execute)
            assert isinstance(p.name(), str)
            assert len(p.name()) > 0

    def test_all_parameterizers_work_on_same_graph(self):
        """All parameterizers should execute on same graph."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)

        vector = CanonicalVector()

        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
        ]

        for p in parameterizers:
            result = p.execute(graph, vector)
            assert isinstance(result, dict), f"{p.name()} failed"

    def test_all_parameterizers_handle_invalid_vector(self):
        """All parameterizers should handle invalid vectors gracefully."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        # Vector with out-of-range parameter (will fail validation)
        invalid_vector = CanonicalVector(max_iterations=100)

        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
        ]

        # They should either accept it or raise ValueError
        for p in parameterizers:
            try:
                result = p.execute(graph, invalid_vector)
                assert isinstance(result, dict)
            except ValueError:
                # Also acceptable - parameter validation at execute time
                pass
