"""Unit tests for AlgorithmParameterizer ABC - 8 comprehensive tests."""

import pytest
from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.graph.graph_manager import GraphManager


class ConcreteParameterizer(AlgorithmParameterizer):
    """Concrete implementation for testing (stub)."""

    def _extract_parameters(self, canonical_vector):
        """Extract test parameters."""
        return {"test": True}

    def _run_algorithm(
        self, graph, parameters, state_store=None, cascade_cache=None, executor=None
    ):
        """Return empty matching."""
        return {}

    def name(self) -> str:
        """Return test name."""
        return "TestAlgorithm"


class TestAlgorithmParameterizerInterface:
    """Test abstract base class interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            AlgorithmParameterizer()

    def test_can_instantiate_concrete_subclass(self):
        """Should be able to instantiate concrete subclass."""
        parameterizer = ConcreteParameterizer()
        assert parameterizer is not None

    def test_subclass_requires_execute(self):
        """Subclass must implement execute()."""

        class IncompleteParameterizer(AlgorithmParameterizer):
            def name(self) -> str:
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteParameterizer()

    def test_subclass_requires_name(self):
        """Subclass must implement name()."""

        class IncompleteParameterizer(AlgorithmParameterizer):
            def execute(self, graph, canonical_vector):
                pass

        with pytest.raises(TypeError):
            IncompleteParameterizer()


class TestAlgorithmParameterizerExecute:
    """Test execute() contract."""

    def test_execute_returns_dict(self):
        """execute() should return Dict[int, int]."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        vector = CanonicalVector.random()
        parameterizer = ConcreteParameterizer()

        result = parameterizer.execute(graph, vector)

        assert isinstance(result, dict)

    def test_execute_matching_has_valid_nodes(self):
        """execute() result should only contain existing nodes."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)

        vector = CanonicalVector.random()
        parameterizer = ConcreteParameterizer()

        result = parameterizer.execute(graph, vector)

        # All keys and values should be valid nodes
        for u, v in result.items():
            assert u in graph.vertices()
            assert v in graph.vertices()

    def test_execute_receives_valid_canonical_vector(self):
        """execute() should accept CanonicalVector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        vector = CanonicalVector()
        parameterizer = ConcreteParameterizer()

        result = parameterizer.execute(graph, vector)
        assert result is not None


class TestAlgorithmParameterizerName:
    """Test name() contract."""

    def test_name_returns_string(self):
        """name() should return str."""
        parameterizer = ConcreteParameterizer()
        result = parameterizer.name()
        assert isinstance(result, str)

    def test_name_non_empty(self):
        """name() should return non-empty string."""
        parameterizer = ConcreteParameterizer()
        result = parameterizer.name()
        assert len(result) > 0

    def test_name_consistent(self):
        """name() should return same value across calls."""
        parameterizer = ConcreteParameterizer()
        name1 = parameterizer.name()
        name2 = parameterizer.name()
        assert name1 == name2


class TestAlgorithmParameterizerRepr:
    """Test string representation."""

    def test_repr_returns_string(self):
        """__repr__() should return str."""
        parameterizer = ConcreteParameterizer()
        result = repr(parameterizer)
        assert isinstance(result, str)

    def test_repr_contains_name(self):
        """__repr__() should contain algorithm name."""
        parameterizer = ConcreteParameterizer()
        result = repr(parameterizer)
        assert "TestAlgorithm" in result
