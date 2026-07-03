"""Unit tests for ParameterizerFactory - 3+ comprehensive tests."""

from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer
from src.meta.parameterizers.factory import ParameterizerFactory


class TestParameterizerFactory:
    """Test ParameterizerFactory static factory methods."""

    def test_create_default_returns_list(self) -> None:
        """create_default should return list of 3 parameterizers."""
        result = ParameterizerFactory.create_default()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_create_default_has_correct_types(self) -> None:
        """create_default should return [Greedy, Itai, Luby] in order."""
        result = ParameterizerFactory.create_default()
        assert isinstance(result[0], UnifiedAlgorithmParameterizer)
        assert isinstance(result[1], UnifiedAlgorithmParameterizer)
        assert isinstance(result[2], UnifiedAlgorithmParameterizer)
        assert result[0].algorithm_type == "greedy"
        assert result[1].algorithm_type == "itai"
        assert result[2].algorithm_type == "luby"

    def test_create_default_multiple_calls_independent(self) -> None:
        """Multiple calls to create_default should return independent instances."""
        set1 = ParameterizerFactory.create_default()
        set2 = ParameterizerFactory.create_default()

        assert set1 is not set2
        assert set1[0] is not set2[0]
        assert len(set1) == len(set2)

    def test_create_luby_only_returns_single(self) -> None:
        """create_luby_only should return list with 1 Luby parameterizer."""
        result = ParameterizerFactory.create_luby_only()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], UnifiedAlgorithmParameterizer)
        assert result[0].algorithm_type == "luby"

    def test_create_greedy_itai_returns_two(self) -> None:
        """create_greedy_itai should return [Greedy, Itai] in order."""
        result = ParameterizerFactory.create_greedy_itai()
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], UnifiedAlgorithmParameterizer)
        assert isinstance(result[1], UnifiedAlgorithmParameterizer)
        assert result[0].algorithm_type == "greedy"
        assert result[1].algorithm_type == "itai"

    def test_all_presets_have_execute_method(self) -> None:
        """All created parameterizers should have execute method."""
        for preset_func in [
            ParameterizerFactory.create_default,
            ParameterizerFactory.create_luby_only,
            ParameterizerFactory.create_greedy_itai,
        ]:
            parameterizers = preset_func()
            for param in parameterizers:
                assert hasattr(param, "execute")
                assert callable(param.execute)
