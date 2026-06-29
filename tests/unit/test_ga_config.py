"""Unit tests for GAConfig - 18+ comprehensive tests."""

import pytest

from src.meta.core.ga_config import GAConfig


class TestGAConfigInitialization:
    """Test GAConfig initialization and defaults."""

    def test_default_initialization(self) -> None:
        """Should initialize with sensible defaults."""
        config = GAConfig()
        assert config.population_size == 20
        assert config.generations == 30
        assert config.mutation_rate == 0.1

    def test_custom_initialization(self) -> None:
        """Should accept custom parameters."""
        config = GAConfig(
            population_size=15,
            generations=20,
            mutation_rate=0.15,
        )
        assert config.population_size == 15
        assert config.generations == 20
        assert config.mutation_rate == 0.15

    def test_all_parameters_settable(self) -> None:
        """Should allow setting all parameters."""
        config = GAConfig(
            population_size=25,
            generations=40,
            mutation_rate=0.12,
            elite_fraction=0.3,
            early_stop_generations=7,
            num_workers=8,
            max_iterations=150,
            convergence_threshold=0.08,
            quorum_threshold=0.6,
            voting_frequency=10,
            gossip_frequency=8,
        )
        assert config.population_size == 25
        assert config.generations == 40
        assert config.mutation_rate == 0.12
        assert config.elite_fraction == 0.3
        assert config.early_stop_generations == 7
        assert config.num_workers == 8
        assert config.max_iterations == 150
        assert config.convergence_threshold == 0.08
        assert config.quorum_threshold == 0.6
        assert config.voting_frequency == 10
        assert config.gossip_frequency == 8


class TestGAConfigValidation:
    """Test GAConfig.validate() method."""

    def test_valid_default_config(self) -> None:
        """Default config should be valid."""
        config = GAConfig()
        is_valid, error = config.validate()
        assert is_valid
        assert error == ""

    def test_invalid_population_size_too_small(self) -> None:
        """Should reject population_size < 2."""
        config = GAConfig(population_size=1)
        is_valid, error = config.validate()
        assert not is_valid
        assert "population_size" in error

    def test_invalid_generations_zero(self) -> None:
        """Should reject generations < 1."""
        config = GAConfig(generations=0)
        is_valid, error = config.validate()
        assert not is_valid
        assert "generations" in error

    def test_invalid_mutation_rate_negative(self) -> None:
        """Should reject mutation_rate < 0."""
        config = GAConfig(mutation_rate=-0.1)
        is_valid, error = config.validate()
        assert not is_valid
        assert "mutation_rate" in error

    def test_invalid_mutation_rate_too_high(self) -> None:
        """Should reject mutation_rate > 1.0."""
        config = GAConfig(mutation_rate=1.5)
        is_valid, error = config.validate()
        assert not is_valid
        assert "mutation_rate" in error

    def test_invalid_elite_fraction_too_low(self) -> None:
        """Should reject elite_fraction < 0.1."""
        config = GAConfig(elite_fraction=0.05)
        is_valid, error = config.validate()
        assert not is_valid
        assert "elite_fraction" in error

    def test_invalid_elite_fraction_too_high(self) -> None:
        """Should reject elite_fraction > 0.9."""
        config = GAConfig(elite_fraction=0.95)
        is_valid, error = config.validate()
        assert not is_valid
        assert "elite_fraction" in error

    def test_invalid_max_iterations_zero(self) -> None:
        """Should reject max_iterations < 1."""
        config = GAConfig(max_iterations=0)
        is_valid, error = config.validate()
        assert not is_valid
        assert "max_iterations" in error

    def test_invalid_max_iterations_too_high(self) -> None:
        """Should reject max_iterations > 1000."""
        config = GAConfig(max_iterations=1001)
        is_valid, error = config.validate()
        assert not is_valid
        assert "max_iterations" in error

    def test_invalid_convergence_threshold_negative(self) -> None:
        """Should reject convergence_threshold < 0."""
        config = GAConfig(convergence_threshold=-0.01)
        is_valid, error = config.validate()
        assert not is_valid
        assert "convergence_threshold" in error

    def test_valid_boundary_values(self) -> None:
        """Should accept boundary values."""
        config = GAConfig(
            population_size=2,
            generations=1,
            mutation_rate=0.0,
            elite_fraction=0.1,
            max_iterations=1,
            convergence_threshold=0.0,
        )
        is_valid, error = config.validate()
        assert is_valid


class TestGAConfigPresets:
    """Test predefined configuration presets."""

    def test_small_graph_preset(self) -> None:
        """Small graph preset should be valid and optimized."""
        config = GAConfig.small_graph()
        is_valid, error = config.validate()
        assert is_valid
        assert config.population_size == 10
        assert config.generations == 10
        assert config.max_iterations == 20

    def test_medium_graph_preset(self) -> None:
        """Medium graph preset should be valid and optimized."""
        config = GAConfig.medium_graph()
        is_valid, error = config.validate()
        assert is_valid
        assert config.population_size == 15
        assert config.generations == 15
        assert config.max_iterations == 50

    def test_large_graph_preset(self) -> None:
        """Large graph preset should be valid and optimized."""
        config = GAConfig.large_graph()
        is_valid, error = config.validate()
        assert is_valid
        assert config.population_size == 20
        assert config.generations == 30
        assert config.max_iterations == 100

    def test_aggressive_exploration_preset(self) -> None:
        """Aggressive exploration preset should have higher mutation, lower elitism."""
        config = GAConfig.aggressive_exploration()
        is_valid, error = config.validate()
        assert is_valid
        assert config.mutation_rate == 0.20
        assert config.elite_fraction == 0.3

    def test_conservative_exploitation_preset(self) -> None:
        """Conservative exploitation preset should have lower mutation, higher elitism."""
        config = GAConfig.conservative_exploitation()
        is_valid, error = config.validate()
        assert is_valid
        assert config.mutation_rate == 0.08
        assert config.elite_fraction == 0.7

    def test_all_presets_valid(self) -> None:
        """All presets should produce valid configs."""
        presets = [
            GAConfig.small_graph(),
            GAConfig.medium_graph(),
            GAConfig.large_graph(),
            GAConfig.aggressive_exploration(),
            GAConfig.conservative_exploitation(),
        ]
        for preset in presets:
            is_valid, error = preset.validate()
            assert is_valid, f"Preset failed validation: {error}"


class TestGAConfigSerialization:
    """Test GAConfig serialization and deserialization."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        config = GAConfig(population_size=15, generations=20)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["population_size"] == 15
        assert data["generations"] == 20
        assert len(data) == 11  # All 11 parameters

    def test_from_dict(self) -> None:
        """Should create config from dictionary."""
        original_data = {
            "population_size": 25,
            "generations": 40,
            "mutation_rate": 0.12,
            "elite_fraction": 0.3,
            "early_stop_generations": 7,
            "num_workers": 8,
            "max_iterations": 150,
            "convergence_threshold": 0.08,
            "quorum_threshold": 0.6,
            "voting_frequency": 10,
            "gossip_frequency": 8,
        }
        config = GAConfig.from_dict(original_data)

        assert config.population_size == 25
        assert config.generations == 40
        assert config.mutation_rate == 0.12

    def test_round_trip_serialization(self) -> None:
        """Should preserve data through to_dict -> from_dict."""
        original = GAConfig(
            population_size=18,
            generations=25,
            mutation_rate=0.13,
            elite_fraction=0.4,
            early_stop_generations=6,
            num_workers=6,
            max_iterations=75,
            convergence_threshold=0.06,
            quorum_threshold=0.55,
            voting_frequency=7,
            gossip_frequency=6,
        )

        data = original.to_dict()
        restored = GAConfig.from_dict(data)

        assert restored.population_size == original.population_size
        assert restored.generations == original.generations
        assert restored.mutation_rate == original.mutation_rate
        assert restored.elite_fraction == original.elite_fraction
        assert restored.early_stop_generations == original.early_stop_generations
        assert restored.num_workers == original.num_workers
        assert restored.max_iterations == original.max_iterations
        assert restored.convergence_threshold == original.convergence_threshold
        assert restored.quorum_threshold == original.quorum_threshold
        assert restored.voting_frequency == original.voting_frequency
        assert restored.gossip_frequency == original.gossip_frequency


class TestGAConfigParameterRanges:
    """Test valid parameter ranges."""

    def test_mutation_rate_bounds_0_0(self) -> None:
        """Mutation rate 0.0 should be valid."""
        config = GAConfig(mutation_rate=0.0)
        is_valid, _ = config.validate()
        assert is_valid

    def test_mutation_rate_bounds_1_0(self) -> None:
        """Mutation rate 1.0 should be valid."""
        config = GAConfig(mutation_rate=1.0)
        is_valid, _ = config.validate()
        assert is_valid

    def test_elite_fraction_bounds_min(self) -> None:
        """Elite fraction 0.1 should be valid."""
        config = GAConfig(elite_fraction=0.1)
        is_valid, _ = config.validate()
        assert is_valid

    def test_elite_fraction_bounds_max(self) -> None:
        """Elite fraction 0.9 should be valid."""
        config = GAConfig(elite_fraction=0.9)
        is_valid, _ = config.validate()
        assert is_valid

    def test_max_iterations_min(self) -> None:
        """Max iterations 1 should be valid."""
        config = GAConfig(max_iterations=1)
        is_valid, _ = config.validate()
        assert is_valid

    def test_max_iterations_max(self) -> None:
        """Max iterations 1000 should be valid."""
        config = GAConfig(max_iterations=1000)
        is_valid, _ = config.validate()
        assert is_valid

    def test_convergence_threshold_bounds(self) -> None:
        """Convergence threshold 0.0-0.5 should be valid."""
        for threshold in [0.0, 0.1, 0.25, 0.5]:
            config = GAConfig(convergence_threshold=threshold)
            is_valid, _ = config.validate()
            assert is_valid, f"threshold {threshold} should be valid"


class TestGAConfigParameterInteraction:
    """Test interactions between parameters."""

    def test_high_elitism_with_low_mutation(self) -> None:
        """High elitism with low mutation is valid but conservative."""
        config = GAConfig(elite_fraction=0.8, mutation_rate=0.05)
        is_valid, _ = config.validate()
        assert is_valid
        assert config.elite_fraction > config.mutation_rate

    def test_low_elitism_with_high_mutation(self) -> None:
        """Low elitism with high mutation is valid but exploratory."""
        config = GAConfig(elite_fraction=0.2, mutation_rate=0.3)
        is_valid, _ = config.validate()
        assert is_valid
        assert config.mutation_rate > config.elite_fraction

    def test_early_stop_less_than_generations(self) -> None:
        """Early stop generations can be less than total generations."""
        config = GAConfig(generations=30, early_stop_generations=5)
        is_valid, _ = config.validate()
        assert is_valid
        assert config.early_stop_generations < config.generations

    def test_early_stop_equal_generations(self) -> None:
        """Early stop generations can equal total generations."""
        config = GAConfig(generations=30, early_stop_generations=30)
        is_valid, _ = config.validate()
        assert is_valid
        assert config.early_stop_generations == config.generations
