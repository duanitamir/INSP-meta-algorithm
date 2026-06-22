"""Unit tests for CanonicalVector parameter container - 15 tests."""

from src.meta.canonical_vector import CanonicalVector


class TestCanonicalVectorInitialization:
    """Test vector creation and initialization."""

    def test_initialization_with_defaults(self):
        """Vector should initialize with default values."""
        vector = CanonicalVector()

        # Verify defaults exist and are in valid ranges
        assert 0.0 <= vector.luby_base_probability <= 1.0
        assert 1 <= vector.itai_timeout_rounds <= 20
        assert 5 <= vector.max_iterations <= 100

    def test_initialization_with_custom_values(self):
        """Vector should accept custom parameter values."""
        vector = CanonicalVector(
            luby_base_probability=0.6,
            luby_coeff_degree=0.2,
            luby_coeff_neighbors_unmatched=-0.1,
            luby_coeff_clustering=0.15,
            luby_coeff_matched=0.05,
            luby_coeff_round=-0.02,
            luby_coeff_weight=0.3,
            itai_timeout_rounds=7,
            max_iterations=50,
        )

        assert vector.luby_base_probability == 0.6
        assert vector.itai_timeout_rounds == 7
        assert vector.max_iterations == 50


class TestCanonicalVectorValidation:
    """Test parameter validation."""

    def test_validation_passes_valid_vector(self):
        """Valid vector should pass validation."""
        vector = CanonicalVector()
        is_valid, error = vector.validate()
        assert is_valid
        assert error is None

    def test_validation_fails_invalid_luby_base(self):
        """Invalid luby_base_probability should fail."""
        vector = CanonicalVector(luby_base_probability=1.5)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "luby_base_probability" in error

    def test_validation_fails_invalid_coefficient(self):
        """Invalid luby coefficient should fail."""
        vector = CanonicalVector(luby_coeff_degree=2.0)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "luby_coeff_degree" in error

    def test_validation_fails_invalid_timeout(self):
        """Invalid timeout_rounds should fail."""
        vector = CanonicalVector(itai_timeout_rounds=25)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "itai_timeout_rounds" in error

    def test_validation_fails_invalid_max_iterations(self):
        """Invalid max_iterations should fail."""
        vector = CanonicalVector(max_iterations=150)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "max_iterations" in error

    def test_validation_fails_invalid_convergence_threshold(self):
        """Invalid convergence_threshold should fail."""
        vector = CanonicalVector(convergence_threshold=0.2)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "convergence_threshold" in error


class TestCanonicalVectorSerialization:
    """Test serialization and deserialization."""

    def test_to_list_converts_to_list(self):
        """to_list() should convert to list."""
        vector = CanonicalVector()
        result = vector.to_list()
        assert isinstance(result, list)
        assert len(result) == 10

    def test_from_list_creates_vector_from_list(self):
        """from_list() should create vector from list."""
        params = [0.5, 0.2, -0.1, 0.15, 0.05, -0.02, 0.3, 7, 50, 0.05]
        vector = CanonicalVector.from_list(params)
        assert vector.luby_base_probability == 0.5
        assert vector.max_iterations == 50
        assert vector.convergence_threshold == 0.05

    def test_to_list_and_from_list_roundtrip(self):
        """Should roundtrip through list."""
        original = CanonicalVector(
            luby_base_probability=0.6,
            luby_coeff_degree=0.2,
            luby_coeff_neighbors_unmatched=-0.1,
            luby_coeff_clustering=0.15,
            luby_coeff_matched=0.05,
            luby_coeff_round=-0.02,
            luby_coeff_weight=0.3,
            itai_timeout_rounds=7,
            max_iterations=50,
            convergence_threshold=0.05,
        )

        params = original.to_list()
        reconstructed = CanonicalVector.from_list(params)

        assert reconstructed.luby_base_probability == original.luby_base_probability
        assert reconstructed.itai_timeout_rounds == original.itai_timeout_rounds
        assert reconstructed.max_iterations == original.max_iterations
        assert reconstructed.convergence_threshold == original.convergence_threshold


class TestCanonicalVectorUtilities:
    """Test utility methods."""

    def test_all_parameters_exist(self):
        """Vector should have all parameters."""
        vector = CanonicalVector()
        assert hasattr(vector, "luby_base_probability")
        assert hasattr(vector, "luby_coeff_degree")
        assert hasattr(vector, "itai_timeout_rounds")
        assert hasattr(vector, "max_iterations")

    def test_random_vector_generation(self):
        """Should generate random valid vector."""
        vector = CanonicalVector.random()
        is_valid, error = vector.validate()
        assert is_valid, error

    def test_string_representation(self):
        """Should have string representation."""
        vector = CanonicalVector()
        result = str(vector)
        assert "CanonicalVector" in result
        assert "luby_base" in result
