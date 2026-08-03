"""Unit tests for CanonicalVector parameter container - 15 tests."""

from src.meta.core.canonical_vector import CanonicalVector


class TestCanonicalVectorInitialization:
    """Test vector creation and initialization."""

    def test_initialization_with_defaults(self):
        """Vector should initialize with default values."""
        vector = CanonicalVector()

        # Verify defaults exist and are in valid ranges
        assert 0.0 <= vector.get("luby_base_probability") <= 1.0
        assert 0.0 <= vector.get("itai_policy_weight") <= 2.0
        assert 5 <= vector.get("max_iterations") <= 100

    def test_initialization_with_custom_values(self):
        """Vector should accept custom parameter values."""
        vector = CanonicalVector(
            luby_base_probability=0.6,
            itai_policy_weight=1.5,
            max_iterations=50,
        )

        assert vector.get("luby_base_probability") == 0.6
        assert vector.get("itai_policy_weight") == 1.5
        assert vector.get("max_iterations") == 50


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

    def test_validation_fails_invalid_policy_weight(self):
        vector = CanonicalVector(itai_policy_weight=3.0)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "itai_policy_weight" in error

    def test_validation_fails_invalid_max_iterations(self):
        """Invalid max_iterations should fail."""
        vector = CanonicalVector(max_iterations=150)
        is_valid, error = vector.validate()
        assert not is_valid
        assert "max_iterations" in error

    def test_validation_fails_invalid_convergence_threshold(self):
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
        assert len(result) == len(vector.parameter_definitions)

    def test_from_list_creates_vector_from_list(self):
        """from_list() should create vector from list."""
        # Create a vector first to determine how many parameters exist
        v = CanonicalVector()
        num_params = len(v.to_list())
        # Create params list with correct number of elements
        params = [0.05] + [100] * (num_params - 1)
        vector = CanonicalVector.from_list(params)
        # Just verify it can be created and basic parameters exist
        assert vector.get("max_iterations") is not None
        assert vector.get("convergence_threshold") is not None

    def test_to_list_and_from_list_roundtrip(self):
        """Should roundtrip through list."""
        original = CanonicalVector(
            luby_base_probability=0.6,
            itai_policy_weight=1.5,
            max_iterations=50,
            convergence_threshold=0.05,
        )

        params = original.to_list()
        reconstructed = CanonicalVector.from_list(params)

        assert reconstructed.get("luby_base_probability") == original.luby_base_probability
        assert reconstructed.get("itai_policy_weight") == original.itai_policy_weight
        assert reconstructed.max_iterations == original.max_iterations
        assert reconstructed.convergence_threshold == original.convergence_threshold


class TestCanonicalVectorFingerprint:
    """Test deterministic vector identity."""

    def test_equivalent_vectors_have_the_same_stable_fingerprint(self):
        first = CanonicalVector(max_iterations=20)
        second = CanonicalVector.from_dict(first.to_dict())

        assert first.fingerprint() == second.fingerprint()


class TestCanonicalVectorUtilities:
    """Test utility methods."""

    def test_all_parameters_exist(self):
        """Vector should have all parameters."""
        vector = CanonicalVector()
        assert hasattr(vector, "luby_base_probability")
        assert hasattr(vector, "itai_policy_weight")
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
