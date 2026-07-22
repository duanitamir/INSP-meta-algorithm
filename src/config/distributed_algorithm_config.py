"""Algorithm configuration that nodes carry and gossip to neighbors.

These parameters define how algorithms behave and are spread via gossip protocol.
Each node carries a local copy and learns updated configs from neighbors.

Parameters come from CanonicalVector during GA optimization.
100% agnostic - stores algorithm parameters dynamically without hardcoding.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List


@dataclass
class DistributedAlgorithmConfig:
    """Algorithm parameters distributed across nodes via gossip.

    Stores all algorithm parameters dynamically in algorithm_parameters dict.
    No hardcoded fields - supports any algorithm combination.

    Structure:
    - algorithm_parameters: Dict[algo_name -> Dict[param_name -> value]]
      Example: {"greedy": {"max_rounds": 100}, "itai": {"timeout_rounds": 5, ...}, ...}
    - Convergence detection thresholds (when to stop)
    - Algorithm list for dynamic discovery
    - Version for ordering gossip updates
    """

    # Convergence detection (shared, not algorithm-specific)
    convergence_threshold: float = 0.05
    quorum_threshold: float = 0.5
    max_iterations: int = 100

    # Algorithm parameters (100% agnostic - dynamic storage, not hardcoded)
    algorithm_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Example: {"greedy": {"max_rounds": 100}, "itai": {...}, "luby": {...}}

    # Versioning for gossip protocol
    version: int = 1

    # Algorithm list version for gossip protocol (algorithms discovered from registry)
    algorithm_list_version: int = 1  # Separate version for algorithm list
    _cached_algorithms: List[str] = field(default_factory=list, init=False, repr=False)

    @property
    def available_algorithms(self) -> List[str]:
        """Get available algorithms from registry (or cache if from gossip).

        Returns:
            List of algorithm names discovered from AlgorithmRegistry
        """
        # If we have cached algorithms (from gossip), use them
        if self._cached_algorithms:
            return self._cached_algorithms

        # Otherwise, discover from registry
        from src.meta.core.algorithm_registry import AlgorithmRegistry
        registry = AlgorithmRegistry.instance()
        return registry.all_algorithm_names()

    @available_algorithms.setter
    def available_algorithms(self, value: List[str]) -> None:
        """Set available algorithms (used when merging from gossip).

        Args:
            value: List of algorithm names
        """
        self._cached_algorithms = value if value else []

    def has_algorithm_updates(self, other: "DistributedAlgorithmConfig") -> bool:
        """Check if another config has different algorithm list.

        Args:
            other: Another DistributedAlgorithmConfig to compare

        Returns:
            True if the algorithm lists differ
        """
        return set(self.available_algorithms or []) != set(other.available_algorithms or [])

    def merge_algorithm_list(self, other: "DistributedAlgorithmConfig") -> None:
        """Merge algorithm list from neighbor config (higher version wins).

        Args:
            other: Another DistributedAlgorithmConfig to merge from
        """
        if other.algorithm_list_version > self.algorithm_list_version:
            self.available_algorithms = (other.available_algorithms or []).copy()
            self.algorithm_list_version = other.algorithm_list_version

    def get_algorithm_params(self, algo_name: str) -> Dict[str, Any]:
        """Get parameters for a specific algorithm.

        Args:
            algo_name: Algorithm name (e.g., "greedy", "itai", "luby")

        Returns:
            Dict of parameters for this algorithm, empty if not found
        """
        return self.algorithm_parameters.get(algo_name, {})

    def set_algorithm_params(self, algo_name: str, params: Dict[str, Any]) -> None:
        """Set parameters for a specific algorithm.

        Args:
            algo_name: Algorithm name
            params: Dict of parameters to set
        """
        self.algorithm_parameters[algo_name] = params

    def get_parameter(self, algo_name: str, param_name: str, default: Any = None) -> Any:
        """Get a specific parameter for an algorithm.

        Args:
            algo_name: Algorithm name
            param_name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        algo_params = self.algorithm_parameters.get(algo_name, {})
        return algo_params.get(param_name, default)

    def has_parameters_for(self, algo_name: str) -> bool:
        """Check if this config has parameters for an algorithm.

        Args:
            algo_name: Algorithm name

        Returns:
            True if parameters exist for this algorithm
        """
        return algo_name in self.algorithm_parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excludes non-init fields)."""
        # Use asdict but filter out non-init fields like _cached_algorithms
        d = asdict(self)
        # Remove fields that aren't part of __init__
        d.pop('_cached_algorithms', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedAlgorithmConfig":
        """Create from dictionary (after deserialization)."""
        # Filter out any non-init fields
        data_copy = data.copy()
        data_copy.pop('_cached_algorithms', None)
        return cls(**data_copy)

    @classmethod
    def from_canonical_vector(cls, vector) -> "DistributedAlgorithmConfig":
        """Create from CanonicalVector for GA optimization.

        Auto-discovers available algorithms and their parameters from AlgorithmRegistry.
        100% agnostic - no hardcoded algorithm extraction.

        Args:
            vector: CanonicalVector with all GA parameters

        Returns:
            DistributedAlgorithmConfig with parameters extracted dynamically
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        registry = AlgorithmRegistry.instance()
        available_algos = registry.all_algorithm_names()  # Auto-discover!

        # Dynamically extract parameters for each algorithm
        algorithm_parameters = {}
        for algo_name in available_algos:
            algo_def = registry.get(algo_name)
            if not algo_def:
                continue

            # Extract parameters for this algorithm from vector
            params = {}
            param_defs = algo_def.get("parameters", {})
            for param_name in param_defs.keys():
                # Build full parameter name (algorithm prefix + parameter name)
                full_param_name = f"{algo_name}_{param_name}"

                # Try to extract from vector
                value = vector.get(full_param_name)
                if value is not None:
                    params[param_name] = value

            if params:  # Only add if we found parameters
                algorithm_parameters[algo_name] = params

        config = cls(
            convergence_threshold=vector.get("convergence_threshold") or 0.05,
            quorum_threshold=0.5,  # Default (not in CanonicalVector)
            max_iterations=int(vector.get("max_iterations") or 100),
            algorithm_parameters=algorithm_parameters,  # Dynamic storage
            version=1,
            algorithm_list_version=1
        )
        # Cache the discovered algorithms
        config.available_algorithms = available_algos
        return config
