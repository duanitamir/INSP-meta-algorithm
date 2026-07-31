"""Frozen startup configuration for one distributed matching run."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class DistributedAlgorithmConfig:
    """Parameters shared by nodes for the lifetime of one runtime execution."""

    convergence_threshold: float = 0.05
    quorum_threshold: float = 0.5
    max_iterations: int = 100
    schema_version: int = 1
    vector_fingerprint: str = ""
    available_algorithms: tuple[str, ...] = ()
    algorithm_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_algorithms", tuple(self.available_algorithms))
        object.__setattr__(
            self,
            "algorithm_parameters",
            MappingProxyType(
                {
                    algorithm: MappingProxyType(dict(parameters))
                    for algorithm, parameters in self.algorithm_parameters.items()
                }
            ),
        )

    def get_algorithm_params(self, algorithm: str) -> Mapping[str, Any]:
        """Return the frozen parameters for one configured algorithm."""
        return self.algorithm_parameters.get(algorithm, MappingProxyType({}))

    def get_parameter(self, algorithm: str, parameter: str, default: Any = None) -> Any:
        """Return one configured parameter or its explicit default."""
        return self.get_algorithm_params(algorithm).get(parameter, default)

    def has_parameters_for(self, algorithm: str) -> bool:
        """Return whether the startup snapshot contains algorithm parameters."""
        return algorithm in self.algorithm_parameters

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the configuration snapshot used by a run."""
        return {
            "convergence_threshold": self.convergence_threshold,
            "quorum_threshold": self.quorum_threshold,
            "max_iterations": self.max_iterations,
            "schema_version": self.schema_version,
            "vector_fingerprint": self.vector_fingerprint,
            "available_algorithms": list(self.available_algorithms),
            "algorithm_parameters": {
                algorithm: dict(parameters)
                for algorithm, parameters in self.algorithm_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistributedAlgorithmConfig":
        """Restore a serialized startup snapshot."""
        return cls(
            convergence_threshold=float(data.get("convergence_threshold", 0.05)),
            quorum_threshold=float(data.get("quorum_threshold", 0.5)),
            max_iterations=int(data.get("max_iterations", 100)),
            schema_version=int(data.get("schema_version", 1)),
            vector_fingerprint=str(data.get("vector_fingerprint", "")),
            available_algorithms=tuple(data.get("available_algorithms", ())),
            algorithm_parameters=data.get("algorithm_parameters", {}),
        )

    @classmethod
    def from_canonical_vector(cls, vector: Any) -> "DistributedAlgorithmConfig":
        """Create one frozen configuration snapshot from a canonical vector."""
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        registry = AlgorithmRegistry.instance()
        available_algorithms = tuple(vector.algorithm_names)
        algorithm_parameters: dict[str, dict[str, Any]] = {}

        for algorithm in available_algorithms:
            definition = registry.get(algorithm)
            if definition is None:
                continue

            parameters = {
                parameter: value
                for parameter in definition.get("parameters", {})
                if (value := vector.get(f"{algorithm}_{parameter}")) is not None
            }
            if parameters:
                algorithm_parameters[algorithm] = parameters

        return cls(
            convergence_threshold=vector.get("convergence_threshold") or 0.05,
            max_iterations=int(vector.get("max_iterations") or 100),
            vector_fingerprint=vector.fingerprint(),
            available_algorithms=available_algorithms,
            algorithm_parameters=algorithm_parameters,
        )
