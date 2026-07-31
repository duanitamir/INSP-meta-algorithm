"""Dynamic local proposal policies for the distributed matching runtime."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import DistributedAlgorithmConfig
from src.simulation.local_node_context import LocalNodeContext


class LocalProposalPolicy(Protocol):
    """One algorithm's local proposal decision for a single node tick."""

    name: str

    def propose(self, context: LocalNodeContext) -> Mapping[int, float]:
        """Return weighted proposals addressed only to direct neighbors."""


@dataclass
class RegisteredProposalPolicy:
    """Adapt one dynamically registered implementation to the local policy API."""

    name: str
    algorithm: Any

    def propose(self, context: LocalNodeContext) -> Mapping[int, float]:
        neighbors = context.graph.neighbors()
        proposals = self.algorithm.propose_to_neighbors(
            context.node_id,
            neighbors,
            context,
        )
        non_neighbors = set(proposals).difference(neighbors)
        if non_neighbors:
            raise ValueError(f"{self.name} proposed non-neighbors: {sorted(non_neighbors)}")
        return proposals


def build_local_proposal_policies(
    config: DistributedAlgorithmConfig,
) -> tuple[LocalProposalPolicy, ...]:
    """Build the configured local policies once for a runtime startup snapshot."""
    from src.meta.core.algorithm_registry import AlgorithmRegistry
    from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

    registry = AlgorithmRegistry.instance()
    policies: list[LocalProposalPolicy] = []
    for name in config.available_algorithms:
        if not registry.is_algorithm_registered(name):
            raise ValueError(f"Configured algorithm is not registered: {name}")
        algorithm_class = AlgorithmRegistryBuilder.get_class(name)
        if algorithm_class is None:
            raise ValueError(f"Registered algorithm has no implementation class: {name}")
        policies.append(
            RegisteredProposalPolicy(
                name=name,
                algorithm=algorithm_class(parameters=dict(config.get_algorithm_params(name))),
            )
        )
    return tuple(policies)
