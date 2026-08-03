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
        return {
            neighbor_id: weight
            for neighbor_id, weight in proposals.items()
            if context.state.is_neighbor_eligible(neighbor_id)
        }


def combine_local_policy_preferences(
    proposals_by_policy: Mapping[str, Mapping[int, float]],
    config: DistributedAlgorithmConfig,
) -> int | None:
    """Select one neighbor by combining normalized local policy preferences.

    Raw proposal values remain local edge scores.  Every policy's scores are
    normalized independently before its registry-derived combination weight is
    applied, so one policy's numeric scale cannot overpower another's weight.
    """
    combined: dict[int, float] = {}
    for policy_name in config.available_algorithms:
        proposals = proposals_by_policy.get(policy_name, {})
        if not proposals:
            continue
        scale = max(abs(score) for score in proposals.values())
        if scale == 0:
            continue
        weight = config.get_policy_weight(policy_name)
        for neighbor_id, score in proposals.items():
            combined[neighbor_id] = combined.get(neighbor_id, 0.0) + weight * score / scale

    if not combined:
        return None
    return max(combined, key=lambda neighbor_id: (combined[neighbor_id], -neighbor_id))


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
