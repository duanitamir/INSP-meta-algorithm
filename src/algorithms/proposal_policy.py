"""Dynamic local proposal policies for the distributed matching runtime."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import DistributedAlgorithmConfig
from src.algorithms.base import EndpointProtocolAlgorithm, ProposalPolicyAlgorithm
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


def select_weight_aware_candidate(
    proposals_by_policy: Mapping[str, Mapping[int, float]],
    edge_weights: Mapping[int, float],
    config: DistributedAlgorithmConfig,
) -> int | None:
    """Choose a locally proposed edge with bounded algorithm influence.

    A materially heavier edge is never overturned.  Among edges within the
    configured relative gap of the heaviest edge, normalized policy preference
    supplies a deliberately small tie-breaking influence.
    """
    if not edge_weights:
        return None
    preference: dict[int, float] = {}
    for policy_name, proposals in proposals_by_policy.items():
        if not proposals:
            continue
        scale = max(abs(score) for score in proposals.values())
        if scale == 0:
            continue
        policy_weight = config.get_policy_weight(policy_name)
        for neighbor_id, score in proposals.items():
            preference[neighbor_id] = preference.get(neighbor_id, 0.0) + policy_weight * score / scale

    maximum_weight = max(edge_weights.values())
    close_candidates = {
        neighbor_id: weight
        for neighbor_id, weight in edge_weights.items()
        if weight >= maximum_weight * (1.0 - config.material_weight_gap)
    }
    maximum_preference = max(preference.values(), default=0.0)
    influence = config.algorithm_influence
    return max(
        close_candidates,
        key=lambda neighbor_id: (
            (1.0 - influence) * (close_candidates[neighbor_id] / maximum_weight)
            + influence * (preference.get(neighbor_id, 0.0) / maximum_preference if maximum_preference else 0.0),
            close_candidates[neighbor_id],
            -neighbor_id,
        ),
    )


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
        algorithm = algorithm_class(parameters=dict(config.get_algorithm_params(name)))
        if not isinstance(algorithm, ProposalPolicyAlgorithm):
            continue
        policies.append(
            RegisteredProposalPolicy(
                name=name,
                algorithm=algorithm,
            )
        )
    return tuple(policies)


def build_endpoint_protocols(
    startup_config: DistributedAlgorithmConfig, **context: Any
) -> tuple[Any, ...]:
    """Create endpoint protocols selected by the immutable startup configuration."""
    from src.meta.core.algorithm_registry import AlgorithmRegistry
    from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

    registry = AlgorithmRegistry.instance()
    protocols: list[Any] = []
    for name in startup_config.available_algorithms:
        if not registry.is_algorithm_registered(name):
            raise ValueError(f"Configured algorithm is not registered: {name}")
        algorithm_class = AlgorithmRegistryBuilder.get_class(name)
        if algorithm_class is None:
            raise ValueError(f"Registered algorithm has no implementation class: {name}")
        algorithm = algorithm_class(parameters=dict(startup_config.get_algorithm_params(name)))
        if isinstance(algorithm, EndpointProtocolAlgorithm):
            protocols.append(algorithm.create_protocol(**context))
    return tuple(protocols)
