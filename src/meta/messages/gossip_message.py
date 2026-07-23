"""Gossip messages for distributed protocol communication."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class GossipMessage:
    """Unified gossip message for all algorithm communication.

    Replaces ConfigGossipMessage, ConvergenceGossipMessage, and ParameterGossipMessage.
    Uses message_subtype to route different data types through same infrastructure.

    Subtypes:
    - "config": Algorithm parameters and configuration (payload = DistributedAlgorithmConfig dict)
    - "convergence": Convergence voting (payload = {"should_stop": bool})
    - "parameter": GA elite vectors (payload = {"elite_vectors": [...], "fitness_values": [...]})
    """

    sender_node_id: int
    payload: Dict[str, Any]  # Structure depends on message_subtype
    message_subtype: str = "config"  # "config" | "convergence" | "parameter"
    message_version: int = 1  # For version-based updates (higher version wins)
    round_num: int = 0
    weight: float = 0.0
    hop_count: int = 0

    def __post_init__(self) -> None:
        """Validate message on creation."""
        if self.round_num < 0:
            raise ValueError(f"round_num must be >= 0, got {self.round_num}")
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    def has_newer_version(self, known_version: int) -> bool:
        """Check if this message has newer version than known.

        Args:
            known_version: The version we currently have

        Returns:
            True if this message's message_version > known_version
        """
        return self.message_version > known_version

    def increment_hop_count(self) -> None:
        """Increment hop count (for future TTL/anti-loop prevention).

        Note: Dataclass is frozen, so this is a no-op. Override in subclass if needed.
        """
        pass  # Frozen dataclass - override in mutable subclass if needed

    @classmethod
    def config_gossip(
        cls,
        sender_node_id: int,
        payload: Dict[str, Any],
        version: int,
        round_num: int = 0,
    ) -> "GossipMessage":
        """Create a config gossip message.

        Args:
            sender_node_id: Node sending config
            payload: DistributedAlgorithmConfig as dict
            version: Config version
            round_num: Algorithm round number

        Returns:
            GossipMessage with subtype="config"
        """
        return cls(
            sender_node_id=sender_node_id,
            payload=payload,
            message_version=version,
            message_subtype="config",
            round_num=round_num,
        )

    @classmethod
    def convergence_gossip(
        cls,
        sender_node_id: int,
        should_stop: bool,
        round_num: int = 0,
        weight: float = 0.0,
    ) -> "GossipMessage":
        """Create a convergence vote message.

        Args:
            sender_node_id: Node casting vote
            should_stop: True to vote for convergence
            round_num: Algorithm round number
            weight: Current weight (diagnostic)

        Returns:
            GossipMessage with subtype="convergence"
        """
        return cls(
            sender_node_id=sender_node_id,
            payload={"should_stop": should_stop},
            message_subtype="convergence",
            round_num=round_num,
            weight=weight,
        )

    @classmethod
    def parameter_gossip(
        cls,
        sender_node_id: int,
        elite_vectors: list,
        fitness_values: list,
        generation: int,
        round_num: int = 0,
    ) -> "GossipMessage":
        """Create a parameter gossip message for GA elite vectors.

        Args:
            sender_node_id: Node sharing vectors
            elite_vectors: List of CanonicalVectors to share
            fitness_values: Corresponding fitness scores
            generation: Generation number
            round_num: Algorithm round number

        Returns:
            GossipMessage with subtype="parameter"
        """
        return cls(
            sender_node_id=sender_node_id,
            payload={
                "elite_vectors": elite_vectors,
                "fitness_values": fitness_values,
                "generation": generation,
            },
            message_subtype="parameter",
            round_num=round_num,
        )
