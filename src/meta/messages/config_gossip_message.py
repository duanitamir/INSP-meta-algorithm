"""Config gossip message for spreading algorithm parameters to neighbors."""

from dataclasses import dataclass
from typing import Dict, Any, List
from src.utils.types import RoundNumber


@dataclass(frozen=True)
class ConfigGossipMessage:
    """Gossip message for spreading algorithm configuration to neighbors.

    Used to ensure all nodes learn about parameter updates across the network
    via gossip protocol. Nodes learn newer config from neighbors and adopt if
    version is higher than their current version.

    Includes version field so nodes only accept updates if version > current.
    Includes algorithm list for dynamic algorithm discovery.
    Includes hop_count for potential TTL/anti-loop prevention (future use).
    """

    sender: int
    recipient: int
    payload: Dict[str, Any]  # DistributedAlgorithmConfig.to_dict()
    available_algorithms: List[str]  # List of algorithm names discovered from registry
    version: int = 1
    algorithm_list_version: int = 1  # NEW: Version of available_algorithms list
    hop_count: int = 0
    round_num: RoundNumber = 0
    message_id: int | None = None

    @property
    def message_type(self) -> str:
        """Message type identifier for routing."""
        return "CONFIG_GOSSIP"

    def has_newer_algorithms(self, known_version: int) -> bool:
        """Check if this message has newer algorithm list than what we know.

        Args:
            known_version: The algorithm list version we currently have

        Returns:
            True if this message's algorithm_list_version > known_version
        """
        return self.algorithm_list_version > known_version

    def __hash__(self) -> int:
        return hash((self.sender, self.recipient, self.round_num, self.message_id, self.version))
