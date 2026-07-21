"""Config gossip message for spreading algorithm parameters to neighbors."""

from dataclasses import dataclass
from typing import Dict, Any
from src.utils.types import RoundNumber


@dataclass(frozen=True)
class ConfigGossipMessage:
    """Gossip message for spreading algorithm configuration to neighbors.

    Used to ensure all nodes learn about parameter updates across the network
    via gossip protocol. Nodes learn newer config from neighbors and adopt if
    version is higher than their current version.

    Includes version field so nodes only accept updates if version > current.
    Includes hop_count for potential TTL/anti-loop prevention (future use).
    """

    sender: int
    recipient: int
    payload: Dict[str, Any]  # DistributedAlgorithmConfig.to_dict()
    version: int = 1
    hop_count: int = 0
    round_num: RoundNumber = 0
    message_id: int | None = None

    @property
    def message_type(self) -> str:
        """Message type identifier for routing."""
        return "CONFIG_GOSSIP"

    def __hash__(self) -> int:
        return hash((self.sender, self.recipient, self.round_num, self.message_id, self.version))
