"""Gossip protocol messages for distributed communication.

Contains:
- GossipMessage: Unified message for config, convergence, and parameter communication
  - Uses message_subtype to route different payload types
  - Includes built-in validation for round_num and weight
"""

from .gossip_message import GossipMessage

__all__ = [
    "GossipMessage",
]
