"""Distributed protocol messages for Phase 3 system.

Contains:
- DistributedMessage: Abstract base class for all distributed messages
- ParameterGossipMessage: Messages for parameter gossip (GA evolution)
- ConvergenceGossipMessage: Messages for convergence voting (termination)
"""

from .base import DistributedMessage
from .parameter_gossip import ParameterGossipMessage
from .convergence_gossip import ConvergenceGossipMessage

__all__ = [
    "DistributedMessage",
    "ParameterGossipMessage",
    "ConvergenceGossipMessage",
]
