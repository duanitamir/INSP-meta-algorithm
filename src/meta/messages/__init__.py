"""Distributed protocol messages for Phase 3 system.

Contains:
- DistributedMessage: Abstract base class for all distributed messages
- EdgeVotingMessage: Messages for edge voting consensus (conflict resolution)
- ParameterGossipMessage: Messages for parameter gossip (GA evolution)
- ConvergenceGossipMessage: Messages for convergence voting (termination)
"""

from .base import DistributedMessage
from .edge_voting import EdgeVotingMessage
from .parameter_gossip import ParameterGossipMessage
from .convergence_gossip import ConvergenceGossipMessage

__all__ = [
    "DistributedMessage",
    "EdgeVotingMessage",
    "ParameterGossipMessage",
    "ConvergenceGossipMessage",
]
