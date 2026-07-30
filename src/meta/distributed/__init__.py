"""Distributed matching runtime.

The runtime exposes a bootstrap/observer orchestrator.  Endpoint nodes own
matching, convergence, and protocol decisions through addressed messages.
"""

from .orchestrator import DistributedOrchestrator

__all__ = [
    "DistributedOrchestrator",
]
