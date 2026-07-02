"""Phase 3 fully distributed implementation (primary production system).

Contains:
- DistributedOrchestrator: Main orchestration, replaces CascadingLoop
- DistributedConvergenceDetector: Quorum-based termination detection
- DistributedParameterEvolver: Gossip-based GA evolution per node

Note: Conflict resolution is now handled by DistributedNode._local_conflict_resolution()
via endpoint voting in Phase 4.
"""

from .orchestrator import DistributedOrchestrator
from .convergence_detector import DistributedConvergenceDetector
from .parameter_evolver import DistributedParameterEvolver

__all__ = [
    "DistributedOrchestrator",
    "DistributedConvergenceDetector",
    "DistributedParameterEvolver",
]
