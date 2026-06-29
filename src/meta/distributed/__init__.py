"""Phase 3 fully distributed implementation (primary production system).

Contains:
- DistributedOrchestrator: Main orchestration, replaces CascadingLoop
- DistributedConflictResolver: Edge voting for conflict resolution (replaces ConflictResolver)
- DistributedConvergenceDetector: Quorum-based termination detection
- DistributedParameterEvolver: Gossip-based GA evolution per node
"""

from .orchestrator import DistributedOrchestrator
from .conflict_resolver import DistributedConflictResolver
from .convergence_detector import DistributedConvergenceDetector
from .parameter_evolver import DistributedParameterEvolver

__all__ = [
    "DistributedOrchestrator",
    "DistributedConflictResolver",
    "DistributedConvergenceDetector",
    "DistributedParameterEvolver",
]
