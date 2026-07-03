"""Coordination layer for distributed protocols.

Handles:
- Edge voting for conflict resolution
- Convergence detection for termination
- Orchestration of distributed execution
"""

from .voting import EdgeVoting, ConvergenceDetection

__all__ = ["EdgeVoting", "ConvergenceDetection"]
