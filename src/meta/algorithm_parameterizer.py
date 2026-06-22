"""Abstract base class for algorithm parameterizers.

Defines extensible interface for wrapping matching algorithms with
parameterization and execution logic.

Each algorithm wrapper:
1. Executes algorithm with parameters from canonical vector
2. Returns matching result
"""

from abc import ABC, abstractmethod
from typing import Dict, Set
from src.meta.canonical_vector import CanonicalVector


class AlgorithmParameterizer(ABC):
    """Base class for algorithm wrappers with parameterization.

    Each algorithm has unique parameters but follows same interface:
    - execute(): Run algorithm with canonical vector parameters
    - name(): Human-readable algorithm name

    Implementation is graph-agnostic: same wrapper works on any graph.
    """

    @abstractmethod
    def execute(
        self,
        graph,
        selected_nodes: Set[int],
        canonical_vector: CanonicalVector,
    ) -> Dict[int, int]:
        """Execute algorithm on selected nodes.

        Args:
            graph: GraphManager instance
            selected_nodes: Nodes to operate on (from select_nodes())
            canonical_vector: 17-parameter chromosome (algorithm-specific params)

        Returns:
            Dict mapping node_id -> matched_partner (matching on selected_nodes).
            Nodes not in selected_nodes should not appear in result.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name.

        Returns:
            String like "Greedy", "Itai-Israeli", "Luby Randomized".
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name()})"
