"""Abstract base class for algorithm parameterizers.

Defines extensible interface for wrapping matching algorithms with
parameterization and execution logic using the Template Method pattern.

Each algorithm wrapper:
1. Extracts parameters from canonical vector
2. Runs the algorithm
3. Validates and returns matching result
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple, List

from src.meta.core.canonical_vector import CanonicalVector
from src.state.node import NodeState

if TYPE_CHECKING:
    from src.simulation.node_context import NodeContext


class AlgorithmParameterizer(ABC):
    """Base class for algorithm wrappers with parameterization.

    Uses Template Method pattern to standardize execution flow:
    1. Validate canonical vector (common)
    2. Extract algorithm-specific parameters (subclass-specific)
    3. Run algorithm (subclass-specific)
    4. Validate output (common)

    Each algorithm has unique parameters but follows same interface:
    - execute(): Run algorithm with canonical vector parameters (template method)
    - _extract_parameters(): Get params for this algorithm (subclass)
    - _run_algorithm(): Execute the algorithm (subclass)
    - name(): Human-readable algorithm name (subclass)
    """

    def execute(
        self,
        graph: Any,
        canonical_vector: CanonicalVector,
        state_store: Any = None,
        cascade_cache: Dict[str, Any] | None = None,
        executor: Any = None,
    ) -> Dict[int, int]:
        """Execute algorithm with canonical vector parameters (Template Method).

        Implements standard flow: validate → extract params → run → validate output.

        Args:
            graph: GraphManager instance
            canonical_vector: 10-parameter chromosome (algorithm-specific params)
            state_store: Optional existing StateStore to reuse (for cascading). If None, creates fresh.
            cascade_cache: Optional distributed cascade cache (for performance). Contains local knowledge only.
            executor: Optional ThreadPoolExecutor to reuse (3D optimization). If None, creates fresh.

        Returns:
            Dict mapping node_id -> matched_partner (maximal matching).

        Raises:
            ValueError: If canonical vector is invalid
        """
        # Template method: validate common fields
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Template method: extract algorithm-specific parameters
        parameters = self._extract_parameters(canonical_vector)

        # Template method: run the algorithm (with optional state_store for cascading, cascade_cache for perf, executor for pooling)
        matching = self._run_algorithm(
            graph,
            parameters,
            state_store=state_store,
            cascade_cache=cascade_cache,
            executor=executor,
        )

        # Template method: validate output
        self._validate_output(matching)

        return matching

    def execute_local_step(
        self, node_context: "NodeContext"
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute algorithm for one node in one round (Phase 1: Per-Node Interface).

        This is the new per-node interface that parameterizers should implement.
        Currently provides default implementation that wraps existing algorithms
        via LocalAlgorithmAdapter, but can be overridden for efficiency.

        Args:
            node_context: NodeContext with node_id, state, messages, graph, vector, etc.

        Returns:
            Tuple of (new_node_state, outgoing_messages)

        Raises:
            NotImplementedError: If subclass doesn't provide optimized implementation
        """

        # Default: Use LocalAlgorithmAdapter to wrap existing algorithm
        # This maintains backward compatibility while providing per-node interface
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement execute_local_step() "
            "for Phase 1 per-node execution. Default wrapper not yet implemented."
        )

    @abstractmethod
    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract algorithm-specific parameters from canonical vector.

        Args:
            canonical_vector: 10-parameter chromosome

        Returns:
            Dict with algorithm-specific parameters (keys are algorithm-dependent)
        """
        pass

    @abstractmethod
    def _run_algorithm(
        self,
        graph: Any,
        parameters: Dict[str, Any],
        state_store: Any = None,
        cascade_cache: Dict[str, Any] | None = None,
        executor: Any = None,
    ) -> Dict[int, int]:
        """Run the matching algorithm with extracted parameters.

        Args:
            graph: GraphManager instance
            parameters: Algorithm-specific parameters from _extract_parameters()
            state_store: Optional existing StateStore to reuse (for cascading). If None, creates fresh.
            cascade_cache: Optional distributed cascade cache for local knowledge. If None, creates fresh.
            executor: Optional ThreadPoolExecutor to reuse (3D optimization). If None, creates fresh.

        Returns:
            Dict mapping node_id -> matched_partner
        """
        pass

    def _validate_output(self, matching: Dict[int, int]) -> None:
        """Validate algorithm output (can override in subclasses).

        Args:
            matching: Output matching from algorithm

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(matching, dict):
            raise ValueError(f"Matching must be dict, got {type(matching)}")

        # No strict validation - algorithms may return partial matchings
        # Each algorithm is responsible for its own output correctness

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
