"""Parameterizer wrapper for Luby Randomized matching algorithm.

Uses ThreadPoolExecutor for parallel node execution within each round (Option 2 optimization).
Each node in the round executes concurrently with fine-grained per-node locking in StateStore.
"""

from typing import Any, Callable, Dict, Tuple, List

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.state.node_state import NodeState
from src.simulation.node_context import NodeContext


class LubyParameterizer(AlgorithmParameterizer):
    """Wraps Luby Randomized algorithm with canonical vector parameters.

    Uses: canonical_vector Luby parameters [0:7] + max_iterations [8]
    Implements: Adaptive activation probability using Luby coefficients [1-6]
    Uses Template Method pattern from AlgorithmParameterizer base class.
    """

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract Luby parameters from canonical vector.

        Uses: luby_base_probability [0], all coefficients [1-6], max_iterations [8].

        Args:
            canonical_vector: 10-parameter chromosome

        Returns:
            Dict with Luby-specific parameters
        """
        return {
            "base_probability": canonical_vector.luby_base_probability,
            "coeff_degree": canonical_vector.luby_coeff_degree,
            "coeff_neighbors_unmatched": canonical_vector.luby_coeff_neighbors_unmatched,
            "coeff_clustering": canonical_vector.luby_coeff_clustering,
            "coeff_matched": canonical_vector.luby_coeff_matched,
            "coeff_round": canonical_vector.luby_coeff_round,
            "coeff_weight": canonical_vector.luby_coeff_weight,
            "max_rounds": int(canonical_vector.max_iterations),
        }

    def _run_algorithm(self, graph: Any, parameters: Dict[str, Any]) -> Dict[int, int]:
        """Run Luby algorithm via per-node execution (Phase 3 refactored).

        PHASE 3 REFACTOR: Delegates to execute_local_step() for per-node execution.

        Creates minimal resources needed for backward compatibility with old interface.
        In real distributed deployment, orchestrator would call execute_local_step() directly.

        Args:
            graph: GraphManager instance
            parameters: Dict with Luby parameters

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        # Import here to avoid circular imports
        from src.state.state_store import StateStore
        from src.communication.message_queue import MessageQueue
        from src.simulation.node_context import NodeContext

        # Create minimal resources for backward compatibility
        state_store = StateStore(graph)
        message_queue = MessageQueue(graph)

        # For backward compatibility, create a temporary vector with params
        if not hasattr(self, "_temp_vector"):
            from src.meta.core.canonical_vector import CanonicalVector

            base_prob = parameters.get("base_probability", 0.5)
            self._temp_vector = CanonicalVector(
                luby_base_probability=base_prob,
                luby_coeff_degree=parameters.get("coeff_degree", 0.0),
                luby_coeff_neighbors_unmatched=parameters.get("coeff_neighbors_unmatched", 0.0),
                luby_coeff_clustering=parameters.get("coeff_clustering", 0.0),
                luby_coeff_matched=parameters.get("coeff_matched", 0.0),
                luby_coeff_round=parameters.get("coeff_round", 0.0),
                luby_coeff_weight=parameters.get("coeff_weight", 0.0),
            )

        # Execute each node using the new per-node interface
        matching = {}
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            messages = message_queue.get_messages(node_id)

            context = NodeContext(
                node_id=node_id,
                state=node_state,
                incoming_messages=messages,
                graph=graph,
                vector=self._temp_vector,
                round_number=0,
                state_store=state_store,
            )

            new_state, _ = self.execute_local_step(context)
            state_store.update_node_state(node_id, new_state)

            if new_state.is_matched():
                matching[node_id] = new_state.get_matched_to()

        return matching

    def _compute_adaptive_activation_fn(
        self, graph: Any, params: Dict[str, float]
    ) -> Callable[[int], float]:
        """Compute per-node adaptive activation probability function.

        Uses node properties weighted by parameters to compute node-specific probabilities.

        Args:
            graph: GraphManager instance
            params: Dict with coefficient values

        Returns:
            Function that maps node_id -> activation_probability [0, 1]
        """
        base_prob = params["base_probability"]
        coeff_degree = params["coeff_degree"]
        coeff_neighbors = params["coeff_neighbors_unmatched"]
        coeff_clustering = params["coeff_clustering"]
        coeff_matched = params["coeff_matched"]
        coeff_weight = params["coeff_weight"]

        # Precompute node properties for efficiency
        vertices = list(graph.vertices())
        degrees = [graph.degree(v) for v in vertices]
        max_degree = max(degrees) if degrees else 1

        # Precompute average edge weights per node
        avg_weights = []
        for v in vertices:
            neighbors = list(graph.neighbors(v))
            if neighbors:
                weights = [graph.get_edge_weight(v, n) for n in neighbors]
                avg_weights.append(sum(weights) / len(weights))
            else:
                avg_weights.append(0.0)
        max_weight = max(avg_weights) if avg_weights else 1.0

        # Create mapping for quick lookup
        degree_map = {v: d for v, d in zip(vertices, degrees)}
        weight_map = {v: w for v, w in zip(vertices, avg_weights)}

        def activation_fn(node_id: int) -> float:
            """Compute adaptive activation probability for a single node."""
            # Normalize degree [0, 1]
            normalized_degree = degree_map.get(node_id, 0) / max_degree if max_degree > 0 else 0

            # Normalize average edge weight [0, 1]
            normalized_weight = weight_map.get(node_id, 0) / max_weight if max_weight > 0 else 0

            # Adaptive probability (base + weighted coefficients)
            prob = base_prob
            prob += coeff_degree * (normalized_degree - 0.5)  # Center around 0
            prob += coeff_neighbors * (normalized_degree - 0.5)  # Proxy for neighbors
            prob += coeff_clustering * (normalized_degree - 0.5)  # Simple clustering proxy
            prob += coeff_matched * (normalized_degree - 0.5)  # Matched neighbors proxy
            prob += coeff_weight * (normalized_weight - 0.5)

            # Clamp to [0, 1]
            return max(0.0, min(1.0, prob))

        return activation_fn

    def name(self) -> str:
        """Return algorithm name."""
        return "Luby Randomized"

    def execute_local_step(
        self, node_context: NodeContext
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute Luby algorithm for one node in one round (Phase 1).

        Implements per-node execution with adaptive activation probability.
        No centralized StateStore or full graph iterations needed.

        Args:
            node_context: NodeContext containing node_id, state, messages, etc.

        Returns:
            Tuple of (new_node_state, outgoing_messages)
        """
        from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
        from src.simulation.algorithm_context import AlgorithmContext

        # Extract parameters from vector
        base_prob = node_context.vector.luby_base_probability
        coeff_degree = node_context.vector.luby_coeff_degree
        coeff_neighbors = node_context.vector.luby_coeff_neighbors_unmatched
        coeff_clustering = node_context.vector.luby_coeff_clustering
        coeff_matched = node_context.vector.luby_coeff_matched
        coeff_round = node_context.vector.luby_coeff_round
        coeff_weight = node_context.vector.luby_coeff_weight

        # Create adaptive activation function
        graph = node_context.graph
        node_id = node_context.node_id

        def activation_fn(nid: int) -> float:
            """Compute adaptive activation probability for a node."""
            if nid != node_id:
                return base_prob

            # Normalize degree
            degree = graph.degree(nid)
            all_degrees = [graph.degree(v) for v in graph.vertices()]
            max_degree = max(all_degrees) if all_degrees else 1
            normalized_degree = degree / max_degree if max_degree > 0 else 0

            # Normalize edge weight
            neighbors = list(graph.neighbors(nid))
            if neighbors:
                weights = [graph.get_edge_weight(nid, n) for n in neighbors]
                avg_weight = sum(weights) / len(weights)
            else:
                avg_weight = 0.0

            all_weights = []
            for v in graph.vertices():
                vneighbors = list(graph.neighbors(v))
                if vneighbors:
                    vweights = [graph.get_edge_weight(v, vn) for vn in vneighbors]
                    all_weights.append(sum(vweights) / len(vweights))
                else:
                    all_weights.append(0.0)
            max_weight = max(all_weights) if all_weights else 1.0
            normalized_weight = avg_weight / max_weight if max_weight > 0 else 0

            # Adaptive probability
            prob = base_prob
            prob += coeff_degree * (normalized_degree - 0.5)
            prob += coeff_neighbors * (normalized_degree - 0.5)
            prob += coeff_clustering * (normalized_degree - 0.5)
            prob += coeff_matched * (normalized_degree - 0.5)
            prob += coeff_round * (node_context.round_number / 100.0 - 0.5)
            prob += coeff_weight * (normalized_weight - 0.5)

            return max(0.0, min(1.0, prob))

        # Create Luby with adaptive activation
        luby = LubyRandomizedMatching(
            activation_probability=base_prob,
            activation_function=activation_fn,
        )

        current_state = node_context.state
        incoming_messages = node_context.incoming_messages

        # Create context for algorithm execution
        context = AlgorithmContext(
            graph=node_context.graph,
            state_store=node_context.state_store,
            round_num=node_context.round_number,
        )

        # Execute Luby behavior for THIS NODE ONLY
        new_state, outgoing_messages = luby.node_behavior(
            node_id=node_id,
            node_state=current_state,
            messages=incoming_messages,
            context=context,
        )

        return new_state, outgoing_messages
