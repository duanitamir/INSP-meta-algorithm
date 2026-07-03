"""Unified parameterizer for all 3 matching algorithms.

Consolidates Greedy, Itai-Israeli, and Luby Randomized parameterizers into
a single generic implementation that dispatches based on algorithm type.
"""

from typing import Any, Dict, Tuple, List

from src.meta.parameterizers.base import AlgorithmParameterizer as BaseParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.state.node import NodeState
from src.simulation.node_context import NodeContext


class UnifiedAlgorithmParameterizer(BaseParameterizer):
    """Unified parameterizer for all 3 matching algorithms.

    Handles parameter extraction and execution for:
    - Greedy: Fast, locally optimal
    - Itai-Israeli: Guaranteed maximal
    - Luby Randomized: Parallel, tunable via coefficients

    Uses Template Method pattern from base class.
    """

    def __init__(self, algorithm_type: str):
        """Initialize parameterizer for specific algorithm.

        Args:
            algorithm_type: One of "greedy", "itai", "luby"

        Raises:
            ValueError: If algorithm_type not recognized
        """
        if algorithm_type not in ["greedy", "itai", "luby"]:
            raise ValueError(
                f"Unknown algorithm: {algorithm_type}. "
                "Must be 'greedy', 'itai', or 'luby'."
            )
        self.algorithm_type = algorithm_type
        self._cached_round = None
        self._max_degree = 1
        self._max_weight = 1.0
        self._max_neighbors = 1

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract algorithm-specific parameters from canonical vector."""
        extractors = {
            "greedy": lambda v: {"max_rounds": int(v.max_iterations)},
            "itai": lambda v: {"timeout_rounds": v.itai_timeout_rounds, "max_rounds": int(v.max_iterations)},
            "luby": lambda v: {
                "base_probability": v.luby_base_probability,
                "coeff_degree": v.luby_coeff_degree,
                "coeff_neighbors_unmatched": v.luby_coeff_neighbors_unmatched,
                "coeff_clustering": v.luby_coeff_clustering,
                "coeff_matched": v.luby_coeff_matched,
                "coeff_round": v.luby_coeff_round,
                "coeff_weight": v.luby_coeff_weight,
                "max_rounds": int(v.max_iterations),
            }
        }
        return extractors[self.algorithm_type](canonical_vector)

    def _create_vector_from_params(self, parameters: Dict[str, Any], max_rounds: int) -> CanonicalVector:
        """Create CanonicalVector from algorithm-specific parameters."""
        if self.algorithm_type == "greedy":
            return CanonicalVector()
        elif self.algorithm_type == "itai":
            return CanonicalVector(itai_timeout_rounds=parameters.get("timeout_rounds", 5))
        elif self.algorithm_type == "luby":
            return CanonicalVector(
                luby_base_probability=parameters.get("base_probability", 0.5),
                luby_coeff_degree=parameters.get("coeff_degree", 0.0),
                luby_coeff_neighbors_unmatched=parameters.get("coeff_neighbors_unmatched", 0.0),
                luby_coeff_clustering=parameters.get("coeff_clustering", 0.0),
                luby_coeff_matched=parameters.get("coeff_matched", 0.0),
                luby_coeff_round=parameters.get("coeff_round", 0.0),
                luby_coeff_weight=parameters.get("coeff_weight", 0.0),
                max_iterations=max_rounds,
            )

    def _run_algorithm(self, graph: Any, parameters: Dict[str, Any]) -> Dict[int, int]:
        """Run algorithm through multiple rounds until convergence."""
        from src.state.store import StateStore
        from src.communication.message_queue import MessageQueue

        state_store = StateStore(graph)
        message_queue = MessageQueue(graph)
        max_rounds = min(parameters.get("max_rounds", 100), 50)
        self._temp_vector = self._create_vector_from_params(parameters, max_rounds)

        # Loop through rounds until convergence or max_rounds
        for round_num in range(max_rounds):
            any_messages_sent = False

            for node_id in graph.vertices():
                node_state = state_store.get_node_state(node_id)
                messages = message_queue.get_messages(node_id)

                context = NodeContext(
                    node_id=node_id,
                    state=node_state,
                    incoming_messages=messages,
                    graph=graph,
                    vector=self._temp_vector,
                    round_number=round_num,
                    state_store=state_store,
                )

                new_state, out_messages = self.execute_local_step(context)
                state_store.update_node_state(node_id, new_state)

                if out_messages:
                    message_queue.send_batch(out_messages)
                    any_messages_sent = True

            if not any_messages_sent:
                break

        # Extract final matching from state store
        matching = {}
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            if node_state.is_matched():
                matching[node_id] = node_state.get_matched_to()

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        names = {
            "greedy": "Greedy",
            "itai": "Itai-Israeli",
            "luby": "Luby Randomized",
        }
        return names.get(self.algorithm_type, "Unknown")

    def execute_local_step(
        self, node_context: NodeContext
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute algorithm for one node in one round.

        Args:
            node_context: NodeContext with node execution context

        Returns:
            Tuple of (new_node_state, outgoing_messages)
        """
        from src.algorithms.implementations.greedy_matching import GreedyMatching
        from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
        from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
        from src.simulation.algorithm_context import AlgorithmContext

        node_id = node_context.node_id
        current_state = node_context.state
        incoming_messages = node_context.incoming_messages
        graph = node_context.graph

        context = AlgorithmContext(
            graph=graph,
            state_store=node_context.state_store,
            round_num=node_context.round_number,
        )

        if self.algorithm_type == "greedy":
            greedy = GreedyMatching()
            new_state, outgoing_messages = greedy.node_behavior(
                node_id=node_id,
                node_state=current_state,
                messages=incoming_messages,
                context=context,
            )

        elif self.algorithm_type == "itai":
            timeout_rounds = node_context.vector.itai_timeout_rounds
            itai = ItaiIsraeliMaximalMatching(timeout_rounds=timeout_rounds)
            new_state, outgoing_messages = itai.node_behavior(
                node_id=node_id,
                node_state=current_state,
                messages=incoming_messages,
                context=context,
            )

        elif self.algorithm_type == "luby":
            # Extract parameters and create adaptive activation function
            base_prob = node_context.vector.luby_base_probability
            coeff_degree = node_context.vector.luby_coeff_degree
            coeff_neighbors = node_context.vector.luby_coeff_neighbors_unmatched
            coeff_clustering = node_context.vector.luby_coeff_clustering
            coeff_matched = node_context.vector.luby_coeff_matched
            coeff_round = node_context.vector.luby_coeff_round
            coeff_weight = node_context.vector.luby_coeff_weight

            # Cache global graph statistics (computed once per round, not per node)
            if (
                not hasattr(self, "_cached_round")
                or self._cached_round != node_context.round_number
            ):
                all_degrees = [graph.degree(v) for v in graph.vertices()]
                self._max_degree = max(all_degrees) if all_degrees else 1

                all_weights = []
                for v in graph.vertices():
                    vneighbors = list(graph.neighbors(v))
                    if vneighbors:
                        vweights = [graph.get_edge_weight(v, vn) for vn in vneighbors]
                        all_weights.append(sum(vweights) / len(vweights))
                    else:
                        all_weights.append(0.0)
                self._max_weight = max(all_weights) if all_weights else 1.0
                self._max_neighbors = max(all_degrees) if all_degrees else 1
                self._cached_round = node_context.round_number

            def activation_fn(nid: int) -> float:
                """Compute adaptive activation probability for a node."""
                if nid != node_id:
                    return base_prob

                # Normalize degree
                degree = graph.degree(nid)
                normalized_degree = (
                    degree / self._max_degree if self._max_degree > 0 else 0
                )

                # Count unmatched neighbors
                neighbors = list(graph.neighbors(nid))
                unmatched_count = sum(
                    1
                    for n in neighbors
                    if not node_context.state_store.get_node_state(n).is_matched()
                )
                normalized_neighbors = unmatched_count / max(self._max_neighbors, 1)

                # Clustering coefficient
                if len(neighbors) > 1:
                    edges_between = 0
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i + 1 :]:
                            if graph.has_edge(n1, n2):
                                edges_between += 1
                    max_edges = len(neighbors) * (len(neighbors) - 1) / 2
                    clustering = edges_between / max_edges if max_edges > 0 else 0.0
                else:
                    clustering = 0.0

                # Count matched neighbors
                matched_count = sum(
                    1
                    for n in neighbors
                    if node_context.state_store.get_node_state(n).is_matched()
                )
                normalized_matched = (
                    matched_count / max(len(neighbors), 1) if neighbors else 0.0
                )

                # Normalize edge weight
                if neighbors:
                    weights = [graph.get_edge_weight(nid, n) for n in neighbors]
                    avg_weight = sum(weights) / len(weights)
                else:
                    avg_weight = 0.0
                normalized_weight = (
                    avg_weight / self._max_weight if self._max_weight > 0 else 0
                )

                # Adaptive probability
                prob = base_prob
                prob += coeff_degree * (normalized_degree - 0.5)
                prob += coeff_neighbors * (normalized_neighbors - 0.5)
                prob += coeff_clustering * (clustering - 0.5)
                prob += coeff_matched * (normalized_matched - 0.5)
                prob += coeff_round * (node_context.round_number / 100.0 - 0.5)
                prob += coeff_weight * (normalized_weight - 0.5)

                return max(0.0, min(1.0, prob))

            luby = LubyRandomizedMatching(
                activation_probability=base_prob,
                activation_function=activation_fn,
            )

            new_state, outgoing_messages = luby.node_behavior(
                node_id=node_id,
                node_state=current_state,
                messages=incoming_messages,
                context=context,
            )

        return new_state, outgoing_messages
