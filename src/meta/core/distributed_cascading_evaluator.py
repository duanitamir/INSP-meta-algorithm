"""Distributed cascading evaluator for GA fitness evaluation.

Implements cascading rounds where algorithms run repeatedly on the same graph,
with matched nodes becoming inactive between rounds. Each node only sees its
unmatched neighbors, creating a logically shrinking graph.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.matching_merger import merge_matchings


class DistributedCascadingEvaluator:
    """Evaluates fitness using distributed cascading rounds.

    For each cascade round:
    1. Run Greedy, Itai, Luby independently (each node sees neighbors only)
    2. Merge results via conflict resolution
    3. Update node states (matched nodes become inactive)
    4. Check convergence (improvement < threshold)
    5. Continue if improvement sufficient, else stop

    This simulates the distributed execution where each cascade round
    operates on a logically smaller graph as matched edges are removed.
    """

    def __init__(self) -> None:
        """Initialize cascading evaluator."""
        # Store cascade details for analysis (optional)
        self.last_num_cascades = 0
        self.last_weights_per_cascade = []
        self._accumulated_weight = 0.0  # Track accumulated weight across evals for debugging

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness using cascading rounds with persistent state.

        Compatible with FitnessEvaluator interface - returns just the fitness weight.
        Cascading details stored in self.last_num_cascades and self.last_weights_per_cascade.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector with parameters including max_iterations and convergence_threshold

        Returns:
            Final weight (fitness score). Cascade details in self.last_* attributes for analysis.
        """
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")

        # Import here to avoid circular imports
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer
        from src.state.store import StateStore
        from src.meta.core.cascade_cache_builder import CascadeCacheBuilder

        # Get parameters from vector
        max_cascades = int(vector.max_iterations)
        convergence_threshold = vector.convergence_threshold

        # Create single StateStore for all cascades (KEY FIX: persistent across cascades)
        state_store = StateStore(graph)

        # Pass thread pool executor to all cascades (3D optimization: algorithm-level pooling)
        # Instead of creating/destroying per cascade, create once and reuse across all cascades
        # This reduces overhead from ~5 executor creations per evaluation to 0
        executor = None

        prev_weight = 0.0
        weight_per_round = []
        cascade_round = 0
        total_weight = 0.0  # Accumulate weight across ALL cascades

        # Cascading rounds with persistent StateStore
        for cascade_round in range(max_cascades):
            # Build distributed cascade cache for this cascade
            # Contains only local knowledge: my_degree, my_neighbors, my_edge_weights,
            # neighbor_degrees, neighbor_state, messages
            cascade_cache = CascadeCacheBuilder.build_cascade_cache(
                graph, state_store, cascade_round
            )

            # CRITICAL: Identify already-matched nodes BEFORE running algorithms
            # This prevents cascades from re-matching previously-matched nodes
            already_matched_nodes = set()
            for node_id in graph.vertices():
                if state_store.get_node_state(node_id).is_matched():
                    already_matched_nodes.add(node_id)

            # Create fresh parameterizers for this cascade round
            parameterizers = [
                UnifiedAlgorithmParameterizer("greedy"),
                UnifiedAlgorithmParameterizer("itai"),
                UnifiedAlgorithmParameterizer("luby"),
            ]

            # Run all 3 parameterizers with SAME state_store (KEY FIX)
            # Each node sees only unmatched neighbors because matched nodes are marked in state_store
            matchings = []
            for parameterizer in parameterizers:
                try:
                    # KEY: Pass state_store AND cascade_cache so algorithms can use local knowledge
                    # Also pass executor so algorithm can reuse it across all rounds and cascades
                    matching = parameterizer.execute(
                        graph,
                        vector,
                        state_store=state_store,
                        cascade_cache=cascade_cache,
                        executor=executor,
                    )
                    matchings.append(matching)
                    # Capture executor on first run (created inside parameterizer if None)
                    if executor is None:
                        executor = parameterizer._executor
                except Exception:
                    matchings.append({})

            # Filter each matching: remove any edge where one endpoint is already matched
            filtered_matchings = []
            for matching in matchings:
                filtered = {}
                for u, v in matching.items():
                    # Only keep if BOTH nodes are unmatched
                    if u not in already_matched_nodes and v not in already_matched_nodes:
                        filtered[u] = v
                filtered_matchings.append(filtered)

            # Merge matchings via conflict resolution
            final_matching = merge_matchings(filtered_matchings, graph)

            # Calculate weight for THIS CASCADE ONLY (new matches in this round)
            curr_weight = 0.0
            if final_matching:
                for u, v in final_matching.items():
                    if u < v:  # Count each edge once
                        curr_weight += graph.get_edge_weight(u, v)

            weight_per_round.append(curr_weight)
            total_weight += curr_weight  # Accumulate across cascades

            # Check convergence
            if cascade_round > 0:
                improvement = (curr_weight - prev_weight) / (prev_weight + 1e-10)
                if improvement < convergence_threshold:
                    # Convergence reached, stop cascading
                    break

            # KEY FIX: Update state_store with matched nodes for next cascade
            # This ensures matched nodes become inactive (not seen as neighbors) in next cascade
            matched_pairs = set()
            for u, v in final_matching.items():
                if u < v:  # Track each pair once
                    matched_pairs.add((u, v))
                # Update matched_to in node's own state
                u_state = state_store.get_node_state(u)
                u_state.set_matched_to(v)
                state_store.update_node_state(u, u_state)

                v_state = state_store.get_node_state(v)
                v_state.set_matched_to(u)
                state_store.update_node_state(v, v_state)

            # Update neighbors dicts: tell all neighbors that u and v are now matched
            for u, v in matched_pairs:
                u_neighbors = graph.neighbors(u)
                v_neighbors = graph.neighbors(v)

                # U knows it's matched to V
                u_state = state_store.get_node_state(u)
                u_state.update_neighbor_status(v, matched=True, matched_to=u)
                state_store.update_node_state(u, u_state)

                # V knows it's matched to U
                v_state = state_store.get_node_state(v)
                v_state.update_neighbor_status(u, matched=True, matched_to=v)
                state_store.update_node_state(v, v_state)

                # Tell u's neighbors that u is matched to v
                for neighbor_u in u_neighbors:
                    neighbor_state = state_store.get_node_state(neighbor_u)
                    neighbor_state.update_neighbor_status(u, matched=True, matched_to=v)
                    state_store.update_node_state(neighbor_u, neighbor_state)

                # Tell v's neighbors that v is matched to u
                for neighbor_v in v_neighbors:
                    neighbor_state = state_store.get_node_state(neighbor_v)
                    neighbor_state.update_neighbor_status(v, matched=True, matched_to=u)
                    state_store.update_node_state(neighbor_v, neighbor_state)

            prev_weight = curr_weight

        # Clean up executor (3D optimization: reused across all cascades)
        if executor is not None:
            executor.shutdown(wait=True)

        # Store details for analysis
        self.last_num_cascades = cascade_round + 1
        self.last_weights_per_cascade = weight_per_round

        # Return total accumulated matched weight across all cascades
        # This represents the total value of all matches found (each cascade finds NEW edges
        # in the shrinking graph after previous cascade removed matched nodes)
        return total_weight

    def name(self) -> str:
        """Return evaluator name."""
        return "DistributedCascadingEvaluator"
