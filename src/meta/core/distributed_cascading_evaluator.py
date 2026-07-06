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

        # Get parameters from vector
        max_cascades = int(vector.max_iterations)
        convergence_threshold = vector.convergence_threshold

        # Create single StateStore for all cascades (KEY FIX: persistent across cascades)
        state_store = StateStore(graph)

        prev_weight = 0.0
        weight_per_round = []
        cascade_round = 0

        # Cascading rounds with persistent StateStore
        for cascade_round in range(max_cascades):
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
                    # KEY: Pass state_store so matched nodes from previous cascades are known
                    matching = parameterizer.execute(graph, vector, state_store=state_store)
                    matchings.append(matching)
                except Exception:
                    matchings.append({})

            # Merge matchings via conflict resolution
            final_matching = merge_matchings(matchings, graph)

            # Calculate weight as fitness score
            curr_weight = 0.0
            if final_matching:
                for u, v in final_matching.items():
                    if u < v:  # Count each edge once
                        curr_weight += graph.get_edge_weight(u, v)

            weight_per_round.append(curr_weight)

            # Check convergence
            if cascade_round > 0:
                improvement = (curr_weight - prev_weight) / (prev_weight + 1e-10)
                if improvement < convergence_threshold:
                    # Convergence reached, stop cascading
                    break

            # KEY FIX: Update state_store with matched nodes for next cascade
            # This ensures matched nodes become inactive (not seen as neighbors) in next cascade
            for u, v in final_matching.items():
                state_store.get_node_state(u).set_matched_to(v)
                state_store.get_node_state(v).set_matched_to(u)

            prev_weight = curr_weight

        # Store details for analysis
        self.last_num_cascades = cascade_round + 1
        self.last_weights_per_cascade = weight_per_round

        return curr_weight

    def name(self) -> str:
        """Return evaluator name."""
        return "DistributedCascadingEvaluator"
