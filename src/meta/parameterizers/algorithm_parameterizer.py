"""Unified parameterizer for all matching algorithms.

Generic, registry-driven implementation that dispatches based on algorithm type.
Discovers and executes all registered algorithms without hardcoded algorithm references.
"""

import os
import random
from typing import Any, Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.meta.parameterizers.base import AlgorithmParameterizer as BaseParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.state.node import NodeState
from src.simulation.node_context import NodeContext


class UnifiedAlgorithmParameterizer(BaseParameterizer):
    """Unified parameterizer for all matching algorithms.

    Generic implementation that handles parameter extraction and execution
    for any registered algorithm via AlgorithmRegistry.

    Uses Template Method pattern from base class.
    """

    @property
    def ALGORITHM_DEFINITIONS(self) -> Dict[str, Dict]:
        """Get algorithm definitions from AlgorithmRegistryBuilder.

        Algorithms self-register when their modules are imported (zero hardcoded imports).

        Returns:
            Dict mapping algorithm name -> definition with "name" and "parameters" keys
        """
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder
        return AlgorithmRegistryBuilder.get_all_definitions()

    def __init__(self, algorithm_type: str):
        """Initialize parameterizer for specific algorithm.

        Args:
            algorithm_type: Algorithm name (discovered from registry)

        Raises:
            ValueError: If algorithm_type not recognized in registry
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        registry = AlgorithmRegistry.instance()
        if algorithm_type not in registry.all_algorithm_names():
            available = ", ".join(registry.all_algorithm_names())
            raise ValueError(
                f"Unknown algorithm: {algorithm_type}. Available: {available}"
            )
        self.algorithm_type = algorithm_type
        self._cached_round = None
        self._max_degree = 1
        self._max_weight = 1.0
        self._max_neighbors = 1

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract algorithm-specific parameters from canonical vector (100% generic).

        Reads from algorithm's PARAMETER_DEFINITION and extracts matching values
        from canonical vector. Uses None for missing parameters (algorithms handle defaults).

        This is completely generic - works with any algorithm without hard-coding.
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        # Get algorithm definition from registry
        registry = AlgorithmRegistry.instance()
        algo_def = registry.get(self.algorithm_type)

        if not algo_def or "parameters" not in algo_def:
            # Fallback: return empty dict (algorithm will use defaults)
            return {}

        # Extract each parameter from canonical vector
        parameters = {}
        for param_name in algo_def["parameters"].keys():
            # Build the full parameter name (with algorithm prefix if needed)
            full_param_name = f"{self.algorithm_type}_{param_name}"

            # Try to get from vector (with algorithm prefix)
            value = canonical_vector.get(full_param_name)

            # If not found and param_name is a base parameter, try without prefix
            if value is None and param_name in ["max_iterations", "convergence_threshold"]:
                value = canonical_vector.get(param_name)

            # Add to parameters dict (None means algorithm will use defaults)
            if value is not None:
                parameters[param_name] = value

        return parameters

    def _create_vector_from_params(
        self, parameters: Dict[str, Any], max_rounds: int
    ) -> CanonicalVector:
        """Create CanonicalVector from parameters (100% generic).

        Builds kwargs dict by prefixing parameter names with algorithm name,
        then creates vector with those parameters. Works with any algorithm.
        """
        # Always set base parameters
        kwargs = {
            "max_iterations": max_rounds,
            "convergence_threshold": 0.05,
        }

        # Add algorithm-specific parameters with algorithm prefix
        for param_name, param_value in parameters.items():
            full_param_name = f"{self.algorithm_type}_{param_name}"
            kwargs[full_param_name] = param_value

        return CanonicalVector(**kwargs)

    def _run_algorithm(
        self,
        graph: Any,
        parameters: Dict[str, Any],
        state_store: Any = None,
        cascade_cache: Dict[str, Any] | None = None,
        executor: Any = None,
    ) -> Dict[int, int]:
        """Run algorithm through multiple rounds until convergence.

        Uses ParallelNodeExecutor for concurrent node execution (3-4x speedup).
        Nodes execute in parallel, message delivery happens sequentially per round.
        Uses cascade_cache for O(1) lookups of local knowledge (my_degree, my_neighbors, etc.)
        Reuses executor across all cascades (3D optimization: algorithm-level pooling).

        Args:
            graph: GraphManager instance
            parameters: Algorithm parameters dict
            state_store: Optional existing StateStore to reuse (for cascading). If None, creates fresh.
            cascade_cache: Optional distributed cascade cache for performance. Contains local knowledge only.
            executor: Optional ThreadPoolExecutor to reuse (3D optimization). If None, creates fresh.

        Returns:
            Dict[int, int] of matching. StateStore state updated in-place if provided.
        """
        from src.state.store import StateStore
        from src.communication.message_queue import MessageQueue

        # Use provided state_store (for cascading) or create fresh one
        if state_store is None:
            state_store = StateStore(graph)

        # ALWAYS initialize algorithm state (for CASCADE 0 to match standard evaluator behavior)
        # For cascading: preserve matched_to from previous cascades, but reset other fields
        # For standard: fresh state for all nodes
        algorithm = self._get_algorithm_instance()

        # Preserve matched_to for cascading re-runs (matched nodes should persist)
        preserved_matched = {}
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            matched_to = node_state.get("matched_to")
            if matched_to is not None:
                preserved_matched[node_id] = matched_to

        # Initialize algorithm state (clears all fields for all nodes)
        algorithm.initialize_state(state_store, graph)

        # Restore matched_to for nodes that were matched in previous cascades
        for node_id, matched_to in preserved_matched.items():
            node_state = state_store.get_node_state(node_id)
            node_state.set("matched_to", matched_to)
            state_store.update_node_state(node_id, node_state)

        message_queue = MessageQueue(graph)
        # Use the full max_rounds value BUT cap at reasonable limit for performance
        # Cap at 10 rounds to balance thoroughness with execution time
        # Allows consecutive_inactive_rounds counter to work: need 2 consecutive for termination
        max_rounds_user = parameters.get("max_rounds", 100)
        max_rounds = min(max_rounds_user, 10)
        self._temp_vector = self._create_vector_from_params(parameters, max_rounds)

        # Determine parallelism: scale with available cores
        # For 1000 nodes: use 8 workers (good balance between parallelism and coordination overhead)
        # For smaller graphs: use fewer workers
        num_nodes = len(list(graph.vertices()))
        num_cores = os.cpu_count() or 4
        # Use 2/3 of available cores, min 4, max 10
        max_workers = min(10, max(4, (num_cores * 2) // 3))
        # Cap at reasonable limit based on node count
        max_workers = min(max_workers, max(4, num_nodes // 50))

        # Use provided executor (3D optimization: algorithm-level pooling) or create fresh one
        # If executor passed from evaluator, reuse across all cascades (will not be shut down here)
        # If None, create fresh (will NOT be shut down by parameterizer - caller responsible)
        from contextlib import nullcontext

        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            # Use null context so we don't shut down the executor here
            # The evaluator or caller is responsible for shutdown
            executor_context = nullcontext(executor)
        else:
            # Executor provided by caller, use null context (caller manages lifecycle)
            executor_context = nullcontext(executor)

        # Store executor as attribute so evaluator can reuse it (3D optimization)
        self._executor = executor

        # Use the executor context (never shuts down executor - caller responsible)
        consecutive_inactive_rounds = 0

        with executor_context:
            # Pre-compute round statistics for algorithms that use them
            # These don't change within a round, so compute once instead of per-node
            self._round_stats = self._compute_round_statistics(graph)

            # Loop through rounds until convergence or max_rounds
            for round_num in range(max_rounds):
                any_messages_sent = False

                # CRITICAL: Get all messages BEFORE parallel execution to avoid race conditions
                # (MessageQueue.get_messages() clears inbox - must be done before parallel threads access it)
                node_ids = list(graph.vertices())
                node_messages = {nid: message_queue.get_messages(nid) for nid in node_ids}

                # Update neighbor status from messages (distributed approach)
                # Each node learns about matches via status messages from neighbors
                # Messages are sent at end of round, so this will update state for next round
                #Skip message processing loop if no STATUS_UPDATE messages exist
                # Most rounds have no STATUS_UPDATE messages, so skip the O(n) loop entirely
                has_status_updates = any(
                    any(hasattr(msg, "payload") and isinstance(msg.payload, dict) and msg.payload.get("type") == "STATUS_UPDATE"
                        for msg in messages)
                    for messages in node_messages.values()
                )

                if has_status_updates:
                    # PHASE 1.2: Process messages and collect state updates for batching
                    message_updates: Dict[int, NodeState] = {}
                    for node_id in node_ids:
                        node_state = state_store.get_node_state(node_id)
                        messages = node_messages[node_id]
                        for msg in messages:
                            # Messages have .payload dict with "type" field
                            if hasattr(msg, "payload") and isinstance(msg.payload, dict):
                                if msg.payload.get("type") == "STATUS_UPDATE":
                                    # Node received status update from a neighbor
                                    neighbor_id = msg.sender
                                    matched = msg.payload.get("matched", False)
                                    matched_to = msg.payload.get("matched_to")
                                    node_state.update_neighbor_status(neighbor_id, matched, matched_to)
                        message_updates[node_id] = node_state

                    # Apply all message processing updates in one batch (reduces lock contention)
                    if message_updates:
                        state_store.batch_update_node_states(message_updates)

                # PARALLELIZED: Execute all nodes concurrently for this round
                def execute_node(node_id: int) -> Tuple[int, List[Dict[str, Any]]]:
                    """Execute one node and return its results."""
                    node_state = state_store.get_node_state(node_id)
                    messages = node_messages[node_id]  # Use pre-fetched messages

                    context = NodeContext(
                        node_id=node_id,
                        state=node_state,
                        incoming_messages=messages,
                        graph=graph,
                        vector=self._temp_vector,
                        round_number=round_num,
                        state_store=state_store,
                        cascade_cache=cascade_cache,  # Restored for proper algorithm execution
                    )

                    new_state, out_messages = self.execute_local_step(context)
                    return node_id, new_state, out_messages

                # Execute all nodes in parallel (reusing executor from cascade level)
                futures = {executor.submit(execute_node, nid): nid for nid in node_ids}

                # Collect all execution results before applying updates (Phase 1.2 batch update)
                execution_updates: Dict[int, NodeState] = {}
                for future in as_completed(futures):
                    try:
                        node_id, new_state, out_messages = future.result()
                        execution_updates[node_id] = new_state
                        if out_messages:
                            message_queue.send_batch(out_messages)
                            any_messages_sent = True
                    except Exception as e:
                        print(f"Error executing node {node_id}: {e}")

                # Apply all execution updates in one batch (Phase 1.2 optimization)
                if execution_updates:
                    state_store.batch_update_node_states(execution_updates)

                # Check if all nodes are inactive (true convergence)
                # Require 2 consecutive rounds of inactivity to be confident all are truly done
                # (not just between protocol phases of negotiation)
                all_inactive = all(
                    not state_store.get_node_state(nid).get("active", False)
                    for nid in node_ids
                )

                if round_num % 3 == 0 and all_inactive:
                    consecutive_inactive_rounds += 1
                    # Require 2 consecutive rounds of inactivity for true convergence
                    # Round 1: Initiators finish, may still have messages to send
                    # Round 2: Recipients finish processing those messages
                    # Round 3: All truly done and no more activity
                    if consecutive_inactive_rounds >= 2:
                        break
                else:
                    consecutive_inactive_rounds = 0

        # Extract final matching from state store
        matching = {}
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            if node_state.is_matched():
                matching[node_id] = node_state.get_matched_to()

        # Validate and enforce matching symmetry
        # In valid matchings, if A is matched to B, then B must be matched to A
        # This can fail in randomized algorithms with protocol latency (e.g., Luby)
        # Remove asymmetric edges where one side didn't reciprocate
        valid_matching = {}
        for node_id, matched_to in matching.items():
            if matched_to in matching and matching[matched_to] == node_id:
                # Symmetric pair - both point to each other
                if node_id not in valid_matching:
                    valid_matching[node_id] = matched_to

        # CRITICAL FIX: Clear invalid (asymmetric) matches from state_store
        # If a node had matched_to set during algorithm execution but the pair is
        # asymmetric and was filtered out, the node's matched_to state persists
        # incorrectly into the next cascade. We must clear these invalid states.
        valid_node_ids = set(valid_matching.keys()) | set(valid_matching.values())
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            if node_state.is_matched() and node_id not in valid_node_ids:
                # This node was marked as matched during algorithm execution,
                # but its match is not in the final valid symmetric matching.
                # Clear the matched_to state so next cascade doesn't see stale data.
                node_state.delete("matched_to")
                state_store.update_node_state(node_id, node_state)

        return valid_matching

    def name(self) -> str:
        """Return algorithm display name (100% agnostic - from registry)."""
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

        return AlgorithmRegistryBuilder.get_display_name(self.algorithm_type)

    def _get_algorithm_instance(self):
        """Get algorithm instance for initialization (from builder, no hardcoded imports).

        Retrieves algorithm class from AlgorithmRegistryBuilder by name.
        """
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

        algo_class = AlgorithmRegistryBuilder.get_class(self.algorithm_type)
        if not algo_class:
            raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")
        return algo_class()

    def execute_local_step(
        self, node_context: NodeContext
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute algorithm for one node in one round.

        Args:
            node_context: NodeContext with node execution context

        Returns:
            Tuple of (new_node_state, outgoing_messages)
        """
        from src.simulation.algorithm_context import AlgorithmContext
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

        node_id = node_context.node_id
        current_state = node_context.state
        incoming_messages = node_context.incoming_messages
        graph = node_context.graph

        context = AlgorithmContext(
            graph=graph,
            state_store=node_context.state_store,
            round_num=node_context.round_number,
        )

        # Extract parameters for this algorithm (100% generic, no hardcoded algorithm names)
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        registry = AlgorithmRegistry.instance()
        algo_def = registry.get(self.algorithm_type)
        if not algo_def:
            raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")

        # Get parameter definitions for this algorithm
        param_defs = algo_def.get("parameters", {})

        # Dynamically extract parameters from CanonicalVector
        params = {}
        for param_name in param_defs.keys():
            full_param_name = f"{self.algorithm_type}_{param_name}"
            value = node_context.vector.get(full_param_name)
            if value is not None:
                params[param_name] = value

        # Get algorithm instance from builder (no hardcoded imports)
        algo_class = AlgorithmRegistryBuilder.get_class(self.algorithm_type)
        if not algo_class:
            raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")

        # Create algorithm instance with extracted parameters (100% generic)
        algo_instance = algo_class(parameters=params if params else None)

        # Call node_behavior (works identically for all algorithms)
        new_state, outgoing_messages = algo_instance.node_behavior(
            node_id=node_id,
            node_state=current_state,
            messages=incoming_messages,
            context=context,
        )

        return new_state, outgoing_messages

    def _compute_round_statistics(self, graph: Any) -> Dict[str, float]:
        """Compute global statistics for adaptive activation function (once per algorithm run).

        These are expensive O(n×d) computations that used to happen
        inside activation_fn (called 1000+ times per cascade). Now computed once and cached.

        Args:
            graph: GraphManager instance

        Returns:
            Dict with max_degree, max_weight, max_neighbors
        """
        all_degrees = [graph.degree(v) for v in graph.vertices()]
        max_degree = max(all_degrees) if all_degrees else 1

        all_weights = []
        for v in graph.vertices():
            vneighbors = list(graph.neighbors(v))
            if vneighbors:
                vweights = [graph.get_edge_weight(v, vn) for vn in vneighbors]
                all_weights.append(sum(vweights) / len(vweights))
            else:
                all_weights.append(0.0)
        max_weight = max(all_weights) if all_weights else 1.0
        max_neighbors = max(all_degrees) if all_degrees else 1

        return {
            "max_degree": max_degree,
            "max_weight": max_weight,
            "max_neighbors": max_neighbors,
        }

    def propose_to_neighbors(
        self, node_id: int, neighbors: List[int], context: Any
    ) -> Dict[int, float]:
        """Get proposals to neighbors (100% generic, no hardcoded algorithm names).

        Args:
            node_id: This node's ID
            neighbors: List of direct neighbors only
            context: Algorithm context

        Returns:
            Dict[neighbor_id, weight] - proposals to send
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

        # Get algorithm definition and class (100% generic)
        registry = AlgorithmRegistry.instance()
        algo_def = registry.get(self.algorithm_type)
        if not algo_def:
            return {}

        algo_class = AlgorithmRegistryBuilder.get_class(self.algorithm_type)
        if not algo_class:
            return {}

        # Extract parameters for this algorithm (dynamically, no hardcoding)
        vector = getattr(context, "vector", None)
        param_defs = algo_def.get("parameters", {})

        params = {}
        for param_name in param_defs.keys():
            full_param_name = f"{self.algorithm_type}_{param_name}"
            if vector:
                value = vector.get(full_param_name)
                if value is not None:
                    params[param_name] = value

        # Create algorithm and get proposals (works identically for all algorithms)
        algo = algo_class(parameters=params if params else None)
        return algo.propose_to_neighbors(node_id, neighbors, context)
