"""Distributed GA for parameter evolution via gossip protocol."""

import random
from typing import Dict, List, Tuple
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.messages.gossip_message import GossipMessage
from src.state.node_parameter_state import NodeParameterState
from src.communication.message import Message
from src.communication.message_queue import MessageQueue
from src.utils.types import RoundNumber


class DistributedParameterEvolver:
    """Distributed GA for parameter evolution using gossip protocol.

    Each node:
    1. Maintains local population of CanonicalVectors
    2. Evaluates fitness (runs algorithms, computes matching weight)
    3. Performs local GA operations (elite selection, crossover, mutation)
    4. Gossips elite vectors to random neighbors periodically
    5. Integrates received elite vectors into population
    6. Converges when all nodes have same elite vector

    This replaces the centralized MetaAlgorithmGA with distributed learning.
    """

    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.15,
        gossip_frequency: int = 5,
        elite_k: int = 3,
    ) -> None:
        """Initialize distributed parameter evolver.

        Args:
            population_size: Number of vectors per node's population
            mutation_rate: Probability to mutate each parameter
            gossip_frequency: Send gossip every N algorithm rounds
            elite_k: Number of elite vectors to gossip to neighbors
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.gossip_frequency = gossip_frequency
        self.elite_k = elite_k

        # Per-node state (will be populated when nodes are known)
        self.node_states: Dict[int, NodeParameterState] = {}

    def initialize(self, node_ids: List[int]) -> None:
        """Initialize GA state for all nodes.

        Args:
            node_ids: List of node IDs in the network
        """
        for node_id in node_ids:
            state = NodeParameterState(node_id=node_id)
            state.initialize(self.population_size)
            self.node_states[node_id] = state

    def get_node_state(self, node_id: int) -> NodeParameterState:
        """Get GA state for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeParameterState for this node
        """
        if node_id not in self.node_states:
            raise ValueError(f"Node {node_id} not initialized")
        return self.node_states[node_id]

    def evaluate_population(
        self, node_id: int, graph: GraphManager, parameterizers, fitness_evaluator
    ) -> None:
        """Evaluate fitness of all vectors in a node's population.

        Args:
            node_id: Node performing evaluation
            graph: Graph to evaluate on
            parameterizers: List of algorithm parameterizers
            fitness_evaluator: FitnessEvaluator instance
        """
        state = self.get_node_state(node_id)

        for vector in state.population:
            # Only evaluate if not already evaluated
            if id(vector) not in state.fitness:
                fitness = fitness_evaluator.evaluate(graph, vector)
                state.update_fitness(vector, fitness)

    def evolve_local_population(self, node_id: int) -> None:
        """Perform one generation of local GA on this node.

        Operations:
        1. Elite selection (keep best K%)
        2. Crossover (blend elite vectors)
        3. Mutation (perturb parameters)

        Args:
            node_id: Node performing evolution
        """
        state = self.get_node_state(node_id)

        # Get elite vectors (top 50% by fitness)
        elite_count = max(1, len(state.population) // 2)
        elite = state.get_elite_k(k=elite_count)

        if not elite:
            elite = state.population[:1]

        # Create new population through crossover and mutation
        new_population: List[CanonicalVector] = []

        # Keep elite (elitism)
        new_population.extend(elite)

        # Generate new vectors through crossover + mutation
        while len(new_population) < self.population_size:
            # Select two random parents from elite
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)

            # Crossover: blend parameters
            child = self._crossover(parent1, parent2)

            # Mutation: perturb parameters
            child = self._mutate(child)

            new_population.append(child)

        # Replace population
        state.population = new_population[:self.population_size]
        state.increment_generation()

    def should_gossip(self, node_id: int) -> bool:
        """Check if this node should gossip this round.

        Args:
            node_id: Node to check

        Returns:
            True if should gossip
        """
        state = self.get_node_state(node_id)
        return state.gossip_round_count >= self.gossip_frequency

    def create_gossip_message(
        self, node_id: int, graph: GraphManager
    ) -> GossipMessage | None:
        """Create gossip message with elite vectors.

        Args:
            node_id: Node creating gossip
            graph: Graph (used to find neighbors)

        Returns:
            GossipMessage with subtype="parameter" or None if no elite to share
        """
        state = self.get_node_state(node_id)

        # Get elite vectors
        elite = state.get_elite_k(k=self.elite_k)
        if not elite:
            return None

        # Get fitness values
        fitness_vals = [state.fitness.get(id(v), 0.0) for v in elite]

        # Create message using GossipMessage factory
        msg = GossipMessage.parameter_gossip(
            sender_node_id=node_id,
            elite_vectors=elite,
            fitness_values=fitness_vals,
            generation=state.generation,
        )

        return msg

    def send_gossip_to_neighbors(
        self,
        node_id: int,
        graph: GraphManager,
        message_queue: MessageQueue,
        round_num: RoundNumber,
    ) -> int:
        """Send gossip messages to random neighbors.

        Args:
            node_id: Node sending gossip
            graph: Graph (used to find neighbors)
            message_queue: Message queue to send through
            round_num: Current round number

        Returns:
            Number of gossip messages sent
        """
        state = self.get_node_state(node_id)

        # Create gossip message
        gossip_msg = self.create_gossip_message(node_id, graph)
        if not gossip_msg:
            return 0

        # Select random neighbors (at most 2)
        neighbors = list(graph.neighbors(node_id))
        if not neighbors:
            return 0

        target_neighbors = random.sample(neighbors, min(2, len(neighbors)))

        # Send gossip to each neighbor
        for neighbor in target_neighbors:
            msg = Message(
                sender=node_id,
                recipient=neighbor,
                payload=gossip_msg,
                round_num=round_num,
            )
            message_queue.send(msg)

        # Reset gossip counter
        state.reset_gossip_round_count()

        return len(target_neighbors)

    def receive_and_integrate_gossip(
        self, node_id: int, gossip_messages: List[GossipMessage]
    ) -> int:
        """Receive gossip from neighbors and integrate into population.

        Args:
            node_id: Node receiving gossip
            gossip_messages: List of GossipMessage with subtype="parameter" received

        Returns:
            Number of new vectors integrated
        """
        state = self.get_node_state(node_id)

        integrated_count = 0
        for gossip_msg in gossip_messages:
            for vector in gossip_msg.elite_vectors:
                if vector not in state.elite_pool:
                    state.add_elite_from_gossip([vector])
                    integrated_count += 1

        # Merge elite pool into population
        if integrated_count > 0:
            state.merge_elite_into_population(elite_pool_size=self.elite_k)

        return integrated_count

    def increment_gossip_round(self, node_id: int) -> None:
        """Increment gossip round counter.

        Args:
            node_id: Node to increment
        """
        state = self.get_node_state(node_id)
        state.increment_gossip_round_count()

    def get_best_vector(self, node_id: int) -> CanonicalVector:
        """Get best vector found by this node.

        Args:
            node_id: Node to query

        Returns:
            Best CanonicalVector found
        """
        state = self.get_node_state(node_id)
        if state.best_local is None:
            # Return first vector if nothing evaluated yet
            return state.population[0] if state.population else CanonicalVector()
        return state.best_local

    def get_best_fitness(self, node_id: int) -> float:
        """Get best fitness found by this node.

        Args:
            node_id: Node to query

        Returns:
            Best fitness value
        """
        state = self.get_node_state(node_id)
        return state.best_local_fitness

    # ===== PRIVATE HELPER METHODS =====

    def _crossover(
        self, parent1: CanonicalVector, parent2: CanonicalVector
    ) -> CanonicalVector:
        """Blend two parent vectors using registry-driven approach (100% agnostic).

        Dynamically iterates over all parameters from AlgorithmRegistry instead
        of hardcoding parameter names. Works with any algorithm combination.

        Args:
            parent1: First parent vector
            parent2: Second parent vector

        Returns:
            Child vector with blended parameters
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        blend_factor = random.random()
        registry = AlgorithmRegistry.instance()
        blended_params = {}

        # Iterate over all algorithms and their parameters
        for algo_name in registry.all_algorithm_names():
            algo_params = registry.get_algorithm_parameters_unprefixed(algo_name)

            for param_name, (min_val, max_val, _) in algo_params.items():
                full_param_name = f"{algo_name}_{param_name}"

                # Get values from parents
                parent1_val = parent1.get(full_param_name)
                parent2_val = parent2.get(full_param_name)

                # Skip if either parent doesn't have this parameter
                if parent1_val is None or parent2_val is None:
                    continue

                # Blend: weighted average of parents
                blended_val = parent1_val * blend_factor + parent2_val * (1 - blend_factor)

                # Handle integer parameters
                if isinstance(min_val, int):
                    blended_params[full_param_name] = int(blended_val)
                else:
                    blended_params[full_param_name] = blended_val

        # Add base parameters (not algorithm-specific)
        base_params = ["max_iterations", "convergence_threshold"]
        for param in base_params:
            parent1_val = parent1.get(param)
            parent2_val = parent2.get(param)
            if parent1_val is not None and parent2_val is not None:
                blended_val = parent1_val * blend_factor + parent2_val * (1 - blend_factor)
                if param == "max_iterations":
                    blended_params[param] = int(blended_val)
                else:
                    blended_params[param] = blended_val

        # Create child from blended parameters
        return CanonicalVector.from_dict({**parent1.to_dict(), **blended_params})

    def _mutate(self, vector: CanonicalVector) -> CanonicalVector:
        """Apply random mutations to a vector using parameter bounds from registry (100% agnostic).

        Dynamically reads parameter bounds from AlgorithmRegistry instead of hardcoding.
        Works with any algorithm combination and parameter definition.

        Args:
            vector: Vector to mutate

        Returns:
            Mutated vector
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        mutation_strength = 0.1  # 10% change per parameter
        registry = AlgorithmRegistry.instance()
        mutated_params = {}

        def mutate_float_param(value: float, lower: float, upper: float) -> float:
            """Mutate a float parameter with bounds."""
            if random.random() < self.mutation_rate:
                delta = random.gauss(0, mutation_strength)
                return max(lower, min(upper, value + delta))
            return value

        def mutate_int_param(value: int, lower: int, upper: int) -> int:
            """Mutate an integer parameter with bounds."""
            if random.random() < self.mutation_rate:
                delta = random.randint(-2, 2)
                return max(lower, min(upper, value + delta))
            return value

        # Mutate algorithm-specific parameters using bounds from registry
        for algo_name in registry.all_algorithm_names():
            algo_params = registry.get_algorithm_parameters_unprefixed(algo_name)

            for param_name, (min_val, max_val, _) in algo_params.items():
                full_param_name = f"{algo_name}_{param_name}"
                current_val = vector.get(full_param_name)

                if current_val is None:
                    continue

                # Use appropriate mutation based on parameter type
                if isinstance(min_val, int):
                    mutated_params[full_param_name] = mutate_int_param(
                        int(current_val), int(min_val), int(max_val)
                    )
                else:
                    mutated_params[full_param_name] = mutate_float_param(
                        float(current_val), float(min_val), float(max_val)
                    )

        # Mutate base parameters using fixed bounds
        base_bounds = {
            "max_iterations": (5, 100, True),      # (min, max, is_int)
            "convergence_threshold": (0.0, 0.1, False),
        }

        for param_name, (min_val, max_val, is_int) in base_bounds.items():
            current_val = vector.get(param_name)
            if current_val is None:
                continue

            if is_int:
                mutated_params[param_name] = mutate_int_param(
                    int(current_val), int(min_val), int(max_val)
                )
            else:
                mutated_params[param_name] = mutate_float_param(
                    float(current_val), float(min_val), float(max_val)
                )

        # Create mutated vector
        return CanonicalVector.from_dict({**vector.to_dict(), **mutated_params})

    def name(self) -> str:
        """Return component name."""
        return "DistributedParameterEvolver"
