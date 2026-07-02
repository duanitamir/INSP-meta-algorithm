"""Enhanced NodeService with distributed parameter evolution.

This version adds parameter learning to the node service. Each node evolves
its own CanonicalVector based on matching results and learns from neighbors
via gossip. No central trainer or pre-computed parameters needed.
"""

from typing import Dict, Any, Tuple, List
from src.node_service import NodeService
from src.node_parameter_controller import NodeParameterController
from src.communication.transport_driver import TransportDriver
from src.communication.drivers.in_memory_driver import InMemoryDriver
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector


class NodeServiceWithLearning(NodeService):
    """NodeService enhanced with distributed parameter learning.

    Adds local GA and gossip-based parameter evolution to the base NodeService.
    Each node evolves its own CanonicalVector independent of other nodes.
    """

    def __init__(
        self,
        node_id: int,
        graph: GraphManager,
        transport_driver: TransportDriver = None,
        max_rounds: int = 1000,
        enable_learning: bool = True,
        population_size: int = 10,
        gossip_frequency: int = 5,
    ):
        """Initialize node service with learning.

        Args:
            node_id: Unique node identifier
            graph: Shared read-only graph
            transport_driver: Message transport (default: in-memory)
            max_rounds: Maximum execution rounds
            enable_learning: Enable parameter learning
            population_size: GA population size
            gossip_frequency: Gossip elite vectors every N rounds
        """
        super().__init__(node_id, graph, transport_driver, max_rounds)

        self.enable_learning = enable_learning
        self.gossip_frequency = gossip_frequency
        self.gossip_round_counter = 0

        # Initialize parameter controller
        self.parameter_controller = NodeParameterController(
            node_id=node_id,
            population_size=population_size,
            generations_per_round=2,
            mutation_rate=0.1,
            elite_fraction=0.5,
        )

        # Current vector being used
        self.current_vector = self.parameter_controller.current_population[0]
        self.current_vector_idx = 0

    def execute_round(self, algorithm) -> Tuple[bool, str]:
        """Execute one round with parameter learning.

        Args:
            algorithm: MatchingAlgorithm to run

        Returns:
            (continue_running, status_message)
        """
        if self.finished or self.rounds_executed >= self.max_rounds:
            return False, "max_rounds_exceeded"

        # Execute with current vector
        should_continue, status = self.node.execute(algorithm)

        # Send/receive messages
        if self.node.outbox:
            messages = self.node.outbox.get_messages(self.node_id)
            self.transport.send_batch(messages)

        received = self.transport.receive_messages(self.node_id)
        if received:
            for msg in received:
                self.node.inbox.send(msg)

        self.rounds_executed += 1

        # Learning phase: Update fitness and gossip
        if self.enable_learning:
            self._learning_phase()

        if not should_continue:
            self.finished = True
            return False, status

        return True, "round_completed"

    def _learning_phase(self) -> None:
        """Execute parameter learning phase.

        Updates fitness for current vector and optionally gossips with neighbors.
        """
        # Get matching weight as fitness
        matching = self.node.get_matching()
        fitness = sum(1 for _ in matching.items())  # Simple: count matched pairs

        # Update fitness for current vector
        self.parameter_controller.update_fitness(self.current_vector_idx, fitness)

        # Every N rounds: evolve and gossip
        self.gossip_round_counter += 1
        if self.gossip_round_counter >= self.gossip_frequency:
            self.gossip_round_counter = 0

            # Evolve population locally
            for _ in range(2):  # 2 generations per gossip round
                self.parameter_controller.evolve_one_generation()

            # Gossip elite vectors (would send via transport in real network)
            elite_vectors = self.parameter_controller.get_elite_vectors_to_share(count=3)
            # In real network, would send via self.transport
            # For now, just update local best
            self.current_vector = self.parameter_controller.get_best_vector()

            # Switch to best vector for next round
            best_vector = self.parameter_controller.get_best_vector()
            # Find best in population
            best_idx = 0
            best_fitness = -1
            for i, vec in enumerate(self.parameter_controller.current_population):
                fitness = getattr(vec, "_fitness", -1)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_idx = i

            self.current_vector_idx = best_idx
            self.current_vector = self.parameter_controller.current_population[best_idx]

    def run_with_learning(self, algorithm, max_learning_rounds: int = 100) -> Dict[str, Any]:
        """Run with continuous parameter learning.

        Args:
            algorithm: MatchingAlgorithm to run
            max_learning_rounds: Maximum rounds of learning

        Returns:
            Execution results dict
        """
        status = "running"
        learning_round = 0

        while (
            not self.finished
            and self.rounds_executed < self.max_rounds
            and learning_round < max_learning_rounds
        ):
            should_continue, status = self.execute_round(algorithm)
            if not should_continue:
                self.finished = True
                break

            # Check convergence
            if self.enable_learning and self.parameter_controller.has_converged():
                learning_round += 1

        return {
            "node_id": self.node_id,
            "rounds_executed": self.rounds_executed,
            "finished": self.finished,
            "final_status": status,
            "best_vector": self.parameter_controller.get_best_vector(),
            "best_fitness": self.parameter_controller.best_fitness,
            "generations_evolved": self.parameter_controller.generation_counter,
            "parameter_controller_summary": self.parameter_controller.summary(),
        }

    def get_parameters_for_export(self) -> Dict[str, Any]:
        """Get best parameters discovered by this node.

        Can be used to share optimal parameters with other nodes or for
        analysis after execution.

        Returns:
            Dict with best_vector and fitness
        """
        best_vector = self.parameter_controller.get_best_vector()
        return {
            "node_id": self.node_id,
            "best_vector": best_vector,
            "best_fitness": self.parameter_controller.best_fitness,
            "generation_counter": self.parameter_controller.generation_counter,
        }
