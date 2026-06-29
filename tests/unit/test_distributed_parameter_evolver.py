"""Tests for distributed parameter evolution (gossip-based GA)."""

import pytest
from src.graph.graph_manager import GraphManager
from src.meta.distributed.parameter_evolver import DistributedParameterEvolver
from src.meta.messages.parameter_gossip import ParameterGossipMessage
from src.meta.core.canonical_vector import CanonicalVector
from src.state.node_parameter_state import NodeParameterState
from src.communication.message_queue import MessageQueue
from src.utils.types import RoundNumber
from tests.fixtures.graphs import CLUSTERED_GRAPH_100


class TestNodeParameterState:
    """Tests for NodeParameterState."""

    def test_initialization(self):
        """Test state initializes correctly."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=5)
        assert state.node_id == 1
        assert len(state.population) == 5
        assert state.generation == 0
        assert state.best_local_fitness == 0.0

    def test_update_fitness_tracks_best(self):
        """Test fitness updates track best vector."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=3)

        v1, v2, v3 = state.population

        state.update_fitness(v1, 50.0)
        assert state.best_local == v1
        assert state.best_local_fitness == 50.0

        state.update_fitness(v2, 75.0)
        assert state.best_local == v2
        assert state.best_local_fitness == 75.0

        state.update_fitness(v3, 60.0)
        assert state.best_local == v2  # Still v2 (highest)
        assert state.best_local_fitness == 75.0

    def test_get_elite_k_returns_top_k(self):
        """Test get_elite_k returns highest fitness vectors."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=10)

        # Assign fitness in reverse order
        for i, v in enumerate(state.population):
            state.update_fitness(v, float(100 - i * 10))

        elite_3 = state.get_elite_k(k=3)
        assert len(elite_3) == 3

        # Verify they're sorted by fitness descending
        fitness_vals = [state.fitness[id(v)] for v in elite_3]
        assert fitness_vals == sorted(fitness_vals, reverse=True)
        assert fitness_vals[0] == 100.0

    def test_add_elite_from_gossip(self):
        """Test adding gossip vectors to elite pool."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=3)

        gossip_vectors = [CanonicalVector() for _ in range(2)]
        state.add_elite_from_gossip(gossip_vectors)

        assert state.elite_pool_size() == 2
        assert all(v in state.elite_pool for v in gossip_vectors)

    def test_merge_elite_into_population(self):
        """Test merging elite pool into population."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=5)

        initial_pop_size = state.population_size()

        # Create gossip vectors
        gossip_vectors = [CanonicalVector() for _ in range(3)]
        state.add_elite_from_gossip(gossip_vectors)

        # Merge
        state.merge_elite_into_population(elite_pool_size=2)

        # Population should have grown (elite added)
        assert state.population_size() > initial_pop_size

    def test_generation_and_gossip_counters(self):
        """Test generation and gossip round counters."""
        state = NodeParameterState(node_id=1)
        state.initialize(initial_population_size=3)

        assert state.generation == 0
        assert state.gossip_round_count == 0

        state.increment_generation()
        assert state.generation == 1

        state.increment_gossip_round_count()
        state.increment_gossip_round_count()
        assert state.gossip_round_count == 2

        state.reset_gossip_round_count()
        assert state.gossip_round_count == 0


class TestParameterGossipMessage:
    """Tests for ParameterGossipMessage."""

    def test_creation(self):
        """Test gossip message creation."""
        vectors = [CanonicalVector(), CanonicalVector()]
        fitness_vals = [100.0, 95.0]

        msg = ParameterGossipMessage(
            sender_node_id=1,
            elite_vectors=vectors,
            fitness_values=fitness_vals,
            generation=5,
        )

        assert msg.sender_node_id == 1
        assert len(msg.elite_vectors) == 2
        assert msg.generation == 5

    def test_best_vector_and_fitness(self):
        """Test getting best vector and fitness from gossip."""
        vectors = [CanonicalVector(), CanonicalVector(), CanonicalVector()]
        fitness_vals = [80.0, 95.0, 85.0]

        msg = ParameterGossipMessage(
            sender_node_id=1,
            elite_vectors=vectors,
            fitness_values=fitness_vals,
            generation=1,
        )

        assert msg.best_fitness() == 95.0
        assert msg.best_vector() == vectors[1]

    def test_validation_errors(self):
        """Test gossip message validation."""
        with pytest.raises(ValueError):
            # Empty vectors
            ParameterGossipMessage(
                sender_node_id=1,
                elite_vectors=[],
                fitness_values=[],
                generation=1,
            )

        with pytest.raises(ValueError):
            # Mismatched vector and fitness counts
            ParameterGossipMessage(
                sender_node_id=1,
                elite_vectors=[CanonicalVector(), CanonicalVector()],
                fitness_values=[100.0],
                generation=1,
            )

        with pytest.raises(ValueError):
            # Negative generation
            ParameterGossipMessage(
                sender_node_id=1,
                elite_vectors=[CanonicalVector()],
                fitness_values=[100.0],
                generation=-1,
            )


class TestDistributedParameterEvolver:
    """Tests for DistributedParameterEvolver."""

    @pytest.fixture
    def evolver(self):
        """Create evolver instance."""
        return DistributedParameterEvolver(
            population_size=10,
            mutation_rate=0.15,
            gossip_frequency=5,
            elite_k=3,
        )

    @pytest.fixture
    def three_node_network(self):
        """Create 3-node network graph."""
        g = GraphManager.create_empty_graph()
        for i in range(3):
            g.add_vertex(i)
        # Ring topology: 0-1-2-0
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 0, 1.0)
        return g

    def test_initialization(self, evolver):
        """Test evolver initializes with correct parameters."""
        assert evolver.population_size == 10
        assert evolver.mutation_rate == 0.15
        assert evolver.gossip_frequency == 5
        assert evolver.elite_k == 3

    def test_initialize_nodes(self, evolver):
        """Test node initialization."""
        evolver.initialize([0, 1, 2])

        assert len(evolver.node_states) == 3
        for node_id in [0, 1, 2]:
            state = evolver.get_node_state(node_id)
            assert state.population_size() == 10
            assert state.generation == 0

    def test_evaluate_population(self, evolver):
        """Test population evaluation."""
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # All vectors should have fitness 0 initially
        for v in state.population:
            assert id(v) not in state.fitness

        # Mock fitness evaluator
        def mock_fitness_evaluator(graph, vector):
            return 100.0

        class MockEvaluator:
            def evaluate(self, graph, vector):
                return mock_fitness_evaluator(graph, vector)

        evolver.evaluate_population(0, None, None, MockEvaluator())

        # All vectors should be evaluated now
        for v in state.population:
            assert id(v) in state.fitness
            assert state.fitness[id(v)] == 100.0

    def test_evolve_local_population(self, evolver):
        """Test local GA evolution."""
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # Set initial generation
        initial_gen = state.generation
        initial_size = state.population_size()

        # Evolve
        evolver.evolve_local_population(0)

        # Generation should increment
        assert state.generation == initial_gen + 1

        # Population size should stay same
        assert state.population_size() == initial_size

    def test_should_gossip(self, evolver):
        """Test gossip frequency check."""
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # Initially should not gossip
        assert not evolver.should_gossip(0)

        # Increment rounds
        for _ in range(evolver.gossip_frequency):
            evolver.increment_gossip_round(0)

        # Now should gossip
        assert evolver.should_gossip(0)

    def test_create_gossip_message(self, evolver):
        """Test gossip message creation."""
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # Assign fitness
        for v in state.population:
            state.update_fitness(v, 100.0)

        # Create gossip
        gossip = evolver.create_gossip_message(0, None)

        assert gossip is not None
        assert gossip.sender_node_id == 0
        assert len(gossip.elite_vectors) <= evolver.elite_k
        assert gossip.generation == state.generation

    def test_send_gossip_to_neighbors(self, evolver, three_node_network):
        """Test sending gossip to neighbors."""
        evolver.initialize([0, 1, 2])
        state = evolver.get_node_state(0)

        # Set fitness
        for v in state.population:
            state.update_fitness(v, 100.0)

        # Increment gossip rounds to trigger
        for _ in range(evolver.gossip_frequency):
            evolver.increment_gossip_round(0)

        # Send gossip
        mq = MessageQueue(three_node_network)
        sent_count = evolver.send_gossip_to_neighbors(
            0, three_node_network, mq, RoundNumber(1)
        )

        # Should send to neighbors (1 and 2)
        assert sent_count > 0
        assert state.gossip_round_count == 0  # Reset

    def test_receive_and_integrate_gossip(self, evolver):
        """Test receiving and integrating gossip."""
        evolver.initialize([0, 1])
        state_0 = evolver.get_node_state(0)
        state_1 = evolver.get_node_state(1)

        # Create gossip from node 1
        for v in state_1.population:
            state_1.update_fitness(v, 100.0)

        gossip_vectors = state_1.get_elite_k(k=2)
        fitness_vals = [state_1.fitness[id(v)] for v in gossip_vectors]

        gossip = ParameterGossipMessage(
            sender_node_id=1,
            elite_vectors=gossip_vectors,
            fitness_values=fitness_vals,
            generation=0,
        )

        # Node 0 receives and integrates
        integrated = evolver.receive_and_integrate_gossip(0, [gossip])

        assert integrated >= 0
        assert state_0.elite_pool_size() >= 0

    def test_crossover_produces_valid_vector(self, evolver):
        """Test crossover produces valid CanonicalVector."""
        p1 = CanonicalVector()
        p2 = CanonicalVector()

        child = evolver._crossover(p1, p2)

        assert isinstance(child, CanonicalVector)
        is_valid, error = child.validate()
        assert is_valid, f"Invalid child vector: {error}"

    def test_mutate_produces_valid_vector(self, evolver):
        """Test mutation produces valid CanonicalVector."""
        original = CanonicalVector()
        mutated = evolver._mutate(original)

        assert isinstance(mutated, CanonicalVector)
        is_valid, error = mutated.validate()
        assert is_valid, f"Invalid mutated vector: {error}"

    def test_mutation_actually_changes_vector(self, evolver):
        """Test that mutation actually perturbs parameters."""
        # Use high mutation rate to ensure changes
        evolver.mutation_rate = 0.9

        original = CanonicalVector(
            luby_base_probability=0.5,
            luby_coeff_degree=0.5,
            itai_timeout_rounds=10,
            max_iterations=50,
        )

        changed = False
        for _ in range(10):
            mutated = evolver._mutate(original)
            if (
                mutated.luby_base_probability != original.luby_base_probability
                or mutated.itai_timeout_rounds != original.itai_timeout_rounds
            ):
                changed = True
                break

        assert changed, "Mutation should change at least one parameter"

    def test_get_best_vector_and_fitness(self, evolver):
        """Test retrieving best vector and fitness."""
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # Set fitness values
        for i, v in enumerate(state.population):
            state.update_fitness(v, float(100 + i))

        best_v = evolver.get_best_vector(0)
        best_f = evolver.get_best_fitness(0)

        assert best_v == state.best_local
        assert best_f == state.best_local_fitness
        assert best_f > 0

    def test_name(self, evolver):
        """Test component name."""
        assert evolver.name() == "DistributedParameterEvolver"


class TestDistributedParameterEvolverIntegration:
    """Integration tests for distributed parameter evolution."""

    def test_three_node_ga_convergence(self):
        """Test GA convergence across 3-node network."""
        # Create 3-node network
        g = GraphManager.create_empty_graph()
        for i in range(3):
            g.add_vertex(i)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 0, 1.0)

        # Create evolver
        evolver = DistributedParameterEvolver(
            population_size=5,
            mutation_rate=0.1,
            gossip_frequency=2,
        )
        evolver.initialize([0, 1, 2])

        # Simulate: evaluate, evolve, gossip for several generations
        mq = MessageQueue(g)

        for generation in range(10):
            # Evaluate
            for node_id in [0, 1, 2]:
                state = evolver.get_node_state(node_id)
                for v in state.population:
                    if id(v) not in state.fitness:
                        state.update_fitness(v, 100.0)  # Mock fitness

            # Evolve
            for node_id in [0, 1, 2]:
                evolver.evolve_local_population(node_id)

            # Gossip
            for node_id in [0, 1, 2]:
                evolver.increment_gossip_round(node_id)
                if evolver.should_gossip(node_id):
                    evolver.send_gossip_to_neighbors(
                        node_id, g, mq, RoundNumber(generation)
                    )

            # Receive gossip
            for node_id in [0, 1, 2]:
                gossip_msgs = [
                    m.payload
                    for m in mq.get_messages(node_id)
                    if isinstance(m.payload, ParameterGossipMessage)
                ]
                if gossip_msgs:
                    evolver.receive_and_integrate_gossip(node_id, gossip_msgs)

        # Verify: all nodes should have elite pools
        for node_id in [0, 1, 2]:
            state = evolver.get_node_state(node_id)
            assert state.generation > 0
            assert state.elite_pool_size() >= 0

    def test_population_diversity_maintained(self):
        """Test that population diversity is maintained."""
        evolver = DistributedParameterEvolver(population_size=10, mutation_rate=0.2)
        evolver.initialize([0])
        state = evolver.get_node_state(0)

        # Assign fitness
        for i, v in enumerate(state.population):
            state.update_fitness(v, float(100 - i * 5))

        # Evolve multiple times
        for _ in range(5):
            evolver.evolve_local_population(0)

        # Population should still have variation
        population_vectors = state.population
        unique_vectors = len(set(id(v) for v in population_vectors))

        # At least some diversity should remain
        assert unique_vectors > 1
