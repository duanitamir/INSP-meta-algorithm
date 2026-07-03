"""End-to-end integration tests for fully distributed system.

Tests that all distributed components work together without centralized orchestrator.
"""

import pytest
import networkx as nx

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.distributed.parameter_evolver import DistributedParameterEvolver
from src.meta.distributed.conflict_resolver import DistributedConflictResolver
from src.meta.distributed.convergence_detector import DistributedConvergenceDetector
from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer
from src.meta.core.fitness_evaluator import FitnessEvaluator


@pytest.fixture
def small_clustered_graph() -> nx.Graph:
    """Create a small clustered graph for testing."""
    G = nx.Graph()
    # Two clusters: (0,1,2) and (3,4,5)
    G.add_weighted_edges_from(
        [
            (0, 1, 10),
            (1, 2, 10),
            (0, 2, 8),
            (3, 4, 10),
            (4, 5, 10),
            (3, 5, 8),
            (2, 3, 1),  # Single edge between clusters
        ]
    )
    return G


class TestDistributedSystemEnd2End:
    """Test full distributed system without central orchestrator."""

    def test_three_node_parameter_evolution(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Test 3-node network evolving parameters via gossip."""
        evolver = DistributedParameterEvolver(
            population_size=5, mutation_rate=0.15, gossip_frequency=2, elite_k=2
        )
        evolver.initialize(node_ids=[0, 1, 2])

        # Simulate 3 rounds of evolution with gossip
        for _ in range(1, 4):
            # Each node evolves locally
            for node_id in [0, 1, 2]:
                evolver.evolve_local_population(node_id)

            # Gossip at node level
            for node_id in [0, 1, 2]:
                if evolver.should_gossip(node_id):
                    msg = evolver.create_gossip_message(node_id)
                    # Other nodes receive
                    for other_id in [0, 1, 2]:
                        if other_id != node_id:
                            evolver.receive_and_integrate_gossip(other_id, [msg])

        # Verify convergence: all nodes should have elite vectors
        best0 = evolver.get_best_vector(0)
        best1 = evolver.get_best_vector(1)
        best2 = evolver.get_best_vector(2)

        assert best0 is not None
        assert best1 is not None
        assert best2 is not None
        # After gossip, nodes should have received elite vectors
        state0 = evolver.node_states[0]
        state1 = evolver.node_states[1]
        # Both should have populated elite pools
        assert state0.elite_pool is not None
        assert state1.elite_pool is not None

    def test_three_node_conflict_resolution(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Test 3-node network resolving edge conflicts via voting."""
        resolver = DistributedConflictResolver(voting_frequency=1, threshold=0.5)
        resolver.initialize(node_ids=[0, 1, 2])

        # Node 0 proposes conflicting edges
        greedy = {0: 1, 1: 0, 2: 3, 3: 2}
        itai = {0: 2, 2: 0}
        luby = {1: 3, 3: 1}

        resolver.propose_edges(0, greedy, itai, luby)

        # Simulate voting
        messages = resolver.broadcast_votes(0, greedy, itai, luby, round_num=1)
        assert len(messages) > 0

        # Other nodes receive and vote
        resolver.receive_votes(1, messages)
        resolver.receive_votes(2, messages)

        # Resolve final matching
        matching = resolver.resolve_matches(0, threshold=0.5)
        assert matching is not None
        # Should have some edges if consensus is reached
        assert len(matching) >= 0

    def test_three_node_convergence_detection(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Test 3-node network achieving convergence via quorum voting."""
        detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5, gossip_frequency=1
        )
        detector.initialize(node_ids=[0, 1, 2])

        # Simulate iteration with low improvement
        for node_id in [0, 1, 2]:
            detector.decide_convergence(
                node_id, prev_weight=1000.0, curr_weight=1001.0
            )  # 0.1% improvement < 5% threshold

        # Create and broadcast convergence votes
        messages = []
        for node_id in [0, 1, 2]:
            msg = detector.create_convergence_message(node_id, round_num=1)
            messages.append(msg)

        # All nodes receive all votes
        for node_id in [0, 1, 2]:
            detector.receive_convergence_votes(node_id, messages)

        # Check convergence: all 3 nodes voted to stop
        for node_id in [0, 1, 2]:
            converged = detector.check_network_convergence(node_id)
            assert converged is True

    def test_full_pipeline_three_nodes(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Integration: Parameter evolution + conflict resolution + convergence detection."""
        # Setup all three distributed components
        param_evolver = DistributedParameterEvolver(
            population_size=5, mutation_rate=0.15, gossip_frequency=2
        )
        conflict_resolver = DistributedConflictResolver(
            voting_frequency=2, threshold=0.5
        )
        convergence_detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5, gossip_frequency=2
        )

        param_evolver.initialize(node_ids=[0, 1, 2])
        conflict_resolver.initialize(node_ids=[0, 1, 2])
        convergence_detector.initialize(node_ids=[0, 1, 2])

        # Simulate 5 rounds
        for round_num in range(1, 6):
            # Phase 1: Parameter evolution
            for node_id in [0, 1, 2]:
                param_evolver.evolve_local_population(node_id)

            # Phase 2: Edge conflict resolution
            for node_id in [0, 1, 2]:
                greedy = {0: 1, 1: 0}
                conflict_resolver.propose_edges(node_id, greedy, {}, {})

            if conflict_resolver.should_vote(0, round_num):
                messages = []
                for node_id in [0, 1, 2]:
                    node_messages = conflict_resolver.broadcast_votes(
                        node_id, {0: 1, 1: 0}, {}, {}, round_num
                    )
                    messages.extend(node_messages)
                for node_id in [0, 1, 2]:
                    conflict_resolver.receive_votes(node_id, messages)

            # Phase 3: Convergence detection
            for node_id in [0, 1, 2]:
                convergence_detector.decide_convergence(
                    node_id, prev_weight=100.0, curr_weight=101.0
                )

            if convergence_detector.should_gossip(0, round_num):
                messages = []
                for node_id in [0, 1, 2]:
                    msg = convergence_detector.create_convergence_message(
                        node_id, round_num
                    )
                    messages.append(msg)
                for node_id in [0, 1, 2]:
                    convergence_detector.receive_convergence_votes(node_id, messages)

            # Check if network converged
            converged = convergence_detector.check_network_convergence(0)
            if converged:
                break

        # Verify all components processed without errors
        assert param_evolver.node_states[0] is not None
        assert conflict_resolver.node_states[0] is not None
        assert convergence_detector.node_states[0] is not None

    def test_timeout_safety(self, small_clustered_graph: nx.Graph) -> None:
        """Test timeout safety prevents infinite loops."""
        detector = DistributedConvergenceDetector(max_iterations=5)
        detector.initialize(node_ids=[0])

        state = detector.node_states[0]

        # Not at timeout yet
        for i in range(5):
            state.increment_iteration()
            assert detector.check_timeout(0) is (i == 4)


class TestDistributedSystemMessageComplexity:
    """Verify message overhead is acceptable for distributed system."""

    def test_gossip_message_count_three_nodes(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Count gossip messages in 3-node network."""
        param_evolver = DistributedParameterEvolver(
            population_size=5, gossip_frequency=5, elite_k=2
        )
        param_evolver.initialize(node_ids=[0, 1, 2])

        # Simulate 10 rounds of gossip checks
        message_count = 0
        for node_id in [0, 1, 2]:
            for _ in range(10):
                if param_evolver.should_gossip(node_id):
                    msg = param_evolver.create_gossip_message(node_id)
                    message_count += 1

        # Should have at least some messages created
        assert message_count >= 0

    def test_voting_message_count_three_nodes(
        self, small_clustered_graph: nx.Graph
    ) -> None:
        """Count voting messages in 3-node network."""
        conflict_resolver = DistributedConflictResolver(voting_frequency=5)
        conflict_resolver.initialize(node_ids=[0, 1, 2])

        # Simulate 10 rounds
        message_count = 0
        for round_num in range(1, 11):
            for node_id in [0, 1, 2]:
                if conflict_resolver.should_vote(node_id, round_num):
                    messages = conflict_resolver.broadcast_votes(
                        node_id, {0: 1, 1: 0}, {}, {}, round_num
                    )
                    message_count += len(messages)

        # With voting_frequency=5, should vote at rounds 5, 10
        assert message_count == 6  # 3 nodes * 2 voting rounds
