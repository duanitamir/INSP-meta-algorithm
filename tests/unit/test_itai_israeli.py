import pytest
from src.graph import GraphManager
from src.algorithms.implementations import ItaiIsraeliMaximalMatching
from src.simulation import Scheduler
from src.simulation.algorithm_context import AlgorithmContext
from src.state import StateStore
from src.communication import Message
from src.utils.types import RoundNumber


class TestItaiIsraeliAlgorithm:
    def test_algorithm_metadata(self):
        algo = ItaiIsraeliMaximalMatching()
        assert algo.metadata.name == "Itai-Israeli Distributed Maximal Matching"
        assert algo.metadata.version == "1.0.0"
        assert algo.metadata.properties["produces_maximal"]

    def test_initialize_state(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(simple_graph)
        algo.initialize_state(store, simple_graph)

        for node_id in simple_graph.vertices():
            state = store.get_node_state(node_id)
            assert state.get("status") == "free"
            assert not state.is_matched()
            assert state.get("active") == (simple_graph.degree(node_id) > 0)

    def test_node_proposal_generation(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(simple_graph)
        algo.initialize_state(store, simple_graph)

        context = AlgorithmContext(simple_graph, store, RoundNumber(0))
        node_state = store.get_node_state(1)

        # No messages, should generate proposal
        new_state, out_messages = algo.node_behavior(1, node_state, [], context)

        assert len(out_messages) > 0
        assert out_messages[0].payload["type"] == "PROPOSE"

    def test_matching_creation(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(simple_graph)
        algo.initialize_state(store, simple_graph)

        # Simulate a proposal and acceptance
        context_node1 = AlgorithmContext(simple_graph, store, RoundNumber(0))
        node1_state = store.get_node_state(1)

        # Node 1 sends proposal
        new_state1, _ = algo.node_behavior(1, node1_state, [], context_node1)

        # Node 2 receives proposal
        context_node2 = AlgorithmContext(simple_graph, store, RoundNumber(0))
        node2_state = store.get_node_state(2)

        proposal_msg = Message(1, 2, {"type": "PROPOSE"}, RoundNumber(0))
        new_state2, accept_msgs = algo.node_behavior(
            2, node2_state, [proposal_msg], context_node2
        )

        assert new_state2.is_matched()
        assert new_state2.get_matched_to() == 1
        assert len(accept_msgs) > 0

    def test_matching_validation(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()

        # Valid matching
        matching = {1: 2, 2: 1, 3: 4, 4: 3}
        is_valid, error = algo.validate_matching(matching, simple_graph)
        assert is_valid

        # Invalid: matched to non-existent edge
        invalid_matching = {1: 3, 3: 1}
        is_valid, error = algo.validate_matching(invalid_matching, simple_graph)
        assert not is_valid

    def test_maximal_matching_check(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()

        matching = {1: 2, 2: 1, 3: 4, 4: 3}
        is_maximal = algo.is_maximal_matching(matching, simple_graph)
        assert is_maximal

    def test_algorithm_termination(self, simple_graph):
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(simple_graph)
        algo.initialize_state(store, simple_graph)

        # Mark all nodes as matched (done)
        for node_id in simple_graph.vertices():
            state = store.get_node_state(node_id)
            state.set("active", False)
            store.update_node_state(node_id, state)

        should_terminate, reason = algo.check_termination(
            store, RoundNumber(1), 0
        )
        assert should_terminate
        assert reason == "all_nodes_inactive"
