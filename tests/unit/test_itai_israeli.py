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


class TestItaiIsraeliStageOperations:
    """Test individual stage operations of the algorithm."""

    def setup_method(self):
        """Setup for each test."""
        self.graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            self.graph.add_vertex(i)
        self.graph.add_edge(1, 2, 2.0)
        self.graph.add_edge(2, 3, 1.5)
        self.graph.add_edge(3, 4, 1.0)

    def test_stage_accept_sends_confirm(self):
        """Test stage_accept sends CONFIRM message."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        context = AlgorithmContext(self.graph, store, RoundNumber(0))
        node_id = 1
        partner = 2

        messages = algo.stage_accept(partner, node_id, context)

        assert len(messages) == 1
        assert messages[0].sender == node_id
        assert messages[0].recipient == partner
        assert messages[0].payload["type"] == "CONFIRM"

    def test_stage_accept_propose_sends_accept(self):
        """Test stage_accept_propose sends ACCEPT message."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        context = AlgorithmContext(self.graph, store, RoundNumber(0))
        node_id = 2
        proposer_id = 1

        messages = algo.stage_accept_propose(proposer_id, node_id, context)

        assert len(messages) == 1
        assert messages[0].sender == node_id
        assert messages[0].recipient == proposer_id
        assert messages[0].payload["type"] == "ACCEPT"

    def test_stage_generate_propose_sends_propose(self):
        """Test stage_generate_propose sends PROPOSE to a neighbor."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        context = AlgorithmContext(self.graph, store, RoundNumber(0))
        node_id = 1
        neighbors = [2]  # Node 1's only neighbor

        messages = algo.stage_generate_propose(neighbors, node_id, context)

        assert len(messages) == 1
        assert messages[0].sender == node_id
        assert messages[0].recipient == 2
        assert messages[0].payload["type"] == "PROPOSE"
        assert messages[0].payload["weight"] == 2.0
        assert messages[0].payload["proposer_id"] == node_id

    def test_stage_generate_propose_picks_best_neighbor(self):
        """Test stage_generate_propose picks highest-weight neighbor."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        context = AlgorithmContext(self.graph, store, RoundNumber(0))
        node_id = 2
        neighbors = [1, 3]  # Node 2 has neighbors 1 (weight 2.0) and 3 (weight 1.5)

        messages = algo.stage_generate_propose(neighbors, node_id, context)

        assert len(messages) == 1
        assert messages[0].recipient == 1  # Should pick node 1 (weight 2.0 > 1.5)
        assert messages[0].payload["type"] == "PROPOSE"
        assert messages[0].payload["weight"] == 2.0

    def test_stage_timeout_no_timeout(self):
        """Test stage_timeout returns unchanged state when timeout not reached."""
        algo = ItaiIsraeliMaximalMatching(timeout_rounds=5)
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        state.set("negotiation_partner", 2)
        state.set("stage_round", 2)

        new_state, partner = algo.stage_timeout(state, 2)

        assert partner == 2
        assert new_state.get("negotiation_partner") == 2
        assert new_state.get("stage_round") == 2

    def test_stage_timeout_timeout_reached(self):
        """Test stage_timeout resets when timeout threshold exceeded."""
        algo = ItaiIsraeliMaximalMatching(timeout_rounds=5)
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        state.set("negotiation_partner", 2)
        state.set("negotiation_stage", "proposed")
        state.set("stage_round", 5)  # At timeout threshold

        new_state, partner = algo.stage_timeout(state, 2)

        assert partner is None
        assert new_state.get("negotiation_partner") is None
        assert new_state.get("negotiation_stage") is None
        assert new_state.get("stage_round") == 0

    def test_stage_timeout_no_partner(self):
        """Test stage_timeout with no active negotiation."""
        algo = ItaiIsraeliMaximalMatching(timeout_rounds=5)
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)

        new_state, partner = algo.stage_timeout(state, None)

        assert partner is None
        assert new_state.get("negotiation_partner") is None

    def test_stage_confirm_finalizes_match(self):
        """Test stage_confirm finalizes a match."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        partner = 2

        new_state = algo.stage_confirm(state, partner)

        assert new_state.is_matched()
        assert new_state.get_matched_to() == partner
        assert new_state.get("status") == "matched"
        assert new_state.get("active") is False


class TestItaiIsraeliPrivateHelpers:
    """Test private helper methods."""

    def setup_method(self):
        """Setup for each test."""
        self.graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            self.graph.add_vertex(i)
        self.graph.add_edge(1, 2, 2.0)
        self.graph.add_edge(2, 3, 1.5)
        self.graph.add_edge(3, 4, 1.0)

    def test_match_nodes_sets_matched_state(self):
        """Test _match_nodes correctly sets matched state."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        partner = 2

        new_state = algo._match_nodes(state, partner)

        assert new_state.is_matched()
        assert new_state.get_matched_to() == partner
        assert new_state.get("status") == "matched"
        assert new_state.get("active") is False
        assert new_state.get("negotiation_partner") is None
        assert new_state.get("negotiation_stage") is None
        assert new_state.get("stage_round") == 0

    def test_start_negotiation_sets_state(self):
        """Test _start_negotiation sets negotiation state."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        partner = 2

        new_state = algo._start_negotiation(state, partner, "proposed")

        assert new_state.get("negotiation_partner") == partner
        assert new_state.get("negotiation_stage") == "proposed"
        assert new_state.get("status") == "proposing"
        assert new_state.get("stage_round") == 0
        assert new_state.get("active") is True

    def test_start_negotiation_accepting_stage(self):
        """Test _start_negotiation with accepting stage."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        partner = 2

        new_state = algo._start_negotiation(state, partner, "accepting")

        assert new_state.get("negotiation_stage") == "accepting"
        assert new_state.get("status") == "accepting"

    def test_find_best_propose_highest_weight(self):
        """Test _find_best_propose selects highest weight."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)

        # Create proposals with different weights
        proposals = [
            Message(1, 2, {"type": "PROPOSE", "weight": 1.0, "proposer_id": 1}, RoundNumber(0)),
            Message(3, 2, {"type": "PROPOSE", "weight": 2.5, "proposer_id": 3}, RoundNumber(0)),
            Message(4, 2, {"type": "PROPOSE", "weight": 1.5, "proposer_id": 4}, RoundNumber(0)),
        ]

        best = algo._find_best_propose(proposals)

        assert best.sender == 3
        assert best.payload["weight"] == 2.5

    def test_find_best_propose_tie_breaking_by_proposer_id(self):
        """Test _find_best_propose tie-breaks by proposer_id (higher wins)."""
        algo = ItaiIsraeliMaximalMatching()
        store = StateStore(self.graph)

        # Create proposals with same weight
        proposals = [
            Message(1, 2, {"type": "PROPOSE", "weight": 2.0, "proposer_id": 1}, RoundNumber(0)),
            Message(4, 2, {"type": "PROPOSE", "weight": 2.0, "proposer_id": 4}, RoundNumber(0)),
            Message(3, 2, {"type": "PROPOSE", "weight": 2.0, "proposer_id": 3}, RoundNumber(0)),
        ]

        best = algo._find_best_propose(proposals)

        assert best.sender == 4
        assert best.payload["proposer_id"] == 4

    def test_find_best_propose_single_proposal(self):
        """Test _find_best_propose with single proposal."""
        algo = ItaiIsraeliMaximalMatching()

        proposals = [
            Message(1, 2, {"type": "PROPOSE", "weight": 1.0, "proposer_id": 1}, RoundNumber(0)),
        ]

        best = algo._find_best_propose(proposals)

        assert best.sender == 1
        assert best.payload["weight"] == 1.0
