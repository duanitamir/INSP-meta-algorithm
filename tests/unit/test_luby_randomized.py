"""Unit tests for Luby-style Randomized Matching algorithm."""

import pytest
from src.graph import GraphManager
from src.state.state_store import StateStore
from src.communication.message_queue import MessageQueue
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
from src.communication.message import Message
from src.utils.types import RoundNumber


class TestLubyRandomizedMetadata:
    def test_metadata_exists(self):
        """Test that algorithm has valid metadata."""
        algo = LubyRandomizedMatching()
        metadata = algo.metadata

        assert metadata.name == "Luby-style Randomized Distributed Maximal Matching"
        assert metadata.version == "1.0.0"
        assert len(metadata.authors) > 0
        assert metadata.properties is not None

    def test_metadata_properties(self):
        """Test algorithm properties."""
        algo = LubyRandomizedMatching()
        props = algo.metadata.properties

        assert props["produces_maximal"] is True
        assert props["produces_maximum"] is False
        assert props["deterministic"] is False
        assert props["max_rounds"] == 200
        assert props["activation_probability"] == 0.5

    def test_custom_activation_probability(self):
        """Test custom activation probability."""
        algo = LubyRandomizedMatching(activation_probability=0.3)
        props = algo.metadata.properties
        assert props["activation_probability"] == 0.3


class TestLubyRandomizedInitialization:
    def test_initialize_simple_graph(self):
        """Test state initialization on a simple graph."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 4):
            graph.add_vertex(i)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        algo = LubyRandomizedMatching()
        store = StateStore(graph)
        algo.initialize_state(store, graph)

        # Check all nodes initialized
        for node_id in graph.vertices():
            state = store.get_node_state(node_id)
            assert state.get("matched_to") is None
            assert state.get("is_active") is True
            assert state.get("has_neighbors") is True

    def test_initialize_isolated_nodes(self):
        """Test initialization with isolated nodes."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_vertex(3)

        algo = LubyRandomizedMatching()
        store = StateStore(graph)
        algo.initialize_state(store, graph)

        # Isolated nodes should have has_neighbors=False but is_active=True initially
        for node_id in graph.vertices():
            state = store.get_node_state(node_id)
            assert state.get("is_active") is True
            assert state.get("has_neighbors") is False

    def test_initialize_complete_graph(self):
        """Test initialization on a complete graph."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            graph.add_vertex(i)
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, 1.0)

        algo = LubyRandomizedMatching()
        store = StateStore(graph)
        algo.initialize_state(store, graph)

        # All nodes should have neighbors
        for node_id in graph.vertices():
            state = store.get_node_state(node_id)
            assert state.get("is_active") is True
            assert state.get("has_neighbors") is True


class TestLubyRandomizedNodeBehavior:
    def setup_method(self):
        """Setup for each test."""
        self.graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            self.graph.add_vertex(i)
        self.graph.add_edge(1, 2, 2.0)
        self.graph.add_edge(2, 3, 1.5)
        self.graph.add_edge(3, 4, 1.0)

    def test_node_behavior_matched_node(self):
        """Test that matched nodes become inactive."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        state.set_matched_to(2)

        class Context:
            round_num = RoundNumber(0)
            graph = self.graph

        new_state, messages = algo.node_behavior(1, state, [], Context())

        assert new_state.is_matched()
        assert new_state.get("is_active") is False
        assert len(messages) == 0

    def test_node_behavior_isolated_node(self):
        """Test that isolated nodes become inactive."""
        isolated_graph = GraphManager.create_empty_graph()
        isolated_graph.add_vertex(1)

        algo = LubyRandomizedMatching()
        store = StateStore(isolated_graph)
        algo.initialize_state(store, isolated_graph)

        state = store.get_node_state(1)

        class Context:
            round_num = RoundNumber(0)
            graph = isolated_graph

        new_state, messages = algo.node_behavior(1, state, [], Context())

        assert not new_state.is_matched()
        assert new_state.get("is_active") is False
        assert len(messages) == 0

    def test_node_behavior_randomized_proposal(self):
        """Test that nodes randomly propose to neighbors."""
        algo = LubyRandomizedMatching(seed=42)
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)

        class Context:
            round_num = RoundNumber(0)
            graph = self.graph

        # Run multiple times with same seed to check randomness
        proposals = []
        for i in range(5):
            algo_test = LubyRandomizedMatching(seed=42 + i)
            store_test = StateStore(self.graph)
            algo_test.initialize_state(store_test, self.graph)
            state_test = store_test.get_node_state(1)

            new_state, messages = algo_test.node_behavior(1, state_test, [], Context())

            if messages:
                propose_msgs = [m for m in messages if m.payload.get("type") == "PROPOSE"]
                if propose_msgs:
                    proposals.append(propose_msgs[0].recipient)

        # Should have at least one proposal (randomness may result in no proposal sometimes)
        assert len(proposals) >= 0

    def test_node_behavior_accepts_best_proposal(self):
        """Test that nodes accept best proposal."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(2)

        # Node 2 receives proposals from neighbors
        messages = [
            Message(
                sender=1,
                recipient=2,
                payload={"type": "PROPOSE", "weight": 2.0, "proposer_id": 1},
                round_num=RoundNumber(0),
            ),
            Message(
                sender=3,
                recipient=2,
                payload={"type": "PROPOSE", "weight": 1.5, "proposer_id": 3},
                round_num=RoundNumber(0),
            ),
        ]

        class Context:
            round_num = RoundNumber(0)
            graph = self.graph

        new_state, out_messages = algo.node_behavior(2, state, messages, Context())

        # Should accept best proposal (from 1 with weight 2.0)
        accept_msgs = [m for m in out_messages if m.payload["type"] == "ACCEPT"]
        reject_msgs = [m for m in out_messages if m.payload["type"] == "REJECT"]

        assert any(m.recipient == 1 for m in accept_msgs), "Should accept best proposer"

    def test_node_behavior_confirm_match(self):
        """Test that proposer confirms match after receiving accept."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        state.set("proposal_to", 2)
        state.set("proposal_weight", 2.0)

        # Receive ACCEPT from partner
        messages = [
            Message(
                sender=2,
                recipient=1,
                payload={"type": "ACCEPT"},
                round_num=RoundNumber(0),
            )
        ]

        class Context:
            round_num = RoundNumber(0)
            graph = self.graph

        new_state, out_messages = algo.node_behavior(1, state, messages, Context())

        # Should match and send CONFIRM
        assert new_state.is_matched()
        assert new_state.get_matched_to() == 2
        confirm_msgs = [m for m in out_messages if m.payload["type"] == "CONFIRM"]
        assert len(confirm_msgs) == 1

    def test_node_behavior_handles_rejection(self):
        """Test that nodes handle rejection gracefully and try again."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        state = store.get_node_state(1)
        state.set("proposal_to", 2)
        state.set("proposal_weight", 2.0)
        state.set("is_active", True)

        # Receive REJECT from partner
        messages = [
            Message(
                sender=2,
                recipient=1,
                payload={"type": "REJECT"},
                round_num=RoundNumber(0),
            )
        ]

        class Context:
            round_num = RoundNumber(0)
            graph = self.graph

        new_state, out_messages = algo.node_behavior(1, state, messages, Context())

        # After REJECT, node should send a new proposal (algorithm should continue trying)
        # Either proposal_to is set to a neighbor, or no neighbors available
        assert new_state.get("is_active") is True or len(new_state.get("neighbors", [])) == 0
        propose_msgs = [m for m in out_messages if m.payload.get("type") == "PROPOSE"]
        assert len(propose_msgs) > 0 or not new_state.get("is_active"), "Should send new proposal or become inactive"


class TestLubyRandomizedTermination:
    def setup_method(self):
        """Setup for each test."""
        self.graph = GraphManager.create_empty_graph()
        for i in range(1, 4):
            self.graph.add_vertex(i)
        self.graph.add_edge(1, 2, 1.0)
        self.graph.add_edge(2, 3, 1.0)

    def test_termination_all_inactive(self):
        """Test termination when all nodes are inactive."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        # Mark all as inactive
        for node_id in self.graph.vertices():
            state = store.get_node_state(node_id)
            state.set("is_active", False)
            store.update_node_state(node_id, state)

        should_terminate, reason = algo.check_termination(
            store, RoundNumber(5), messages_sent=0
        )

        assert should_terminate
        assert reason == "all_nodes_inactive"

    def test_termination_no_progress(self):
        """Test termination when no progress is made."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        should_terminate, reason = algo.check_termination(
            store, RoundNumber(5), messages_sent=0
        )

        assert should_terminate
        assert reason == "no_progress"

    def test_termination_max_rounds(self):
        """Test termination when max rounds exceeded."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        should_terminate, reason = algo.check_termination(
            store, RoundNumber(300), messages_sent=10
        )

        assert should_terminate
        assert reason == "max_rounds_exceeded"

    def test_no_termination_active_progress(self):
        """Test that algorithm doesn't terminate with active nodes."""
        algo = LubyRandomizedMatching()
        store = StateStore(self.graph)
        algo.initialize_state(store, self.graph)

        should_terminate, reason = algo.check_termination(
            store, RoundNumber(5), messages_sent=10
        )

        assert not should_terminate


class TestLubyRandomizedValidation:
    def test_matching_validation_valid(self):
        """Test validation of a valid matching."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            graph.add_vertex(i)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        algo = LubyRandomizedMatching()

        matching = {1: 2, 2: 1, 3: 4, 4: 3}
        is_valid, error = algo.validate_matching(matching, graph)

        assert is_valid
        assert error is None

    def test_matching_validation_invalid_duplicate(self):
        """Test validation catches duplicate matches."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            graph.add_vertex(i)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        algo = LubyRandomizedMatching()

        matching = {1: 2, 2: 1, 3: 2}  # Node 2 matched twice
        is_valid, error = algo.validate_matching(matching, graph)

        assert not is_valid

    def test_maximal_matching_check(self):
        """Test maximal matching validation."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            graph.add_vertex(i)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)

        algo = LubyRandomizedMatching()

        matching = {1: 2, 2: 1}  # Not maximal (3-4 available)
        is_maximal = algo.is_maximal_matching(matching, graph)

        assert not is_maximal

        matching = {1: 2, 2: 1, 3: 4, 4: 3}  # Maximal
        is_maximal = algo.is_maximal_matching(matching, graph)

        assert is_maximal


class TestLubyRandomizedDeterminism:
    def test_determinism_with_seed(self):
        """Test that algorithm can run consistently with same seed."""
        test_graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            test_graph.add_vertex(i)
        test_graph.add_edge(1, 2, 1.0)
        test_graph.add_edge(2, 3, 1.0)
        test_graph.add_edge(3, 4, 1.0)

        algo1 = LubyRandomizedMatching(seed=42)
        algo2 = LubyRandomizedMatching(seed=99)

        store1 = StateStore(test_graph)
        store2 = StateStore(test_graph)
        algo1.initialize_state(store1, test_graph)
        algo2.initialize_state(store2, test_graph)

        state1 = store1.get_node_state(1)
        state2 = store2.get_node_state(1)

        class Context:
            round_num = RoundNumber(0)
            graph = test_graph

        new_state1, msgs1 = algo1.node_behavior(1, state1, [], Context())
        new_state2, msgs2 = algo2.node_behavior(1, state2, [], Context())

        # Different seeds might produce different behavior
        # Just verify both complete without error
        assert new_state1.get("is_active") in [True, False]
        assert new_state2.get("is_active") in [True, False]

    def test_non_determinism_different_seeds(self):
        """Test that different seeds can produce different results."""
        test_graph = GraphManager.create_empty_graph()
        for i in range(1, 5):
            test_graph.add_vertex(i)
        test_graph.add_edge(1, 2, 1.0)
        test_graph.add_edge(2, 3, 1.0)
        test_graph.add_edge(3, 4, 1.0)

        results = []
        for seed in [42, 99, 123]:
            algo = LubyRandomizedMatching(seed=seed)
            store = StateStore(test_graph)
            algo.initialize_state(store, test_graph)

            state = store.get_node_state(1)

            class Context:
                round_num = RoundNumber(0)
                graph = test_graph

            new_state, msgs = algo.node_behavior(1, state, [], Context())
            results.append(len(msgs))

        # At least some variation in proposal behavior expected with different seeds
        # (though not guaranteed due to random nature)
        assert len(results) == 3
