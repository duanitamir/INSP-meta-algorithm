"""Unit tests for DistributedNode - autonomous distributed execution."""

import pytest
from src.graph.graph_manager import GraphManager
from src.simulation.distributed_node import DistributedNode
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.communication.message import Message
from src.state.node_state_store_adapter import NodeStateStoreAdapter


@pytest.fixture
def simple_graph():
    """Create a simple 4-node graph: 0-1-2-3."""
    graph = GraphManager()
    for i in range(4):
        graph.add_vertex(i)
    graph.add_edge(0, 1, weight=10)
    graph.add_edge(1, 2, weight=20)
    graph.add_edge(2, 3, weight=15)
    return graph


@pytest.fixture
def greedy_algorithm():
    """Create Greedy matching algorithm."""
    return GreedyMatching()


class TestDistributedNodeInitialization:
    """Test DistributedNode creation and setup."""

    def test_node_creation(self, simple_graph):
        """Should create node with required attributes."""
        node = DistributedNode(0, simple_graph)
        assert node.id == 0
        assert node.graph is simple_graph
        assert node.round_number == 0
        assert not node.finished

    def test_node_has_state(self, simple_graph):
        """Should have NodeState object."""
        node = DistributedNode(0, simple_graph)
        assert node.state is not None
        assert node.state.node_id == 0

    def test_node_has_communication(self, simple_graph):
        """Should have inbox and outbox."""
        node = DistributedNode(0, simple_graph)
        assert node.inbox is not None
        assert node.outbox is not None

    def test_node_has_metrics(self, simple_graph):
        """Should have local metrics collector."""
        node = DistributedNode(0, simple_graph)
        assert node.local_metrics is not None

    def test_node_convergence_defaults(self, simple_graph):
        """Should have convergence tracking initialized."""
        node = DistributedNode(0, simple_graph)
        assert node.convergence_vote is None
        assert len(node.known_convergence_votes) == 0
        assert node.convergence_threshold == 0.05
        assert node.quorum_threshold == 0.5


class TestDistributedNodeExecution:
    """Test node execution and algorithm running."""

    def test_execute_runs_algorithm(self, simple_graph, greedy_algorithm):
        """Should execute algorithm and return status."""
        from src.state.node_state_store_adapter import NodeStateStoreAdapter
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        continue_running, status = node.execute(greedy_algorithm)

        # Should return execution status
        assert isinstance(continue_running, bool)
        assert isinstance(status, str)
        assert node.round_number == 1

    def test_execute_increments_round(self, simple_graph, greedy_algorithm):
        """Should increment round number after execution."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        assert node.round_number == 0
        node.execute(greedy_algorithm)
        assert node.round_number == 1
        node.execute(greedy_algorithm)
        assert node.round_number == 2

    def test_execute_tracks_metrics(self, simple_graph, greedy_algorithm):
        """Should track local metrics during execution."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.execute(greedy_algorithm)

        # Should record metrics
        metrics = node.local_metrics.get_all_metrics()
        assert len(metrics) > 0
        assert metrics[0].round_num == 0

    def test_execute_returns_false_when_finished(self, simple_graph, greedy_algorithm):
        """Should return False if node already finished."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.finished = True
        continue_running, status = node.execute(greedy_algorithm)

        assert not continue_running
        assert status == "already_finished"


class TestDistributedNodeMessaging:
    """Test message sending and receiving."""

    def test_send_message_to_outbox(self, simple_graph):
        """Should send messages to outbox."""
        node = DistributedNode(0, simple_graph)

        msg = Message(
            sender=0,
            recipient=1,
            payload={"type": "TEST"},
            round_num=0
        )
        node.outbox.send(msg)

        # Check outbox has message (messages go to recipient's inbox)
        # So we check recipient node 1's inbox
        outgoing = node.outbox.peek_messages(1)
        assert len(outgoing) == 1
        assert outgoing[0].sender == 0

    def test_receive_message_in_inbox(self, simple_graph):
        """Should receive messages in inbox."""
        node = DistributedNode(0, simple_graph)

        msg = Message(
            sender=1,
            recipient=0,
            payload={"type": "TEST"},
            round_num=0
        )
        node.inbox.send(msg)

        # Check inbox has message
        messages = node.inbox.get_messages(0)
        assert len(messages) == 1
        assert messages[0].sender == 1


class TestDistributedNodeConvergence:
    """Test convergence voting and quorum detection."""

    def test_convergence_vote_initialized_false(self, simple_graph, greedy_algorithm):
        """First round should vote False (continue)."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.execute(greedy_algorithm)

        # First round: no prior weight, should vote False
        assert node.convergence_vote is False

    def test_process_convergence_messages(self, simple_graph):
        """Should extract convergence votes from messages."""
        node = DistributedNode(0, simple_graph)

        # Create convergence vote messages
        msg1 = Message(
            sender=1,
            recipient=0,
            payload={"type": "CONVERGENCE_VOTE", "vote": True},
            round_num=5
        )
        msg2 = Message(
            sender=2,
            recipient=0,
            payload={"type": "CONVERGENCE_VOTE", "vote": False},
            round_num=5
        )

        messages = [msg1, msg2]
        node._process_coordination_messages(messages)

        # Should have learned about both votes
        assert node.known_convergence_votes[1] is True
        assert node.known_convergence_votes[2] is False

    def test_quorum_check_insufficient_votes(self, simple_graph):
        """Should not stop with only 1 stop vote (need >50%)."""
        node = DistributedNode(0, simple_graph)

        # Simulate learning about 2 nodes, only 1 voting to stop
        node.known_convergence_votes = {1: True, 2: False}

        should_stop = node._should_stop_based_on_quorum()
        assert not should_stop

    def test_quorum_check_sufficient_votes(self, simple_graph):
        """Should stop when quorum (>50%) votes to stop."""
        node = DistributedNode(0, simple_graph)
        node.quorum_threshold = 0.5

        # Simulate learning about 3 nodes, 2 voting to stop (67%)
        node.known_convergence_votes = {1: True, 2: True, 3: False}

        should_stop = node._should_stop_based_on_quorum()
        assert should_stop

    def test_quorum_check_empty_votes(self, simple_graph):
        """Should not stop if no votes known yet."""
        node = DistributedNode(0, simple_graph)

        should_stop = node._should_stop_based_on_quorum()
        assert not should_stop

    def test_gossip_convergence_vote(self, simple_graph, greedy_algorithm):
        """Should gossip convergence vote to neighbors."""
        node = DistributedNode(1, simple_graph)  # Node 1 has neighbors 0 and 2
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.execute(greedy_algorithm)
        node._gossip_convergence_vote()

        # Check that messages were sent to some neighbor
        neighbors = list(simple_graph.neighbors(1))
        assert len(neighbors) > 0

        # Should have sent gossip to at least one neighbor
        outgoing_messages_found = False
        for neighbor in neighbors:
            outgoing = node.outbox.peek_messages(neighbor)
            convergence_msgs = [m for m in outgoing if m.payload.get("type") == "CONVERGENCE_VOTE"]
            if len(convergence_msgs) > 0:
                outgoing_messages_found = True
                break

        assert outgoing_messages_found


class TestDistributedNodeState:
    """Test state management."""

    def test_get_matching_when_matched(self, simple_graph):
        """Should return matching dict when matched."""
        node = DistributedNode(0, simple_graph)
        node.state.set_matched_to(1)

        matching = node.get_matching()

        assert matching == {0: 1}

    def test_get_matching_when_unmatched(self, simple_graph):
        """Should return empty dict when unmatched."""
        node = DistributedNode(0, simple_graph)

        matching = node.get_matching()

        assert matching == {}

    def test_metrics_summary(self, simple_graph, greedy_algorithm):
        """Should provide metrics summary for gossip."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.execute(greedy_algorithm)

        summary = node.metrics_summary

        assert "node_id" in summary
        assert "round_number" in summary
        assert "active" in summary
        assert "matched" in summary
        assert "convergence_vote" in summary
        assert summary["node_id"] == 0


class TestDistributedNodeReset:
    """Test reset functionality."""

    def test_reset_clears_state(self, simple_graph, greedy_algorithm):
        """Should reset all state to initial."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        # Execute a few rounds
        node.execute(greedy_algorithm)
        node.execute(greedy_algorithm)
        node.finished = True
        node.convergence_vote = True

        assert node.round_number > 0
        assert node.finished

        # Reset
        node.reset()

        assert node.round_number == 0
        assert not node.finished
        assert node.convergence_vote is None
        assert len(node.known_convergence_votes) == 0

    def test_reset_clears_metrics(self, simple_graph, greedy_algorithm):
        """Should clear metrics on reset."""
        node = DistributedNode(0, simple_graph)
        adapter = NodeStateStoreAdapter(node.state, node.id)
        greedy_algorithm.initialize_state(adapter, simple_graph)

        node.execute(greedy_algorithm)
        node.execute(greedy_algorithm)

        initial_total = node.local_metrics.total_messages

        node.reset()

        assert node.local_metrics.total_messages == 0


class TestDistributedNodeMultipleNodes:
    """Test interactions between multiple nodes."""

    def test_two_nodes_exchange_messages(self, simple_graph):
        """Two nodes should exchange convergence votes."""
        node0 = DistributedNode(0, simple_graph)
        node1 = DistributedNode(1, simple_graph)

        # Node 0 sends convergence vote to node 1
        msg = Message(
            sender=0,
            recipient=1,
            payload={"type": "CONVERGENCE_VOTE", "vote": True},
            round_num=0
        )
        node0.outbox.send(msg)

        # Deliver message from node0 outbox to node1 inbox
        # Note: get_messages retrieves from the recipient's inbox
        messages = node0.outbox.get_messages(1)  # Get messages for recipient 1
        for m in messages:
            node1.inbox.send(m)

        # Node 1 receives and processes
        messages = node1.inbox.get_messages(1)
        node1._process_coordination_messages(messages)

        # Node 1 should know about node 0's vote
        assert 0 in node1.known_convergence_votes
        assert node1.known_convergence_votes[0] is True
