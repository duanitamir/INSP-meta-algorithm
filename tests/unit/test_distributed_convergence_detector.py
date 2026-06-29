"""Tests for distributed convergence detection via quorum voting."""

import pytest
from src.meta.distributed.convergence_detector import DistributedConvergenceDetector
from src.meta.messages.convergence_gossip import ConvergenceGossipMessage
from src.state.convergence_state import ConvergenceState


class TestConvergenceState:
    """Tests for ConvergenceState."""

    def test_initialization(self) -> None:
        """Test state initialization."""
        state = ConvergenceState(node_id=1)
        assert state.node_id == 1
        assert state.iteration_count == 0
        assert state.prev_weight == 0.0
        assert state.curr_weight == 0.0
        assert state.improvement == 0.0
        assert state.convergence_votes == {}
        assert state.total_nodes == 0
        assert state.should_stop is False

    def test_update_weights(self) -> None:
        """Test updating iteration weights."""
        state = ConvergenceState(node_id=1)
        state.update_weights(prev_weight=100.0, curr_weight=105.0)
        assert state.prev_weight == 100.0
        assert state.curr_weight == 105.0
        assert pytest.approx(state.improvement) == 0.05  # 5/100 = 0.05

    def test_compute_improvement_zero_previous_weight(self) -> None:
        """Test improvement computation with zero previous weight."""
        state = ConvergenceState(node_id=1)
        state.update_weights(prev_weight=0.0, curr_weight=10.0)
        # When prev is 0, improvement is 1.0 (special case)
        assert state.improvement == 1.0

    def test_decide_convergence_above_threshold(self) -> None:
        """Test convergence decision when improvement is above threshold."""
        state = ConvergenceState(node_id=1)
        state.update_weights(prev_weight=100.0, curr_weight=110.0)
        vote = state.decide_convergence(convergence_threshold=0.05)
        # 10/100 = 0.1 > 0.05, so should not stop
        assert vote is False

    def test_decide_convergence_below_threshold(self) -> None:
        """Test convergence decision when improvement is below threshold."""
        state = ConvergenceState(node_id=1)
        state.update_weights(prev_weight=100.0, curr_weight=101.0)
        vote = state.decide_convergence(convergence_threshold=0.05)
        # 1/100 = 0.01 < 0.05, so should stop
        assert vote is True

    def test_add_convergence_vote(self) -> None:
        """Test recording convergence votes from other nodes."""
        state = ConvergenceState(node_id=1)
        state.add_convergence_vote(node_id=0, vote=True)
        assert state.convergence_votes[0] is True

    def test_has_quorum_consensus_stop(self) -> None:
        """Test quorum detection when >50% vote to stop."""
        state = ConvergenceState(node_id=1, total_nodes=3)
        state.add_convergence_vote(0, True)
        state.add_convergence_vote(1, True)
        state.add_convergence_vote(2, False)
        # 2/3 > 50%, so consensus to stop
        assert state.has_quorum_consensus(consensus_threshold=0.5) is True

    def test_has_quorum_consensus_continue(self) -> None:
        """Test quorum detection when <50% vote to stop."""
        state = ConvergenceState(node_id=1, total_nodes=3)
        state.add_convergence_vote(0, False)
        state.add_convergence_vote(1, False)
        state.add_convergence_vote(2, True)
        # 1/3 < 50%, so consensus to continue
        assert state.has_quorum_consensus(consensus_threshold=0.5) is False

    def test_has_quorum_consensus_exact_threshold(self) -> None:
        """Test quorum at exact threshold boundary (should not trigger)."""
        state = ConvergenceState(node_id=1, total_nodes=2)
        state.add_convergence_vote(0, True)
        state.add_convergence_vote(1, False)
        # 1/2 = 50%, needs > 50% to trigger
        assert state.has_quorum_consensus(consensus_threshold=0.5) is False

    def test_reset_votes(self) -> None:
        """Test resetting convergence votes for new round."""
        state = ConvergenceState(node_id=1)
        state.add_convergence_vote(0, True)
        state.add_convergence_vote(1, True)
        assert len(state.convergence_votes) == 2
        state.reset_votes()
        assert len(state.convergence_votes) == 0

    def test_increment_iteration(self) -> None:
        """Test incrementing iteration counter."""
        state = ConvergenceState(node_id=1)
        assert state.iteration_count == 0
        state.increment_iteration()
        assert state.iteration_count == 1
        state.increment_iteration()
        assert state.iteration_count == 2


class TestConvergenceGossipMessage:
    """Tests for ConvergenceGossipMessage."""

    def test_message_creation(self) -> None:
        """Test creating a convergence message."""
        msg = ConvergenceGossipMessage(
            sender_node_id=1, should_stop=True, round_num=5, weight=100.0
        )
        assert msg.sender_node_id == 1
        assert msg.should_stop is True
        assert msg.round_num == 5
        assert msg.weight == 100.0

    def test_message_immutable(self) -> None:
        """Test message is immutable."""
        msg = ConvergenceGossipMessage(
            sender_node_id=1, should_stop=True, round_num=5, weight=100.0
        )
        with pytest.raises(AttributeError):
            msg.should_stop = False  # type: ignore

    def test_message_negative_round_invalid(self) -> None:
        """Test validation: negative round invalid."""
        with pytest.raises(ValueError):
            ConvergenceGossipMessage(
                sender_node_id=1, should_stop=True, round_num=-1, weight=100.0
            )

    def test_message_negative_weight_invalid(self) -> None:
        """Test validation: negative weight invalid."""
        with pytest.raises(ValueError):
            ConvergenceGossipMessage(
                sender_node_id=1, should_stop=True, round_num=5, weight=-10.0
            )


class TestDistributedConvergenceDetector:
    """Tests for DistributedConvergenceDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = DistributedConvergenceDetector()
        assert detector.node_states == {}

    def test_initialize_nodes(self) -> None:
        """Test initializing multiple nodes."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0, 1, 2])
        assert 0 in detector.node_states
        assert 1 in detector.node_states
        assert 2 in detector.node_states

    def test_should_gossip_convergence(self) -> None:
        """Test gossip frequency check for convergence."""
        detector = DistributedConvergenceDetector(gossip_frequency=5)
        detector.initialize(node_ids=[0, 1])

        # First few rounds: no gossip
        assert detector.should_gossip(0, round_num=1) is False
        assert detector.should_gossip(0, round_num=4) is False
        # At gossip_frequency: gossip
        assert detector.should_gossip(0, round_num=5) is True

    def test_decide_convergence_node(self) -> None:
        """Test deciding convergence at a single node."""
        detector = DistributedConvergenceDetector(convergence_threshold=0.05)
        detector.initialize(node_ids=[0])

        # High improvement: should continue
        detector.decide_convergence(0, prev_weight=100.0, curr_weight=115.0)
        state = detector.node_states[0]
        assert state.should_stop is False

        # Low improvement: should stop
        detector.decide_convergence(0, prev_weight=100.0, curr_weight=100.5)
        state = detector.node_states[0]
        assert state.should_stop is True

    def test_create_convergence_message(self) -> None:
        """Test creating convergence gossip message."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0])

        detector.decide_convergence(0, prev_weight=100.0, curr_weight=101.0)
        msg = detector.create_convergence_message(0, round_num=5)

        assert msg.sender_node_id == 0
        assert isinstance(msg.should_stop, bool)
        assert msg.round_num == 5

    def test_broadcast_convergence_votes(self) -> None:
        """Test broadcasting convergence votes to neighbors."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0, 1, 2])

        # Node 0 decides to stop
        detector.decide_convergence(0, prev_weight=100.0, curr_weight=100.1)
        messages = detector.broadcast_convergence_votes(0, round_num=5)

        assert len(messages) >= 1
        assert messages[0].sender_node_id == 0

    def test_receive_convergence_votes(self) -> None:
        """Test receiving and tallying convergence votes."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0, 1, 2])

        # Create convergence messages
        msg1 = ConvergenceGossipMessage(
            sender_node_id=0, should_stop=True, round_num=5, weight=100.0
        )
        msg2 = ConvergenceGossipMessage(
            sender_node_id=1, should_stop=False, round_num=5, weight=100.0
        )

        detector.receive_convergence_votes(0, [msg1, msg2])
        state = detector.node_states[0]

        assert 0 in state.convergence_votes
        assert 1 in state.convergence_votes

    def test_check_network_convergence_with_quorum(self) -> None:
        """Test detecting network-wide convergence with quorum."""
        detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5
        )
        detector.initialize(node_ids=[0, 1, 2])

        # All nodes vote to stop
        msg1 = ConvergenceGossipMessage(
            sender_node_id=0, should_stop=True, round_num=5, weight=100.0
        )
        msg2 = ConvergenceGossipMessage(
            sender_node_id=1, should_stop=True, round_num=5, weight=100.0
        )
        msg3 = ConvergenceGossipMessage(
            sender_node_id=2, should_stop=True, round_num=5, weight=100.0
        )

        detector.receive_convergence_votes(0, [msg1, msg2, msg3])
        converged = detector.check_network_convergence(0)

        assert converged is True

    def test_check_network_convergence_without_quorum(self) -> None:
        """Test that convergence requires quorum."""
        detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5
        )
        detector.initialize(node_ids=[0, 1, 2])

        # Only 1 node votes to stop (need >50%)
        msg1 = ConvergenceGossipMessage(
            sender_node_id=0, should_stop=True, round_num=5, weight=100.0
        )
        msg2 = ConvergenceGossipMessage(
            sender_node_id=1, should_stop=False, round_num=5, weight=100.0
        )

        detector.receive_convergence_votes(0, [msg1, msg2])
        converged = detector.check_network_convergence(0)

        assert converged is False

    def test_timeout_safety(self) -> None:
        """Test timeout safety: max iterations stops regardless of convergence."""
        detector = DistributedConvergenceDetector(max_iterations=10)
        detector.initialize(node_ids=[0])

        state = detector.node_states[0]
        for _ in range(9):
            state.increment_iteration()

        # Not yet at max
        assert detector.check_timeout(0) is False

        # At max iterations
        state.increment_iteration()
        assert detector.check_timeout(0) is True

    def test_multiple_nodes_independent_state(self) -> None:
        """Test that each node maintains independent convergence state."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0, 1, 2])

        # Different weights per node
        detector.decide_convergence(0, prev_weight=100.0, curr_weight=115.0)  # Continue
        detector.decide_convergence(1, prev_weight=100.0, curr_weight=100.1)  # Stop

        state0 = detector.node_states[0]
        state1 = detector.node_states[1]

        assert state0.should_stop is False
        assert state1.should_stop is True

    def test_reset_convergence_votes_per_round(self) -> None:
        """Test that convergence votes are reset each gossip round."""
        detector = DistributedConvergenceDetector()
        detector.initialize(node_ids=[0, 1])

        # Round 1: receive votes
        msg1 = ConvergenceGossipMessage(
            sender_node_id=0, should_stop=True, round_num=5, weight=100.0
        )
        detector.receive_convergence_votes(0, [msg1])
        assert len(detector.node_states[0].convergence_votes) == 1

        # Reset votes for next round
        detector.node_states[0].reset_votes()
        assert len(detector.node_states[0].convergence_votes) == 0

    def test_convergence_threshold_tuning(self) -> None:
        """Test that convergence threshold is tunable."""
        detector_strict = DistributedConvergenceDetector(convergence_threshold=0.05)
        detector_loose = DistributedConvergenceDetector(convergence_threshold=0.1)

        detector_strict.initialize(node_ids=[0])
        detector_loose.initialize(node_ids=[1])

        # Improvement of 2% (0.02)
        detector_strict.decide_convergence(0, prev_weight=100.0, curr_weight=102.0)
        detector_loose.decide_convergence(1, prev_weight=100.0, curr_weight=102.0)

        # Strict detector should stop (0.02 < 0.05), loose should also stop (0.02 < 0.1)
        assert detector_strict.node_states[0].should_stop is True
        assert detector_loose.node_states[1].should_stop is True

        # Test with higher improvement (15%)
        detector_strict2 = DistributedConvergenceDetector(convergence_threshold=0.1)
        detector_strict2.initialize(node_ids=[2])
        detector_strict2.decide_convergence(2, prev_weight=100.0, curr_weight=115.0)
        # 15% > 10% threshold, so should not stop (continue improving)
        assert detector_strict2.node_states[2].should_stop is False

    def test_integration_three_node_convergence(self) -> None:
        """Integration test: 3-node network achieving convergence."""
        detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5
        )
        detector.initialize(node_ids=[0, 1, 2])

        # Simulate iteration where all nodes have low improvement
        for node_id in [0, 1, 2]:
            detector.decide_convergence(node_id, prev_weight=1000.0, curr_weight=1001.0)

        # Gossip: create messages
        messages = []
        for node_id in [0, 1, 2]:
            msg = detector.create_convergence_message(node_id, round_num=5)
            messages.append(msg)

        # All nodes receive all votes
        detector.receive_convergence_votes(0, messages)

        # Check convergence
        converged = detector.check_network_convergence(0)
        assert converged is True
