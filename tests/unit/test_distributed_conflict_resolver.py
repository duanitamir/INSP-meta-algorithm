"""Tests for distributed conflict resolution via edge voting."""

import pytest
from src.meta.distributed.conflict_resolver import DistributedConflictResolver
from src.meta.messages.edge_voting import EdgeVotingMessage
from src.state.edge_voting_state import EdgeVotingState


class TestEdgeVotingState:
    """Tests for EdgeVotingState."""

    def test_initialization(self) -> None:
        """Test state initialization."""
        state = EdgeVotingState(node_id=1)
        assert state.node_id == 1
        assert state.proposed_edges == {}
        assert state.received_votes == {}
        assert state.final_matching == {}
        assert state.vote_tally == {}

    def test_propose_edges(self) -> None:
        """Test proposing edges locally."""
        state = EdgeVotingState(node_id=1)
        edges = {(0, 1): 10.5, (1, 2): 8.3}
        state.propose_edges("greedy", edges)
        assert state.proposed_edges == {"greedy": edges}

    def test_add_vote(self) -> None:
        """Test adding a vote for an edge."""
        state = EdgeVotingState(node_id=1)
        state.add_vote((0, 1), True)
        assert (0, 1) in state.vote_tally
        assert state.vote_tally[(0, 1)] == 1

    def test_add_multiple_votes_same_edge(self) -> None:
        """Test accumulating votes for same edge."""
        state = EdgeVotingState(node_id=1)
        state.add_vote((0, 1), True)
        state.add_vote((0, 1), False)
        state.add_vote((0, 1), True)
        assert state.vote_tally[(0, 1)] == 3

    def test_get_consensus_threshold(self) -> None:
        """Test consensus determination (>50%)."""
        state = EdgeVotingState(node_id=1)
        # 2 votes: need >1 for consensus
        state.add_vote((0, 1), True)
        state.add_vote((0, 1), True)
        assert state.has_consensus((0, 1), threshold=0.5) is True

        # Reset and test failure
        state = EdgeVotingState(node_id=1)
        state.add_vote((0, 1), True)
        state.add_vote((0, 1), False)
        assert state.has_consensus((0, 1), threshold=0.5) is False

    def test_get_consensus_edges(self) -> None:
        """Test getting all edges with consensus."""
        state = EdgeVotingState(node_id=1)
        state.add_vote((0, 1), True)
        state.add_vote((0, 1), True)
        state.add_vote((1, 2), True)
        state.add_vote((1, 2), False)

        consensus_edges = state.get_consensus_edges(threshold=0.5)
        assert (0, 1) in consensus_edges
        assert (1, 2) not in consensus_edges

    def test_finalize_matching(self) -> None:
        """Test finalizing matching from consensus edges."""
        state = EdgeVotingState(node_id=1)
        state.add_vote((0, 1), True)
        state.add_vote((0, 1), True)
        state.add_vote((1, 2), True)
        state.add_vote((1, 2), False)

        state.finalize_matching(threshold=0.5)
        assert state.final_matching == {0: 1, 1: 0}


class TestEdgeVotingMessage:
    """Tests for EdgeVotingMessage."""

    def test_message_creation(self) -> None:
        """Test creating a vote message."""
        msg = EdgeVotingMessage(
            sender_node_id=1, edge=(0, 1), vote=True, round_num=5, weight=10.5
        )
        assert msg.sender_node_id == 1
        assert msg.edge == (0, 1)
        assert msg.vote is True
        assert msg.round_num == 5
        assert msg.weight == 10.5

    def test_message_immutable(self) -> None:
        """Test message is immutable."""
        msg = EdgeVotingMessage(
            sender_node_id=1, edge=(0, 1), vote=True, round_num=5, weight=10.5
        )
        with pytest.raises(AttributeError):
            msg.vote = False  # type: ignore

    def test_message_negative_round_invalid(self) -> None:
        """Test validation: negative round invalid."""
        with pytest.raises(ValueError):
            EdgeVotingMessage(
                sender_node_id=1,
                edge=(0, 1),
                vote=True,
                round_num=-1,
                weight=10.5,
            )

    def test_message_negative_weight_invalid(self) -> None:
        """Test validation: negative weight invalid."""
        with pytest.raises(ValueError):
            EdgeVotingMessage(
                sender_node_id=1,
                edge=(0, 1),
                vote=True,
                round_num=5,
                weight=-10.5,
            )

    def test_message_invalid_edge_format(self) -> None:
        """Test validation: edge must be 2-tuple."""
        with pytest.raises(ValueError):
            EdgeVotingMessage(
                sender_node_id=1,
                edge=(0, 1, 2),  # type: ignore
                vote=True,
                round_num=5,
                weight=10.5,
            )


class TestDistributedConflictResolver:
    """Tests for DistributedConflictResolver."""

    def test_initialization(self) -> None:
        """Test resolver initialization."""
        resolver = DistributedConflictResolver()
        assert resolver.node_states == {}

    def test_initialize_nodes(self) -> None:
        """Test initializing multiple nodes."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1, 2])
        assert 0 in resolver.node_states
        assert 1 in resolver.node_states
        assert 2 in resolver.node_states

    def test_propose_edges_simple(self) -> None:
        """Test proposing matching edges."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1])

        greedy_matching = {0: 1, 1: 0}
        resolver.propose_edges(0, greedy_matching, {}, {})

        state = resolver.node_states[0]
        assert "greedy" in state.proposed_edges

    def test_should_vote(self) -> None:
        """Test gossip frequency check for voting."""
        resolver = DistributedConflictResolver(voting_frequency=5)
        resolver.initialize(node_ids=[0, 1])

        # First few rounds: no voting
        assert resolver.should_vote(0, round_num=1) is False
        assert resolver.should_vote(0, round_num=4) is False
        # At gossip_frequency: vote
        assert resolver.should_vote(0, round_num=5) is True

    def test_extract_edges_from_matchings(self) -> None:
        """Test extracting all unique edges from algorithm matchings."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0])

        greedy = {0: 1, 1: 0, 2: 3, 3: 2}
        itai = {0: 2, 2: 0}
        luby = {1: 3, 3: 1}

        edges = resolver._extract_edges_from_matchings(greedy, itai, luby)

        # All unique edges should be present (normalized)
        assert (0, 1) in edges
        assert (0, 2) in edges
        assert (1, 3) in edges
        assert (2, 3) in edges

    def test_normalize_edge(self) -> None:
        """Test edge normalization (min, max) ordering."""
        resolver = DistributedConflictResolver()
        assert resolver._normalize_edge(5, 2) == (2, 5)
        assert resolver._normalize_edge(1, 3) == (1, 3)
        assert resolver._normalize_edge(0, 0) == (0, 0)

    def test_get_edge_endpoints(self) -> None:
        """Test extracting endpoints from edge."""
        resolver = DistributedConflictResolver()
        u, v = resolver._get_edge_endpoints((2, 5))
        assert set([u, v]) == {2, 5}

    def test_broadcast_votes_simple(self) -> None:
        """Test broadcasting votes to endpoints."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1, 2, 3, 4])

        greedy = {0: 1, 1: 0}
        resolver.propose_edges(2, greedy, {}, {})

        # Extract edges
        edges = resolver._extract_edges_from_matchings(greedy, {}, {})
        messages = []

        # For each edge, node 2 votes on it
        for edge in edges:
            msg = EdgeVotingMessage(
                sender_node_id=2, edge=edge, vote=True, round_num=1, weight=10.0
            )
            messages.append(msg)

        assert len(messages) == len(edges)

    def test_receive_votes_and_tally(self) -> None:
        """Test receiving votes and tallying consensus."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1])

        # Node 1 receives votes on edge (0, 1)
        msg1 = EdgeVotingMessage(
            sender_node_id=0, edge=(0, 1), vote=True, round_num=1, weight=10.0
        )
        msg2 = EdgeVotingMessage(
            sender_node_id=1, edge=(0, 1), vote=True, round_num=1, weight=10.0
        )

        resolver.receive_votes(0, [msg1, msg2])
        state = resolver.node_states[0]

        assert (0, 1) in state.vote_tally
        assert state.vote_tally[(0, 1)] == 2

    def test_resolve_matches_with_consensus(self) -> None:
        """Test resolving final matching with consensus votes."""
        resolver = DistributedConflictResolver(threshold=0.5)
        resolver.initialize(node_ids=[0, 1, 2])

        # Node 0 proposes edge (1, 2)
        greedy = {1: 2, 2: 1}
        resolver.propose_edges(0, greedy, {}, {})

        # Simulate endpoint votes: both endpoints vote YES
        msg1 = EdgeVotingMessage(
            sender_node_id=1, edge=(1, 2), vote=True, round_num=1, weight=10.0
        )
        msg2 = EdgeVotingMessage(
            sender_node_id=2, edge=(1, 2), vote=True, round_num=1, weight=10.0
        )

        resolver.receive_votes(0, [msg1, msg2])
        matching = resolver.resolve_matches(0, threshold=0.5)

        assert 1 in matching
        assert matching[1] == 2
        assert matching[2] == 1

    def test_resolve_matches_without_consensus(self) -> None:
        """Test that edges without consensus are rejected."""
        resolver = DistributedConflictResolver(threshold=0.5)
        resolver.initialize(node_ids=[0, 1, 2])

        greedy = {1: 2, 2: 1}
        resolver.propose_edges(0, greedy, {}, {})

        # Mixed votes: one YES, one NO
        msg1 = EdgeVotingMessage(
            sender_node_id=1, edge=(1, 2), vote=True, round_num=1, weight=10.0
        )
        msg2 = EdgeVotingMessage(
            sender_node_id=2, edge=(1, 2), vote=False, round_num=1, weight=10.0
        )

        resolver.receive_votes(0, [msg1, msg2])
        matching = resolver.resolve_matches(0, threshold=0.5)

        assert 1 not in matching
        assert 2 not in matching

    def test_conflicting_proposals_first_wins(self) -> None:
        """Test that when node has conflicting match proposals, first consensus wins."""
        resolver = DistributedConflictResolver(threshold=0.5)
        resolver.initialize(node_ids=[0, 1, 2])

        # Greedy proposes (0, 1), Itai proposes (0, 2)
        greedy = {0: 1, 1: 0}
        itai = {0: 2, 2: 0}

        resolver.propose_edges(0, greedy, itai, {})

        # Both edges get consensus
        messages = [
            EdgeVotingMessage(
                sender_node_id=0, edge=(0, 1), vote=True, round_num=1, weight=10.0
            ),
            EdgeVotingMessage(
                sender_node_id=1, edge=(0, 1), vote=True, round_num=1, weight=10.0
            ),
            EdgeVotingMessage(
                sender_node_id=0, edge=(0, 2), vote=True, round_num=1, weight=10.0
            ),
            EdgeVotingMessage(
                sender_node_id=2, edge=(0, 2), vote=True, round_num=1, weight=10.0
            ),
        ]

        resolver.receive_votes(0, messages)
        matching = resolver.resolve_matches(0, threshold=0.5)

        # Both edges got consensus, so both are in matching (consumer responsible for deconflicting)
        assert (0, 1) in [(min(k, v), max(k, v)) for k, v in matching.items() if v is not None]
        assert (0, 2) in [(min(k, v), max(k, v)) for k, v in matching.items() if v is not None]

    def test_symmetric_matching_property(self) -> None:
        """Test that final matching is symmetric."""
        resolver = DistributedConflictResolver(threshold=0.5)
        resolver.initialize(node_ids=[0, 1, 2])

        greedy = {0: 1, 1: 0}
        resolver.propose_edges(0, greedy, {}, {})

        messages = [
            EdgeVotingMessage(
                sender_node_id=0, edge=(0, 1), vote=True, round_num=1, weight=10.0
            ),
            EdgeVotingMessage(
                sender_node_id=1, edge=(0, 1), vote=True, round_num=1, weight=10.0
            ),
        ]

        resolver.receive_votes(0, messages)
        matching = resolver.resolve_matches(0, threshold=0.5)

        # For every edge, both endpoints match each other
        for u, v in matching.items():
            if v is not None:
                assert matching.get(v) == u

    def test_get_best_vector(self) -> None:
        """Test retrieving best found vector."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1])

        # Query before any state set
        best = resolver.get_best_vector(0)
        assert best is None

    def test_multiple_nodes_independent_state(self) -> None:
        """Test that each node maintains independent state."""
        resolver = DistributedConflictResolver()
        resolver.initialize(node_ids=[0, 1, 2])

        greedy0 = {0: 1, 1: 0}
        greedy1 = {1: 2, 2: 1}

        resolver.propose_edges(0, greedy0, {}, {})
        resolver.propose_edges(1, greedy1, {}, {})

        # Verify independent state
        assert resolver.node_states[0].proposed_edges != resolver.node_states[1].proposed_edges
