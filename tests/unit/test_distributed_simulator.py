"""Unit tests for DistributedSimulator - minimal orchestrator."""

import pytest
from src.graph.graph_manager import GraphManager
from src.simulation.distributed_simulator import DistributedSimulator
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching


@pytest.fixture
def simple_graph():
    """Create a simple 4-node graph."""
    graph = GraphManager()
    for i in range(4):
        graph.add_vertex(i)
    graph.add_edge(0, 1, weight=10)
    graph.add_edge(1, 2, weight=20)
    graph.add_edge(2, 3, weight=15)
    return graph


@pytest.fixture
def clustered_graph():
    """Create clustered graph with two groups."""
    graph = GraphManager()
    for i in range(6):
        graph.add_vertex(i)
    # Group 1: 0-1-2
    graph.add_edge(0, 1, weight=10)
    graph.add_edge(1, 2, weight=10)
    graph.add_edge(0, 2, weight=8)
    # Group 2: 3-4-5
    graph.add_edge(3, 4, weight=10)
    graph.add_edge(4, 5, weight=10)
    graph.add_edge(3, 5, weight=8)
    # Inter-group edge
    graph.add_edge(2, 3, weight=5)
    return graph


@pytest.fixture
def greedy_algorithm():
    """Create Greedy matching algorithm."""
    return GreedyMatching()


class TestDistributedSimulatorInitialization:
    """Test simulator creation and setup."""

    def test_simulator_creation(self, simple_graph, greedy_algorithm):
        """Should create simulator with nodes."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        assert sim.graph is simple_graph
        assert sim.algorithm is greedy_algorithm
        assert len(sim.nodes) == 4
        assert all(node_id in sim.nodes for node_id in range(4))

    def test_simulator_config(self, simple_graph, greedy_algorithm):
        """Should apply config to nodes."""
        config = {
            "max_rounds": 500,
            "convergence_threshold": 0.1,
            "quorum_threshold": 0.7
        }
        sim = DistributedSimulator(simple_graph, greedy_algorithm, config)

        # Check config applied to nodes
        node = sim.nodes[0]
        assert node.convergence_threshold == 0.1
        assert node.quorum_threshold == 0.7
        assert sim.max_rounds == 500

    def test_simulator_creates_all_nodes(self, simple_graph, greedy_algorithm):
        """Should create node for each vertex."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        node_ids = set(sim.nodes.keys())
        graph_vertices = set(simple_graph.vertices())

        assert node_ids == graph_vertices


class TestDistributedSimulatorExecution:
    """Test simulation execution."""

    def test_run_executes_algorithm(self, simple_graph, greedy_algorithm):
        """Should run algorithm and return results."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results = sim.run()

        assert isinstance(results, dict)
        assert "final_matching" in results
        assert "total_rounds" in results
        assert "all_finished" in results

    def test_run_convergence_detection(self, simple_graph, greedy_algorithm):
        """Should detect convergence and stop."""
        config = {"max_rounds": 1000, "convergence_threshold": 0.05}
        sim = DistributedSimulator(simple_graph, greedy_algorithm, config)

        results = sim.run()

        # Should converge before max rounds on small graph
        assert results["total_rounds"] < 100

    def test_run_respects_max_rounds(self, simple_graph, greedy_algorithm):
        """Should not exceed max_rounds."""
        config = {"max_rounds": 5}
        sim = DistributedSimulator(simple_graph, greedy_algorithm, config)

        results = sim.run()

        assert results["total_rounds"] <= 5

    def test_run_produces_valid_matching(self, simple_graph, greedy_algorithm):
        """Should produce symmetric, valid matching."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results = sim.run()
        matching = results["final_matching"]

        # Check symmetry
        for u, v in matching.items():
            if v in matching:
                assert matching[v] == u, f"Asymmetric: {u}-{v} but {v}-{matching[v]}"


class TestDistributedSimulatorMessageDelivery:
    """Test message delivery between nodes."""

    def test_deliver_messages(self, simple_graph, greedy_algorithm):
        """Should deliver messages from outbox to inbox."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        # Manually place message
        from src.communication.message import Message
        msg = Message(
            sender=0,
            recipient=1,
            payload={"type": "TEST"},
            round_num=0
        )
        sim.nodes[0].outbox.send(msg)

        # Deliver
        sim._deliver_messages()

        # Check recipient has message
        received = sim.nodes[1].inbox.peek_messages(1)
        assert len(received) > 0
        assert received[0].sender == 0

    def test_deliver_to_all_recipients(self, simple_graph, greedy_algorithm):
        """Should deliver to all specified recipients."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        from src.communication.message import Message

        # Send from node 0 to multiple recipients
        for recipient in [1, 2]:
            msg = Message(
                sender=0,
                recipient=recipient,
                payload={"type": "TEST", "id": recipient},
                round_num=0
            )
            sim.nodes[0].outbox.send(msg)

        sim._deliver_messages()

        # Each recipient should have received message
        assert sim.nodes[1].inbox.has_messages(1)
        assert sim.nodes[2].inbox.has_messages(2)


class TestDistributedSimulatorMetrics:
    """Test metrics collection."""

    def test_collect_metrics(self, simple_graph, greedy_algorithm):
        """Should collect global metrics each round."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        # Execute one round manually
        for node in sim.nodes.values():
            node.execute(greedy_algorithm)

        sim._collect_global_metrics()

        # Check metrics recorded
        metrics = sim.global_metrics.get_all_metrics()
        assert len(metrics) > 0

    def test_metrics_track_progress(self, simple_graph, greedy_algorithm):
        """Should track metrics across rounds."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        # Run a few rounds
        for _ in range(5):
            for node in sim.nodes.values():
                if not node.finished:
                    node.execute(greedy_algorithm)
            sim._deliver_messages()
            sim._collect_global_metrics()

        metrics = sim.global_metrics.get_all_metrics()

        # Should have 5 metric snapshots
        assert len(metrics) == 5


class TestDistributedSimulatorResults:
    """Test result extraction."""

    def test_extract_final_matching(self, simple_graph, greedy_algorithm):
        """Should extract final matching from nodes."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results = sim.run()
        matching = results["final_matching"]

        # Should have matching for each node
        assert len(matching) > 0

        # Each matched node should be in graph
        for u, v in matching.items():
            assert u in simple_graph.vertices()
            assert v in simple_graph.vertices()

    def test_extract_node_metrics(self, simple_graph, greedy_algorithm):
        """Should extract per-node metrics."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results = sim.run()

        assert "node_metrics" in results
        node_metrics = results["node_metrics"]

        # Should have metrics for each node
        assert len(node_metrics) == 4

        # Each node should have summary
        for node_id in range(4):
            assert node_id in node_metrics
            summary = node_metrics[node_id]
            assert "round_number" in summary
            assert "messages_sent" in summary

    def test_extract_convergence_votes(self, simple_graph, greedy_algorithm):
        """Should extract final convergence votes."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results = sim.run()

        assert "convergence_votes" in results
        votes = results["convergence_votes"]

        # Should have vote for each node
        assert len(votes) == 4


class TestDistributedSimulatorReset:
    """Test reset functionality."""

    def test_reset_clears_nodes(self, simple_graph, greedy_algorithm):
        """Should reset all nodes."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        # Run simulation
        sim.run()

        initial_rounds = sim.nodes[0].round_number
        assert initial_rounds > 0

        # Reset
        sim.reset()

        # All nodes should be reset
        for node in sim.nodes.values():
            assert node.round_number == 0
            assert not node.finished

    def test_reset_clears_metrics(self, simple_graph, greedy_algorithm):
        """Should reset global metrics."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        sim.run()

        initial_metrics = len(sim.global_metrics.get_all_metrics())
        assert initial_metrics > 0

        sim.reset()

        final_metrics = len(sim.global_metrics.get_all_metrics())
        assert final_metrics == 0


class TestDistributedSimulatorNodeAccess:
    """Test accessing nodes."""

    def test_get_node(self, simple_graph, greedy_algorithm):
        """Should return specific node."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        node = sim.get_node(2)

        assert node.id == 2
        assert node is sim.nodes[2]

    def test_get_all_nodes(self, simple_graph, greedy_algorithm):
        """Should return all nodes."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        nodes = sim.get_all_nodes()

        assert len(nodes) == 4
        assert all(node_id in nodes for node_id in range(4))


class TestDistributedSimulatorDifferentGraphs:
    """Test with different graph types."""

    def test_clustered_graph(self, clustered_graph, greedy_algorithm):
        """Should work with clustered graph."""
        sim = DistributedSimulator(clustered_graph, greedy_algorithm)

        results = sim.run()

        assert "final_matching" in results
        assert results["all_finished"]

    def test_large_graph(self, greedy_algorithm):
        """Should work with larger graph."""
        # Create 10-node graph
        graph = GraphManager()
        for i in range(10):
            graph.add_vertex(i)
        for i in range(10):
            for j in range(i + 1, min(i + 4, 10)):
                graph.add_edge(i, j, weight=10)

        sim = DistributedSimulator(graph, greedy_algorithm)

        results = sim.run()

        assert "final_matching" in results
        assert len(results["node_metrics"]) == 10


class TestDistributedSimulatorDifferentAlgorithms:
    """Test with different algorithms."""

    def test_itai_algorithm(self, simple_graph):
        """Should work with Itai-Israeli algorithm."""
        algorithm = ItaiIsraeliMaximalMatching()
        sim = DistributedSimulator(simple_graph, algorithm)

        results = sim.run()

        assert "final_matching" in results
        assert results["all_finished"]

    def test_multiple_runs(self, simple_graph, greedy_algorithm):
        """Should support multiple sequential runs."""
        sim = DistributedSimulator(simple_graph, greedy_algorithm)

        results1 = sim.run()
        sim.reset()
        results2 = sim.run()

        # Both should produce valid results
        assert "final_matching" in results1
        assert "final_matching" in results2
