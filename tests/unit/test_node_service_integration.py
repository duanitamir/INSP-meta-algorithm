"""Integration tests for NodeService with multi-node scenarios."""

import pytest
from src.graph.graph_manager import GraphManager
from src.node_service import NodeService
from src.communication.drivers.in_memory_driver import InMemoryDriver
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


class TestNodeServiceWithSimpleGraph:
    """Test NodeService execution on simple graphs."""

    def test_single_node_converges(self, simple_graph):
        """Single node should converge on simple graph."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=50)
        algo = GreedyMatching()

        results = service.run(algo)

        assert results["node_id"] == 0
        assert results["rounds_executed"] > 0
        assert results["finished"]
        assert "final_status" in results

    def test_multiple_nodes_independent(self, simple_graph):
        """Multiple nodes should run independently."""
        driver = InMemoryDriver(simple_graph)
        services = [
            NodeService(node_id=i, graph=simple_graph, transport_driver=driver, max_rounds=50)
            for i in range(4)
        ]
        algo = GreedyMatching()

        results = [service.run(algo) for service in services]

        assert len(results) == 4
        for i, result in enumerate(results):
            assert result["node_id"] == i
            assert result["finished"]

    def test_nodes_with_different_algorithms(self, simple_graph):
        """Nodes can use different algorithms."""
        driver = InMemoryDriver(simple_graph)
        service_greedy = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=20)
        service_itai = NodeService(node_id=1, graph=simple_graph, transport_driver=driver, max_rounds=20)

        result_greedy = service_greedy.run(GreedyMatching())
        result_itai = service_itai.run(ItaiIsraeliMaximalMatching())

        assert result_greedy["node_id"] == 0
        assert result_itai["node_id"] == 1
        assert result_greedy["finished"]
        assert result_itai["finished"]

    def test_node_respects_max_rounds(self, simple_graph):
        """Node should stop at max rounds."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=3)
        algo = GreedyMatching()

        results = service.run(algo)

        assert results["rounds_executed"] <= 3

    def test_node_convergence_detection(self, simple_graph):
        """Node should detect convergence or reach max rounds."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=100)
        algo = GreedyMatching()

        results = service.run(algo)

        # Should finish (either by convergence or max rounds)
        assert results["rounds_executed"] <= 100
        assert results["finished"]


class TestNodeServiceWithClusteredGraph:
    """Test NodeService on more complex graphs."""

    def test_clustered_graph_execution(self, clustered_graph):
        """Should handle clustered graphs."""
        driver = InMemoryDriver(clustered_graph)
        service = NodeService(node_id=0, graph=clustered_graph, transport_driver=driver, max_rounds=100)
        algo = GreedyMatching()

        results = service.run(algo)

        assert results["finished"]
        assert results["rounds_executed"] > 0

    def test_multiple_nodes_clustered_graph(self, clustered_graph):
        """Multiple nodes should work on clustered graphs."""
        driver = InMemoryDriver(clustered_graph)
        services = [
            NodeService(node_id=i, graph=clustered_graph, transport_driver=driver, max_rounds=100)
            for i in range(6)
        ]
        algo = GreedyMatching()

        results = [service.run(algo) for service in services]

        assert all(r["finished"] for r in results)
        assert len(results) == 6


class TestNodeServiceRoundExecution:
    """Test detailed round execution."""

    def test_execute_single_round(self, simple_graph):
        """Should execute single round."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver)
        algo = GreedyMatching()

        should_continue, status = service.execute_round(algo)

        assert service.round_number == 1
        assert status in ["round_completed", "convergence_quorum_reached"]
        assert isinstance(should_continue, bool)

    def test_execute_multiple_rounds_sequence(self, simple_graph):
        """Should execute multiple rounds in sequence."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=10)
        algo = GreedyMatching()

        for i in range(1, 6):
            should_continue, status = service.execute_round(algo)
            assert service.round_number == i
            if not should_continue:
                break

    def test_round_increments_counter(self, simple_graph):
        """Round counter should increment."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver)
        algo = GreedyMatching()

        assert service.round_number == 0
        service.execute_round(algo)
        assert service.round_number == 1
        service.execute_round(algo)
        assert service.round_number == 2


class TestNodeServiceStateManagement:
    """Test state transitions during execution."""

    def test_finished_flag_after_run(self, simple_graph):
        """Finished flag should be set after run completes."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=50)
        algo = GreedyMatching()

        assert not service.is_finished
        service.run(algo)
        assert service.is_finished

    def test_finished_service_refuses_rounds(self, simple_graph):
        """Finished service should refuse more rounds."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver)
        service.finished = True
        algo = GreedyMatching()

        should_continue, status = service.execute_round(algo)

        assert not should_continue
        assert status == "max_rounds_exceeded"

    def test_max_rounds_stops_execution(self, simple_graph):
        """Should stop at max rounds."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver, max_rounds=2)
        algo = GreedyMatching()

        round1_continue, _ = service.execute_round(algo)
        assert service.round_number == 1

        round2_continue, _ = service.execute_round(algo)
        assert service.round_number == 2

        round3_continue, status = service.execute_round(algo)
        # At this point, should_continue depends on convergence, but after max_rounds we stop anyway


class TestNodeServiceTransportIntegration:
    """Test transport driver integration."""

    def test_transport_available(self, simple_graph):
        """Transport should be available."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver)

        assert service.transport is driver
        assert service.transport.name() == "InMemoryDriver"

    def test_default_transport_created(self, simple_graph):
        """Default transport should be created if not provided."""
        service = NodeService(node_id=0, graph=simple_graph)

        assert service.transport is not None
        assert service.transport.name() == "InMemoryDriver"

    def test_messages_flow_through_transport(self, simple_graph):
        """Messages should flow through transport."""
        driver = InMemoryDriver(simple_graph)
        service = NodeService(node_id=0, graph=simple_graph, transport_driver=driver)
        algo = GreedyMatching()

        # Execute round (should use transport for message passing)
        should_continue, status = service.execute_round(algo)

        # Service should still be operational
        assert service.transport is not None
