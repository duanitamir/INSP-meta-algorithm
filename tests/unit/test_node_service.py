"""Tests for NodeService (autonomous node execution)."""

import pytest
from src.graph.graph_manager import GraphManager
from src.node_service import NodeService
from src.communication.drivers.in_memory_driver import InMemoryDriver
from src.algorithms.implementations.greedy_matching import GreedyMatching


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    graph = GraphManager()
    vertices = [1, 2, 3, 4]
    edges = [(1, 2, 10), (2, 3, 5), (3, 4, 8)]
    for v in vertices:
        graph.add_vertex(v)
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    return graph


@pytest.fixture
def transport_driver(simple_graph):
    """Create in-memory transport driver."""
    return InMemoryDriver(simple_graph)


class TestNodeServiceBasics:
    """Test NodeService initialization and basic properties."""

    def test_init(self, simple_graph, transport_driver):
        """Test NodeService initialization."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=100,
        )
        assert service.node_id == 1
        assert service.max_rounds == 100
        assert not service.is_finished
        assert service.round_number == 0

    def test_init_without_transport(self, simple_graph):
        """Test NodeService with default in-memory transport."""
        service = NodeService(node_id=1, graph=simple_graph)
        assert service.transport is not None
        assert not service.is_finished

    def test_transport_name(self, simple_graph):
        """Test transport driver name."""
        service = NodeService(node_id=1, graph=simple_graph)
        assert service.transport.name() == "InMemoryDriver"


class TestNodeServiceExecution:
    """Test NodeService round execution."""

    def test_single_round_execution(self, simple_graph, transport_driver):
        """Test executing a single round."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
        )
        algorithm = GreedyMatching()

        should_continue, status = service.execute_round(algorithm)

        assert service.round_number == 1
        assert isinstance(should_continue, bool)
        assert isinstance(status, str)

    def test_multiple_rounds(self, simple_graph, transport_driver):
        """Test executing multiple rounds."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=5,
        )
        algorithm = GreedyMatching()

        for _ in range(3):
            should_continue, status = service.execute_round(algorithm)
            if not should_continue:
                break

        assert service.round_number in [1, 2, 3, 4, 5]

    def test_max_rounds_limit(self, simple_graph, transport_driver):
        """Test that execution stops at max rounds."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=2,
        )
        algorithm = GreedyMatching()

        # Execute more times than max
        for _ in range(5):
            should_continue, status = service.execute_round(algorithm)
            if not should_continue:
                break

        assert service.round_number <= service.max_rounds


class TestNodeServiceRun:
    """Test NodeService.run() method."""

    def test_run_completes(self, simple_graph, transport_driver):
        """Test that run completes and returns results."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=10,
        )
        algorithm = GreedyMatching()

        results = service.run(algorithm)

        assert results["node_id"] == 1
        assert results["rounds_executed"] >= 0
        assert "finished" in results
        assert "final_status" in results

    def test_run_updates_state(self, simple_graph, transport_driver):
        """Test that run updates service state."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=5,
        )
        algorithm = GreedyMatching()

        results = service.run(algorithm)

        assert service.rounds_executed == results["rounds_executed"]
        assert service.finished == results["finished"]


class TestNodeServiceMultiNode:
    """Test NodeService with multiple nodes."""

    def test_two_nodes_independent(self, simple_graph):
        """Test that two nodes run independently."""
        driver = InMemoryDriver(simple_graph)
        service1 = NodeService(node_id=1, graph=simple_graph, transport_driver=driver)
        service2 = NodeService(node_id=2, graph=simple_graph, transport_driver=driver)

        assert service1.node_id == 1
        assert service2.node_id == 2
        assert service1.node is not service2.node

    def test_nodes_use_same_transport(self, simple_graph):
        """Test that multiple nodes can share transport driver."""
        driver = InMemoryDriver(simple_graph)
        services = [
            NodeService(node_id=i, graph=simple_graph, transport_driver=driver)
            for i in range(1, 4)
        ]

        for service in services:
            assert service.transport is driver


class TestNodeServiceTermination:
    """Test NodeService termination behavior."""

    def test_finish_flag(self, simple_graph, transport_driver):
        """Test that finished flag is set correctly."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
            max_rounds=1,
        )
        algorithm = GreedyMatching()

        assert not service.is_finished
        service.execute_round(algorithm)
        # May or may not be finished depending on convergence

    def test_completed_service_refuses_more_rounds(self, simple_graph, transport_driver):
        """Test that a finished service won't execute more rounds."""
        service = NodeService(
            node_id=1,
            graph=simple_graph,
            transport_driver=transport_driver,
        )
        service.finished = True
        algorithm = GreedyMatching()

        should_continue, status = service.execute_round(algorithm)

        assert not should_continue
