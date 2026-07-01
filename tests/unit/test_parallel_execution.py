"""Tests for parallel node execution capability."""

import pytest
from src.graph.graph_manager import GraphManager
from src.simulation.distributed_simulator import DistributedSimulator
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching


@pytest.fixture
def medium_graph():
    """Create a 50-node test graph."""
    graph = GraphManager()
    for i in range(50):
        graph.add_vertex(i)

    # Sparse graph to allow parallelism benefits
    for i in range(50):
        for j in range(i + 1, min(i + 3, 50)):
            graph.add_edge(i, j, weight=10 + (i * j) % 5)

    return graph


class TestParallelExecution:
    """Test parallel node execution capability."""

    def test_parallel_enabled_by_default(self, medium_graph):
        """Should enable parallel execution by default."""
        sim = DistributedSimulator(medium_graph, GreedyMatching())
        assert sim.use_parallel is True
        assert sim.num_workers > 0

    def test_parallel_execution_produces_valid_results(self, medium_graph):
        """Parallel execution should produce valid results."""
        config = {"use_parallel": True, "num_workers": 2, "max_rounds": 100}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results = sim.run()

        # Should have valid matching
        assert "final_matching" in results
        assert len(results["final_matching"]) > 0

    def test_sequential_execution_produces_valid_results(self, medium_graph):
        """Sequential execution should still work."""
        config = {"use_parallel": False, "max_rounds": 100}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results = sim.run()

        # Should have valid matching
        assert "final_matching" in results
        assert len(results["final_matching"]) > 0

    def test_parallel_and_sequential_same_results(self, medium_graph):
        """Parallel and sequential should produce same results (deterministic)."""
        # Sequential run
        sim_seq = DistributedSimulator(medium_graph, GreedyMatching(), {"use_parallel": False})
        results_seq = sim_seq.run()

        # Parallel run
        sim_par = DistributedSimulator(medium_graph, GreedyMatching(), {"use_parallel": True})
        results_par = sim_par.run()

        # Results should be identical
        assert results_seq["final_matching"] == results_par["final_matching"]
        assert results_seq["all_finished"] == results_par["all_finished"]
        assert results_seq["total_rounds"] == results_par["total_rounds"]

    def test_timing_info_included(self, medium_graph):
        """Results should include timing information."""
        config = {"use_parallel": True}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results = sim.run()

        # Timing info
        assert "execution_time_seconds" in results
        assert "parallel_execution_time_seconds" in results
        assert "message_delivery_time_seconds" in results
        assert "parallelization" in results

        # Parallelization info
        assert results["parallelization"]["enabled"] is True
        assert results["parallelization"]["num_workers"] > 0
        assert results["parallelization"]["num_nodes"] == 50

    def test_custom_num_workers(self, medium_graph):
        """Should respect custom num_workers setting."""
        config = {"use_parallel": True, "num_workers": 2}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)

        assert sim.num_workers == 2
        results = sim.run()
        assert results["parallelization"]["num_workers"] == 2

    def test_parallel_execution_completes(self, medium_graph):
        """Parallel execution should complete successfully."""
        config = {"use_parallel": True, "max_rounds": 100}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results = sim.run()

        # Should complete
        assert results["total_rounds"] > 0
        assert results["execution_time_seconds"] >= 0
        assert results["parallel_execution_time_seconds"] >= 0

    def test_different_algorithms_parallel(self, medium_graph):
        """Parallel execution should work with different algorithms."""
        config = {"use_parallel": True, "max_rounds": 100}

        # Greedy
        sim_greedy = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results_greedy = sim_greedy.run()
        assert len(results_greedy["final_matching"]) > 0

        # Itai-Israeli
        sim_itai = DistributedSimulator(medium_graph, ItaiIsraeliMaximalMatching(), config)
        results_itai = sim_itai.run()
        assert len(results_itai["final_matching"]) > 0

    def test_execution_timing_reasonable(self, medium_graph):
        """Execution timing should be reasonable."""
        config = {"use_parallel": True, "max_rounds": 20}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)
        results = sim.run()

        # Timing should be reasonable (not negative or zero)
        assert results["execution_time_seconds"] > 0
        assert results["parallel_execution_time_seconds"] > 0
        assert results["message_delivery_time_seconds"] >= 0

        # Total should account for most of execution time
        total_work = (
            results["parallel_execution_time_seconds"]
            + results["message_delivery_time_seconds"]
        )
        assert total_work <= results["execution_time_seconds"] * 1.1  # Allow 10% overhead

    def test_reset_clears_timing(self, medium_graph):
        """Reset should clear timing information."""
        config = {"use_parallel": True}
        sim = DistributedSimulator(medium_graph, GreedyMatching(), config)

        # Run once
        sim.run()
        assert sim.execution_time > 0

        # Reset
        sim.reset()
        assert sim.execution_time == 0.0
        assert sim.parallel_execution_time == 0.0
        assert sim.message_delivery_time == 0.0

    def test_parallel_toggle_works(self, medium_graph):
        """Should be able to toggle parallel execution."""
        # First: parallel
        sim = DistributedSimulator(medium_graph, GreedyMatching(), {"use_parallel": True})
        assert sim.use_parallel is True
        sim.reset()

        # Toggle off
        sim.use_parallel = False
        assert sim.use_parallel is False
        results = sim.run()
        assert results["parallelization"]["enabled"] is False


class TestParallelPerformance:
    """Test that parallel execution doesn't break performance characteristics."""

    def test_convergence_speed_maintained(self):
        """Convergence speed should be maintained or improved."""
        # Small graph should converge quickly
        graph = GraphManager()
        for i in range(10):
            graph.add_vertex(i)
        for i in range(9):
            graph.add_edge(i, i + 1, weight=10)

        sim = DistributedSimulator(graph, GreedyMatching(), {"use_parallel": True})
        results = sim.run()

        # Should converge in a few rounds
        assert results["total_rounds"] <= 10
        assert results["all_finished"]

    def test_message_count_unchanged(self):
        """Total message count should be similar (same algorithm behavior with gossip variance)."""
        graph = GraphManager()
        for i in range(20):
            graph.add_vertex(i)
        for i in range(20):
            for j in range(i + 1, min(i + 3, 20)):
                graph.add_edge(i, j, weight=10)

        config = {"max_rounds": 50}

        # Sequential
        sim_seq = DistributedSimulator(graph, GreedyMatching(), {**config, "use_parallel": False})
        results_seq = sim_seq.run()

        # Parallel
        sim_par = DistributedSimulator(graph, GreedyMatching(), {**config, "use_parallel": True})
        results_par = sim_par.run()

        # Message counts should be similar (within 5% - gossip has randomness)
        # The key is that algorithm produces same results, not same message counts
        assert abs(results_seq["total_messages"] - results_par["total_messages"]) < max(
            5, results_seq["total_messages"] * 0.05
        )
