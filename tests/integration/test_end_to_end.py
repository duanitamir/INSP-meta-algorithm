import pytest
from src.graph import GraphManager
from src.state import StateStore
from src.communication import Message, MessageQueue
from src.simulation import Scheduler, SimulationConfig
from src.utils.types import RoundNumber


class TestEndToEnd:
    def test_graph_creation_and_visualization(self, simple_graph):
        """Test creating and visualizing a graph."""
        assert simple_graph.num_vertices() == 4
        assert simple_graph.num_edges() == 3
        assert simple_graph.is_connected()

    def test_state_management(self, simple_graph):
        """Test state store initialization and updates."""
        store = StateStore(simple_graph)
        for vertex_id in simple_graph.vertices():
            state = store.get_node_state(vertex_id)
            assert state.node_id == vertex_id
            state.set_matched_to(None)
            store.update_node_state(vertex_id, state)

    def test_message_passing(self, simple_graph):
        """Test message creation and queuing."""
        queue = MessageQueue(simple_graph)
        for node_id in simple_graph.vertices():
            for neighbor in simple_graph.neighbors(node_id):
                msg = Message(node_id, neighbor, "test", RoundNumber(0))
                queue.send(msg)
        assert queue.total_messages_sent() > 0

    def test_scheduler_initialization(self, simple_graph):
        """Test scheduler setup."""
        scheduler = Scheduler(simple_graph)
        assert scheduler.state_store.graph == simple_graph
        assert scheduler.message_queue.graph == simple_graph
        assert not scheduler.is_running
        scheduler.initialize()
        assert scheduler.is_running

    def test_full_simulation_flow(self, simple_graph, simulation_config):
        """Test a complete simulation flow."""
        scheduler = Scheduler(simple_graph, simulation_config)

        def dummy_termination(state_store, round_num, messages_sent):
            return round_num >= RoundNumber(5), "test complete"

        rounds = scheduler.run_until_termination(
            termination_callback=dummy_termination
        )
        assert rounds >= RoundNumber(5)
        assert scheduler.is_terminated

    def test_snapshot_and_restoration(self, simple_graph):
        """Test state snapshot and restoration."""
        store = StateStore(simple_graph)

        state1 = store.get_node_state(1)
        state1.set_matched_to(2)
        store.update_node_state(1, state1)

        snapshot = store.create_snapshot(RoundNumber(0))
        assert snapshot.round_num == RoundNumber(0)

        state2 = store.get_node_state(1)
        state2.set_matched_to(None)
        store.update_node_state(1, state2)
        assert not store.get_node_state(1).is_matched()

        store.restore_snapshot(snapshot)
        assert store.get_node_state(1).is_matched()

    def test_metrics_collection_during_simulation(self, simple_graph):
        """Test metrics collection during simulation."""
        scheduler = Scheduler(simple_graph)
        scheduler.initialize()

        for _ in range(3):
            scheduler.execute_round()

        metrics = scheduler.metrics.get_all_metrics()
        assert len(metrics) == 3

    def test_multiple_simulation_runs(self, simple_graph, simulation_config):
        """Test running multiple simulations."""
        config1 = SimulationConfig(max_rounds=5)
        scheduler1 = Scheduler(simple_graph, config1)
        rounds1 = scheduler1.run_until_termination()

        scheduler1.reset()
        rounds2 = scheduler1.run_until_termination()

        assert rounds1 == rounds2

    def test_graph_with_multiple_components(self):
        """Test graph with disconnected components."""
        graph = GraphManager.create_empty_graph()
        for i in range(1, 7):
            graph.add_vertex(i)

        edges = [(1, 2, 1.0), (2, 3, 1.0), (4, 5, 1.0)]
        for u, v, w in edges:
            graph.add_edge(u, v, w)

        assert not graph.is_connected()
        components = graph.get_connected_components()
        assert len(components) == 3

        scheduler = Scheduler(graph)
        scheduler.initialize()
        scheduler.execute_round()
        assert scheduler.state_store.graph == graph

    def test_visualization_with_state_and_matching(self, simple_graph):
        """Test visualization with state and matching."""
        from src.visualization import GraphVisualizer

        viz = GraphVisualizer(simple_graph)

        store = StateStore(simple_graph)
        state = store.get_node_state(1)
        state.set_matched_to(2)
        store.update_node_state(1, state)

        output = viz.render_to_ascii(state_store=store)
        assert "matched to 2" in output

        matching = {1: 2, 2: 1, 3: 4, 4: 3}
        matching_output = viz.render_matching_to_ascii(matching)
        assert "MATCHED" in matching_output or "Matching size" in matching_output
