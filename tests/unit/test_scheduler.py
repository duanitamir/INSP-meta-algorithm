import pytest
from src.graph import GraphManager
from src.simulation import Scheduler, SimulationConfig
from src.utils.types import RoundNumber


class TestScheduler:
    def test_create_scheduler(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        assert scheduler.graph == simple_graph
        assert scheduler.current_round == RoundNumber(0)

    def test_reset(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        scheduler.initialize()
        scheduler.execute_round()
        assert scheduler.current_round > RoundNumber(0)
        scheduler.reset()
        assert scheduler.current_round == RoundNumber(0)

    def test_initialize(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        assert not scheduler.is_running
        scheduler.initialize()
        assert scheduler.is_running

    def test_execute_round(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        scheduler.initialize()
        result = scheduler.execute_round()
        assert result is True
        assert scheduler.current_round == RoundNumber(1)

    def test_max_rounds_termination(self, simple_graph):
        config = SimulationConfig(max_rounds=2)
        scheduler = Scheduler(simple_graph, config)
        rounds = scheduler.run_until_termination()
        assert rounds <= config.max_rounds

    def test_check_no_progress(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        scheduler.initialize()
        # Initially no messages sent
        assert scheduler.check_no_progress()

    def test_metrics_collection(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        scheduler.initialize()
        scheduler.execute_round()
        metrics = scheduler.metrics.get_metrics_snapshot()
        assert metrics is not None

    def test_state_store_access(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        state = scheduler.state_store.get_node_state(1)
        assert state.node_id == 1

    def test_message_queue_access(self, simple_graph):
        scheduler = Scheduler(simple_graph)
        from src.communication import Message
        msg = Message(1, 2, "test", RoundNumber(0))
        scheduler.message_queue.send(msg)
        assert scheduler.message_queue.inbox_size(2) == 1

    def test_termination_callback(self, simple_graph):
        scheduler = Scheduler(simple_graph)

        def check_termination(state_store, round_num, messages_sent):
            return round_num >= RoundNumber(3), "reached round 3"

        rounds = scheduler.run_until_termination(termination_callback=check_termination)
        assert rounds >= RoundNumber(3)
        assert scheduler.is_terminated
