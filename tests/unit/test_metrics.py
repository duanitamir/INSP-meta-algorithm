import pytest
from src.metrics import MetricsCollector, MetricsSnapshot
from src.utils.types import RoundNumber


class TestMetricsSnapshot:
    def test_create_snapshot(self):
        snapshot = MetricsSnapshot(
            round_num=RoundNumber(1),
            messages_sent=10,
            messages_received=5,
            active_nodes=3,
        )
        assert snapshot.round_num == RoundNumber(1)
        assert snapshot.messages_sent == 10

    def test_to_dict(self):
        snapshot = MetricsSnapshot(
            round_num=RoundNumber(1),
            messages_sent=10,
        )
        d = snapshot.to_dict()
        assert d["round_num"] == RoundNumber(1)
        assert d["messages_sent"] == 10


class TestMetricsCollector:
    def test_create_collector(self):
        collector = MetricsCollector()
        assert len(collector.round_metrics) == 0

    def test_record_round(self):
        collector = MetricsCollector()
        collector.record_round(RoundNumber(0), messages_sent=5)
        assert len(collector.round_metrics) == 1
        assert collector.round_metrics[0].messages_sent == 5

    def test_total_messages(self):
        collector = MetricsCollector()
        collector.record_round(RoundNumber(0), messages_sent=5)
        collector.record_round(RoundNumber(1), messages_sent=3)
        assert collector.total_messages == 8

    def test_custom_metric(self):
        collector = MetricsCollector()
        collector.record_custom_metric("matching_size", 10)
        collector.record_custom_metric("matching_size", 12)
        metrics = collector.get_custom_metric("matching_size")
        assert metrics == [10, 12]

    def test_get_metrics_snapshot(self):
        collector = MetricsCollector()
        collector.record_round(RoundNumber(0), messages_sent=5)
        snapshot = collector.get_metrics_snapshot()
        assert snapshot is not None
        assert snapshot.messages_sent == 5

    def test_get_all_metrics(self):
        collector = MetricsCollector()
        for i in range(3):
            collector.record_round(RoundNumber(i), messages_sent=i + 1)
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) == 3

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_round(RoundNumber(0), messages_sent=5)
        collector.reset()
        assert len(collector.round_metrics) == 0
        assert collector.total_messages == 0
