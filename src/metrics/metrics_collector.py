from dataclasses import dataclass
from typing import Dict, Any, List
import time
from src.utils.types import RoundNumber


@dataclass
class MetricsSnapshot:
    """Metrics for a single round."""

    round_num: RoundNumber
    messages_sent: int
    messages_received: int = 0
    active_nodes: int = 0
    matched_nodes: int = 0
    dormant_nodes: int = 0
    converged: bool = False
    unmatched_vertices: int = 0
    round_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_num": self.round_num,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "active_nodes": self.active_nodes,
            "matched_nodes": self.matched_nodes,
            "dormant_nodes": self.dormant_nodes,
            "converged": self.converged,
            "unmatched_vertices": self.unmatched_vertices,
            "round_duration_ms": self.round_duration_ms,
        }


class MetricsCollector:
    """Collects metrics throughout simulation execution."""

    def __init__(self):
        self.round_metrics: List[MetricsSnapshot] = []
        self.custom_metrics: Dict[str, List[Any]] = {}
        self.start_time = time.time()
        self.total_messages = 0

    def record_round(
        self,
        round_num: RoundNumber,
        messages_sent: int,
        **kwargs: Any,
    ) -> None:
        """Record metrics for a round."""
        snapshot = MetricsSnapshot(
            round_num=round_num,
            messages_sent=messages_sent,
            messages_received=kwargs.get("messages_received", 0),
            active_nodes=kwargs.get("active_nodes", 0),
            matched_nodes=kwargs.get("matched_nodes", 0),
            dormant_nodes=kwargs.get("dormant_nodes", 0),
            converged=kwargs.get("converged", False),
            unmatched_vertices=kwargs.get("unmatched_vertices", 0),
            round_duration_ms=0.0,
        )
        self.round_metrics.append(snapshot)
        self.total_messages += messages_sent

    def record_custom_metric(self, metric_name: str, value: Any) -> None:
        """Record a custom metric."""
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []
        self.custom_metrics[metric_name].append(value)

    def get_metrics_snapshot(self) -> MetricsSnapshot | None:
        """Get current metrics."""
        return self.round_metrics[-1] if self.round_metrics else None

    def get_all_metrics(self) -> List[MetricsSnapshot]:
        """Get all round metrics."""
        return self.round_metrics.copy()

    def get_custom_metric(self, metric_name: str) -> List[Any]:
        """Get custom metric values."""
        return self.custom_metrics.get(metric_name, [])

    def get_total_runtime_seconds(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time

    def reset(self) -> None:
        """Clear all metrics."""
        self.round_metrics.clear()
        self.custom_metrics.clear()
        self.start_time = time.time()
        self.total_messages = 0
