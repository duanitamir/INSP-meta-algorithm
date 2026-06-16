# Metrics Module

## Overview

The Metrics module collects performance metrics throughout simulation execution.

## Responsibilities

- Collect per-round metrics
- Track custom metrics
- Provide metrics snapshots
- Calculate aggregate statistics

## Key Classes

### MetricsSnapshot

Immutable metrics for a single round.

```python
snapshot = MetricsSnapshot(
    round_num=RoundNumber(1),
    messages_sent=45,
    messages_received=35,
    active_nodes=20,
    matched_nodes=15,
)
```

### MetricsCollector

Accumulates metrics during simulation.

```python
collector = MetricsCollector()
collector.record_round(RoundNumber(0), messages_sent=45)
metrics = collector.get_metrics_snapshot()
```

## Public API

### MetricsSnapshot

**Fields:**
- `round_num: RoundNumber` - Round number
- `messages_sent: int` - Messages sent this round
- `messages_received: int` (default: 0) - Messages received
- `active_nodes: int` (default: 0) - Nodes that sent messages
- `matched_nodes: int` (default: 0) - Nodes that are matched
- `dormant_nodes: int` (default: 0) - Inactive nodes
- `converged: bool` (default: False) - Algorithm converged
- `unmatched_vertices: int` (default: 0) - Unmatched count
- `round_duration_ms: float` (default: 0.0) - Round execution time

**Methods:**
- `to_dict() -> Dict[str, Any]` - Convert to dictionary

### MetricsCollector

**Recording:**
- `record_round(round_num, messages_sent, **kwargs)` - Record round metrics
- `record_custom_metric(metric_name, value)` - Record custom metric

**Retrieval:**
- `get_metrics_snapshot()` - Get current round metrics
- `get_all_metrics()` - Get all round metrics
- `get_custom_metric(metric_name)` - Get custom metric values

**Information:**
- `get_total_runtime_seconds()` - Get elapsed time
- `total_messages` - Total messages sent

**Maintenance:**
- `reset()` - Clear all metrics

## Examples

### Record Metrics During Simulation

```python
collector = MetricsCollector()

for round in range(10):
    messages_sent = run_round()
    collector.record_round(
        round_num=RoundNumber(round),
        messages_sent=messages_sent,
        active_nodes=get_active_count(),
        matched_nodes=get_matched_count(),
    )
```

### Query Metrics

```python
# Get latest metrics
latest = collector.get_metrics_snapshot()
print(f"Messages this round: {latest.messages_sent}")

# Get all metrics
all_metrics = collector.get_all_metrics()
total_msgs = sum(m.messages_sent for m in all_metrics)
print(f"Total messages: {total_msgs}")
```

### Record Custom Metrics

```python
collector.record_custom_metric("matching_size", 45)
collector.record_custom_metric("matching_size", 48)
collector.record_custom_metric("matching_size", 49)

sizes = collector.get_custom_metric("matching_size")
print(f"Matching sizes: {sizes}")
```

### Compute Statistics

```python
all_metrics = collector.get_all_metrics()

# Total runtime
runtime = collector.get_total_runtime_seconds()

# Average messages per round
avg_msgs = sum(m.messages_sent for m in all_metrics) / len(all_metrics)

# Peak messages in a round
peak_msgs = max(m.messages_sent for m in all_metrics)

print(f"Runtime: {runtime:.2f}s")
print(f"Average messages/round: {avg_msgs:.1f}")
print(f"Peak messages: {peak_msgs}")
```

## Standard Metrics

### Per-Round

- `messages_sent` - Messages sent this round
- `active_nodes` - Nodes that participated
- `matched_nodes` - Nodes in a matching
- `converged` - Convergence status

### Aggregate

- `total_messages` - Sum of messages across all rounds
- `total_runtime_seconds` - Elapsed wall-clock time
- Custom metrics via `record_custom_metric()`

## Design Notes

- Metrics collected automatically during simulation
- No performance penalty (metrics are cheap)
- Snapshots provide point-in-time view
- Custom metrics allow algorithm-specific tracking
- Collector resets between simulations

## Performance

- O(1) to record metric
- O(n) to get all metrics (n = rounds)
- Memory: ~200 bytes per round

## Future Extensions

- Streaming metrics export
- Real-time metric visualization
- Metric aggregation across multiple runs
- Statistical analysis (mean, variance, percentiles)
- Performance profiling integration
