# Scheduler Module

## Overview

The Scheduler module implements round-based synchronous execution. It orchestrates graph, state, communication, and metrics components for distributed algorithm simulation.

## Responsibilities

- Execute round-by-round synchronous simulation
- Manage simulation lifecycle (init, run, terminate)
- Check termination conditions
- Coordinate state and message updates

## Key Classes

### Scheduler

Main orchestrator for simulation.

```python
scheduler = Scheduler(graph, config)
scheduler.initialize()

while scheduler.execute_round():
    # Each round executes
    pass

# Or:
rounds = scheduler.run_until_termination(termination_callback=check_func)
```

### SimulationConfig

Configuration for scheduler behavior.

```python
config = SimulationConfig(
    max_rounds=1000,
    max_messages=None,
    debug=False,
    random_seed=42,
    collect_snapshots=True,
    collect_message_traces=True,
)
scheduler = Scheduler(graph, config)
```

## Public API

### Scheduler

**Properties:**
- `current_round: RoundNumber` - Current round
- `is_running: bool` - Execution status
- `is_terminated: bool` - Termination status

**Initialization:**
- `reset()` - Reset to initial state
- `initialize()` - Prepare for execution

**Execution:**
- `execute_round() -> bool` - Execute one round, return True if continue
- `run_until_termination(callback) -> RoundNumber` - Run to completion

**Accessors:**
- `state_store: StateStore` - Access to all node states
- `message_queue: MessageQueue` - Access to message queue
- `metrics: MetricsCollector` - Access to metrics

**Termination Check:**
- `check_no_progress() -> bool` - Check if no messages sent

### SimulationConfig

**Fields:**
- `max_rounds: int` (default: 1000) - Maximum rounds before timeout
- `max_messages: int | None` (default: None) - Maximum total messages
- `debug: bool` (default: False) - Enable debug output
- `random_seed: int | None` (default: None) - Random seed for reproducibility
- `collect_snapshots: bool` (default: True) - Save state snapshots
- `collect_message_traces: bool` (default: True) - Track messages

## Examples

### Basic Execution

```python
scheduler = Scheduler(graph)
scheduler.initialize()

for _ in range(100):
    if not scheduler.execute_round():
        break

print(f"Completed {scheduler.current_round} rounds")
```

### Execute to Completion

```python
def check_termination(state_store, round_num, messages_sent):
    # Check custom termination condition
    if messages_sent == 0:
        return True, "no_progress"
    return False, None

rounds = scheduler.run_until_termination(
    termination_callback=check_termination
)
```

### Access Results

```python
# Get state at current round
state = scheduler.state_store.get_node_state(node_id)

# Get messages from current round
messages = scheduler.message_queue.get_messages(node_id)

# Get metrics
metrics = scheduler.metrics.get_all_metrics()
for m in metrics:
    print(f"Round {m.round_num}: {m.messages_sent} messages")
```

### Snapshot and Restore

```python
# Before running
initial_snapshot = scheduler.state_store.create_snapshot(RoundNumber(0))

# After running
scheduler.run_until_termination()

# Reset and restore
scheduler.reset()
scheduler.state_store.restore_snapshot(initial_snapshot)

# Can continue from any point
```

## Round Execution Flow

```
execute_round():
  1. Record metrics for current round
  2. Increment round counter
  3. Check max_rounds limit
  4. Check max_messages limit
  5. Create snapshot (if enabled)
  6. Return True to continue, False to stop
```

## Termination Conditions

Termination occurs when:

1. `max_rounds` exceeded
2. `max_messages` exceeded
3. `termination_callback` returns True
4. `execute_round()` returns False

## Design Notes

- Synchronous execution (not async)
- All nodes execute in lockstep
- No concurrent node execution (single-threaded)
- Deterministic (same input = same output)
- Optional snapshots for memory efficiency

## Performance

- O(1) per round execution
- Overhead minimal (~1ms per round)
- Memory: snapshot size = O(n) where n = vertices

## Future Extensions

- Asynchronous execution modes
- Parallel node execution
- Network latency simulation
- Fault injection
- Checkpointing/resumption
