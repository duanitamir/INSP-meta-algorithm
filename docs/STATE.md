# State Management Module

## Overview

The State module manages externalized node state for the distributed simulation. All mutable state lives in StateStore, not in node objects.

## Key Concept: Externalized State

Traditional OOP stores state in objects:
```python
node.matched_to = 5  # state in node object
```

This framework externalizes state:
```python
state_store.get_node_state(node_id).set_matched_to(5)  # state in store
```

**Why?** Enables snapshots, serialization, state sharing across algorithms, and time-travel debugging.

## Responsibilities

- Manage per-node state (NodeState)
- Manage global state (MetaState)
- Create/restore state snapshots
- Provide transactional updates

## Key Classes

### NodeState

Mutable state for a single node.

```python
state = NodeState(node_id=1)
state.set("phase", "dormant")
state.set_matched_to(None)
phase = state.get("phase")
state.set_matched_to(5)
```

### StateStore

Central repository for all node states.

```python
store = StateStore(graph)
state = store.get_node_state(1)  # Defensive copy
store.update_node_state(1, state)
snapshot = store.create_snapshot(RoundNumber(0))
store.restore_snapshot(snapshot)
```

### MetaState

Global simulation state (round, convergence).

```python
meta = MetaState(round_num=RoundNumber(0))
meta_converged = meta.with_convergence("reason")
```

### StateSnapshot

Immutable snapshot of all state at a round.

```python
snapshot = store.create_snapshot(RoundNumber(5))
state = snapshot.get_node_state(node_id)
```

## Public API

### NodeState

- `set(key, value)` - Set state variable
- `get(key, default=None)` - Get state variable
- `update(key, fn)` - Apply function to value
- `exists(key)` - Check if exists
- `delete(key)` - Delete variable
- `keys()` - Get all keys as frozenset
- `clone()` - Deep copy
- `set_matched_to(vertex_id)` - Set match
- `get_matched_to()` - Get match
- `is_matched()` - Check if matched
- `to_dict()` - Convert to dict

### StateStore

- `get_node_state(node_id)` - Get state (copy)
- `get_all_states()` - Get all states (copies)
- `update_node_state(node_id, state)` - Update node state
- `update_all_states(states)` - Atomic update all
- `create_snapshot(round_num)` - Save snapshot
- `restore_snapshot(snapshot)` - Restore to snapshot
- `get_snapshots()` - Get all saved snapshots
- `clear_snapshots()` - Clear all snapshots
- `get_meta_state()` - Get global state
- `update_meta_state(meta)` - Update global state

## Examples

### Initialize state for a node

```python
store = StateStore(graph)
state = store.get_node_state(node_id)
state.set("phase", "dormant")
state.set("matched_to", None)
store.update_node_state(node_id, state)
```

### Create and restore snapshots

```python
# Save current state
snapshot = store.create_snapshot(RoundNumber(5))

# Modify state
state = store.get_node_state(1)
state.set_matched_to(2)
store.update_node_state(1, state)

# Restore to snapshot
store.restore_snapshot(snapshot)
# Now matched_to is None again
```

### Check matching status

```python
state = store.get_node_state(1)
if state.is_matched():
    matched_to = state.get_matched_to()
    print(f"Node 1 is matched to {matched_to}")
```

## Design Notes

- StateStore is initialized with all nodes from graph
- Defensive copies returned to prevent external modification
- Snapshots are immutable and composable
- MetaState uses immutable value pattern (with_* methods)
- Snapshot restoration is point-in-time restore
- All node states are cloned for safety

## Performance

- O(1) state lookup
- O(n) for full snapshot (n = vertices)
- O(n) for restore
- Memory: ~1MB per snapshot for 10k nodes

## Future Extensions

- Differential snapshots (only changed state)
- Distributed serialization (JSON export)
- State versioning (track state history)
- Circular buffer of snapshots (bounded memory)
