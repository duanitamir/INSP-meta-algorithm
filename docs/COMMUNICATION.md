# Communication Module

## Overview

The Communication module implements asynchronous message passing between nodes using FIFO message queues.

## Responsibilities

- Define immutable Message class
- Manage per-node message inboxes
- Provide FIFO message delivery
- Track message statistics

## Key Classes

### Message

Immutable message sent between nodes.

```python
msg = Message(sender=1, recipient=2, payload="data", round_num=RoundNumber(0))
print(msg.sender)      # 1
print(msg.recipient)   # 2
print(msg.payload)     # "data"
print(msg.round_num)   # RoundNumber(0)
```

### MessageQueue

FIFO message queue with per-node inboxes.

```python
queue = MessageQueue(graph)
queue.send(msg)
messages = queue.get_messages(node_id)  # Gets and clears inbox
```

## Public API

### Message

- `sender: int` - Source node
- `recipient: int` - Destination node
- `payload: Any` - Message content (any type)
- `round_num: RoundNumber` - Round sent
- `message_id: int | None` - Optional message ID

### MessageQueue

- `send(message)` - Send message to inbox
- `send_batch(messages)` - Send multiple messages
- `get_messages(node_id)` - Get all messages and clear inbox
- `peek_messages(node_id)` - View without clearing
- `has_messages(node_id)` - Check if messages pending
- `inbox_size(node_id)` - Get inbox size
- `total_messages_sent()` - Total sent this session
- `total_messages_pending()` - Total in all inboxes
- `reset()` - Clear all messages

## Examples

### Send Point-to-Point Message

```python
msg = Message(sender=1, recipient=2, payload="PROPOSE", round_num=RoundNumber(1))
queue.send(msg)
```

### Receive Messages

```python
messages = queue.get_messages(2)
for msg in messages:
    print(f"From {msg.sender}: {msg.payload}")
```

### Peek Messages Without Clearing

```python
messages = queue.peek_messages(2)
print(f"Will receive {len(messages)} messages")
messages = queue.get_messages(2)  # Still get them
```

### Batch Send

```python
messages = [
    Message(1, 2, "msg1", RoundNumber(0)),
    Message(1, 3, "msg2", RoundNumber(0)),
    Message(1, 4, "msg3", RoundNumber(0)),
]
queue.send_batch(messages)
```

## Design Notes

- Messages are immutable (frozen dataclasses)
- FIFO semantics per node
- No message routing (direct node-to-node only)
- Payload can be any Python type (not limited to strings)
- Messages are stamped with round number for tracking

## Semantics

### FIFO Ordering

Messages to same node are processed in order sent:

```
Sent: [A, B, C]
Received: [A, B, C]
```

### Per-Node Isolation

Each node has separate inbox:

```
send(1→2, "msg1")
send(1→3, "msg2")
get_messages(2)  # ["msg1"]
get_messages(3)  # ["msg2"]
```

### Inbox Clearing

`get_messages()` empties inbox:

```
send(msg)
messages1 = get_messages(2)  # ["msg"]
messages2 = get_messages(2)  # [] - cleared
```

## Performance

- O(1) send
- O(k) receive (k = messages in inbox)
- O(n) total_messages_pending (n = all nodes)
- Memory: ~100 bytes per message

## Future Extensions

- Network latency simulation
- Message loss/failure
- Broadcast messages (one sender, many recipients)
- Message filtering/routing
