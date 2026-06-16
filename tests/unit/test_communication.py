import pytest
from src.graph import GraphManager
from src.communication import Message, MessageQueue
from src.utils.types import RoundNumber


class TestMessage:
    def test_create_message(self):
        msg = Message(1, 2, "test_payload", RoundNumber(0))
        assert msg.sender == 1
        assert msg.recipient == 2
        assert msg.payload == "test_payload"
        assert msg.round_num == RoundNumber(0)

    def test_message_immutable(self):
        msg = Message(1, 2, "test", RoundNumber(0))
        with pytest.raises(AttributeError):
            msg.sender = 3


class TestMessageQueue:
    def test_create_queue(self, simple_graph):
        queue = MessageQueue(simple_graph)
        assert queue.graph == simple_graph

    def test_send_message(self, simple_graph):
        queue = MessageQueue(simple_graph)
        msg = Message(1, 2, "test", RoundNumber(0))
        queue.send(msg)
        assert queue.inbox_size(2) == 1

    def test_send_invalid_recipient(self, simple_graph):
        queue = MessageQueue(simple_graph)
        msg = Message(1, 99, "test", RoundNumber(0))
        with pytest.raises(ValueError):
            queue.send(msg)

    def test_get_messages(self, simple_graph):
        queue = MessageQueue(simple_graph)
        msg = Message(1, 2, "test", RoundNumber(0))
        queue.send(msg)
        messages = queue.get_messages(2)
        assert len(messages) == 1
        assert messages[0].payload == "test"

    def test_get_messages_clears_inbox(self, simple_graph):
        queue = MessageQueue(simple_graph)
        msg = Message(1, 2, "test", RoundNumber(0))
        queue.send(msg)
        queue.get_messages(2)
        messages = queue.get_messages(2)
        assert len(messages) == 0

    def test_peek_messages(self, simple_graph):
        queue = MessageQueue(simple_graph)
        msg = Message(1, 2, "test", RoundNumber(0))
        queue.send(msg)
        peeked = queue.peek_messages(2)
        assert len(peeked) == 1
        # Inbox should still have message
        messages = queue.get_messages(2)
        assert len(messages) == 1

    def test_has_messages(self, simple_graph):
        queue = MessageQueue(simple_graph)
        assert not queue.has_messages(2)
        msg = Message(1, 2, "test", RoundNumber(0))
        queue.send(msg)
        assert queue.has_messages(2)

    def test_fifo_order(self, simple_graph):
        queue = MessageQueue(simple_graph)
        for i in range(5):
            msg = Message(1, 2, f"msg_{i}", RoundNumber(0))
            queue.send(msg)
        messages = queue.get_messages(2)
        for i, msg in enumerate(messages):
            assert msg.payload == f"msg_{i}"

    def test_send_batch(self, simple_graph):
        queue = MessageQueue(simple_graph)
        messages = [
            Message(1, 2, f"msg_{i}", RoundNumber(0))
            for i in range(3)
        ]
        queue.send_batch(messages)
        assert queue.inbox_size(2) == 3

    def test_total_messages_sent(self, simple_graph):
        queue = MessageQueue(simple_graph)
        assert queue.total_messages_sent() == 0
        for i in range(3):
            msg = Message(1, 2, f"msg_{i}", RoundNumber(0))
            queue.send(msg)
        assert queue.total_messages_sent() == 3

    def test_total_messages_pending(self, simple_graph):
        queue = MessageQueue(simple_graph)
        for i in range(3):
            msg = Message(1, 2, f"msg_{i}", RoundNumber(0))
            queue.send(msg)
        for i in range(2):
            msg = Message(1, 3, f"msg_{i}", RoundNumber(0))
            queue.send(msg)
        assert queue.total_messages_pending() == 5
