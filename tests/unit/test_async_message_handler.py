"""Tests for AsyncMessageHandler."""

import asyncio
import pytest
from src.graph.graph_manager import GraphManager
from src.communication.drivers.in_memory_driver import InMemoryDriver
from src.communication.async_message_handler import AsyncMessageHandler
from src.communication.message import Message


@pytest.fixture
def simple_graph():
    """Create simple graph."""
    graph = GraphManager()
    for i in range(4):
        graph.add_vertex(i)
    graph.add_edge(0, 1, weight=10)
    return graph


@pytest.fixture
def transport_driver(simple_graph):
    """Create transport driver."""
    return InMemoryDriver(simple_graph)


class TestAsyncMessageHandlerBasics:
    """Test basic async message handler functionality."""

    def test_handler_initialization(self, transport_driver):
        """Should initialize async message handler."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        assert handler.node_id == 0
        assert handler.transport is transport_driver
        assert isinstance(handler.incoming_queue, asyncio.Queue)
        assert isinstance(handler.outgoing_queue, asyncio.Queue)

    def test_synchronous_send(self, transport_driver, simple_graph):
        """Should send message synchronously."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)
        msg = Message(
            sender=0,
            recipient=1,
            round_num=1,
            payload={},
        )

        # Should not raise
        handler.synchronous_send(1, msg)

    def test_synchronous_receive(self, transport_driver):
        """Should receive messages synchronously."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        messages = handler.synchronous_receive()

        assert isinstance(messages, list)

    def test_async_send_message(self, transport_driver):
        """Should queue message for async send."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)
        msg = Message(
            sender=0,
            recipient=1,
            round_num=1,
            payload={},
        )

        # Run async method synchronously for testing
        async def run_test():
            await handler.send_message_async(1, msg)
            return not handler.outgoing_queue.empty()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result

    def test_async_send_batch(self, transport_driver):
        """Should queue batch for async send."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)
        messages = [
            Message(
                sender=0,
                recipient=1,
                round_num=1,
                payload={},
            ),
            Message(
                sender=0,
                recipient=2,
                round_num=1,
                payload={},
            ),
        ]

        async def run_test():
            await handler.send_batch_async(messages)
            return handler.outgoing_queue.qsize()

        loop = asyncio.new_event_loop()
        count = loop.run_until_complete(run_test())
        loop.close()

        assert count >= 2


class TestAsyncMessageHandlerRegistration:
    """Test message handler registration."""

    def test_register_handler(self, transport_driver):
        """Should register message handler."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        def dummy_handler(msg):
            pass

        handler.register_handler("test_type", dummy_handler)

        assert "test_type" in handler.message_handlers
        assert handler.message_handlers["test_type"] == dummy_handler

    def test_unregister_handler(self, transport_driver):
        """Should unregister message handler."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        def dummy_handler(msg):
            pass

        handler.register_handler("test_type", dummy_handler)
        handler.unregister_handler("test_type")

        assert "test_type" not in handler.message_handlers

    def test_handle_message_with_registered_handler(self, transport_driver):
        """Should call registered handler for message."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        processed_messages = []

        def sync_handler(msg):
            processed_messages.append(msg)

        # Register with the key that will be generated
        msg = Message(
            sender=1,
            recipient=0,
            round_num=1,
            payload={},
        )
        msg_key = f"msg_{msg.message_id}"
        handler.register_handler(msg_key, sync_handler)

        async def run_test():
            await handler.handle_message(msg)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_test())
        loop.close()

        assert len(processed_messages) == 1
        assert processed_messages[0] == msg

    def test_handle_message_without_handler(self, transport_driver):
        """Should queue message if no handler registered."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        msg = Message(
            sender=1,
            recipient=0,
            round_num=1,
            payload={},
        )

        async def run_test():
            await handler.handle_message(msg)
            return not handler.incoming_queue.empty()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result


class TestAsyncMessageHandlerQueue:
    """Test message queue operations."""

    def test_incoming_queue_operations(self, transport_driver):
        """Should handle incoming queue operations."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        msg = Message(
            sender=1,
            recipient=0,
            round_num=1,
            payload={},
        )

        async def run_test():
            await handler.incoming_queue.put(msg)
            retrieved = await handler.incoming_queue.get()
            return retrieved == msg

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result

    def test_outgoing_queue_operations(self, transport_driver):
        """Should handle outgoing queue operations."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        msg = Message(
            sender=0,
            recipient=1,
            round_num=1,
            payload={},
        )

        async def run_test():
            await handler.outgoing_queue.put(msg)
            retrieved = await handler.outgoing_queue.get()
            return retrieved == msg

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result

    def test_receive_messages_async(self, transport_driver):
        """Should receive messages asynchronously."""
        handler = AsyncMessageHandler(transport_driver, node_id=0)

        async def run_test():
            messages = await handler.receive_messages_async()
            return isinstance(messages, list)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()

        assert result
