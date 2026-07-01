"""Asynchronous message handler for distributed nodes."""

import asyncio
from typing import List, Dict, Optional, Callable
from src.communication.transport_driver import TransportDriver
from src.communication.message import Message


class AsyncMessageHandler:
    """Handles asynchronous message sending and receiving.

    Decouples message delivery from round execution, allowing nodes to
    communicate without blocking the algorithm execution.
    """

    def __init__(self, transport_driver: TransportDriver, node_id: int):
        """Initialize async message handler.

        Args:
            transport_driver: Transport layer for message delivery
            node_id: This node's ID
        """
        self.transport = transport_driver
        self.node_id = node_id
        self.incoming_queue: asyncio.Queue = asyncio.Queue()
        self.outgoing_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}

    async def send_message_async(self, recipient_id: int, message: Message) -> None:
        """Send message asynchronously.

        Args:
            recipient_id: Target node ID
            message: Message to send
        """
        await self.outgoing_queue.put((recipient_id, message))
        # Process immediately
        self.transport.send_message(recipient_id, message)

    async def send_batch_async(self, messages: List[Message]) -> None:
        """Send multiple messages asynchronously.

        Args:
            messages: List of messages to send
        """
        for msg in messages:
            await self.outgoing_queue.put(msg)
        self.transport.send_batch(messages)

    async def receive_messages_async(self) -> List[Message]:
        """Receive messages asynchronously.

        Returns:
            List of received messages
        """
        received = self.transport.receive_messages(self.node_id)
        for msg in received:
            await self.incoming_queue.put(msg)
        return received

    async def process_message_loop(self) -> None:
        """Process messages in background loop.

        This allows message handling without blocking other operations.
        """
        try:
            while True:
                # Check for incoming messages (non-blocking)
                received = self.transport.receive_messages(self.node_id)
                for msg in received:
                    await self.handle_message(msg)

                # Small delay to prevent tight loop
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

    async def handle_message(self, message: Message) -> None:
        """Handle incoming message with registered handler.

        Args:
            message: Message to process
        """
        # Use message_id as default handler key (or implement custom type field)
        msg_key = getattr(message, "message_type", f"msg_{message.message_id}")
        if msg_key in self.message_handlers:
            handler = self.message_handlers[msg_key]
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        else:
            # Default: just queue for later processing
            await self.incoming_queue.put(message)

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register message type handler.

        Args:
            message_type: Type of message to handle
            handler: Callable to process message (can be async)
        """
        self.message_handlers[message_type] = handler

    def unregister_handler(self, message_type: str) -> None:
        """Unregister message type handler.

        Args:
            message_type: Type of message to stop handling
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]

    def synchronous_send(self, recipient_id: int, message: Message) -> None:
        """Synchronous message send (fallback).

        Args:
            recipient_id: Target node ID
            message: Message to send
        """
        self.transport.send_message(recipient_id, message)

    def synchronous_receive(self) -> List[Message]:
        """Synchronous message receive (fallback).

        Returns:
            List of received messages
        """
        return self.transport.receive_messages(self.node_id)
