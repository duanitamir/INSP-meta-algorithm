"""In-memory transport driver using MessageQueue (for testing)."""

from typing import List, Dict
from src.communication.transport_driver import TransportDriver
from src.communication.message import Message
from src.communication.message_queue import MessageQueue
from src.graph.graph_manager import GraphManager


class InMemoryDriver(TransportDriver):
    """In-memory message transport using existing MessageQueue.

    Used for testing. In production, swap with TCP or gRPC driver.
    """

    def __init__(self, graph: GraphManager):
        """Initialize in-memory driver.

        Args:
            graph: Shared graph reference
        """
        self.graph = graph
        self.queue = MessageQueue(graph)

    def send_message(self, recipient_id: int, message: Message) -> None:
        """Send single message via queue.

        Args:
            recipient_id: Target node ID
            message: Message to send
        """
        self.queue.send(message)

    def receive_messages(self, recipient_id: int = None) -> List[Message]:
        """Receive messages for a node.

        Args:
            recipient_id: Node to receive messages for

        Returns:
            List of messages
        """
        if recipient_id is None:
            return []
        return self.queue.get_messages(recipient_id)

    def send_batch(self, messages: List[Message]) -> None:
        """Send multiple messages.

        Args:
            messages: List of messages to send
        """
        if messages:
            self.queue.send_batch(messages)
