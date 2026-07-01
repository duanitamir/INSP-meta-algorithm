"""Abstract transport driver for message passing between nodes."""

from abc import ABC, abstractmethod
from typing import List, Dict
from src.communication.message import Message


class TransportDriver(ABC):
    """Abstract base class for message transport between nodes.

    Allows swapping in-memory queue for real network (TCP, gRPC, etc.)
    without changing application logic.
    """

    @abstractmethod
    def send_message(self, recipient_id: int, message: Message) -> None:
        """Send message to a recipient.

        Args:
            recipient_id: Target node ID
            message: Message to send
        """
        pass

    @abstractmethod
    def receive_messages(self, recipient_id: int = None) -> List[Message]:
        """Receive messages destined for this node.

        Args:
            recipient_id: Filter messages for specific recipient (optional)

        Returns:
            List of received messages
        """
        pass

    @abstractmethod
    def send_batch(self, messages: List[Message]) -> None:
        """Send multiple messages in one batch.

        Args:
            messages: List of messages to send
        """
        pass

    def name(self) -> str:
        """Return transport driver name."""
        return self.__class__.__name__
