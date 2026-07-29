"""Node-bound facade over recipient-scoped transport."""

from typing import Any, Dict, Iterable, List

from src.communication.message import Message, SemanticMessage
from src.communication.transport import InMemoryTransport


class NodeCommunicator:
    """Send only from one node and receive only that node's addressed mail."""

    def __init__(self, node_id: int, transport: InMemoryTransport):
        self.node_id = node_id
        self._transport = transport

    def send_to(self, recipient_id: int, message_type: str, payload: Dict[str, Any]) -> None:
        self._transport.send(
            self.node_id,
            recipient_id,
            SemanticMessage(
                sender=self.node_id,
                recipient=recipient_id,
                message_type=message_type,
                payload=payload,
            ),
        )

    def send_message(self, message: Message) -> None:
        """Send a low-level endpoint-protocol message from this node."""
        self._transport.send(self.node_id, message.recipient, message)

    def broadcast_to_neighbors(
        self, neighbors: Iterable[int], message_type: str, payload: Dict[str, Any]
    ) -> None:
        for neighbor_id in neighbors:
            self.send_to(neighbor_id, message_type, payload)

    def receive_messages(self) -> List[Message | SemanticMessage]:
        return self._transport.receive(self.node_id)
