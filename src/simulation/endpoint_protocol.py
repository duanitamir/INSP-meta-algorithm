"""Endpoint-owned proposal and confirmation protocol."""

from collections.abc import Callable, Mapping, Sequence

from src.communication.message import Message
from src.graph.local_graph import LocalGraph
from src.state.node import NodeState


class EndpointProtocol:
    """Own one node's tentative matching state and protocol messages."""

    def __init__(
        self,
        node_id: int,
        graph: LocalGraph,
        state: NodeState,
        send_message: Callable[[Message], None],
        proposal_timeout: int,
    ) -> None:
        self.node_id = node_id
        self.graph = graph
        self.state = state
        self._send_message = send_message
        self.proposal_timeout = proposal_timeout

    def reset(self, state: NodeState) -> None:
        """Use replacement node state after a node reset."""
        self.state = state

    def start_proposal(self, neighbor_id: int, weight: float, local_time: int, round_number: int) -> None:
        """Begin a local negotiation with one direct neighbor."""
        if neighbor_id not in self.graph.neighbors():
            raise ValueError(f"Node {neighbor_id} is not a neighbor of node {self.node_id}")
        if self.state.is_matched() or self.state.get(NodeState.TENTATIVE_PARTNER) is not None:
            return
        attempt = self.state.begin_tentative_match(
            neighbor_id,
            local_time + self.proposal_timeout,
            proposing=True,
        )
        self._send(neighbor_id, "PROPOSE", round_number, attempt=attempt, weight=weight)

    def process_messages(
        self, messages: Sequence[Message], local_time: int, round_number: int
    ) -> None:
        """Process addressed endpoint messages and resolve received proposals."""
        proposals: dict[int, tuple[float, int]] = {}
        for message in messages:
            if message.sender not in self.graph.neighbors() or not isinstance(message.payload, dict):
                continue
            message_type = message.payload.get("type")
            if message_type == "PROPOSE":
                attempt = int(message.payload.get("attempt", 0))
                previous = proposals.get(message.sender)
                if previous is None or attempt > previous[1]:
                    proposals[message.sender] = (float(message.payload["weight"]), attempt)
            elif message_type == "ACCEPT":
                self._receive_accept(
                    message.sender,
                    int(message.payload.get("attempt", 0)),
                    local_time,
                    round_number,
                )
            elif message_type == "REJECT":
                self._receive_reject(message.sender, int(message.payload.get("attempt", 0)))
            elif message_type == "CONFIRM":
                self._receive_confirm(
                    message.sender,
                    bool(message.payload.get("acknowledgement", False)),
                    int(message.payload.get("attempt", 0)),
                    round_number,
                )
            elif message_type == "CANCEL":
                self._receive_cancel(message.sender, int(message.payload.get("attempt", 0)))

        if proposals:
            self._resolve_received_proposals(
                [(sender_id, weight, attempt) for sender_id, (weight, attempt) in proposals.items()],
                local_time,
                round_number,
            )

    def select_proposal(
        self, proposals: Mapping[int, float], local_time: int, round_number: int
    ) -> None:
        """Start the locally best proposal without committing a final match."""
        if proposals and not self.state.is_matched():
            neighbor_id, weight = max(proposals.items(), key=lambda item: (item[1], -item[0]))
            self.start_proposal(neighbor_id, weight, local_time, round_number)

    def expire_if_needed(self, local_time: int, round_number: int) -> None:
        """Cancel an unanswered tentative negotiation after its local deadline."""
        partner_id = self.state.get(NodeState.TENTATIVE_PARTNER)
        deadline = self.state.get(NodeState.PROPOSAL_DEADLINE)
        if partner_id is None or deadline is None or local_time < deadline:
            return
        attempt = int(self.state.get(NodeState.PROPOSAL_ATTEMPT, 0))
        self.state.clear_tentative_match()
        self._send(partner_id, "CANCEL", round_number, attempt=attempt)

    def _resolve_received_proposals(
        self, proposals: Sequence[tuple[int, float, int]], local_time: int, round_number: int
    ) -> None:
        if self.state.is_matched():
            for sender_id, _, attempt in proposals:
                self._send(sender_id, "REJECT", round_number, attempt=attempt)
            return

        partner_id = self.state.get(NodeState.TENTATIVE_PARTNER)
        active_attempt = self.state.get(NodeState.PROPOSAL_ATTEMPT)
        if partner_id is not None:
            for sender_id, weight, attempt in proposals:
                if (
                    sender_id == partner_id
                    and attempt == active_attempt
                    and not self.state.get(NodeState.PROPOSAL_PENDING)
                ):
                    self._send(sender_id, "ACCEPT", round_number, attempt=attempt, weight=weight)
                else:
                    self._send(sender_id, "REJECT", round_number, attempt=attempt)
            return

        winner_id, winner_weight, winner_attempt = max(
            proposals, key=lambda item: (item[1], -item[0])
        )
        self.state.begin_tentative_match(
            winner_id,
            local_time + self.proposal_timeout,
            proposing=False,
            attempt=winner_attempt,
        )
        for sender_id, _, attempt in proposals:
            if sender_id == winner_id:
                self._send(
                    sender_id,
                    "ACCEPT",
                    round_number,
                    attempt=attempt,
                    weight=winner_weight,
                )
            else:
                self._send(sender_id, "REJECT", round_number, attempt=attempt)

    def _receive_accept(
        self, sender_id: int, attempt: int, local_time: int, round_number: int
    ) -> None:
        if (
            self.state.get(NodeState.TENTATIVE_PARTNER) != sender_id
            or not self.state.get(NodeState.PROPOSAL_PENDING)
            or self.state.get(NodeState.PROPOSAL_ATTEMPT) != attempt
        ):
            return
        self.state.begin_tentative_match(
            sender_id,
            local_time + self.proposal_timeout,
            proposing=False,
            attempt=attempt,
        )
        self._send(sender_id, "CONFIRM", round_number, attempt=attempt, acknowledgement=False)

    def _receive_reject(self, sender_id: int, attempt: int) -> None:
        if (
            self.state.get(NodeState.TENTATIVE_PARTNER) == sender_id
            and self.state.get(NodeState.PROPOSAL_ATTEMPT) == attempt
        ):
            self.state.clear_tentative_match()

    def _receive_confirm(
        self, sender_id: int, acknowledgement: bool, attempt: int, round_number: int
    ) -> None:
        if (
            self.state.get(NodeState.TENTATIVE_PARTNER) != sender_id
            or self.state.get(NodeState.PROPOSAL_ATTEMPT) != attempt
        ):
            return
        self.state.set_matched_to(sender_id)
        self.state.clear_tentative_match()
        if not acknowledgement:
            self._send(sender_id, "CONFIRM", round_number, attempt=attempt, acknowledgement=True)

    def _receive_cancel(self, sender_id: int, attempt: int) -> None:
        if (
            self.state.get(NodeState.TENTATIVE_PARTNER) == sender_id
            and self.state.get(NodeState.PROPOSAL_ATTEMPT) == attempt
        ):
            self.state.clear_tentative_match()

    def _send(
        self, recipient_id: int, message_type: str, round_number: int, *, attempt: int, **payload: object
    ) -> None:
        self._send_message(
            Message(
                sender=self.node_id,
                recipient=recipient_id,
                payload={"type": message_type, "attempt": attempt, **payload},
                round_num=round_number,
            )
        )
