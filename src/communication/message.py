from dataclasses import dataclass
from typing import Any
from src.utils.types import RoundNumber


@dataclass(frozen=True)
class Message:
    """Immutable message between nodes."""

    sender: int
    recipient: int
    payload: Any
    round_num: RoundNumber
    message_id: int | None = None

    def __hash__(self) -> int:
        return hash((self.sender, self.recipient, self.round_num, self.message_id))
