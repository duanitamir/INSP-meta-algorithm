"""Base class for all distributed protocol messages."""

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedMessage(ABC):
    """Base class for all immutable distributed protocol messages.

    All distributed messages share common validation logic for standard fields.
    Subclasses define their own fields and must have sender_node_id and round_num.
    """

    def validate_base_fields(self, round_num: int, weight: float) -> None:
        """Validate common message fields.

        Args:
            round_num: Algorithm round number
            weight: Weight/fitness value

        Raises:
            ValueError: If round_num < 0 or weight < 0
        """
        if round_num < 0:
            raise ValueError(f"round_num must be >= 0, got {round_num}")

        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
