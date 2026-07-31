"""Local proposal-policy contract shared by registered algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.simulation.local_node_context import LocalNodeContext


@dataclass
class AlgorithmMetadata:
    """Registry-facing description of one local proposal policy."""

    name: str
    description: str
    version: str
    authors: List[str]
    references: List[str]
    properties: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = {}


class MatchingAlgorithm(ABC):
    """An algorithm that ranks or selects direct neighbours locally."""

    @property
    @abstractmethod
    def metadata(self) -> AlgorithmMetadata:
        """Return descriptive registry metadata."""

    @abstractmethod
    def propose_to_neighbors(
        self, node_id: int, neighbors: List[int], context: "LocalNodeContext"
    ) -> Dict[int, float]:
        """Return weighted proposals addressed only to direct neighbours."""
