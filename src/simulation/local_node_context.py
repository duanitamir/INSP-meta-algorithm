"""Node-local inputs exposed to distributed proposal algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.communication.message import Message
from src.config import DistributedAlgorithmConfig
from src.graph.local_graph import LocalGraph
from src.state.node import NodeState


@dataclass(frozen=True)
class LocalNodeContext:
    """All and only the inputs one distributed node may use for a proposal."""

    node_id: int
    graph: LocalGraph
    state: NodeState
    messages: Sequence[Message]
    config: DistributedAlgorithmConfig = field(default_factory=DistributedAlgorithmConfig)
    round_number: int = 0
    logical_time: int = 0
