"""Wattenhofer-inspired weighted local proposal policy."""

import random
from collections.abc import Callable
from typing import Dict, List, Tuple

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm
from src.communication.message import Message
from src.graph.local_graph import LocalGraph
from src.state.node import NodeState


class WattenhoferProtocol:
    """Endpoint-local state for the paper's weighted graph protocol."""

    CANDIDATE = "WATTENHOFER_CANDIDATE"
    SELECT = "WATTENHOFER_SELECT"
    ELIMINATE = "WATTENHOFER_ELIMINATE"
    MATCH = "WATTENHOFER_MATCH"
    CLEANUP = "WATTENHOFER_CLEANUP"
    SYNC = "WATTENHOFER_SYNC"
    execution_mode = "synchronous_rounds"
    VALID_STAGE = "valid"
    SELECT_STAGE = "select"
    ELIMINATE_STAGE = "eliminate"
    MATCH_STAGE = "match"
    CLEANUP_STAGE = "cleanup"

    def __init__(
        self,
        node_id: int,
        graph: LocalGraph,
        state: NodeState,
        send_message: Callable[[Message], None],
        node_count: int,
        rng: random.Random,
    ) -> None:
        self.node_id = node_id
        self.graph = graph
        self.state = state
        self.send_message = send_message
        self.node_count = node_count
        self.max_phases = max(1, (node_count - 1).bit_length())
        self.rng = rng
        self.phase = 0
        self.round = 0
        self.started = False
        self.local_candidate_neighbors: set[int] = set()
        self.valid_neighbors: set[int] = set()
        self.selected_neighbor: int | None = None
        self.imposed_neighbors: set[int] = set()
        self.chosen_imposed_neighbors: set[int] = set()
        self.retained_neighbors: set[int] = set()
        self.match_choice: int | None = None
        self.stage = self.VALID_STAGE
        self.stage_notices: set[int] = set()
        self.sync_notices: dict[tuple[int, int, str], set[int]] = {}
        self.terminal = False

    def begin_phase(self) -> None:
        """Execute Algorithm 6's local half-maximum candidate step."""
        eligible = [] if self.state.is_matched() else [
            neighbor
            for neighbor in self.graph.neighbors()
            if self.state.is_neighbor_eligible(neighbor)
        ]
        self.local_candidate_neighbors = set()
        self.valid_neighbors = set()
        self.stage = self.VALID_STAGE
        self.stage_notices = self._received_syncs(self.VALID_STAGE)
        self.imposed_neighbors = set()
        if not eligible:
            self._sync(self.VALID_STAGE)
            return
        weights = {neighbor: self.graph.get_edge_weight(self.node_id, neighbor) for neighbor in eligible}
        local_maximum = max(weights.values())
        self.local_candidate_neighbors = {
            neighbor for neighbor, weight in weights.items() if weight >= local_maximum / 2
        }
        for neighbor in self.local_candidate_neighbors:
            self.send_message(
                Message(
                    sender=self.node_id,
                    recipient=neighbor,
                    payload={"type": self.CANDIDATE, "phase": self.phase},
                    round_num=self.round,
                )
            )
        self._sync(self.VALID_STAGE)

    def tick(self, messages: List[Message]) -> None:
        """Consume addressed messages for the current protocol phase."""
        for message in messages:
            if (
                message.sender in self.local_candidate_neighbors
                and message.payload.get("type") == self.CANDIDATE
                and message.payload.get("phase") == self.phase
            ):
                self.valid_neighbors.add(message.sender)
            elif self._is_current_round(message, self.SELECT):
                self.imposed_neighbors.add(message.sender)
            elif self._is_current_round(message, self.ELIMINATE):
                self.retained_neighbors.add(message.sender)
            elif (
                self.match_choice == message.sender
                and self._is_current_round(message, self.MATCH)
            ):
                self._finalize_match(message.sender)
            elif message.payload.get("type") == self.CLEANUP:
                self.valid_neighbors.discard(message.sender)
                self.state.mark_neighbor_unavailable(message.sender)
            elif message.payload.get("type") == self.SYNC:
                key = (
                    message.payload.get("phase"),
                    message.payload.get("round"),
                    message.payload.get("stage"),
                )
                self.sync_notices.setdefault(key, set()).add(message.sender)
                if key == (self.phase, self.round, self.stage):
                    self.stage_notices.add(message.sender)
        self._advance_if_synchronized()

    def begin_select(self) -> None:
        """Execute Algorithm 8: choose one valid incident edge uniformly."""
        self.selected_neighbor = None
        self.chosen_imposed_neighbors = set()
        self.retained_neighbors = set()
        self.stage = self.SELECT_STAGE
        self.stage_notices = self._received_syncs(self.SELECT_STAGE)
        if not self.valid_neighbors:
            self._sync(self.SELECT_STAGE)
            return
        self.selected_neighbor = self.rng.choice(sorted(self.valid_neighbors))
        self._send(self.selected_neighbor, self.SELECT)
        self._sync(self.SELECT_STAGE)

    def begin_eliminate(self) -> None:
        """Execute Algorithm 9: choose at most one imposed edge uniformly."""
        self.chosen_imposed_neighbors = set()
        self.stage = self.ELIMINATE_STAGE
        self.stage_notices = self._received_syncs(self.ELIMINATE_STAGE)
        if not self.imposed_neighbors:
            self._sync(self.ELIMINATE_STAGE)
            return
        neighbor = self.rng.choice(sorted(self.imposed_neighbors))
        self.chosen_imposed_neighbors.add(neighbor)
        self.retained_neighbors.add(neighbor)
        self._send(neighbor, self.ELIMINATE)
        self._sync(self.ELIMINATE_STAGE)

    def begin_matching(self) -> None:
        """Execute Algorithm 10: make one random retained-edge choice."""
        self.match_choice = None
        self.stage = self.MATCH_STAGE
        self.stage_notices = self._received_syncs(self.MATCH_STAGE)
        if not self.retained_neighbors or self.state.is_matched():
            self._sync(self.MATCH_STAGE)
            return
        self.match_choice = self.rng.choice(sorted(self.retained_neighbors))
        self._send(self.match_choice, self.MATCH)
        self._sync(self.MATCH_STAGE)

    def begin_cleanup(self) -> None:
        """Synchronize cleanup before deciding whether the phase continues."""
        self.stage = self.CLEANUP_STAGE
        self.stage_notices = self._received_syncs(self.CLEANUP_STAGE)
        self._sync(self.CLEANUP_STAGE)

    def _finalize_match(self, neighbor: int) -> None:
        if self.state.is_matched():
            return
        self.state.set_matched_to(neighbor)
        self.valid_neighbors = set()
        for other in self.graph.neighbors():
            if other != neighbor:
                self.state.mark_neighbor_unavailable(other)
                self._send(other, self.CLEANUP)

    def finalize_match(self, neighbor: int) -> None:
        """Finalize a reciprocal match and execute its cleanup notification."""
        self._finalize_match(neighbor)

    def _is_current_round(self, message: Message, message_type: str) -> bool:
        return (
            message.sender in self.graph.neighbors()
            and message.payload.get("type") == message_type
            and message.payload.get("phase") == self.phase
            and message.payload.get("round") == self.round
        )

    def _send(self, recipient: int, message_type: str) -> None:
        self.send_message(
            Message(
                sender=self.node_id,
                recipient=recipient,
                payload={"type": message_type, "phase": self.phase, "round": self.round},
                round_num=self.round,
            )
        )

    def _sync(self, stage: str) -> None:
        for neighbor in self.graph.neighbors():
            self.send_message(
                Message(
                    sender=self.node_id,
                    recipient=neighbor,
                    payload={
                        "type": self.SYNC,
                        "phase": self.phase,
                        "round": self.round,
                        "stage": stage,
                    },
                    round_num=self.round,
                )
            )

    def _received_syncs(self, stage: str) -> set[int]:
        return set(self.sync_notices.get((self.phase, self.round, stage), set()))

    def _advance_if_synchronized(self) -> None:
        if self.terminal or self.stage_notices != set(self.graph.neighbors()):
            return
        if self.stage == self.VALID_STAGE:
            self.begin_select()
        elif self.stage == self.SELECT_STAGE:
            self.begin_eliminate()
        elif self.stage == self.ELIMINATE_STAGE:
            self.begin_matching()
        elif self.stage == self.MATCH_STAGE:
            self.begin_cleanup()
        else:
            self.round += 1
            self.imposed_neighbors = set()
            if self.valid_neighbors and not self.state.is_matched():
                self.begin_select()
            else:
                self.phase += 1
                if self.phase >= self.max_phases:
                    self.terminal = True
                else:
                    self.begin_phase()


class WattenhoferGraphMatching(MatchingAlgorithm):
    PARAMETERS = {
        "policy_weight": {"min": 0.0, "max": 2.0, "default": 1.0, "type": "float", "description": "Local policy combination weight"},
        "watt_phase_count": {"min": 1, "max": 20, "default": 10, "type": "integer", "description": "Local filtering phases"},
        "watt_rounds_per_phase": {"min": 2, "max": 20, "default": 10, "type": "integer", "description": "Local rounds per phase"},
        "watt_valid_threshold": {"min": 0.1, "max": 1.0, "default": 0.5, "type": "float", "description": "Normalized local edge threshold"},
        "watt_select_probability": {"min": 0.1, "max": 1.0, "default": 1.0, "type": "float", "description": "Local activation probability"},
        "watt_eliminate_probability": {"min": 0.1, "max": 1.0, "default": 0.5, "type": "float", "description": "Local candidate elimination probability"},
    }
    PARAMETER_DEFINITION = {"name": "wattenhofer_graph", "display_name": "Wattenhofer Weighted Matching", "parameters": {name: (spec["min"], spec["max"], lambda spec=spec: random.randint(spec["min"], spec["max"]) if spec["type"] == "integer" else random.uniform(spec["min"], spec["max"])) for name, spec in PARAMETERS.items()}}
    PARAMETER_DEFAULTS = {name: spec["default"] for name, spec in PARAMETERS.items()}

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {**self.PARAMETER_DEFAULTS, **(parameters or {})}
        self._metadata = AlgorithmMetadata("Wattenhofer Weighted Matching", "Rank all direct weighted edges locally.", "1.0.0", ["Mirjam Wattenhofer", "Roger Wattenhofer"], ["Wattenhofer & Wattenhofer (2003)"], {"deterministic": False})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def create_protocol(self, **context):
        """Provide the algorithm-owned endpoint protocol to a generic runtime."""
        if len(context["config"].available_algorithms) != 1:
            return None
        return WattenhoferProtocol(
            node_id=context["node_id"],
            graph=context["graph"],
            state=context["state"],
            send_message=context["send_message"],
            node_count=context["node_count"],
            rng=random.Random(f"{context['config'].vector_fingerprint}:{context['node_id']}"),
        )

    def _get_valid_edges(self, node_id: int, graph, state) -> List[Tuple[int, float]]:
        return [(neighbor_id, graph.get_edge_weight(node_id, neighbor_id)) for neighbor_id in graph.neighbors()]

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        if not neighbors:
            return {}
        weights = {neighbor: context.graph.get_edge_weight(node_id, neighbor) for neighbor in neighbors}
        phase = min(context.round_number // self.parameters["watt_rounds_per_phase"], self.parameters["watt_phase_count"] - 1)
        threshold = self.parameters["watt_valid_threshold"] * (1 - phase / self.parameters["watt_phase_count"])
        best_weight = max(weights.values())
        candidates = {neighbor: weight for neighbor, weight in weights.items() if weight / best_weight >= threshold}
        seed = f"{context.config.vector_fingerprint}:{node_id}:{context.round_number}"
        if random.Random(f"select:{seed}").random() >= self.parameters["watt_select_probability"]:
            return {}
        retained = {neighbor: weight for neighbor, weight in candidates.items() if random.Random(f"eliminate:{seed}:{neighbor}").random() >= self.parameters["watt_eliminate_probability"]}
        return retained or {max(candidates, key=candidates.get): max(candidates.values())}
