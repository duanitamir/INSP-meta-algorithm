"""Wattenhofer and Wattenhofer's synchronous weighted matching protocol."""

import random
from collections.abc import Callable
from typing import Any, Dict, List, Mapping

from src.algorithms.base import AlgorithmMetadata, EndpointProtocolAlgorithm
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
    VALID_STAGE = "valid"
    SELECT_STAGE = "select"
    ELIMINATE_STAGE = "eliminate"
    MATCH_STAGE = "match"
    CLEANUP_STAGE = "cleanup"
    algorithm_name = "wattenhofer_graph"

    def __init__(
        self,
        node_id: int,
        graph: LocalGraph,
        state: NodeState,
        send_message: Callable[[Message], None],
        node_count: int,
        rng: random.Random,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.node_id = node_id
        self.graph = graph
        self.state = state
        self.send_message = send_message
        self.node_count = node_count
        self.parameters = dict(parameters or {})
        logarithmic_budget = max(1, (node_count - 1).bit_length())
        self.max_phases = int(
            self.parameters.get(
                "watt_phase_count",
                max(1, round(logarithmic_budget * float(self.parameters.get("phase_multiplier", 1.0)))),
            )
        )
        self.rounds_per_phase = int(
            self.parameters.get(
                "watt_rounds_per_phase",
                max(1, round(logarithmic_budget * float(self.parameters.get("round_multiplier", 1.0)))),
            )
        )
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
        self.match_notices: set[int] = set()
        self.advised_neighbor: int | None = None
        self._private_matched = False
        self._private_unavailable_neighbors: set[int] = set()
        self.stage = self.VALID_STAGE
        self.stage_notices: set[int] = set()
        self.sync_notices: dict[tuple[int, int, str], set[int]] = {}
        self.terminal = False

    def begin_phase(self) -> None:
        """Execute Algorithm 6's local half-maximum candidate step."""
        eligible = [] if self._private_matched or self.state.is_matched() else [
            neighbor
            for neighbor in self.graph.neighbors()
            if self.state.is_neighbor_eligible(neighbor)
            and neighbor not in self._private_unavailable_neighbors
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
        threshold = float(self.parameters.get("watt_valid_threshold", 0.5))
        threshold *= max(0.0, 1.0 - float(self.parameters.get("threshold_decay", 0.0)) * self.phase)
        self.local_candidate_neighbors = {
            neighbor for neighbor, weight in weights.items() if weight >= local_maximum * threshold
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
            elif self._is_current_round(message, self.MATCH):
                self.match_notices.add(message.sender)
                if self.match_choice == message.sender:
                    self._finalize_match(message.sender)
            elif message.payload.get("type") == self.CLEANUP:
                self.mark_neighbor_unavailable(message.sender)
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
        self.match_notices = set()
        self.chosen_imposed_neighbors = set()
        self.retained_neighbors = set()
        self.stage = self.SELECT_STAGE
        self.stage_notices = self._received_syncs(self.SELECT_STAGE)
        if not self.valid_neighbors:
            self._sync(self.SELECT_STAGE)
            return
        activation = float(self.parameters.get("watt_select_probability", 1.0))
        activation *= float(self.parameters.get("activation_probability", 1.0))
        if self.rng.random() >= min(1.0, activation):
            self._sync(self.SELECT_STAGE)
            return
        self.selected_neighbor = self._choose_weighted(
            self.valid_neighbors,
            float(self.parameters.get("select_weight_bias", 0.0)),
        )
        self._send(self.selected_neighbor, self.SELECT)
        self._sync(self.SELECT_STAGE)

    def begin_eliminate(self) -> None:
        """Execute Algorithm 9: choose at most one imposed edge uniformly."""
        self.chosen_imposed_neighbors = set()
        self.stage = self.ELIMINATE_STAGE
        self.stage_notices = self._received_syncs(self.ELIMINATE_STAGE)
        if not self.imposed_neighbors or self.rng.random() >= float(
            self.parameters.get("watt_eliminate_probability", 1.0)
        ):
            self._sync(self.ELIMINATE_STAGE)
            return
        neighbor = self._choose_weighted(
            self.imposed_neighbors,
            float(self.parameters.get("eliminate_weight_bias", 0.0)),
        )
        self.chosen_imposed_neighbors.add(neighbor)
        self.retained_neighbors.add(neighbor)
        self._send(neighbor, self.ELIMINATE)
        self._sync(self.ELIMINATE_STAGE)

    def begin_matching(self) -> None:
        """Execute Algorithm 10: make one random retained-edge choice."""
        self.match_choice = None
        self.stage = self.MATCH_STAGE
        self.stage_notices = self._received_syncs(self.MATCH_STAGE)
        if not self.retained_neighbors or self._private_matched or self.state.is_matched():
            self._sync(self.MATCH_STAGE)
            return
        self.match_choice = self.rng.choice(sorted(self.retained_neighbors))
        self._send(self.match_choice, self.MATCH)
        if self.match_choice in self.match_notices:
            self._finalize_match(self.match_choice)
        self._sync(self.MATCH_STAGE)

    def begin_cleanup(self) -> None:
        """Synchronize cleanup before deciding whether the phase continues."""
        self.stage = self.CLEANUP_STAGE
        self.stage_notices = self._received_syncs(self.CLEANUP_STAGE)
        self._sync(self.CLEANUP_STAGE)

    def _finalize_match(self, neighbor: int) -> None:
        if self._private_matched or self.state.is_matched():
            return
        self._private_matched = True
        self.advised_neighbor = neighbor
        self.valid_neighbors = set()
        for other in self.graph.neighbors():
            if other != neighbor:
                self._private_unavailable_neighbors.add(other)
                self._send(other, self.CLEANUP)

    def finalize_match(self, neighbor: int) -> None:
        """Finalize a private advisory match and notify protocol neighbours."""
        self._finalize_match(neighbor)

    def recommendations(self) -> Dict[int, float]:
        """Return this protocol's current local advice without committing a match."""
        neighbor = self.advised_neighbor or self.match_choice or self.selected_neighbor
        if neighbor is not None:
            if self.state.is_matched() or not self.state.is_neighbor_eligible(neighbor):
                return {}
            return {neighbor: self.graph.get_edge_weight(self.node_id, neighbor)}
        return {
            candidate: self.graph.get_edge_weight(self.node_id, candidate)
            for candidate in self.local_candidate_neighbors
            if self.state.is_neighbor_eligible(candidate)
        }

    def mark_neighbor_unavailable(self, neighbor: int) -> None:
        """Remove a real-world unavailable neighbour from this advisory round."""
        self.valid_neighbors.discard(neighbor)
        self._private_unavailable_neighbors.add(neighbor)

    def retire_from_shared_matcher(self) -> None:
        """Tell advisory neighbours this endpoint has been committed elsewhere."""
        if self.terminal:
            return
        self._private_matched = True
        self.terminal = True
        for neighbor in self.graph.neighbors():
            self._send(neighbor, self.CLEANUP)

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

    def _choose_weighted(self, neighbors: set[int], bias: float) -> int:
        """Choose uniformly at bias zero, or favour heavier local edges otherwise."""
        ordered = sorted(neighbors)
        exponent = bias + float(self.parameters.get("policy_weight", 1.0)) - 1.0
        if exponent == 0:
            return self.rng.choice(ordered)
        weights = [self.graph.get_edge_weight(self.node_id, neighbor) ** exponent for neighbor in ordered]
        return self.rng.choices(ordered, weights=weights, k=1)[0]

    def _advance_if_synchronized(self) -> None:
        expected_notices = set(self.graph.neighbors()).difference(self._private_unavailable_neighbors)
        if self.terminal or self.stage_notices != expected_notices:
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
            if self.round < self.rounds_per_phase:
                self.begin_select()
            else:
                self.phase += 1
                if self.phase >= self.max_phases:
                    self.terminal = True
                else:
                    self.round = 0
                    self.begin_phase()


class WattenhoferGraphMatching(EndpointProtocolAlgorithm):
    PARAMETERS = {
        "policy_weight": {"min": 0.0, "max": 2.0, "default": 1.0, "type": "number", "description": "Weight-selection bias offset"},
        "watt_phase_count": {"min": 1, "max": 20, "default": None, "type": "integer", "description": "Fixed number of matching phases; omitted uses the logarithmic budget"},
        "watt_rounds_per_phase": {"min": 1, "max": 20, "default": None, "type": "integer", "description": "Fixed Uniform-Matching rounds per phase; omitted uses the logarithmic budget"},
        "watt_valid_threshold": {"min": 0.1, "max": 1.0, "default": 0.5, "type": "number", "description": "Minimum fraction of the local maximum edge weight"},
        "watt_select_probability": {"min": 0.0, "max": 1.0, "default": 1.0, "type": "number", "description": "Probability of selecting a valid candidate"},
        "watt_eliminate_probability": {"min": 0.0, "max": 1.0, "default": 1.0, "type": "number", "description": "Probability of retaining one imposed edge"},
        "phase_multiplier": {"min": 0.25, "max": 4.0, "default": 1.0, "type": "number", "description": "Logarithmic phase-budget multiplier when phase count is omitted"},
        "round_multiplier": {"min": 0.25, "max": 4.0, "default": 1.0, "type": "number", "description": "Logarithmic round-budget multiplier when rounds are omitted"},
        "select_weight_bias": {"min": -2.0, "max": 4.0, "default": 0.0, "type": "number", "description": "Exponent bias for Select edge sampling"},
        "eliminate_weight_bias": {"min": -2.0, "max": 4.0, "default": 0.0, "type": "number", "description": "Exponent bias for Eliminate edge sampling"},
        "activation_probability": {"min": 0.0, "max": 1.0, "default": 1.0, "type": "number", "description": "Additional local Select activation probability"},
        "threshold_decay": {"min": 0.0, "max": 0.25, "default": 0.0, "type": "number", "description": "Per-phase valid-threshold decay"},
    }
    PARAMETER_DEFINITION = {
        "name": "wattenhofer_graph",
        "display_name": "Wattenhofer Weighted Matching",
        "parameters": {
            name: (
                spec["min"],
                spec["max"],
                lambda spec=spec: random.randint(spec["min"], spec["max"])
                if spec["type"] == "integer"
                else random.uniform(spec["min"], spec["max"]),
            )
            for name, spec in PARAMETERS.items()
        },
    }

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {
            name: spec["default"]
            for name, spec in self.PARAMETERS.items()
            if spec["default"] is not None
        }
        self.parameters.update(parameters or {})
        self._metadata = AlgorithmMetadata("Wattenhofer Weighted Matching", "Wattenhofer-inspired staged advisory protocol for the hybrid ensemble.", "2.1.0", ["Mirjam Wattenhofer", "Roger Wattenhofer"], ["Wattenhofer & Wattenhofer (2003)"], {"deterministic": False, "advisory": True})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def create_protocol(self, **context):
        """Provide the algorithm-owned endpoint protocol to a generic runtime."""
        return WattenhoferProtocol(
            node_id=context["node_id"],
            graph=context["graph"],
            state=context["state"],
            send_message=context["send_message"],
            node_count=context["node_count"],
            rng=random.Random(f"{context['config'].vector_fingerprint}:{context['node_id']}"),
            parameters=self.parameters,
        )
