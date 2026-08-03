"""Itai-Israeli-inspired directional local proposal policy."""

import random
from typing import Dict, List

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


class ItaiIsraeliMaximalMatching(MatchingAlgorithm):
    PARAMETERS = {
        "policy_weight": {"min": 0.0, "max": 2.0, "default": 1.0, "type": "number", "description": "Local policy combination weight"},
        "timeout_rounds": {"min": 1, "max": 20, "default": 5, "type": "integer", "description": "Local orientation rotation interval"},
        "max_rounds": {"min": 5, "max": 100, "default": 100, "type": "integer", "description": "Local proposal horizon"},
    }
    PARAMETER_DEFINITION = {"name": "itai", "display_name": "Itai-Israeli Maximal Matching", "parameters": {name: (spec["min"], spec["max"], lambda spec=spec: random.uniform(spec["min"], spec["max"]) if spec["type"] == "number" else random.randint(spec["min"], spec["max"])) for name, spec in PARAMETERS.items()}}
    PARAMETER_DEFAULTS = {name: spec["default"] for name, spec in PARAMETERS.items()}

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {**self.PARAMETER_DEFAULTS, **(parameters or {})}
        self._metadata = AlgorithmMetadata("Itai-Israeli Maximal Matching", "Propose downwards by node ID to the best direct neighbour.", "1.0.0", ["Adi Itai", "Michael Rodeh"], ["Itai & Rodeh (1978)"], {"deterministic": True})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        if context.round_number >= self.parameters["max_rounds"]:
            return {}
        orient_downward = (context.round_number // self.parameters["timeout_rounds"]) % 2 == 0
        candidates = [neighbor for neighbor in neighbors if (neighbor < node_id) == orient_downward]
        if not candidates:
            return {}
        neighbor_id = max(candidates, key=lambda neighbor: (context.graph.get_edge_weight(node_id, neighbor), -neighbor))
        return {neighbor_id: context.graph.get_edge_weight(node_id, neighbor_id)}
