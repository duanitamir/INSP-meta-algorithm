"""Wattenhofer-inspired weighted local proposal policy."""

import random
from typing import Dict, List, Tuple

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


class WattenhoferGraphMatching(MatchingAlgorithm):
    PARAMETERS = {
        "watt_phase_count": {"min": 1, "max": 20, "default": 10, "type": "integer", "description": "Filtering phases"},
        "watt_rounds_per_phase": {"min": 2, "max": 20, "default": 10, "type": "integer", "description": "Rounds per phase"},
        "watt_valid_threshold": {"min": 0.1, "max": 1.0, "default": 0.5, "type": "float", "description": "Local weight threshold"},
        "watt_select_probability": {"min": 0.1, "max": 1.0, "default": 1.0, "type": "float", "description": "Selection probability"},
        "watt_eliminate_probability": {"min": 0.1, "max": 1.0, "default": 0.5, "type": "float", "description": "Tie-break probability"},
    }
    PARAMETER_DEFINITION = {"name": "wattenhofer_graph", "display_name": "Wattenhofer Weighted Matching", "parameters": {name: (spec["min"], spec["max"], lambda spec=spec: random.randint(spec["min"], spec["max"]) if spec["type"] == "integer" else random.uniform(spec["min"], spec["max"])) for name, spec in PARAMETERS.items()}}
    PARAMETER_DEFAULTS = {name: spec["default"] for name, spec in PARAMETERS.items()}

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {**self.PARAMETER_DEFAULTS, **(parameters or {})}
        self._metadata = AlgorithmMetadata("Wattenhofer Weighted Matching", "Rank all direct weighted edges locally.", "1.0.0", ["Mirjam Wattenhofer", "Roger Wattenhofer"], ["Wattenhofer & Wattenhofer (2003)"], {"deterministic": False})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def _get_valid_edges(self, node_id: int, graph, state) -> List[Tuple[int, float]]:
        return [(neighbor_id, graph.get_edge_weight(node_id, neighbor_id)) for neighbor_id in graph.neighbors()]

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        return {neighbor_id: context.graph.get_edge_weight(node_id, neighbor_id) for neighbor_id in neighbors}
