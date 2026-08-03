"""Wattenhofer-inspired weighted local proposal policy."""

import random
from typing import Dict, List, Tuple

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


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
