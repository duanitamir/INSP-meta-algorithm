"""Luby-style randomized local proposal policy."""

import random
from typing import Dict, List

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


class LubyRandomizedMatching(MatchingAlgorithm):
    PARAMETERS = {
        "base_probability": {"min": 0.0, "max": 1.0, "default": 0.5, "type": "number", "description": "Activation probability"},
        "coeff_degree": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Degree coefficient"},
        "coeff_neighbors_unmatched": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Neighbour coefficient"},
        "coeff_clustering": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Clustering coefficient"},
        "coeff_matched": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Matched coefficient"},
        "coeff_round": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Round coefficient"},
        "coeff_weight": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Weight coefficient"},
        "max_rounds": {"min": 5, "max": 100, "default": 100, "type": "integer", "description": "Maximum execution rounds"},
    }
    PARAMETER_DEFINITION = {"name": "luby", "display_name": "Luby Randomized Matching", "parameters": {name: (spec["min"], spec["max"], lambda spec=spec: random.uniform(spec["min"], spec["max"]) if spec["type"] == "number" else random.randint(spec["min"], spec["max"])) for name, spec in PARAMETERS.items()}}
    PARAMETER_DEFAULTS = {name: spec["default"] for name, spec in PARAMETERS.items()}

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {**self.PARAMETER_DEFAULTS, **(parameters or {})}
        self._metadata = AlgorithmMetadata("Luby Randomized Matching", "Activate probabilistically, then select the best direct neighbour.", "1.0.0", ["Michael Luby"], ["Luby (1986)"], {"deterministic": False})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        activation_seed = f"{context.config.vector_fingerprint}:{node_id}:{context.round_number}"
        if not neighbors or random.Random(activation_seed).random() >= self.parameters["base_probability"]:
            return {}
        neighbor_id = max(neighbors, key=lambda neighbor: (context.graph.get_edge_weight(node_id, neighbor), -neighbor))
        return {neighbor_id: context.graph.get_edge_weight(node_id, neighbor_id)}
