"""Luby-style randomized local proposal policy."""

import random
from typing import Dict, List

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


class LubyRandomizedMatching(MatchingAlgorithm):
    PARAMETERS = {
        "policy_weight": {"min": 0.0, "max": 2.0, "default": 1.0, "type": "number", "description": "Local policy combination weight"},
        "base_probability": {"min": 0.0, "max": 1.0, "default": 0.5, "type": "number", "description": "Activation probability"},
        "coeff_degree": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Local degree coefficient"},
        "coeff_neighbors_unmatched": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Eligible-neighbor ratio coefficient"},
        "coeff_clustering": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Incident-weight concentration coefficient"},
        "coeff_matched": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Unavailable-neighbor ratio coefficient"},
        "coeff_round": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Local round-progress coefficient"},
        "coeff_weight": {"min": -1.0, "max": 1.0, "default": 0.1, "type": "number", "description": "Best incident-edge coefficient"},
        "max_rounds": {"min": 5, "max": 100, "default": 100, "type": "integer", "description": "Local proposal horizon"},
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
        if not neighbors or context.round_number >= self.parameters["max_rounds"]:
            return {}
        weights = [context.graph.get_edge_weight(node_id, neighbor) for neighbor in neighbors]
        degree = len(neighbors)
        eligible = sum(context.state.is_neighbor_eligible(neighbor) for neighbor in neighbors)
        features = {
            "coeff_degree": degree / (degree + 1),
            "coeff_neighbors_unmatched": eligible / degree,
            "coeff_clustering": max(weights) / sum(weights),
            "coeff_matched": 1 - eligible / degree,
            "coeff_round": context.round_number / self.parameters["max_rounds"],
            "coeff_weight": max(weights) / sum(weights),
        }
        probability = max(0.0, min(1.0, self.parameters["base_probability"] + sum(self.parameters[name] * value for name, value in features.items())))
        activation_seed = f"{context.config.vector_fingerprint}:{node_id}:{context.round_number}"
        if random.Random(activation_seed).random() >= probability:
            return {}
        neighbor_id = max(neighbors, key=lambda neighbor: (context.graph.get_edge_weight(node_id, neighbor), -neighbor))
        return {neighbor_id: context.graph.get_edge_weight(node_id, neighbor_id)}
