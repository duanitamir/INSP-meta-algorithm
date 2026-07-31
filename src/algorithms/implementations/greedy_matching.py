"""Greedy local proposal policy."""

from typing import Dict, List

from src.algorithms.base import AlgorithmMetadata, MatchingAlgorithm


class GreedyMatching(MatchingAlgorithm):
    PARAMETERS = {"max_rounds": {"min": 5, "max": 100, "default": 100, "type": "integer", "description": "Maximum execution rounds"}}
    PARAMETER_DEFINITION = {"name": "greedy", "display_name": "Greedy Matching", "parameters": {name: (spec["min"], spec["max"], lambda spec=spec: __import__("random").randint(spec["min"], spec["max"])) for name, spec in PARAMETERS.items()}}
    PARAMETER_DEFAULTS = {name: spec["default"] for name, spec in PARAMETERS.items()}

    def __init__(self, parameters: Dict | None = None) -> None:
        self.parameters = {**self.PARAMETER_DEFAULTS, **(parameters or {})}
        self._metadata = AlgorithmMetadata("Greedy Matching", "Select the highest-weight direct neighbour.", "2.0.0", ["Distributed Systems"], ["Greedy matching"], {"deterministic": True})

    @property
    def metadata(self) -> AlgorithmMetadata:
        return self._metadata

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        if not neighbors:
            return {}
        neighbor_id = max(neighbors, key=lambda neighbor: (context.graph.get_edge_weight(node_id, neighbor), -neighbor))
        return {neighbor_id: context.graph.get_edge_weight(node_id, neighbor_id)}
