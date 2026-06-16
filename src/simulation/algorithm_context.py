from src.graph.graph_manager import GraphManager
from src.state.state_store import StateStore
from src.utils.types import RoundNumber


class AlgorithmContext:
    """Runtime context provided to nodes during execution."""

    def __init__(
        self,
        graph: GraphManager,
        state_store: StateStore,
        round_num: RoundNumber,
    ):
        self.graph = graph
        self.state_store = state_store
        self.round_num = round_num

    def get_neighbor_states(self, node_id: int):
        """Get states of all neighbors."""
        neighbors = self.graph.neighbors(node_id)
        return {
            neighbor: self.state_store.get_node_state(neighbor)
            for neighbor in neighbors
        }

    def get_incident_edges(self, node_id: int):
        """Get all edges incident to a node."""
        neighbors = self.graph.neighbors(node_id)
        return frozenset(
            (node_id, neighbor)
            for neighbor in neighbors
        )
