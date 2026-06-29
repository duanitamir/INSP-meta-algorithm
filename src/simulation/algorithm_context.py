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
