from src.graph.graph_manager import GraphManager
from src.state.store import StateStore
from src.utils.types import RoundNumber


class AlgorithmContext:
    """Legacy centralized context retained for the offline baseline only.

    Distributed proposal algorithms receive ``LocalNodeContext`` instead.
    """

    def __init__(
        self,
        graph: GraphManager,
        state_store: StateStore,
        round_num: RoundNumber,
    ):
        self.graph = graph
        self.state_store = state_store
        self.round_num = round_num
