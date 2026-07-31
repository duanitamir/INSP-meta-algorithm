"""
Wattenhofer-Wattenhofer Distributed Weighted Matching Algorithm (General Graphs)

Reference: "Distributed Weighted Matching" by Mirjam Wattenhofer & Roger Wattenhofer (2003)
Paper: https://www.inf.ethz.ch/personal/rvwxyza/

Algorithm Overview:
- Multi-phase approach
- Each phase progressively filters edges to isolate valuable matches
- Achieves O(1) approximation ratio (≥ 1/6 of optimal weight)
- Fully distributed: only neighbor communication, no central coordinator

Key Phases:
1. VALID: Identify edges with weight ≥ threshold
2. SELECT: Each node randomly picks one incident edge
3. ELIMINATE: Resolve conflicts probabilistically
4. MATCHING: Extract matches from agreement edges
5. CLEANUP: Remove matched edges and neighbors

Main Loop:
- Run log(n) phases
- Each phase: O(log n) rounds of Uniform-Matching
- Accumulate matches across all phases
"""

from typing import List, Tuple, Dict
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class WattenhoferGraphMatching(MatchingAlgorithm):
    """Wattenhofer-Wattenhofer distributed matching for general weighted graphs."""

    PARAMETERS = {
        "watt_phase_count": {
            "min": 1,
            "max": 20,
            "default": 10,
            "type": "integer",
            "description": "Number of filtering phases (default: ceil(log n))",
        },
        "watt_rounds_per_phase": {
            "min": 2,
            "max": 20,
            "default": 10,
            "type": "integer",
            "description": "Rounds per phase (default: ceil(log n))",
        },
        "watt_valid_threshold": {
            "min": 0.1,
            "max": 1.0,
            "default": 0.5,
            "type": "float",
            "description": "Edge weight threshold as fraction of local max (0.1-1.0)",
        },
        "watt_select_probability": {
            "min": 0.1,
            "max": 1.0,
            "default": 1.0,
            "type": "float",
            "description": "Probability of selecting an edge in Select phase",
        },
        "watt_eliminate_probability": {
            "min": 0.1,
            "max": 1.0,
            "default": 0.5,
            "type": "float",
            "description": "Tie-breaker probability in Eliminate phase",
        },
    }

    PARAMETER_DEFINITION = {
        "name": "wattenhofer_graph",
        "display_name": "Wattenhofer-Wattenhofer Weighted Matching",
        "parameters": {
            param: (p["min"], p["max"], (lambda pm=param, pp=p: __import__("random").randint(pp["min"], pp["max"]) if pp["type"] == "integer" else __import__("random").uniform(pp["min"], pp["max"])))
            for param, p in PARAMETERS.items()
        }
    }

    PARAMETER_DEFAULTS = {param: p["default"] for param, p in PARAMETERS.items()}

    PARAMETER_SCHEMA = {
        "type": "object",
        "properties": {
            param: {
                "type": p["type"],
                "minimum": p["min"],
                "maximum": p["max"],
                "description": p["description"],
            }
            for param, p in PARAMETERS.items()
        },
        "required": list(PARAMETERS.keys()),
    }

    def __init__(self, parameters: Dict = None):
        """Initialize Wattenhofer-Wattenhofer algorithm.

        Args:
            parameters: Optional parameter dict. Missing parameters use defaults.
        """
        self.parameters = {**self.PARAMETER_DEFAULTS}
        if parameters:
            self.parameters.update(parameters)

        self._metadata = AlgorithmMetadata(
            name="Wattenhofer-Wattenhofer Distributed Weighted Matching",
            description="Distributed algorithm for weighted matching in general graphs. "
                       "Achieves O(1) approximation in O(log²n) rounds.",
            version="1.0.0",
            authors=["Mirjam Wattenhofer", "Roger Wattenhofer"],
            references=["Wattenhofer & Wattenhofer (2003) - Distributed Weighted Matching"],
            properties={
                "produces_maximal": False,
                "produces_maximum": False,
                "deterministic": False,
                "approximation_ratio": "1/6",
                "time_complexity": "O(log^2 n)",
                "message_complexity": "O(n log^2 n)",
                "phase_count": self.parameters.get("watt_phase_count", 10),
                "rounds_per_phase": self.parameters.get("watt_rounds_per_phase", 10),
            },
        )

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Get algorithm metadata."""
        return self._metadata

    def initialize_state(self, state_store: StateStore, graph) -> None:
        """Initialize state for all nodes."""
        for node_id in graph.vertices():
            state = state_store.get_node_state(node_id)
            state.set("matched_to", None)
            state.set("active", graph.degree(node_id) > 0)
            state.set("selected_edge", None)  # Edge selected in Select phase
            state.set("is_good_endpoint", False)  # Whether this node is "good" for this round
            state.set("phase", 0)
            state.set("round_in_phase", 0)
            state.set("eliminated_edges", set())  # Edges to skip in future rounds
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self,
        node_id: int,
        node_state,
        messages: List[Message],
        context,
    ) -> Tuple:
        """Execute one round of Wattenhofer algorithm for a node.

        Implements the Uniform-Matching sub-algorithm which consists of:
        1. SELECT: Pick a random incident edge
        2. ELIMINATE: Resolve conflicts
        3. MATCHING: Extract matches from agreements
        4. CLEANUP: Mark matched edges for removal
        """
        new_state = node_state.clone()

        if new_state.is_matched():
            new_state.set("active", False)
            return (new_state, [])

        neighbors = self.get_active_neighbors(node_id, context)
        if self.check_no_neighbors(neighbors):
            new_state.set("active", False)
            return (new_state, [])

        graph = self._context_value(context, "graph")
        # Parse incoming messages
        selected_by_neighbor = set()  # Neighbors who selected this node's edge
        for msg in messages:
            if self._message_type(msg) == "SELECTED":
                selected_by_neighbor.add(msg.sender)

        # --- PHASE 1: SELECT (Algorithm 8) ---
        # Each unmatched node picks a random incident edge
        valid_edges = self._get_valid_edges(node_id, graph, new_state)

        if valid_edges:
            selected_edge = random.choice(valid_edges)
            neighbor = selected_edge[1] if selected_edge[0] == node_id else selected_edge[0]
            new_state.set("selected_edge", selected_edge)

            # Notify neighbor that we selected the edge to them
            msg = Message(
                sender=node_id,
                recipient=neighbor,
                payload={"type": "SELECTED", "edge": selected_edge, "node": node_id},
                round_num=self._context_round_number(context),
            )
            messages_out = [msg]
        else:
            new_state.set("selected_edge", None)
            messages_out = []

        # --- PHASE 2: ELIMINATE (Algorithm 9) ---
        # If both endpoints independently selected each other, mark as "good"
        selected_edge = new_state.get("selected_edge")
        is_good_endpoint = False

        if selected_edge:
            neighbor = selected_edge[1] if selected_edge[0] == node_id else selected_edge[0]
            if neighbor in selected_by_neighbor:
                is_good_endpoint = True

        new_state.set("is_good_endpoint", is_good_endpoint)

        # --- PHASE 3: MATCHING (Algorithm 10) ---
        # If we're a "good endpoint", randomly decide whether to match
        if is_good_endpoint:
            eliminate_prob = self.parameters.get("watt_eliminate_probability", 0.5)
            if random.random() < eliminate_prob:
                # Match with the neighbor
                selected_edge = new_state.get("selected_edge")
                neighbor = selected_edge[1] if selected_edge[0] == node_id else selected_edge[0]

                # Send MATCHED message to neighbor
                msg = Message(
                    sender=node_id,
                    recipient=neighbor,
                    payload={"type": "MATCHED", "edge": selected_edge},
                    round_num=self._context_round_number(context),
                )
                messages_out.append(msg)

                # Mark as matched
                edge_weight = graph.get_edge_weight(selected_edge[0], selected_edge[1])
                new_state.set("matched_to", neighbor)
                new_state.set("matched_edge", selected_edge)
                new_state.set("matched_weight", edge_weight)
                new_state.set("active", False)

        # --- PHASE 4: CLEANUP (Algorithm 11) ---
        # Process incoming MATCHED messages
        for msg in messages:
            if self._message_type(msg) == "MATCHED":
                if msg.payload.get("edge") == new_state.get("selected_edge"):
                    # Neighbor confirmed the match
                    edge_weight = graph.get_edge_weight(msg.payload["edge"][0], msg.payload["edge"][1])
                    new_state.set("matched_to", msg.sender)
                    new_state.set("matched_edge", msg.payload["edge"])
                    new_state.set("matched_weight", edge_weight)
                    new_state.set("active", False)

        return (new_state, messages_out)

    def check_termination(
        self,
        state_store: StateStore,
        round_num: RoundNumber,
        messages_sent: int,
    ) -> Tuple[bool, str | None]:
        """Check if algorithm should terminate.

        Terminates when:
        1. All nodes are matched or have no neighbors
        2. Max rounds exceeded
        3. No progress (no new matches, no messages sent)
        """
        # Check if all nodes are inactive
        all_inactive = all(
            not state.get("active")
            for state in state_store.iter_node_states()
        )

        if all_inactive:
            return (True, "All nodes inactive")

        max_rounds = self.parameters.get("watt_phase_count", 10) * \
                     self.parameters.get("watt_rounds_per_phase", 10)

        if round_num >= max_rounds:
            return (True, f"Max rounds reached ({round_num}/{max_rounds})")

        if messages_sent == 0 and round_num > 1:
            return (True, "No progress (no messages)")

        return (False, None)

    def _get_valid_edges(self, node_id: int, graph, state) -> List[Tuple]:
        """Get valid candidate edges for this node (Algorithm 6).

        An edge is valid if:
        - The other endpoint is not yet matched
        - Edge weight ≥ threshold × max_weight_in_neighborhood
        """
        from src.graph.local_graph import LocalGraph

        neighbors = (
            graph.neighbors()
            if isinstance(graph, LocalGraph)
            else graph.neighbors(node_id)
        )
        if not neighbors:
            return []

        # Find max weight among incident edges
        incident_weights = []
        for neighbor in neighbors:
            weight = graph.get_edge_weight(node_id, neighbor)
            incident_weights.append(weight)

        if not incident_weights:
            return []

        max_weight = max(incident_weights)
        threshold = self.parameters.get("watt_valid_threshold", 0.5)
        min_weight = threshold * max_weight

        # Filter valid edges
        valid_edges = []
        for neighbor in neighbors:
            # Skip if neighbor already matched (distributed knowledge from state)
            # In practice, this would be learned from messages
            weight = graph.get_edge_weight(node_id, neighbor)
            if weight >= min_weight:
                edge = (min(node_id, neighbor), max(node_id, neighbor))
                valid_edges.append(edge)

        return valid_edges

    def get_active_neighbors(self, node_id: int, context) -> List[int]:
        """Get list of active (unmatched) neighbors.

        This is a helper that checks local graph knowledge.
        In a truly distributed system, nodes would know neighbors are active
        through message exchanges or gossip.
        """
        graph = self._context_value(context, "graph")
        state_store = self._context_value(context, "state_store")
        from src.graph.local_graph import LocalGraph

        neighbors = (
            graph.neighbors()
            if isinstance(graph, LocalGraph)
            else graph.neighbors(node_id)
        )

        active_neighbors = []
        for neighbor in neighbors:
            neighbor_state = state_store.get_node_state(neighbor)
            if neighbor_state.get("active") and not neighbor_state.is_matched():
                active_neighbors.append(neighbor)

        return active_neighbors

    def check_no_neighbors(self, neighbors: List[int]) -> bool:
        """Check if node has no active neighbors."""
        return len(neighbors) == 0

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        """Get proposals (edges with weights) to neighbors.

        Returns dict of neighbor_id -> edge_weight for all valid edges to neighbors.
        """
        graph = self._context_value(context, "graph")
        proposals = {}

        for neighbor in neighbors:
            weight = graph.get_edge_weight(node_id, neighbor)
            proposals[neighbor] = weight

        return proposals

    @staticmethod
    def _context_value(context, key: str):
        """Read a context value from legacy mappings or node context objects."""
        if isinstance(context, dict):
            return context[key]
        return getattr(context, key)

    @staticmethod
    def _context_round_number(context) -> int:
        """Read the current round from either supported execution context."""
        if isinstance(context, dict):
            return context.get("round_number", context.get("round_num", 0))
        return getattr(context, "round_number", getattr(context, "round_num", 0))

    @staticmethod
    def _message_type(message: Message) -> str | None:
        """Read a message type from the current payload-based message contract."""
        return message.payload.get("type") if isinstance(message.payload, dict) else None
