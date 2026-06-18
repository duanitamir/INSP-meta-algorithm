from typing import NewType, NamedTuple

VertexId = NewType("VertexId", int)
EdgeWeight = NewType("EdgeWeight", float)
RoundNumber = NewType("RoundNumber", int)
MessageId = NewType("MessageId", int)


class Edge(NamedTuple):
    """Canonical edge representation (u, v) where u <= v."""
    u: int
    v: int

    @staticmethod
    def from_nodes(u: int, v: int) -> "Edge":
        """Create canonical edge (always u <= v)."""
        return Edge(min(u, v), max(u, v))

    def other(self, node: int) -> int:
        """Get the other endpoint of the edge."""
        if node == self.u:
            return self.v
        elif node == self.v:
            return self.u
        else:
            raise ValueError(f"Node {node} not in edge {self}")


class MatchedEdge(NamedTuple):
    """Matched edge with weight."""
    edge: Edge
    weight: float
