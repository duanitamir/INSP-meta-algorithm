import pytest
from src.graph import GraphManager


class TestGraphManager:
    def test_create_empty_graph(self):
        graph = GraphManager.create_empty_graph()
        assert graph.num_vertices() == 0
        assert graph.num_edges() == 0

    def test_add_vertex(self):
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        assert graph.num_vertices() == 1
        assert 1 in graph.vertices()

    def test_add_multiple_vertices(self):
        graph = GraphManager.create_empty_graph()
        for i in range(1, 6):
            graph.add_vertex(i)
        assert graph.num_vertices() == 5
        assert graph.vertices() == frozenset(range(1, 6))

    def test_add_edge(self):
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.5)
        assert graph.num_edges() == 1
        assert graph.has_edge(1, 2)
        assert graph.get_edge_weight(1, 2) == 1.5

    def test_add_edge_without_vertices_fails(self):
        graph = GraphManager.create_empty_graph()
        with pytest.raises(ValueError):
            graph.add_edge(1, 2, 1.0)

    def test_neighbors(self, simple_graph):
        neighbors_2 = simple_graph.neighbors(2)
        assert neighbors_2 == frozenset([1, 3])

    def test_degree(self, simple_graph):
        assert simple_graph.degree(1) == 1
        assert simple_graph.degree(2) == 2
        assert simple_graph.degree(3) == 2
        assert simple_graph.degree(4) == 1

    def test_max_degree(self, simple_graph):
        assert simple_graph.max_degree() == 2

    def test_create_from_edges(self):
        vertices = [1, 2, 3, 4]
        edges = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
        graph = GraphManager.create_from_edges(vertices, edges)
        assert graph.num_vertices() == 4
        assert graph.num_edges() == 3

    def test_is_connected(self, simple_graph):
        assert simple_graph.is_connected()

    def test_is_disconnected(self):
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_vertex(3)
        graph.add_vertex(4)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)
        assert not graph.is_connected()

    def test_connected_components(self):
        graph = GraphManager.create_empty_graph()
        for i in range(1, 7):
            graph.add_vertex(i)
        edges = [(1, 2, 1.0), (2, 3, 1.0), (4, 5, 1.0)]
        for u, v, w in edges:
            graph.add_edge(u, v, w)
        components = graph.get_connected_components()
        assert len(components) == 3
        assert frozenset([1, 2, 3]) in components
        assert frozenset([4, 5]) in components
        assert frozenset([6]) in components
