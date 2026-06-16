import pytest
from src.graph import GraphManager
from src.visualization import GraphVisualizer
from src.state import StateStore


class TestGraphVisualizer:
    def test_create_visualizer(self, simple_graph):
        viz = GraphVisualizer(simple_graph)
        assert viz.graph == simple_graph

    def test_render_to_ascii(self, simple_graph):
        viz = GraphVisualizer(simple_graph)
        output = viz.render_to_ascii()
        assert "Graph:" in output
        assert "vertices" in output
        assert "edges" in output

    def test_render_with_state(self, simple_graph):
        viz = GraphVisualizer(simple_graph)
        state_store = StateStore(simple_graph)
        output = viz.render_to_ascii(state_store=state_store)
        assert "Vertex Summary" in output

    def test_render_matching(self, simple_graph):
        viz = GraphVisualizer(simple_graph)
        matching = {1: 2, 2: 1, 3: 4, 4: 3}
        output = viz.render_matching_to_ascii(matching)
        assert "MATCHING ANALYSIS" in output
        assert "Matched vertices" in output

    def test_summary(self, simple_graph):
        viz = GraphVisualizer(simple_graph)
        summary = viz.summary()
        assert "GRAPH SUMMARY" in summary
        assert "Vertices" in summary
        assert "Edges" in summary
