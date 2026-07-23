import pytest
from src.graph import GraphManager
from src.state import NodeState, StateStore
from src.utils.types import RoundNumber


class TestNodeState:
    def test_create_node_state(self):
        state = NodeState(1)
        assert state.node_id == 1

    def test_set_and_get(self):
        state = NodeState(1)
        state.set("key", "value")
        assert state.get("key") == "value"

    def test_get_nonexistent_key(self):
        state = NodeState(1)
        assert state.get("missing") is None
        assert state.get("missing", "default") == "default"

    def test_exists(self):
        state = NodeState(1)
        state.set("key", "value")
        assert state.exists("key")
        assert not state.exists("missing")

    def test_delete(self):
        state = NodeState(1)
        state.set("key", "value")
        state.delete("key")
        assert not state.exists("key")

    def test_update(self):
        state = NodeState(1)
        state.set("count", 5)
        state.update("count", lambda x: x + 1)
        assert state.get("count") == 6

    def test_keys(self):
        state = NodeState(1)
        state.set("a", 1)
        state.set("b", 2)
        assert state.keys() == frozenset(["a", "b"])

    def test_clone(self):
        state = NodeState(1)
        state.set("key", "value")
        cloned = state.clone()
        cloned.set("key", "modified")
        assert state.get("key") == "value"
        assert cloned.get("key") == "modified"

    def test_matched_to(self):
        state = NodeState(1)
        assert not state.is_matched()
        state.set_matched_to(5)
        assert state.is_matched()
        assert state.get_matched_to() == 5

    def test_unmatched(self):
        state = NodeState(1)
        state.set_matched_to(5)
        state.set_matched_to(None)
        assert not state.is_matched()

    def test_to_dict(self):
        state = NodeState(1)
        state.set("a", 1)
        state.set("b", "test")
        d = state.to_dict()
        assert d["a"] == 1
        assert d["b"] == "test"


class TestStateStore:
    def test_create_state_store(self, simple_graph):
        store = StateStore(simple_graph)
        assert store.graph == simple_graph

    def test_get_node_state(self, simple_graph):
        store = StateStore(simple_graph)
        for vertex_id in simple_graph.vertices():
            state = store.get_node_state(vertex_id)
            assert state.node_id == vertex_id

    def test_update_node_state(self, simple_graph):
        store = StateStore(simple_graph)
        state = store.get_node_state(1)
        state.set("key", "value")
        store.update_node_state(1, state)
        retrieved = store.get_node_state(1)
        assert retrieved.get("key") == "value"

    def test_get_all_states(self, simple_graph):
        store = StateStore(simple_graph)
        all_states = store.get_all_states()
        assert len(all_states) == simple_graph.num_vertices()

    def test_meta_state(self, simple_graph):
        store = StateStore(simple_graph)
        assert store.round_num == RoundNumber(0)
        assert not store.converged
        assert store.termination_reason is None
