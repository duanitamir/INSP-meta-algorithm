#!/usr/bin/env python3
"""
Detailed trace of Greedy algorithm execution to identify the bug.
"""

import sys
sys.path.insert(0, '.')

from src.graph import GraphManager
from src.simulation import Scheduler, SimulationConfig
from src.algorithms.implementations import GreedyMatching


def trace_algorithm():
    # Create test graph
    graph = GraphManager.create_empty_graph()
    vertices = [1, 2, 3, 4, 5, 6]
    for v in vertices:
        graph.add_vertex(v)

    edges = [
        (1, 2, 10.0),
        (1, 3, 5.0),
        (2, 3, 8.0),
        (2, 4, 7.0),
        (3, 4, 9.0),
        (4, 5, 6.0),
        (5, 6, 11.0),
        (4, 6, 4.0),
    ]
    for u, v, w in edges:
        graph.add_edge(u, v, w)

    # Create algorithm
    algo = GreedyMatching(seed=42)

    # Manual execution with tracing
    from src.state.state_store import StateStore
    from src.communication.message_queue import MessageQueue
    from src.simulation.algorithm_context import AlgorithmContext
    from src.utils.types import RoundNumber

    state_store = StateStore(graph)
    message_queue = MessageQueue(graph)

    # Initialize
    algo.initialize_state(state_store, graph)

    print("=" * 80)
    print("INITIAL STATE")
    print("=" * 80)
    all_states = state_store.get_all_states()
    for nid in sorted(all_states.keys()):
        state = all_states[nid]
        print(f"Node {nid}: neighbors={state.get('neighbors')}, active={state.get('active')}")

    # Run a few rounds manually
    for round_num in range(1, 4):
        print("\n" + "=" * 80)
        print(f"ROUND {round_num}")
        print("=" * 80)

        # Execute each node
        for node_id in sorted(graph.vertices()):
            node_state = state_store.get_node_state(node_id)
            messages = message_queue.get_messages(node_id)

            print(f"\n--- Node {node_id} ---")
            print(f"  State before: matched_to={node_state.get_matched_to()}, "
                  f"current_bid_partner={node_state.get('current_bid_partner')}, "
                  f"active={node_state.get('active')}")
            print(f"  Received messages: {len(messages)}")
            for msg in messages:
                print(f"    - From {msg.sender}: {msg.payload}")

            context = AlgorithmContext(
                graph=graph,
                round_num=RoundNumber(round_num),
                state_store=state_store
            )

            new_state, out_messages = algo.node_behavior(node_id, node_state, messages, context)

            print(f"  State after: matched_to={new_state.get_matched_to()}, "
                  f"current_bid_partner={new_state.get('current_bid_partner')}, "
                  f"active={new_state.get('active')}")
            print(f"  Sent messages: {len(out_messages)}")
            for msg in out_messages:
                print(f"    - To {msg.recipient}: {msg.payload}")

            # Update state and queue messages
            state_store.update_node_state(node_id, new_state)
            for msg in out_messages:
                message_queue.send(msg)

        # Check termination
        messages_sent = message_queue._total_sent
        should_terminate, reason = algo.check_termination(state_store, RoundNumber(round_num), messages_sent)
        print(f"\nTermination check: should_terminate={should_terminate}, reason={reason}")

        if should_terminate:
            break

    print("\n" + "=" * 80)
    print("FINAL MATCHING")
    print("=" * 80)
    matching = algo.extract_matching(state_store, graph)
    print(f"Matching: {matching}")

    is_valid, error = algo.validate_matching(matching, graph)
    print(f"Valid: {is_valid}, Error: {error}")


if __name__ == "__main__":
    trace_algorithm()
