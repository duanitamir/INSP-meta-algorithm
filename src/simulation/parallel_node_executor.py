"""Independent ready-node scheduler for the distributed simulator."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Set


@dataclass(frozen=True)
class RuntimeOutcome:
    """Operational scheduler result; it contains no algorithmic decision."""

    scheduled_ticks: int
    active_node_ids: Set[int]
    watchdog_exhausted: bool


class ParallelNodeExecutor:
    """Run independent node ticks without a round-wide synchronization barrier."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.last_outcome: RuntimeOutcome | None = None

    def run_until_idle(
        self,
        nodes: Dict[int, object],
        max_ticks: int,
        tick: Callable[[object], object] | None = None,
    ) -> RuntimeOutcome:
        """Schedule ready active nodes, resubmitting each one as it completes.

        ``tick`` lets the bootstrapper provide immutable run configuration.  The
        executor observes only ``is_active`` and never reads a node vote, state,
        or matching result.
        """
        scheduled_ticks = 0
        in_flight: Dict[Future, int] = {}
        ready = deque(node_id for node_id, node in nodes.items() if node.is_active())

        def submit(pool: ThreadPoolExecutor, node_id: int) -> bool:
            nonlocal scheduled_ticks
            if scheduled_ticks >= max_ticks or not nodes[node_id].is_active():
                return False
            operation = (lambda: tick(nodes[node_id])) if tick else nodes[node_id].tick
            in_flight[pool.submit(operation)] = node_id
            scheduled_ticks += 1
            return True

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while ready and len(in_flight) < self.max_workers and scheduled_ticks < max_ticks:
                submit(pool, ready.popleft())

            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    node_id = in_flight.pop(future)
                    future.result()
                    if nodes[node_id].is_active():
                        ready.append(node_id)

                while ready and len(in_flight) < self.max_workers and scheduled_ticks < max_ticks:
                    submit(pool, ready.popleft())

        active_node_ids = {node_id for node_id, node in nodes.items() if node.is_active()}
        outcome = RuntimeOutcome(
            scheduled_ticks=scheduled_ticks,
            active_node_ids=active_node_ids,
            watchdog_exhausted=scheduled_ticks >= max_ticks and bool(active_node_ids),
        )
        self.last_outcome = outcome
        return outcome

    def name(self) -> str:
        return f"ParallelNodeExecutor(workers={self.max_workers})"
