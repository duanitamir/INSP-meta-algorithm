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
        modes = {getattr(node, "execution_mode", "asynchronous") for node in nodes.values()}
        if modes == {"synchronous_rounds"}:
            return self._run_synchronous_rounds(nodes, max_ticks, tick)
        if "synchronous_rounds" in modes:
            raise ValueError("Synchronous-round and asynchronous nodes cannot share one execution")
        scheduled_ticks = 0
        in_flight: Dict[Future, int] = {}
        ready = deque(node_id for node_id, node in nodes.items() if node.is_active())

        if self.max_workers == 1:
            while ready and scheduled_ticks < max_ticks:
                node_id = ready.popleft()
                if not nodes[node_id].is_active():
                    continue
                operation = (lambda: tick(nodes[node_id])) if tick else nodes[node_id].tick
                operation()
                scheduled_ticks += 1
                if nodes[node_id].is_active():
                    ready.append(node_id)
            return self._record_outcome(nodes, scheduled_ticks, max_ticks)

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

        return self._record_outcome(nodes, scheduled_ticks, max_ticks)

    def _run_synchronous_rounds(
        self,
        nodes: Dict[int, object],
        max_ticks: int,
        tick: Callable[[object], object] | None,
    ) -> RuntimeOutcome:
        """Run one tick per active node, publishing sends only between rounds."""
        scheduled_ticks = 0
        while scheduled_ticks < max_ticks:
            active = [node for node in nodes.values() if node.is_active()]
            if not active:
                break
            transport = getattr(active[0], "transport", None)
            if transport is None:
                raise ValueError("Synchronous-round nodes must expose a shared transport")
            transport.begin_synchronous_round()
            try:
                for node in active:
                    if scheduled_ticks >= max_ticks:
                        break
                    operation = (lambda node=node: tick(node)) if tick else node.tick
                    operation()
                    scheduled_ticks += 1
            finally:
                transport.commit_synchronous_round()
        return self._record_outcome(nodes, scheduled_ticks, max_ticks)

    def _record_outcome(
        self,
        nodes: Dict[int, object],
        scheduled_ticks: int,
        max_ticks: int,
    ) -> RuntimeOutcome:
        """Create and retain the scheduler-only outcome."""
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
