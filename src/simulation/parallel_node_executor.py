"""Parallel node executor - executes N nodes concurrently via ThreadPoolExecutor.

Phase 6: Extract parallelism from parameterizers to orchestrator level.
This enables efficient multi-core execution at the node level (not within algorithms).

Key Design:
- Each node executes independently (no shared state during execution)
- ThreadPoolExecutor runs all nodes in parallel
- Message delivery still happens sequentially (architectural constraint)
- Tests pass without modification (behavior unchanged, just faster)
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from src.simulation.distributed_node import DistributedNode
from src.meta.core.canonical_vector import CanonicalVector


class ParallelNodeExecutor:
    """Execute multiple DistributedNodes in parallel.

    Uses ThreadPoolExecutor for concurrent node execution. Each node:
    1. Runs its algorithms independently (no shared state during execution)
    2. Produces continue_flag and status
    3. Results collected after all nodes complete
    """

    def __init__(self, max_workers: int = 4):
        """Initialize executor with thread pool size.

        Args:
            max_workers: Number of concurrent worker threads (default 4)
        """
        self.max_workers = max_workers

    def execute_all_nodes(
        self,
        nodes: Dict[int, DistributedNode],
        canonical_vector: CanonicalVector,
    ) -> List[bool]:
        """Execute all nodes in parallel, return continue flags.

        Args:
            nodes: Dict of node_id -> DistributedNode
            canonical_vector: Shared immutable parameter vector

        Returns:
            List of continue_flags from each node
        """
        if not nodes:
            return []

        # Execute all nodes in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(node.execute_distributed_round, canonical_vector): node_id
                for node_id, node in nodes.items()
            }

            # Collect results as they complete
            results = {}
            for future in futures:
                node_id = futures[future]
                try:
                    continue_flag, status = future.result()
                    results[node_id] = continue_flag
                except Exception as e:
                    print(f"Error executing node {node_id}: {e}")
                    results[node_id] = False

        # Return continue flags in order
        return [results.get(node_id, False) for node_id in sorted(nodes.keys())]

    def name(self) -> str:
        """Return executor name."""
        return f"ParallelNodeExecutor(workers={self.max_workers})"
