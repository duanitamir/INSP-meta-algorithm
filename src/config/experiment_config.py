from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Infrastructure configuration for running experiments (centralized).

    These settings control how the experiment is executed:
    - How many rounds to run
    - Debugging and tracing options
    - Snapshot collection

    Orchestrator uses this to control experiment flow.
    Nodes do NOT need this - they only care about algorithm parameters.
    """

    max_rounds: int = 1000
    max_messages: int | None = None
    debug: bool = False
    random_seed: int | None = None
    collect_snapshots: bool = True
    collect_message_traces: bool = True
