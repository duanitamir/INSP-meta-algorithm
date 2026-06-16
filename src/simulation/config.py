from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""

    max_rounds: int = 1000
    max_messages: int | None = None
    debug: bool = False
    random_seed: int | None = None
    collect_snapshots: bool = True
    collect_message_traces: bool = True
