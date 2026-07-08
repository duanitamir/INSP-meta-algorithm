"""Meta-algorithm configuration system.

Provides centralized configuration for GA runs, including:
- Algorithm selection via Algorithms enum
- GA parameters via GAConfig dataclass
- Combined configuration via MetaConfig dataclass
"""

from .meta_config import Algorithms, GAConfig, MetaConfig

__all__ = [
    "Algorithms",
    "GAConfig",
    "MetaConfig",
]
