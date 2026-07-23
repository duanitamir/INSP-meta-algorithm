"""Type schema for NodeState._state dictionary.

This module defines the structure and types for the generic state dictionary
stored in NodeState. Using TypedDict allows static type checkers to provide
IDE autocomplete and catch type errors, while remaining flexible at runtime.

All fields are optional (total=False) since nodes don't necessarily set every
state variable. The keys and types below document what values can be stored.
"""

from typing import TypedDict, Optional


class NodeStateSchema(TypedDict, total=False):
    """Schema for NodeState._state dictionary.

    All fields are optional since different algorithms use different state variables.
    Static type checkers will use this to validate state access patterns.
    """

    # Matching state (all algorithms)
    matched_to: Optional[int]

    # Algorithm-specific: Luby Randomized
    active: bool
    is_active: bool
    proposal_to: Optional[int]
    proposal_weight: Optional[float]
    has_neighbors: bool

    # Algorithm-specific: Luby Negotiations
    negotiation_stage: str
    negotiation_partner: Optional[int]
    accept_from: Optional[int]
    accept_confirmed: bool
    stage_round: int
    last_bid_weight: Optional[float]

    # Matching results
    status: str
    matched_edges: dict
