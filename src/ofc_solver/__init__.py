"""Public scaffolding exports for the solver layer.

This package is reserved for rollout policies, hidden-state sampling, and move
ranking. It must stay separate from the engine package so rule logic remains in
``src/ofc/`` and solver experimentation remains isolated.
"""

from ofc_solver.models import MoveAnalysis, MoveEstimate
from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state

__all__ = [
    "MoveAnalysis",
    "MoveEstimate",
    "rank_actions_from_observation",
    "rank_actions_from_state",
]
