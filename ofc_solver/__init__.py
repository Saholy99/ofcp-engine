"""Compatibility wrapper for importing the src-layout solver package."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ofc_solver"
__path__ = [str(_SRC_PACKAGE)]

from ofc_solver.models import MoveAnalysis, MoveEstimate
from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state

__all__ = [
    "MoveAnalysis",
    "MoveEstimate",
    "rank_actions_from_observation",
    "rank_actions_from_state",
]
