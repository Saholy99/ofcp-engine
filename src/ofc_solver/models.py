"""Public result models for Monte Carlo move ranking."""

from __future__ import annotations

from dataclasses import dataclass

from ofc.state import HandPhase, PlayerId
from ofc_analysis.action_codec import EncodedAction


@dataclass(frozen=True)
class MoveEstimate:
    """Aggregate rollout statistics for one root action."""

    action_index: int
    action: EncodedAction
    mean_value: float
    stddev: float
    sample_count: int
    min_value: float
    max_value: float


@dataclass(frozen=True)
class MoveAnalysis:
    """Ranked Monte Carlo output for one root-state move analysis."""

    observer: PlayerId
    phase: HandPhase
    rollouts_per_action: int
    rng_seed: int | str | None
    ranked_actions: tuple[MoveEstimate, ...]


SUPPORTED_ROOT_PHASES = frozenset({HandPhase.INITIAL_DEAL, HandPhase.DRAW})


__all__ = ["MoveAnalysis", "MoveEstimate", "SUPPORTED_ROOT_PHASES"]
