"""Public result models for Monte Carlo move ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ofc.state import HandPhase, PlayerId

if TYPE_CHECKING:
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
    rollout_mean_value: float | None = None
    root_risk_score: float = 0.0
    root_risk_reasons: tuple[str, ...] = ()
    pattern_score: float = 0.0
    pattern_reasons: tuple[str, ...] = ()
    selection_reasons: tuple[str, ...] = ()
    candidate_rank: int | None = None
    final_score: float | None = None
    late_search_activated: bool = False
    late_search_mode: str | None = None
    late_search_nodes: int = 0
    late_search_depth: int = 0
    late_search_candidate_count: int = 0
    late_search_terminal_evaluations: int = 0
    late_search_fallback_reason: str | None = None
    phase_auto_search_activated: bool = False
    phase_auto_search_reason: str | None = None
    phase_auto_search_tree_nodes: int = 0
    phase_auto_search_depth: int = 0
    late_search_runtime_seconds: float = 0.0
    final_draw_continuation_aware: bool = False
    final_draw_continuation_triggered: bool = False
    final_draw_continuation_rollouts: int = 0
    final_draw_current_hand_value: float = 0.0
    final_draw_continuation_value: float = 0.0
    final_draw_total_value: float = 0.0
    final_draw_continuation_reason: str | None = None
    final_draw_continuation_runtime_seconds: float = 0.0


@dataclass(frozen=True)
class MoveAnalysis:
    """Ranked Monte Carlo output for one root-state move analysis."""

    observer: PlayerId
    phase: HandPhase
    rollouts_per_action: int
    rng_seed: int | str | None
    ranked_actions: tuple[MoveEstimate, ...]
    early_search_enabled: bool = False
    total_legal_actions: int | None = None
    candidate_count: int | None = None
    beam_size: int | None = None
    candidate_extra_rollouts: int = 0
    draw_safe_candidates: bool = True
    draw_baseline_keep: int = 0
    draw_safety_keep: int = 0
    late_search_enabled: bool = False
    late_search_mode: str | None = None
    late_search_max_depth: int | None = None
    late_search_max_nodes: int | None = None
    late_search_beam_size: int | None = None
    final_draw_auto_search_enabled: bool = False
    final_draw_auto_max_depth: int | None = None
    final_draw_auto_max_nodes: int | None = None
    final_draw_auto_include_continuation: bool = False
    final_draw_continuation_rollouts: int = 0


SUPPORTED_ROOT_PHASES = frozenset({HandPhase.INITIAL_DEAL, HandPhase.DRAW})


__all__ = ["MoveAnalysis", "MoveEstimate", "SUPPORTED_ROOT_PHASES"]
