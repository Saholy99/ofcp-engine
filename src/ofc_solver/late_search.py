"""Bounded late-street root evaluation for DRAW states."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import TYPE_CHECKING, Literal

from ofc.actions import GameAction
from ofc.engine import showdown
from ofc.scoring import TerminalResult
from ofc.state import GameState, HandPhase, PlayerId, player_index
from ofc.transitions import apply_action, legal_actions
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy

if TYPE_CHECKING:
    from ofc_solver.rollout import RolloutResult
    from ofc_solver.rollout_policy import RolloutPolicy


LateSearchMode = Literal["auto", "exact", "beam"]


@dataclass(frozen=True)
class LateSearchConfig:
    """Runtime limits and mode selection for late-street search."""

    mode: LateSearchMode = "auto"
    max_depth: int = 4
    max_nodes: int = 500
    beam_size: int = 8

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "exact", "beam"}:
            raise ValueError("late_search mode must be one of: auto, exact, beam")
        if self.max_depth < 0:
            raise ValueError("late_search max_depth must be non-negative")
        if self.max_nodes <= 0:
            raise ValueError("late_search max_nodes must be positive")
        if self.beam_size <= 0:
            raise ValueError("late_search beam_size must be positive")


@dataclass(frozen=True)
class LateSearchResult:
    """Value and diagnostics for one searched root action."""

    value: float
    rollout_result: RolloutResult
    activated: bool
    mode: str
    nodes_searched: int
    depth_reached: int
    candidate_count: int
    terminal_evaluations: int
    fallback_reason: str | None = None
    runtime_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True)
class RankedLateSearchAction:
    """One root action result with its original engine action index."""

    action_index: int
    action: GameAction
    result: LateSearchResult


@dataclass
class _SearchStats:
    max_nodes: int
    nodes: int = 0
    depth_reached: int = 0
    candidate_count: int = 0
    terminal_evaluations: int = 0

    def visit(self, depth: int) -> None:
        self.nodes += 1
        self.depth_reached = max(self.depth_reached, depth)
        if self.nodes > self.max_nodes:
            raise _SearchBudgetExceeded


@dataclass(frozen=True)
class _SearchOutcome:
    value: float
    terminal_state: GameState
    terminal_result: TerminalResult


class _SearchBudgetExceeded(Exception):
    pass


class _SearchDepthExceeded(Exception):
    pass


class _SearchUnsupported(Exception):
    pass


def rank_late_root_actions(
    state: GameState,
    *,
    perspective: PlayerId,
    rng: random.Random,
    config: LateSearchConfig | None = None,
    policy: RolloutPolicy | None = None,
) -> tuple[RankedLateSearchAction, ...]:
    """Evaluate and rank all legal root DRAW actions with late search."""

    effective_config = config or LateSearchConfig()
    results = tuple(
        RankedLateSearchAction(
            action_index=action_index,
            action=action,
            result=evaluate_late_root_action(
                state,
                action,
                perspective=perspective,
                rng=rng,
                config=effective_config,
                policy=policy,
            ),
        )
        for action_index, action in enumerate(tuple(legal_actions(state)))
    )
    return tuple(sorted(results, key=lambda item: (-item.result.value, item.action_index)))


def evaluate_late_root_action(
    state: GameState,
    action: GameAction,
    *,
    perspective: PlayerId,
    rng: random.Random,
    config: LateSearchConfig | None = None,
    policy: RolloutPolicy | None = None,
) -> LateSearchResult:
    """Evaluate one legal root action with bounded late-street search."""

    effective_config = config or LateSearchConfig()
    start = time.perf_counter()
    if state.phase != HandPhase.DRAW:
        return _fallback_result(
            state,
            action,
            perspective=perspective,
            rng=rng,
            policy=policy,
            reason="unsupported-phase",
            mode="fallback",
            runtime_start=start,
        )

    if effective_config.mode in {"auto", "exact"}:
        stats = _SearchStats(max_nodes=effective_config.max_nodes)
        try:
            outcome = _search_value(
                apply_action(state, action),
                perspective=perspective,
                config=effective_config,
                stats=stats,
                depth=0,
                beam_size=None,
            )
            return _searched_result(
                outcome,
                perspective=perspective,
                stats=stats,
                mode="exact",
                runtime_start=start,
            )
        except _SearchBudgetExceeded:
            if effective_config.mode == "exact":
                return _fallback_result(
                    state,
                    action,
                    perspective=perspective,
                    rng=rng,
                    policy=policy,
                    reason="exact-budget-exceeded",
                    mode="fallback",
                    runtime_start=start,
                    stats=stats,
                )
        except _SearchDepthExceeded:
            if effective_config.mode == "exact":
                return _fallback_result(
                    state,
                    action,
                    perspective=perspective,
                    rng=rng,
                    policy=policy,
                    reason="exact-depth-exceeded",
                    mode="fallback",
                    runtime_start=start,
                    stats=stats,
                )
        except _SearchUnsupported:
            if effective_config.mode == "exact":
                return _fallback_result(
                    state,
                    action,
                    perspective=perspective,
                    rng=rng,
                    policy=policy,
                    reason="exact-unsupported-state",
                    mode="fallback",
                    runtime_start=start,
                    stats=stats,
                )

    if effective_config.mode in {"auto", "beam"}:
        stats = _SearchStats(max_nodes=effective_config.max_nodes)
        try:
            outcome = _search_value(
                apply_action(state, action),
                perspective=perspective,
                config=effective_config,
                stats=stats,
                depth=0,
                beam_size=effective_config.beam_size,
            )
            return _searched_result(
                outcome,
                perspective=perspective,
                stats=stats,
                mode="beam",
                runtime_start=start,
            )
        except _SearchBudgetExceeded:
            reason = "beam-budget-exceeded"
        except _SearchDepthExceeded:
            reason = "beam-depth-exceeded"
        except _SearchUnsupported:
            reason = "beam-unsupported-state"
        if effective_config.mode == "auto" and effective_config.beam_size > 1:
            narrow_stats = _SearchStats(max_nodes=effective_config.max_nodes)
            try:
                outcome = _search_value(
                    apply_action(state, action),
                    perspective=perspective,
                    config=effective_config,
                    stats=narrow_stats,
                    depth=0,
                    beam_size=1,
                )
                return _searched_result(
                    outcome,
                    perspective=perspective,
                    stats=narrow_stats,
                    mode="beam",
                    runtime_start=start,
                )
            except _SearchBudgetExceeded:
                stats = narrow_stats
                reason = "beam-budget-exceeded"
            except _SearchDepthExceeded:
                stats = narrow_stats
                reason = "beam-depth-exceeded"
            except _SearchUnsupported:
                stats = narrow_stats
                reason = "beam-unsupported-state"
        return _fallback_result(
            state,
            action,
            perspective=perspective,
            rng=rng,
            policy=policy,
            reason=reason,
            mode="fallback",
            runtime_start=start,
            stats=stats,
        )

    return _fallback_result(
        state,
        action,
        perspective=perspective,
        rng=rng,
        policy=policy,
        reason="late-search-disabled",
        mode="fallback",
        runtime_start=start,
    )


def _search_value(
    state: GameState,
    *,
    perspective: PlayerId,
    config: LateSearchConfig,
    stats: _SearchStats,
    depth: int,
    beam_size: int | None,
) -> _SearchOutcome:
    stats.visit(depth)
    if state.phase in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        terminal_state, result = showdown(state)
        stats.terminal_evaluations += 1
        return _SearchOutcome(
            value=_value_for_player(result, perspective),
            terminal_state=terminal_state,
            terminal_result=result,
        )
    if state.phase != HandPhase.DRAW:
        raise _SearchUnsupported
    if depth >= config.max_depth:
        raise _SearchDepthExceeded

    actions = _search_actions(state, beam_size=beam_size)
    if not actions:
        raise _SearchUnsupported
    stats.candidate_count += len(actions)
    child_outcomes = tuple(
        _search_value(
            apply_action(state, child_action),
            perspective=perspective,
            config=config,
            stats=stats,
            depth=depth + 1,
            beam_size=beam_size,
        )
        for child_action in actions
    )
    if state.acting_player == perspective:
        return max(child_outcomes, key=lambda outcome: outcome.value)
    return min(child_outcomes, key=lambda outcome: outcome.value)


def _search_actions(state: GameState, *, beam_size: int | None) -> tuple[GameAction, ...]:
    if beam_size is None:
        return tuple(legal_actions(state))
    scored = HeuristicRolloutPolicy().rank_actions(state)
    return tuple(item.action for item in scored[:beam_size])


def _searched_result(
    outcome: _SearchOutcome,
    *,
    perspective: PlayerId,
    stats: _SearchStats,
    mode: str,
    runtime_start: float,
) -> LateSearchResult:
    return LateSearchResult(
        value=outcome.value,
        rollout_result=_rollout_result_from_terminal(
            outcome.terminal_state,
            outcome.terminal_result,
            perspective=perspective,
            value=outcome.value,
            late_search_mode=mode,
            activated=True,
            stats=stats,
        ),
        activated=True,
        mode=mode,
        nodes_searched=stats.nodes,
        depth_reached=stats.depth_reached,
        candidate_count=stats.candidate_count,
        terminal_evaluations=stats.terminal_evaluations,
        runtime_seconds=time.perf_counter() - runtime_start,
    )


def _fallback_result(
    state: GameState,
    action: GameAction,
    *,
    perspective: PlayerId,
    rng: random.Random,
    policy: RolloutPolicy | None,
    reason: str,
    mode: str,
    runtime_start: float,
    stats: _SearchStats | None = None,
) -> LateSearchResult:
    from ofc_solver.rollout import run_rollout

    fallback_policy = policy or HeuristicRolloutPolicy()
    rollout_result = run_rollout(
        state,
        root_action=action,
        root_player=perspective,
        rng=rng,
        policy=fallback_policy,
    )
    nodes = 0 if stats is None else min(stats.nodes, stats.max_nodes)
    return LateSearchResult(
        value=rollout_result.total_value,
        rollout_result=rollout_result.with_late_search(
            activated=False,
            mode=mode,
            nodes=nodes,
            depth=0 if stats is None else stats.depth_reached,
            candidate_count=0 if stats is None else stats.candidate_count,
            terminal_evaluations=0 if stats is None else stats.terminal_evaluations,
            fallback_reason=reason,
            runtime_seconds=time.perf_counter() - runtime_start,
        ),
        activated=False,
        mode=mode,
        nodes_searched=nodes,
        depth_reached=0 if stats is None else stats.depth_reached,
        candidate_count=0 if stats is None else stats.candidate_count,
        terminal_evaluations=0 if stats is None else stats.terminal_evaluations,
        fallback_reason=reason,
        runtime_seconds=time.perf_counter() - runtime_start,
    )


def _rollout_result_from_terminal(
    terminal_state: GameState,
    result: TerminalResult,
    *,
    perspective: PlayerId,
    value: float,
    late_search_mode: str,
    activated: bool,
    stats: _SearchStats,
) -> RolloutResult:
    from ofc_solver.rollout import RolloutResult

    root_breakdown, opponent_breakdown = _breakdowns_for_player(result, perspective)
    return RolloutResult(
        root_player=perspective,
        total_value=value,
        current_hand_value=value,
        continuation_value=0.0,
        continuation_hands_simulated=0,
        root_player_fouled=root_breakdown.fouled,
        opponent_fouled=opponent_breakdown.fouled,
        both_players_fouled=root_breakdown.fouled and opponent_breakdown.fouled,
        root_player_next_fantasyland=terminal_state.next_hand_fantasyland[player_index(perspective)],
        opponent_next_fantasyland=terminal_state.next_hand_fantasyland[1 - player_index(perspective)],
        late_search_activated=activated,
        late_search_mode=late_search_mode,
        late_search_nodes=stats.nodes,
        late_search_depth=stats.depth_reached,
        late_search_candidate_count=stats.candidate_count,
        late_search_terminal_evaluations=stats.terminal_evaluations,
    )


def _value_for_player(result: TerminalResult, player_id: PlayerId) -> float:
    root_breakdown, _ = _breakdowns_for_player(result, player_id)
    return float(root_breakdown.total_points)


def _breakdowns_for_player(result: TerminalResult, player_id: PlayerId):
    if PlayerId(result.left.player_id) == player_id:
        return result.left, result.right
    if PlayerId(result.right.player_id) == player_id:
        return result.right, result.left
    raise ValueError(f"Terminal result does not contain player {player_id.value}")


__all__ = [
    "LateSearchConfig",
    "LateSearchResult",
    "RankedLateSearchAction",
    "evaluate_late_root_action",
    "rank_late_root_actions",
]
