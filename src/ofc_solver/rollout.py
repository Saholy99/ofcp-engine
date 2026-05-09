"""Rollout driver for the baseline Monte Carlo solver."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import random
import time
from typing import Any, Callable, cast

from ofc.actions import GameAction
from ofc.engine import showdown
from ofc.scoring import TerminalResult
from ofc.state import GameState, HandPhase, PlayerId, player_index
from ofc.transitions import advance_after_showdown, apply_action
from ofc_solver.models import SUPPORTED_ROOT_PHASES
from ofc_solver.rollout_policy import RolloutPolicy
from ofc_solver.sampler import sample_next_deck


@dataclass(frozen=True)
class RolloutResult:
    """Terminal-value result for one Monte Carlo rollout."""

    root_player: PlayerId
    total_value: float
    current_hand_value: float
    continuation_value: float
    continuation_hands_simulated: int
    root_player_fouled: bool = False
    opponent_fouled: bool = False
    both_players_fouled: bool = False
    root_player_next_fantasyland: bool = False
    opponent_next_fantasyland: bool = False
    policy_decision_count: int = 0
    exact_late_search_decision_count: int = 0
    exact_late_search_node_count: int = 0
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
    final_draw_continuation_enabled: bool = False
    final_draw_continuation_triggered: bool = False
    final_draw_continuation_rollouts: int = 0
    final_draw_continuation_value: float = 0.0
    final_draw_current_hand_value: float = 0.0
    final_draw_total_value: float = 0.0
    final_draw_continuation_runtime_seconds: float = 0.0
    final_draw_continuation_reason: str | None = None

    def with_late_search(
        self,
        *,
        activated: bool,
        mode: str,
        nodes: int,
        depth: int,
        candidate_count: int,
        terminal_evaluations: int,
        fallback_reason: str | None = None,
        runtime_seconds: float = 0.0,
    ) -> "RolloutResult":
        """Return a copy annotated with root late-search diagnostics."""

        return replace(
            self,
            late_search_activated=activated,
            late_search_mode=mode,
            late_search_nodes=nodes,
            late_search_depth=depth,
            late_search_candidate_count=candidate_count,
            late_search_terminal_evaluations=terminal_evaluations,
            late_search_fallback_reason=fallback_reason,
            late_search_runtime_seconds=runtime_seconds,
        )

    def with_phase_auto_search(
        self,
        *,
        activated: bool,
        reason: str | None,
        tree_nodes: int,
        depth: int,
    ) -> "RolloutResult":
        """Return a copy annotated with final-draw auto gate diagnostics."""

        return replace(
            self,
            phase_auto_search_activated=activated,
            phase_auto_search_reason=reason,
            phase_auto_search_tree_nodes=tree_nodes,
            phase_auto_search_depth=depth,
        )

    def with_final_draw_continuation(
        self,
        *,
        enabled: bool,
        triggered: bool,
        rollouts: int,
        continuation_value: float,
        current_hand_value: float,
        total_value: float,
        runtime_seconds: float = 0.0,
        reason: str | None = None,
    ) -> "RolloutResult":
        """Return a copy annotated with final-draw continuation diagnostics."""

        return replace(
            self,
            total_value=total_value,
            current_hand_value=current_hand_value,
            continuation_value=continuation_value,
            continuation_hands_simulated=rollouts if triggered else 0,
            final_draw_continuation_enabled=enabled,
            final_draw_continuation_triggered=triggered,
            final_draw_continuation_rollouts=rollouts if triggered else 0,
            final_draw_continuation_value=continuation_value,
            final_draw_current_hand_value=current_hand_value,
            final_draw_total_value=total_value,
            final_draw_continuation_runtime_seconds=runtime_seconds,
            final_draw_continuation_reason=reason,
        )


@dataclass(frozen=True)
class ContinuationSimulationResult:
    """One immediate Fantasyland continuation simulation result."""

    value: float
    hands_simulated: int
    policy_trace: RolloutPolicyTrace
    triggered: bool
    reason: str | None = None
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class RolloutPolicyTrace:
    """Diagnostics accumulated while a rollout policy plays one simulated hand."""

    policy_decision_count: int = 0
    exact_late_search_decision_count: int = 0
    exact_late_search_node_count: int = 0

    def combine(self, other: "RolloutPolicyTrace") -> "RolloutPolicyTrace":
        return RolloutPolicyTrace(
            policy_decision_count=self.policy_decision_count + other.policy_decision_count,
            exact_late_search_decision_count=(
                self.exact_late_search_decision_count + other.exact_late_search_decision_count
            ),
            exact_late_search_node_count=self.exact_late_search_node_count + other.exact_late_search_node_count,
        )


def run_rollout(
    state: GameState,
    *,
    root_action: GameAction,
    root_player: PlayerId,
    rng: random.Random,
    policy: RolloutPolicy,
) -> RolloutResult:
    """Run one simulated continuation from a root state and action."""

    if state.phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(
            "Unsupported root phase for solver rollout: "
            f"{state.phase.value}. Supported phases: initial_deal, draw."
        )

    state_after_root = apply_action(state, root_action)
    terminal_state, current_result, current_trace = _simulate_to_terminal(state_after_root, rng=rng, policy=policy)
    current_value = _value_for_player(current_result, root_player)
    root_breakdown, opponent_breakdown = _breakdowns_for_player(current_result, root_player)

    continuation = simulate_one_fantasyland_continuation(
        terminal_state,
        current_result,
        root_player=root_player,
        rng=rng,
        policy=policy,
    )
    continuation_value = continuation.value
    continuation_hands_simulated = continuation.hands_simulated
    continuation_trace = continuation.policy_trace
    combined_trace = current_trace.combine(continuation_trace)

    return RolloutResult(
        root_player=root_player,
        total_value=current_value + continuation_value,
        current_hand_value=current_value,
        continuation_value=continuation_value,
        continuation_hands_simulated=continuation_hands_simulated,
        root_player_fouled=root_breakdown.fouled,
        opponent_fouled=opponent_breakdown.fouled,
        both_players_fouled=root_breakdown.fouled and opponent_breakdown.fouled,
        root_player_next_fantasyland=terminal_state.next_hand_fantasyland[player_index(root_player)],
        opponent_next_fantasyland=terminal_state.next_hand_fantasyland[1 - player_index(root_player)],
        policy_decision_count=combined_trace.policy_decision_count,
        exact_late_search_decision_count=combined_trace.exact_late_search_decision_count,
        exact_late_search_node_count=combined_trace.exact_late_search_node_count,
    )


def simulate_one_fantasyland_continuation(
    terminal_state: GameState,
    current_result: TerminalResult,
    *,
    root_player: PlayerId,
    rng: random.Random,
    policy: RolloutPolicy,
) -> ContinuationSimulationResult:
    """Simulate the rollout convention's single immediate Fantasyland hand.

    The baseline solver advances exactly one hand when the current showdown creates
    any next-hand Fantasyland flag.  The next deck is sampled through
    :func:`sample_next_deck`, the continuation is played by the rollout policy, and
    its zero-sum terminal value is added from the original root player's
    perspective.  This helper intentionally does not recurse if that continuation
    hand creates another Fantasyland entry or stay.
    """

    start = time.perf_counter()
    if not any(terminal_state.next_hand_fantasyland):
        return ContinuationSimulationResult(
            value=0.0,
            hands_simulated=0,
            policy_trace=RolloutPolicyTrace(),
            triggered=False,
            reason="no-fantasyland-continuation",
            runtime_seconds=time.perf_counter() - start,
        )

    next_state = advance_after_showdown(terminal_state, current_result, sample_next_deck(rng=rng))
    _, continuation_result, continuation_trace = _simulate_to_terminal(next_state, rng=rng, policy=policy)
    return ContinuationSimulationResult(
        value=_value_for_player(continuation_result, root_player),
        hands_simulated=1,
        policy_trace=continuation_trace,
        triggered=True,
        reason="simulated",
        runtime_seconds=time.perf_counter() - start,
    )


def _simulate_to_terminal(
    state: GameState,
    *,
    rng: random.Random,
    policy: RolloutPolicy,
) -> tuple[GameState, TerminalResult, RolloutPolicyTrace]:
    trace = RolloutPolicyTrace()
    while state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        action, decision_trace = _choose_policy_action(policy, state, rng=rng)
        trace = trace.combine(decision_trace)
        state = apply_action(state, action)
    terminal_state, result = showdown(state)
    return terminal_state, result, trace


def _choose_policy_action(
    policy: RolloutPolicy,
    state: GameState,
    *,
    rng: random.Random,
) -> tuple[GameAction, RolloutPolicyTrace]:
    diagnostic_chooser = getattr(policy, "choose_action_with_diagnostics", None)
    if callable(diagnostic_chooser):
        action, diagnostics = cast(Callable[..., Any], diagnostic_chooser)(state, rng=rng)
        return action, RolloutPolicyTrace(
            policy_decision_count=1,
            exact_late_search_decision_count=1 if getattr(diagnostics, "used_exact_late_search", False) else 0,
            exact_late_search_node_count=int(getattr(diagnostics, "exact_late_search_node_count", 0)),
        )
    return policy.choose_action(state, rng=rng), RolloutPolicyTrace(policy_decision_count=1)


def _value_for_player(result: TerminalResult, player_id: PlayerId) -> float:
    if PlayerId(result.left.player_id) == player_id:
        return float(result.left.total_points)
    if PlayerId(result.right.player_id) == player_id:
        return float(result.right.total_points)
    raise ValueError(f"Terminal result does not contain player {player_id.value}")


def _breakdowns_for_player(result: TerminalResult, player_id: PlayerId):
    if PlayerId(result.left.player_id) == player_id:
        return result.left, result.right
    if PlayerId(result.right.player_id) == player_id:
        return result.right, result.left
    raise ValueError(f"Terminal result does not contain player {player_id.value}")


__all__ = [
    "ContinuationSimulationResult",
    "RolloutPolicyTrace",
    "RolloutResult",
    "run_rollout",
    "simulate_one_fantasyland_continuation",
]
