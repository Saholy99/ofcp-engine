"""Rollout driver for the baseline Monte Carlo solver."""

from __future__ import annotations

from dataclasses import dataclass
import random
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

    continuation_value = 0.0
    continuation_hands_simulated = 0
    continuation_trace = RolloutPolicyTrace()
    if any(terminal_state.next_hand_fantasyland):
        next_state = advance_after_showdown(terminal_state, current_result, sample_next_deck(rng=rng))
        _, continuation_result, continuation_trace = _simulate_to_terminal(next_state, rng=rng, policy=policy)
        continuation_value = _value_for_player(continuation_result, root_player)
        continuation_hands_simulated = 1
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


__all__ = ["RolloutPolicyTrace", "RolloutResult", "run_rollout"]
