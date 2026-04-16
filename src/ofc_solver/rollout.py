"""Rollout driver for the baseline Monte Carlo solver."""

from __future__ import annotations

from dataclasses import dataclass
import random

from ofc.actions import GameAction
from ofc.engine import showdown
from ofc.scoring import TerminalResult
from ofc.state import GameState, HandPhase, PlayerId
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
    terminal_state, current_result = _simulate_to_terminal(state_after_root, rng=rng, policy=policy)
    current_value = _value_for_player(current_result, root_player)

    continuation_value = 0.0
    continuation_hands_simulated = 0
    if any(terminal_state.next_hand_fantasyland):
        next_state = advance_after_showdown(terminal_state, current_result, sample_next_deck(rng=rng))
        _, continuation_result = _simulate_to_terminal(next_state, rng=rng, policy=policy)
        continuation_value = _value_for_player(continuation_result, root_player)
        continuation_hands_simulated = 1

    return RolloutResult(
        root_player=root_player,
        total_value=current_value + continuation_value,
        current_hand_value=current_value,
        continuation_value=continuation_value,
        continuation_hands_simulated=continuation_hands_simulated,
    )


def _simulate_to_terminal(
    state: GameState,
    *,
    rng: random.Random,
    policy: RolloutPolicy,
) -> tuple[GameState, TerminalResult]:
    while state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        action = policy.choose_action(state, rng=rng)
        state = apply_action(state, action)
    return showdown(state)


def _value_for_player(result: TerminalResult, player_id: PlayerId) -> float:
    if PlayerId(result.left.player_id) == player_id:
        return float(result.left.total_points)
    if PlayerId(result.right.player_id) == player_id:
        return float(result.right.total_points)
    raise ValueError(f"Terminal result does not contain player {player_id.value}")


__all__ = ["RolloutResult", "run_rollout"]
