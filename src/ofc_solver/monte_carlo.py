"""Monte Carlo current-move ranking for the solver MVP."""

from __future__ import annotations

import math
import random
from typing import Callable

from ofc.actions import GameAction
from ofc.state import GameState, PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_action
from ofc_analysis.observation import PlayerObservation
from ofc_solver.models import SUPPORTED_ROOT_PHASES, MoveAnalysis, MoveEstimate
from ofc_solver.rollout import run_rollout
from ofc_solver.rollout_policy import RandomRolloutPolicy, RolloutPolicy
from ofc_solver.sampler import sample_state


def rank_actions_from_state(
    state: GameState,
    *,
    observer: PlayerId,
    rollouts_per_action: int,
    rng_seed: int | str | None,
    policy: RolloutPolicy | None = None,
) -> MoveAnalysis:
    """Rank legal root actions from an exact engine state."""

    _validate_rank_request(state.phase, state.acting_player, observer, rollouts_per_action)
    rng = random.Random(rng_seed)
    rollout_policy = policy or RandomRolloutPolicy()
    root_actions = tuple(legal_actions(state))
    estimates = _rank_actions(
        root_actions,
        rollouts_per_action=rollouts_per_action,
        rollout_fn=lambda action: run_rollout(
            state,
            root_action=action,
            root_player=observer,
            rng=rng,
            policy=rollout_policy,
        ).total_value,
    )
    return MoveAnalysis(
        observer=observer,
        phase=state.phase,
        rollouts_per_action=rollouts_per_action,
        rng_seed=rng_seed,
        ranked_actions=estimates,
    )


def rank_actions_from_observation(
    observation: PlayerObservation,
    *,
    rollouts_per_action: int,
    rng_seed: int | str | None,
    policy: RolloutPolicy | None = None,
) -> MoveAnalysis:
    """Rank legal root actions from an observer-facing information set."""

    _validate_rank_request(observation.phase, observation.acting_player, observation.observer, rollouts_per_action)
    rng = random.Random(rng_seed)
    rollout_policy = policy or RandomRolloutPolicy()
    enumeration_state = sample_state(observation, rng=random.Random(rng_seed)).state
    root_actions = tuple(legal_actions(enumeration_state))
    estimates = _rank_actions(
        root_actions,
        rollouts_per_action=rollouts_per_action,
        rollout_fn=lambda action: run_rollout(
            sample_state(observation, rng=rng).state,
            root_action=action,
            root_player=observation.observer,
            rng=rng,
            policy=rollout_policy,
        ).total_value,
    )
    return MoveAnalysis(
        observer=observation.observer,
        phase=observation.phase,
        rollouts_per_action=rollouts_per_action,
        rng_seed=rng_seed,
        ranked_actions=estimates,
    )


def _validate_rank_request(
    phase,
    acting_player: PlayerId,
    observer: PlayerId,
    rollouts_per_action: int,
) -> None:
    if phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(
            f"Unsupported root phase for solver ranking: {phase.value}. Supported phases: initial_deal, draw."
        )
    if observer != acting_player:
        raise ValueError("Solver MVP can only rank moves for the acting player")
    if rollouts_per_action <= 0:
        raise ValueError("rollouts_per_action must be positive")


def _rank_actions(
    actions: tuple[GameAction, ...],
    *,
    rollouts_per_action: int,
    rollout_fn: Callable[[GameAction], float],
) -> tuple[MoveEstimate, ...]:
    estimates = []
    for action_index, action in enumerate(actions):
        values = tuple(float(rollout_fn(action)) for _ in range(rollouts_per_action))
        estimates.append(_estimate(action_index, action, values))
    return tuple(sorted(estimates, key=lambda estimate: (-estimate.mean_value, estimate.action_index)))


def _estimate(action_index: int, action: GameAction, values: tuple[float, ...]) -> MoveEstimate:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return MoveEstimate(
        action_index=action_index,
        action=encode_action(action_index, action),
        mean_value=mean_value,
        stddev=math.sqrt(variance),
        sample_count=len(values),
        min_value=min(values),
        max_value=max(values),
    )


__all__ = ["rank_actions_from_observation", "rank_actions_from_state"]
