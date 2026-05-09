"""Monte Carlo current-move ranking for the solver MVP."""

from __future__ import annotations

import math
import random
from typing import Callable, NamedTuple

from ofc.actions import GameAction
from ofc.state import GameState, HandPhase, PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_action
from ofc_analysis.observation import PlayerObservation
from ofc_solver.early_search import EarlySearchCandidate, EarlySearchConfig, select_early_search_candidates
from ofc_solver.late_search import (
    FinalDrawAutoSearchConfig,
    LateSearchConfig,
    LateSearchResult,
    evaluate_final_draw_auto_root_action,
    evaluate_late_root_action,
)
from ofc_solver.models import SUPPORTED_ROOT_PHASES, MoveAnalysis, MoveEstimate
from ofc_solver.recommended import (
    RecommendedSolverConfig,
    RecommendedSolverDecision,
    choose_recommended_solver_policy,
)
from ofc_solver.root_action_risk import RootActionRiskAssessment, RootRiskConfig, score_root_action
from ofc_solver.rollout import run_rollout
from ofc_solver.rollout_policy import RandomRolloutPolicy, RolloutPolicy
from ofc_solver.sampler import sample_state


class _ActionEvaluation(NamedTuple):
    value: float
    late_search: LateSearchResult | None = None


def rank_actions_from_state(
    state: GameState,
    *,
    observer: PlayerId,
    rollouts_per_action: int,
    rng_seed: int | str | None,
    policy: RolloutPolicy | None = None,
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
    early_search: bool = False,
    early_search_config: EarlySearchConfig | None = None,
    late_search: bool = False,
    late_search_config: LateSearchConfig | None = None,
    final_draw_auto_search: bool = False,
    final_draw_auto_search_config: FinalDrawAutoSearchConfig | None = None,
    solver_mode: str = "manual",
    recommended_root_risk: bool = True,
    recommended_initial_early_search: bool = True,
    recommended_final_draw_auto: bool = True,
) -> MoveAnalysis:
    """Rank legal root actions from an exact engine state."""

    _validate_rank_request(state.phase, state.acting_player, observer, rollouts_per_action)
    rng = random.Random(rng_seed)
    rollout_policy = policy or RandomRolloutPolicy()
    effective_early_search_config = early_search_config or EarlySearchConfig()
    effective_late_search_config = late_search_config or LateSearchConfig()
    effective_final_draw_auto_config = final_draw_auto_search_config or FinalDrawAutoSearchConfig()
    recommended_decision = _recommended_decision(
        state,
        solver_mode=solver_mode,
        final_draw_auto_config=effective_final_draw_auto_config,
        recommended_root_risk=recommended_root_risk,
        recommended_initial_early_search=recommended_initial_early_search,
        recommended_final_draw_auto=recommended_final_draw_auto,
    )
    effective_early_search = early_search or recommended_decision.early_search
    effective_root_action_risk = root_action_risk or recommended_decision.root_action_risk
    effective_final_draw_auto_search = final_draw_auto_search or recommended_decision.final_draw_auto_search
    candidate_set = (
        select_early_search_candidates(state, config=effective_early_search_config)
        if effective_early_search
        else None
    )
    root_actions = tuple(candidate.action for candidate in candidate_set.candidates) if candidate_set else tuple(legal_actions(state))
    action_indices = tuple(candidate.action_index for candidate in candidate_set.candidates) if candidate_set else None
    rollouts = (
        rollouts_per_action + effective_early_search_config.candidate_extra_rollouts
        if effective_early_search
        else rollouts_per_action
    )
    estimates = _rank_actions(
        root_actions,
        rollouts_per_action=rollouts,
        root_risk_fn=_root_risk_fn(state, root_action_risk_config) if effective_root_action_risk else None,
        action_indices=action_indices,
        candidates=candidate_set.candidates if candidate_set else None,
        evaluate_fn=lambda action: _evaluate_root_action(
            state,
            action,
            root_player=observer,
            rng=rng,
            policy=rollout_policy,
            late_search=late_search,
            late_search_config=effective_late_search_config,
            final_draw_auto_search=effective_final_draw_auto_search,
            final_draw_auto_search_config=effective_final_draw_auto_config,
        ),
    )
    return MoveAnalysis(
        observer=observer,
        phase=state.phase,
        rollouts_per_action=rollouts,
        rng_seed=rng_seed,
        ranked_actions=estimates,
        solver_mode=solver_mode,
        recommended_solver_enabled=recommended_decision.enabled,
        recommended_sub_policy=recommended_decision.sub_policy,
        recommended_root_risk_enabled=recommended_decision.root_action_risk,
        recommended_initial_early_search_enabled=recommended_decision.early_search,
        recommended_final_draw_auto_enabled=recommended_decision.final_draw_auto_search,
        recommended_final_draw_auto_candidate_count=recommended_decision.final_draw_auto_candidate_count,
        root_action_risk_enabled=effective_root_action_risk,
        early_search_enabled=effective_early_search,
        total_legal_actions=candidate_set.total_legal_actions if candidate_set else len(root_actions),
        candidate_count=len(root_actions),
        beam_size=effective_early_search_config.beam_size if effective_early_search else None,
        candidate_extra_rollouts=effective_early_search_config.candidate_extra_rollouts if effective_early_search else 0,
        draw_safe_candidates=effective_early_search_config.draw_safe_candidates if effective_early_search else False,
        draw_baseline_keep=effective_early_search_config.draw_baseline_keep if effective_early_search else 0,
        draw_safety_keep=effective_early_search_config.draw_safety_keep if effective_early_search else 0,
        late_search_enabled=late_search,
        late_search_mode=effective_late_search_config.mode if late_search else None,
        late_search_max_depth=effective_late_search_config.max_depth if late_search else None,
        late_search_max_nodes=effective_late_search_config.max_nodes if late_search else None,
        late_search_beam_size=effective_late_search_config.beam_size if late_search else None,
        final_draw_auto_search_enabled=effective_final_draw_auto_search,
        final_draw_auto_max_depth=effective_final_draw_auto_config.max_depth if effective_final_draw_auto_search else None,
        final_draw_auto_max_nodes=effective_final_draw_auto_config.max_nodes if effective_final_draw_auto_search else None,
        final_draw_auto_include_continuation=(
            effective_final_draw_auto_config.include_continuation if effective_final_draw_auto_search else False
        ),
        final_draw_continuation_rollouts=(
            effective_final_draw_auto_config.continuation_rollouts if effective_final_draw_auto_search else 0
        ),
    )


def rank_actions_from_observation(
    observation: PlayerObservation,
    *,
    rollouts_per_action: int,
    rng_seed: int | str | None,
    policy: RolloutPolicy | None = None,
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
    early_search: bool = False,
    early_search_config: EarlySearchConfig | None = None,
    late_search: bool = False,
    late_search_config: LateSearchConfig | None = None,
    final_draw_auto_search: bool = False,
    final_draw_auto_search_config: FinalDrawAutoSearchConfig | None = None,
    solver_mode: str = "manual",
    recommended_root_risk: bool = True,
    recommended_initial_early_search: bool = True,
    recommended_final_draw_auto: bool = True,
) -> MoveAnalysis:
    """Rank legal root actions from an observer-facing information set."""

    _validate_rank_request(observation.phase, observation.acting_player, observation.observer, rollouts_per_action)
    rng = random.Random(rng_seed)
    rollout_policy = policy or RandomRolloutPolicy()
    enumeration_state = sample_state(observation, rng=random.Random(rng_seed)).state
    effective_early_search_config = early_search_config or EarlySearchConfig()
    effective_late_search_config = late_search_config or LateSearchConfig()
    effective_final_draw_auto_config = final_draw_auto_search_config or FinalDrawAutoSearchConfig()
    recommended_decision = _recommended_decision(
        enumeration_state,
        solver_mode=solver_mode,
        final_draw_auto_config=effective_final_draw_auto_config,
        recommended_root_risk=recommended_root_risk,
        recommended_initial_early_search=recommended_initial_early_search,
        recommended_final_draw_auto=recommended_final_draw_auto,
    )
    effective_early_search = early_search or recommended_decision.early_search
    effective_root_action_risk = root_action_risk or recommended_decision.root_action_risk
    effective_final_draw_auto_search = final_draw_auto_search or recommended_decision.final_draw_auto_search
    candidate_set = (
        select_early_search_candidates(enumeration_state, config=effective_early_search_config)
        if effective_early_search
        else None
    )
    root_actions = tuple(candidate.action for candidate in candidate_set.candidates) if candidate_set else tuple(legal_actions(enumeration_state))
    action_indices = tuple(candidate.action_index for candidate in candidate_set.candidates) if candidate_set else None
    rollouts = (
        rollouts_per_action + effective_early_search_config.candidate_extra_rollouts
        if effective_early_search
        else rollouts_per_action
    )
    estimates = _rank_actions(
        root_actions,
        rollouts_per_action=rollouts,
        root_risk_fn=_root_risk_fn(enumeration_state, root_action_risk_config) if effective_root_action_risk else None,
        action_indices=action_indices,
        candidates=candidate_set.candidates if candidate_set else None,
        evaluate_fn=lambda action: _evaluate_root_action(
            sample_state(observation, rng=rng).state,
            action,
            root_player=observation.observer,
            rng=rng,
            policy=rollout_policy,
            late_search=late_search,
            late_search_config=effective_late_search_config,
            final_draw_auto_search=effective_final_draw_auto_search,
            final_draw_auto_search_config=effective_final_draw_auto_config,
        ),
    )
    return MoveAnalysis(
        observer=observation.observer,
        phase=observation.phase,
        rollouts_per_action=rollouts,
        rng_seed=rng_seed,
        ranked_actions=estimates,
        solver_mode=solver_mode,
        recommended_solver_enabled=recommended_decision.enabled,
        recommended_sub_policy=recommended_decision.sub_policy,
        recommended_root_risk_enabled=recommended_decision.root_action_risk,
        recommended_initial_early_search_enabled=recommended_decision.early_search,
        recommended_final_draw_auto_enabled=recommended_decision.final_draw_auto_search,
        recommended_final_draw_auto_candidate_count=recommended_decision.final_draw_auto_candidate_count,
        root_action_risk_enabled=effective_root_action_risk,
        early_search_enabled=effective_early_search,
        total_legal_actions=candidate_set.total_legal_actions if candidate_set else len(root_actions),
        candidate_count=len(root_actions),
        beam_size=effective_early_search_config.beam_size if effective_early_search else None,
        candidate_extra_rollouts=effective_early_search_config.candidate_extra_rollouts if effective_early_search else 0,
        draw_safe_candidates=effective_early_search_config.draw_safe_candidates if effective_early_search else False,
        draw_baseline_keep=effective_early_search_config.draw_baseline_keep if effective_early_search else 0,
        draw_safety_keep=effective_early_search_config.draw_safety_keep if effective_early_search else 0,
        late_search_enabled=late_search,
        late_search_mode=effective_late_search_config.mode if late_search else None,
        late_search_max_depth=effective_late_search_config.max_depth if late_search else None,
        late_search_max_nodes=effective_late_search_config.max_nodes if late_search else None,
        late_search_beam_size=effective_late_search_config.beam_size if late_search else None,
        final_draw_auto_search_enabled=effective_final_draw_auto_search,
        final_draw_auto_max_depth=effective_final_draw_auto_config.max_depth if effective_final_draw_auto_search else None,
        final_draw_auto_max_nodes=effective_final_draw_auto_config.max_nodes if effective_final_draw_auto_search else None,
        final_draw_auto_include_continuation=(
            effective_final_draw_auto_config.include_continuation if effective_final_draw_auto_search else False
        ),
        final_draw_continuation_rollouts=(
            effective_final_draw_auto_config.continuation_rollouts if effective_final_draw_auto_search else 0
        ),
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


def _recommended_decision(
    state: GameState,
    *,
    solver_mode: str,
    final_draw_auto_config: FinalDrawAutoSearchConfig,
    recommended_root_risk: bool,
    recommended_initial_early_search: bool,
    recommended_final_draw_auto: bool,
):
    if solver_mode not in {"manual", "recommended"}:
        raise ValueError("solver_mode must be one of: manual, recommended")
    if solver_mode != "recommended":
        return RecommendedSolverDecision(enabled=False, sub_policy=None)
    return choose_recommended_solver_policy(
        state,
        config=RecommendedSolverConfig(
            use_initial_early_search=recommended_initial_early_search,
            use_root_risk=recommended_root_risk,
            use_final_draw_auto=recommended_final_draw_auto,
        ),
        final_draw_auto_config=final_draw_auto_config,
    )


def _rank_actions(
    actions: tuple[GameAction, ...],
    *,
    rollouts_per_action: int,
    evaluate_fn: Callable[[GameAction], _ActionEvaluation],
    root_risk_fn: Callable[[GameAction], RootActionRiskAssessment] | None = None,
    action_indices: tuple[int, ...] | None = None,
    candidates: tuple[EarlySearchCandidate, ...] | None = None,
) -> tuple[MoveEstimate, ...]:
    estimates = []
    candidate_by_index = {} if candidates is None else {candidate.action_index: candidate for candidate in candidates}
    effective_indices = action_indices or tuple(range(len(actions)))
    for action_index, action in zip(effective_indices, actions, strict=True):
        evaluations = tuple(evaluate_fn(action) for _ in range(rollouts_per_action))
        values = tuple(evaluation.value for evaluation in evaluations)
        late_results = tuple(
            evaluation.late_search
            for evaluation in evaluations
            if evaluation.late_search is not None
        )
        root_risk = root_risk_fn(action) if root_risk_fn is not None else None
        estimates.append(
            _estimate(
                action_index,
                action,
                values,
                root_risk=root_risk,
                candidate=candidate_by_index.get(action_index),
                late_search_results=late_results,
            )
        )
    return tuple(sorted(estimates, key=lambda estimate: (-estimate.mean_value, estimate.action_index)))


def _root_risk_fn(
    state: GameState,
    config: RootRiskConfig | None,
) -> Callable[[GameAction], RootActionRiskAssessment]:
    effective_config = config or RootRiskConfig.default()
    return lambda action: score_root_action(state, action, config=effective_config)


def _estimate(
    action_index: int,
    action: GameAction,
    values: tuple[float, ...],
    *,
    root_risk: RootActionRiskAssessment | None = None,
    candidate: EarlySearchCandidate | None = None,
    late_search_results: tuple[LateSearchResult, ...] = (),
) -> MoveEstimate:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    risk_score = 0.0 if root_risk is None else root_risk.contribution
    pattern_score = 0.0 if candidate is None else candidate.pattern_score
    final_score = mean_value + risk_score
    late_modes = {result.mode for result in late_search_results}
    fallback_reasons = tuple(
        reason
        for reason in (result.fallback_reason for result in late_search_results)
        if reason is not None
    )
    phase_auto_reasons = tuple(
        reason
        for reason in (result.phase_auto_search_reason for result in late_search_results)
        if reason is not None
    )
    continuation_reasons = tuple(
        reason
        for reason in (result.continuation_reason for result in late_search_results)
        if reason is not None
    )
    return MoveEstimate(
        action_index=action_index,
        action=encode_action(action_index, action),
        mean_value=final_score,
        stddev=math.sqrt(variance),
        sample_count=len(values),
        min_value=min(values),
        max_value=max(values),
        rollout_mean_value=mean_value,
        root_risk_score=risk_score,
        root_risk_reasons=() if root_risk is None else root_risk.reasons,
        pattern_score=pattern_score,
        pattern_reasons=() if candidate is None else candidate.reasons,
        selection_reasons=() if candidate is None else candidate.selection_reasons,
        candidate_rank=None if candidate is None else candidate.candidate_rank,
        final_score=final_score,
        late_search_activated=any(result.activated for result in late_search_results),
        late_search_mode=late_modes.pop() if len(late_modes) == 1 else "mixed" if late_modes else None,
        late_search_nodes=sum(result.nodes_searched for result in late_search_results),
        late_search_depth=max((result.depth_reached for result in late_search_results), default=0),
        late_search_candidate_count=sum(result.candidate_count for result in late_search_results),
        late_search_terminal_evaluations=sum(result.terminal_evaluations for result in late_search_results),
        late_search_fallback_reason=";".join(dict.fromkeys(fallback_reasons)) if fallback_reasons else None,
        phase_auto_search_activated=any(result.phase_auto_search_activated for result in late_search_results),
        phase_auto_search_reason=";".join(dict.fromkeys(phase_auto_reasons)) if phase_auto_reasons else None,
        phase_auto_search_tree_nodes=sum(result.phase_auto_search_tree_nodes for result in late_search_results),
        phase_auto_search_depth=max((result.phase_auto_search_depth for result in late_search_results), default=0),
        late_search_runtime_seconds=sum(result.runtime_seconds for result in late_search_results),
        final_draw_continuation_aware=any(result.continuation_aware for result in late_search_results),
        final_draw_continuation_triggered=any(result.continuation_triggered for result in late_search_results),
        final_draw_continuation_rollouts=sum(result.continuation_rollouts for result in late_search_results),
        final_draw_current_hand_value=_mean_or_zero(result.current_hand_value for result in late_search_results),
        final_draw_continuation_value=_mean_or_zero(result.continuation_value for result in late_search_results),
        final_draw_total_value=_mean_or_zero(result.value for result in late_search_results),
        final_draw_continuation_reason=(
            ";".join(dict.fromkeys(continuation_reasons)) if continuation_reasons else None
        ),
        final_draw_continuation_runtime_seconds=sum(
            result.continuation_runtime_seconds for result in late_search_results
        ),
    )


def _mean_or_zero(values) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _evaluate_root_action(
    state: GameState,
    action: GameAction,
    *,
    root_player: PlayerId,
    rng: random.Random,
    policy: RolloutPolicy,
    late_search: bool,
    late_search_config: LateSearchConfig,
    final_draw_auto_search: bool,
    final_draw_auto_search_config: FinalDrawAutoSearchConfig,
) -> _ActionEvaluation:
    if late_search and state.phase == HandPhase.DRAW:
        result = evaluate_late_root_action(
            state,
            action,
            perspective=root_player,
            rng=rng,
            config=late_search_config,
            policy=policy,
        )
        return _ActionEvaluation(value=result.value, late_search=result)
    if final_draw_auto_search and state.phase == HandPhase.DRAW:
        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=root_player,
            rng=rng,
            config=final_draw_auto_search_config,
            policy=policy,
        )
        return _ActionEvaluation(value=result.value, late_search=result)
    rollout = run_rollout(
        state,
        root_action=action,
        root_player=root_player,
        rng=rng,
        policy=policy,
    )
    return _ActionEvaluation(value=rollout.total_value)


__all__ = ["rank_actions_from_observation", "rank_actions_from_state"]
