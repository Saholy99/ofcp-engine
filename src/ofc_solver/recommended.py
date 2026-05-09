"""Phase-gated recommended solver policy decisions."""

from __future__ import annotations

from dataclasses import dataclass

from ofc.state import GameState, HandPhase
from ofc.transitions import legal_actions
from ofc_solver.late_search import FinalDrawAutoSearchConfig, assess_final_draw_auto_search


@dataclass(frozen=True)
class RecommendedSolverConfig:
    """Opt-in recommended solver component gates."""

    use_initial_early_search: bool = True
    use_root_risk: bool = True
    use_final_draw_auto: bool = True


@dataclass(frozen=True)
class RecommendedSolverDecision:
    """Concrete per-root policy selected by recommended mode."""

    enabled: bool
    sub_policy: str | None
    early_search: bool = False
    root_action_risk: bool = False
    final_draw_auto_search: bool = False
    final_draw_auto_candidate_count: int = 0


def choose_recommended_solver_policy(
    state: GameState,
    *,
    config: RecommendedSolverConfig,
    final_draw_auto_config: FinalDrawAutoSearchConfig,
) -> RecommendedSolverDecision:
    """Return the recommended sub-policy for this exact root state.

    The gate uses engine state and legal root actions only. Benchmark tags are
    intentionally not part of this decision.
    """

    if state.phase == HandPhase.INITIAL_DEAL:
        if config.use_initial_early_search:
            return RecommendedSolverDecision(
                enabled=True,
                sub_policy="initial_deal_early_search",
                early_search=True,
                root_action_risk=config.use_root_risk,
            )
        return RecommendedSolverDecision(
            enabled=True,
            sub_policy="initial_deal_root_risk" if config.use_root_risk else "baseline_initial_deal",
            root_action_risk=config.use_root_risk,
        )

    if state.phase == HandPhase.DRAW:
        candidate_count = _final_draw_auto_candidate_count(state, final_draw_auto_config)
        if config.use_final_draw_auto and candidate_count > 0:
            return RecommendedSolverDecision(
                enabled=True,
                sub_policy="final_draw_auto",
                final_draw_auto_search=True,
                final_draw_auto_candidate_count=candidate_count,
            )
        return RecommendedSolverDecision(enabled=True, sub_policy="baseline_draw")

    return RecommendedSolverDecision(enabled=True, sub_policy="unsupported_phase")


def _final_draw_auto_candidate_count(
    state: GameState,
    config: FinalDrawAutoSearchConfig,
) -> int:
    if state.phase != HandPhase.DRAW:
        return 0
    return sum(
        1
        for action in legal_actions(state)
        if assess_final_draw_auto_search(state, action, config=config).eligible
    )


__all__ = [
    "RecommendedSolverConfig",
    "RecommendedSolverDecision",
    "choose_recommended_solver_policy",
]
