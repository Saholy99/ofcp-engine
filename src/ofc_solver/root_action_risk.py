"""Cheap root-only risk scoring for early OFC solver decisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ofc.actions import GameAction
from ofc.board import Board, RowName, board_card_count, row_capacity
from ofc.cards import Rank
from ofc.config import VariantConfig
from ofc.evaluator import HandCategory, evaluate_five_card_row, evaluate_top_row
from ofc.state import GameState, HandPhase, get_player
from ofc.transitions import apply_action


@dataclass(frozen=True)
class RootRiskComponent:
    """One interpretable contribution to a root action risk score."""

    name: str
    contribution: float
    detail: str = ""


@dataclass(frozen=True)
class RootActionRiskAssessment:
    """Root-only adjustment and debug reasons for one candidate action."""

    contribution: float
    reasons: tuple[str, ...]
    components: tuple[RootRiskComponent, ...] = ()


def score_root_action(state: GameState, action: GameAction) -> RootActionRiskAssessment:
    """Return a deterministic root-only risk adjustment for an early action.

    The score is intentionally solver-side and heuristic. Engine legality,
    foul detection, royalties, and terminal scoring still come from ``ofc``.
    Negative contributions reduce the action's root ranking score.
    """

    if state.phase not in {HandPhase.INITIAL_DEAL, HandPhase.DRAW}:
        return _neutral()
    if state.phase == HandPhase.DRAW and _is_final_draw_root(state):
        return _neutral()

    next_state = apply_action(state, action)
    player_after = get_player(next_state, state.acting_player)
    board = player_after.board
    components = (
        _unsupported_top_pressure(board, state.config)
        + _middle_over_bottom_pressure(board, state.config)
        + _bottom_underbuilt_pressure(board, state.config)
        + _top_slot_pressure(board, state.config)
    )
    if not components:
        return _neutral()
    contribution = sum(component.contribution for component in components)
    reasons = tuple(component.name for component in components)
    return RootActionRiskAssessment(
        contribution=contribution,
        reasons=reasons,
        components=components,
    )


def _neutral() -> RootActionRiskAssessment:
    return RootActionRiskAssessment(contribution=0.0, reasons=(), components=())


def _is_final_draw_root(state: GameState) -> bool:
    player = get_player(state, state.acting_player)
    return player.normal_draws_taken >= state.config.normal_draw_turns_per_player - 1


def _unsupported_top_pressure(board: Board, config: VariantConfig) -> tuple[RootRiskComponent, ...]:
    if not board.top:
        return ()

    counts = Counter(int(card.rank) for card in board.top)
    middle_support = _support_rank(board.middle)
    bottom_support = _support_rank(board.bottom)

    trips = [rank for rank, count in counts.items() if count == 3]
    if trips:
        trip_rank = max(trips)
        needed_support = trip_rank + 4
        support_gap = max(0, needed_support - min(middle_support, bottom_support))
        if support_gap > 0:
            return (
                RootRiskComponent(
                    name="unsupported-top-trips",
                    contribution=-(4.0 + 0.22 * support_gap),
                    detail=f"top trips rank {trip_rank} need lower-row support",
                ),
            )
        return ()

    pairs = [rank for rank, count in counts.items() if count == 2]
    if not pairs:
        return ()

    pair_rank = max(pairs)
    if pair_rank < int(Rank.QUEEN):
        return ()
    if middle_support >= pair_rank and bottom_support >= pair_rank:
        return ()

    support_gap = max(0, pair_rank - min(middle_support, bottom_support))
    full_top_extra = 0.75 if len(board.top) == config.top_row_capacity else 0.0
    return (
        RootRiskComponent(
            name="unsupported-top-pair",
            contribution=-(2.6 + full_top_extra + 0.18 * support_gap),
            detail=f"top pair rank {pair_rank} outruns lower-row support",
        ),
    )


def _middle_over_bottom_pressure(board: Board, config: VariantConfig) -> tuple[RootRiskComponent, ...]:
    if not board.middle or not board.bottom:
        return ()
    if len(board.bottom) >= row_capacity(RowName.BOTTOM, config):
        return ()

    middle_support = _support_rank(board.middle)
    bottom_support = _support_rank(board.bottom)
    support_gap = middle_support - bottom_support
    if support_gap < 4:
        return ()
    return (
        RootRiskComponent(
            name="middle-over-bottom-pressure",
            contribution=-(1.4 + 0.18 * support_gap),
            detail=f"middle support {middle_support} exceeds bottom support {bottom_support}",
        ),
    )


def _bottom_underbuilt_pressure(board: Board, config: VariantConfig) -> tuple[RootRiskComponent, ...]:
    cards_on_board = board_card_count(board)
    if cards_on_board > 10:
        return ()
    if len(board.bottom) >= len(board.middle):
        return ()

    row_gap = len(board.middle) - len(board.bottom)
    support_gap = max(0, _support_rank(board.middle) - _support_rank(board.bottom))
    return (
        RootRiskComponent(
            name="bottom-underbuilt",
            contribution=-(0.7 * row_gap + 0.08 * support_gap),
            detail=f"bottom has {len(board.bottom)} cards behind middle {len(board.middle)}",
        ),
    )


def _top_slot_pressure(board: Board, config: VariantConfig) -> tuple[RootRiskComponent, ...]:
    if len(board.top) != config.top_row_capacity:
        return ()
    if board_card_count(board) > 7:
        return ()

    top_value = evaluate_top_row(board.top)
    if top_value.category == HandCategory.THREE_OF_A_KIND:
        return ()
    if top_value.category == HandCategory.ONE_PAIR and top_value.tiebreak[0] >= int(Rank.QUEEN):
        return ()

    high_cards = sum(1 for card in board.top if card.rank >= Rank.QUEEN)
    contribution = -(0.9 + 0.25 * high_cards)
    return (
        RootRiskComponent(
            name="top-slots-closed",
            contribution=contribution,
            detail="top row filled before lower rows are structurally stable",
        ),
    )


def _support_rank(cards) -> int:
    if not cards:
        return 0
    counts = Counter(int(card.rank) for card in cards)
    if len(cards) == 5:
        value = evaluate_five_card_row(tuple(cards))
        if value.category >= HandCategory.THREE_OF_A_KIND:
            return 18 + int(value.category)
        if value.category == HandCategory.TWO_PAIR:
            return max(rank for rank, count in counts.items() if count == 2) + 2
        if value.category == HandCategory.ONE_PAIR:
            return value.tiebreak[0]
        return max(counts) - 3
    if any(count >= 3 for count in counts.values()):
        return max(rank for rank, count in counts.items() if count >= 3) + 4
    if any(count == 2 for count in counts.values()):
        return max(rank for rank, count in counts.items() if count == 2)
    return max(counts) - 3


__all__ = [
    "RootActionRiskAssessment",
    "RootRiskComponent",
    "score_root_action",
]
