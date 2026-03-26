"""Foul detection, royalties, and terminal scoring."""

from __future__ import annotations

from dataclasses import dataclass

from ofc.board import Board, RowName, board_full
from ofc.config import DEFAULT_CONFIG, VariantConfig
from ofc.evaluator import (
    HandCategory,
    compare_cross_rows_for_foul,
    compare_same_size_rows,
    evaluate_five_card_row,
    evaluate_top_row,
)


@dataclass(frozen=True)
class RowOutcome:
    """Row results from the left player's perspective."""

    top: int
    middle: int
    bottom: int

    @property
    def total(self) -> int:
        return self.top + self.middle + self.bottom


@dataclass(frozen=True)
class PlayerScoreBreakdown:
    """Terminal score components for one player."""

    player_id: str
    fouled: bool
    row_points: int
    sweep_bonus: int
    royalties: int
    total_points: int


@dataclass(frozen=True)
class TerminalResult:
    """Zero-sum terminal hand result."""

    left: PlayerScoreBreakdown
    right: PlayerScoreBreakdown
    row_outcome: RowOutcome


def _require_full_board(board: Board) -> None:
    if not board_full(board):
        raise ValueError("Board must be complete before foul or scoring checks")


def is_foul(board: Board) -> bool:
    """Return whether a completed board fouls."""

    _require_full_board(board)
    middle_vs_top = compare_cross_rows_for_foul(board.middle, board.top)
    bottom_vs_middle = compare_cross_rows_for_foul(board.bottom, board.middle)
    return middle_vs_top < 0 or bottom_vs_middle < 0


def _top_row_royalties(board: Board, config: VariantConfig) -> int:
    top_value = evaluate_top_row(board.top)
    if top_value.category == HandCategory.ONE_PAIR:
        return config.top_pair_royalties.get(top_value.tiebreak[0], 0)  # type: ignore[union-attr]
    if top_value.category == HandCategory.THREE_OF_A_KIND:
        return config.top_trips_royalties[top_value.tiebreak[0]]  # type: ignore[index]
    return 0


def _five_card_row_royalties(row: RowName, cards: tuple, config: VariantConfig) -> int:
    value = evaluate_five_card_row(cards)
    if row is RowName.MIDDLE:
        royalty_table = config.middle_royalties  # type: ignore[assignment]
    else:
        royalty_table = config.bottom_royalties  # type: ignore[assignment]

    lookup = {
        HandCategory.THREE_OF_A_KIND: "three_of_a_kind",
        HandCategory.STRAIGHT: "straight",
        HandCategory.FLUSH: "flush",
        HandCategory.FULL_HOUSE: "full_house",
        HandCategory.FOUR_OF_A_KIND: "four_of_a_kind",
        HandCategory.STRAIGHT_FLUSH: "straight_flush",
        HandCategory.ROYAL_FLUSH: "royal_flush",
    }
    royalty_key = lookup.get(value.category)
    if royalty_key is None:
        return 0
    if row is RowName.BOTTOM and value.category == HandCategory.THREE_OF_A_KIND:
        return 0
    return royalty_table.get(royalty_key, 0)


def royalties_for_board(board: Board, config: VariantConfig = DEFAULT_CONFIG) -> int:
    """Return total royalties for a completed legal board."""

    _require_full_board(board)
    if is_foul(board):
        return 0
    return (
        _top_row_royalties(board, config)
        + _five_card_row_royalties(RowName.MIDDLE, board.middle, config)
        + _five_card_row_royalties(RowName.BOTTOM, board.bottom, config)
    )


def score_rows(left_board: Board, right_board: Board) -> RowOutcome:
    """Compare corresponding rows for two completed legal boards."""

    _require_full_board(left_board)
    _require_full_board(right_board)
    return RowOutcome(
        top=compare_same_size_rows(left_board.top, right_board.top, RowName.TOP),
        middle=compare_same_size_rows(left_board.middle, right_board.middle, RowName.MIDDLE),
        bottom=compare_same_size_rows(left_board.bottom, right_board.bottom, RowName.BOTTOM),
    )


def score_terminal(
    left_player_id: str,
    left_board: Board,
    right_player_id: str,
    right_board: Board,
    config: VariantConfig = DEFAULT_CONFIG,
) -> TerminalResult:
    """Score a completed hand from both players' perspectives."""

    _require_full_board(left_board)
    _require_full_board(right_board)

    left_foul = is_foul(left_board)
    right_foul = is_foul(right_board)

    if left_foul and right_foul:
        empty_outcome = RowOutcome(top=0, middle=0, bottom=0)
        return TerminalResult(
            left=PlayerScoreBreakdown(left_player_id, True, 0, 0, 0, 0),
            right=PlayerScoreBreakdown(right_player_id, True, 0, 0, 0, 0),
            row_outcome=empty_outcome,
        )

    if left_foul or right_foul:
        left_royalties = 0 if left_foul else royalties_for_board(left_board, config)
        right_royalties = 0 if right_foul else royalties_for_board(right_board, config)
        winner_total = 6 + left_royalties + right_royalties
        if left_foul:
            return TerminalResult(
                left=PlayerScoreBreakdown(left_player_id, True, -3, -3, 0, -winner_total),
                right=PlayerScoreBreakdown(right_player_id, False, 3, 3, right_royalties, winner_total),
                row_outcome=RowOutcome(top=-1, middle=-1, bottom=-1),
            )
        return TerminalResult(
            left=PlayerScoreBreakdown(left_player_id, False, 3, 3, left_royalties, winner_total),
            right=PlayerScoreBreakdown(right_player_id, True, -3, -3, 0, -winner_total),
            row_outcome=RowOutcome(top=1, middle=1, bottom=1),
        )

    row_outcome = score_rows(left_board, right_board)
    if row_outcome.top == row_outcome.middle == row_outcome.bottom == 1:
        left_sweep = 3
        right_sweep = -3
    elif row_outcome.top == row_outcome.middle == row_outcome.bottom == -1:
        left_sweep = -3
        right_sweep = 3
    else:
        left_sweep = 0
        right_sweep = 0
    left_royalties = royalties_for_board(left_board, config)
    right_royalties = royalties_for_board(right_board, config)
    left_total = row_outcome.total + left_sweep + left_royalties - right_royalties
    right_total = -left_total
    return TerminalResult(
        left=PlayerScoreBreakdown(left_player_id, False, row_outcome.total, left_sweep, left_royalties, left_total),
        right=PlayerScoreBreakdown(right_player_id, False, -row_outcome.total, right_sweep, right_royalties, right_total),
        row_outcome=row_outcome,
    )
