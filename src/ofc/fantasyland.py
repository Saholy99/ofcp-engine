"""Fantasyland qualification helpers."""

from __future__ import annotations

from ofc.board import Board
from ofc.evaluator import HandCategory, evaluate_five_card_row, evaluate_top_row
from ofc.scoring import is_foul


def qualifies_for_fantasyland(board: Board) -> bool:
    """Return whether a completed legal hand enters Fantasyland next hand."""

    if is_foul(board):
        return False
    top_value = evaluate_top_row(board.top)
    if top_value.category == HandCategory.THREE_OF_A_KIND:
        return True
    return top_value.category == HandCategory.ONE_PAIR and top_value.tiebreak[0] >= 12


def qualifies_to_stay_in_fantasyland(board: Board) -> bool:
    """Return whether a completed Fantasyland hand stays in Fantasyland."""

    if is_foul(board):
        return False
    top_value = evaluate_top_row(board.top)
    middle_value = evaluate_five_card_row(board.middle)
    bottom_value = evaluate_five_card_row(board.bottom)
    return (
        top_value.category == HandCategory.THREE_OF_A_KIND
        or middle_value.category >= HandCategory.FULL_HOUSE
        or bottom_value.category >= HandCategory.FOUR_OF_A_KIND
    )


def resolve_next_hand_fantasyland_flags(
    current_fantasyland_flags: tuple[bool, bool],
    boards: tuple[Board, Board],
) -> tuple[bool, bool]:
    """Resolve next-hand Fantasyland status for both players."""

    next_flags = []
    for is_active, board in zip(current_fantasyland_flags, boards, strict=True):
        if is_active:
            next_flags.append(qualifies_to_stay_in_fantasyland(board))
        else:
            next_flags.append(qualifies_for_fantasyland(board))
    return tuple(next_flags)  # type: ignore[return-value]
