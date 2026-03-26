"""Board structures and placement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ofc.cards import Card
from ofc.config import DEFAULT_CONFIG, VariantConfig


class RowName(str, Enum):
    """Board row names."""

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


ROW_ORDER = (RowName.TOP, RowName.MIDDLE, RowName.BOTTOM)


@dataclass(frozen=True)
class Board:
    """Immutable board state for a single player."""

    top: tuple[Card, ...] = ()
    middle: tuple[Card, ...] = ()
    bottom: tuple[Card, ...] = ()


def row_cards(board: Board, row: RowName) -> tuple[Card, ...]:
    """Return the cards currently in the requested row."""

    return getattr(board, row.value)


def row_capacity(row: RowName, config: VariantConfig = DEFAULT_CONFIG) -> int:
    """Return the maximum capacity for a row."""

    if row is RowName.TOP:
        return config.top_row_capacity
    if row is RowName.MIDDLE:
        return config.middle_row_capacity
    return config.bottom_row_capacity


def row_capacity_remaining(board: Board, row: RowName, config: VariantConfig = DEFAULT_CONFIG) -> int:
    """Return the number of remaining slots in a row."""

    return row_capacity(row, config) - len(row_cards(board, row))


def board_card_count(board: Board) -> int:
    """Return the total number of cards on the board."""

    return len(board.top) + len(board.middle) + len(board.bottom)


def board_full(board: Board, config: VariantConfig = DEFAULT_CONFIG) -> bool:
    """Return whether all three rows are filled to capacity."""

    return (
        len(board.top) == config.top_row_capacity
        and len(board.middle) == config.middle_row_capacity
        and len(board.bottom) == config.bottom_row_capacity
    )


def visible_cards(board: Board) -> tuple[Card, ...]:
    """Return all visible cards on the board."""

    return board.top + board.middle + board.bottom


def place_cards(
    board: Board,
    placements: Iterable[tuple[RowName, Card]],
    config: VariantConfig = DEFAULT_CONFIG,
) -> Board:
    """Return a new board with the requested cards appended to the requested rows."""

    top = list(board.top)
    middle = list(board.middle)
    bottom = list(board.bottom)
    seen_existing = set(visible_cards(board))
    new_cards: set[Card] = set()

    for row, card in placements:
        if card in seen_existing or card in new_cards:
            raise ValueError(f"Card {card} is already on the board")
        target = top if row is RowName.TOP else middle if row is RowName.MIDDLE else bottom
        if len(target) >= row_capacity(row, config):
            raise ValueError(f"Row {row.value} is already full")
        target.append(card)
        new_cards.add(card)

    return Board(top=tuple(top), middle=tuple(middle), bottom=tuple(bottom))
