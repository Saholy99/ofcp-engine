"""Structured action types for explicit game progression."""

from __future__ import annotations

from dataclasses import dataclass

from ofc.board import RowName
from ofc.cards import Card
from ofc.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class Placement:
    """Place a specific card into a specific row."""

    row: RowName
    card: Card


def _validate_unique_cards(cards: tuple[Card, ...], expected_count: int) -> None:
    if len(cards) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} cards, got {len(cards)}")
    if len(set(cards)) != expected_count:
        raise ValueError("Action cards must be unique")


@dataclass(frozen=True)
class PlaceInitialFiveAction:
    """Place all five initially dealt cards."""

    player_id: str
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        _validate_unique_cards(tuple(placement.card for placement in self.placements), DEFAULT_CONFIG.initial_deal_count)


@dataclass(frozen=True)
class PlaceDrawAction:
    """Place exactly two of three drawn cards and discard one."""

    player_id: str
    placements: tuple[Placement, ...]
    discard: Card

    def __post_init__(self) -> None:
        cards = tuple(placement.card for placement in self.placements) + (self.discard,)
        _validate_unique_cards(cards, DEFAULT_CONFIG.normal_draw_count)
        if len(self.placements) != DEFAULT_CONFIG.normal_draw_placements:
            raise ValueError("A Pineapple draw action must place exactly two cards")


@dataclass(frozen=True)
class SetFantasylandHandAction:
    """Set a concealed 13-card Fantasyland hand and discard one."""

    player_id: str
    placements: tuple[Placement, ...]
    discard: Card

    def __post_init__(self) -> None:
        cards = tuple(placement.card for placement in self.placements) + (self.discard,)
        _validate_unique_cards(cards, DEFAULT_CONFIG.fantasyland_deal_count)
        if len(self.placements) != DEFAULT_CONFIG.fantasyland_placements:
            raise ValueError("A Fantasyland set action must place exactly thirteen cards")


GameAction = PlaceInitialFiveAction | PlaceDrawAction | SetFantasylandHandAction
