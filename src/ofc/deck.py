"""Deterministic deck construction and drawing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from ofc.cards import Card, full_deck, parse_card


@dataclass(frozen=True)
class DeckState:
    """Remaining undealt deck order for a single hand."""

    undealt_cards: tuple[Card, ...]

    @property
    def cards_remaining(self) -> int:
        return len(self.undealt_cards)


def _normalize_cards(cards: Iterable[Card | str]) -> tuple[Card, ...]:
    normalized = tuple(parse_card(card) if isinstance(card, str) else card for card in cards)
    if len(normalized) != 52:
        raise ValueError("A full OFC deck must contain exactly 52 cards")
    if len(set(normalized)) != 52:
        raise ValueError("Deck order must contain 52 unique cards")
    return normalized


def make_deck(*, seed: int | str | None = None, preset_order: Iterable[Card | str] | None = None) -> DeckState:
    """Build a deterministic deck from a seed or explicit card order."""

    if seed is not None and preset_order is not None:
        raise ValueError("Pass either seed or preset_order, not both")
    if preset_order is not None:
        return DeckState(undealt_cards=_normalize_cards(preset_order))
    cards = list(full_deck())
    if seed is not None:
        random.Random(seed).shuffle(cards)
    return DeckState(undealt_cards=tuple(cards))


def draw_n(deck: DeckState, count: int) -> tuple[tuple[Card, ...], DeckState]:
    """Draw the next ``count`` cards from the undealt deck."""

    if count < 0:
        raise ValueError("Cannot draw a negative number of cards")
    if count > deck.cards_remaining:
        raise ValueError("Cannot draw more cards than remain in the deck")
    drawn = deck.undealt_cards[:count]
    remaining = deck.undealt_cards[count:]
    return drawn, DeckState(undealt_cards=remaining)


def remaining_cards(deck: DeckState) -> tuple[Card, ...]:
    """Return the undealt cards in order."""

    return deck.undealt_cards
