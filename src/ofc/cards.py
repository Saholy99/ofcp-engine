"""Card types and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(str, Enum):
    """Card suits."""

    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Rank(IntEnum):
    """Card ranks."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


_RANK_TOKEN_TO_RANK = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

_RANK_TO_TOKEN = {rank: token for token, rank in _RANK_TOKEN_TO_RANK.items()}
_SUIT_TOKEN_TO_SUIT = {suit.value: suit for suit in Suit}


@dataclass(frozen=True, order=True)
class Card:
    """Immutable playing card."""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return format_card(self)


def parse_card(token: str) -> Card:
    """Parse a two-character card token like ``Ah`` or ``Tc``."""

    if len(token) != 2:
        raise ValueError(f"Card token must be two characters: {token!r}")
    rank_token = token[0].upper()
    suit_token = token[1].lower()
    try:
        rank = _RANK_TOKEN_TO_RANK[rank_token]
        suit = _SUIT_TOKEN_TO_SUIT[suit_token]
    except KeyError as exc:
        raise ValueError(f"Invalid card token: {token!r}") from exc
    return Card(rank=rank, suit=suit)


def format_card(card: Card) -> str:
    """Format a card as a two-character token."""

    return f"{_RANK_TO_TOKEN[card.rank]}{card.suit.value}"


def full_deck() -> tuple[Card, ...]:
    """Return the canonical 52-card deck order."""

    return tuple(Card(rank=rank, suit=suit) for suit in Suit for rank in Rank)
