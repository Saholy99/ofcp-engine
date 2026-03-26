"""Row evaluation and comparison helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from ofc.board import RowName
from ofc.cards import Card, Rank


class HandCategory(IntEnum):
    """Comparable hand categories shared across OFC rows."""

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


@dataclass(frozen=True)
class TopRowValue:
    """Comparable top-row value."""

    category: HandCategory
    tiebreak: tuple[int, ...]
    cards: tuple[Card, ...]


@dataclass(frozen=True)
class FiveCardValue:
    """Comparable five-card row value."""

    category: HandCategory
    tiebreak: tuple[int, ...]
    cards: tuple[Card, ...]


ComparableRowValue = TopRowValue | FiveCardValue


def _validate_unique_cards(cards: Iterable[Card], expected: int) -> tuple[Card, ...]:
    normalized = tuple(cards)
    if len(normalized) != expected:
        raise ValueError(f"Expected exactly {expected} cards, got {len(normalized)}")
    if len(set(normalized)) != expected:
        raise ValueError("Rows may not contain duplicate cards")
    return normalized


def _straight_high_card(ranks: Iterable[int]) -> int | None:
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return None
    if unique == [2, 3, 4, 5, 14]:
        return 5
    if unique[-1] - unique[0] == 4:
        return unique[-1]
    return None


def _compare_keys(left: tuple[int, tuple[int, ...]], right: tuple[int, tuple[int, ...]]) -> int:
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def evaluate_top_row(cards: Iterable[Card]) -> TopRowValue:
    """Evaluate a 3-card OFC top row."""

    normalized = _validate_unique_cards(cards, 3)
    ranks = sorted((int(card.rank) for card in normalized), reverse=True)
    counts = Counter(ranks)
    if 3 in counts.values():
        trip_rank = max(rank for rank, count in counts.items() if count == 3)
        return TopRowValue(HandCategory.THREE_OF_A_KIND, (trip_rank,), normalized)
    if 2 in counts.values():
        pair_rank = max(rank for rank, count in counts.items() if count == 2)
        kicker = max(rank for rank, count in counts.items() if count == 1)
        return TopRowValue(HandCategory.ONE_PAIR, (pair_rank, kicker), normalized)
    return TopRowValue(HandCategory.HIGH_CARD, tuple(ranks), normalized)


def evaluate_five_card_row(cards: Iterable[Card]) -> FiveCardValue:
    """Evaluate a standard 5-card poker hand."""

    normalized = _validate_unique_cards(cards, 5)
    ranks = [int(card.rank) for card in normalized]
    rank_counts = Counter(ranks)
    sorted_ranks_desc = tuple(sorted(ranks, reverse=True))
    is_flush = len({card.suit for card in normalized}) == 1
    straight_high = _straight_high_card(ranks)

    if straight_high is not None and is_flush:
        if set(ranks) == {10, 11, 12, 13, 14}:
            return FiveCardValue(HandCategory.ROYAL_FLUSH, (int(Rank.ACE),), normalized)
        return FiveCardValue(HandCategory.STRAIGHT_FLUSH, (straight_high,), normalized)

    if 4 in rank_counts.values():
        quad_rank = max(rank for rank, count in rank_counts.items() if count == 4)
        kicker = max(rank for rank, count in rank_counts.items() if count == 1)
        return FiveCardValue(HandCategory.FOUR_OF_A_KIND, (quad_rank, kicker), normalized)

    if sorted(rank_counts.values()) == [2, 3]:
        trip_rank = max(rank for rank, count in rank_counts.items() if count == 3)
        pair_rank = max(rank for rank, count in rank_counts.items() if count == 2)
        return FiveCardValue(HandCategory.FULL_HOUSE, (trip_rank, pair_rank), normalized)

    if is_flush:
        return FiveCardValue(HandCategory.FLUSH, sorted_ranks_desc, normalized)

    if straight_high is not None:
        return FiveCardValue(HandCategory.STRAIGHT, (straight_high,), normalized)

    if 3 in rank_counts.values():
        trip_rank = max(rank for rank, count in rank_counts.items() if count == 3)
        kickers = tuple(sorted((rank for rank, count in rank_counts.items() if count == 1), reverse=True))
        return FiveCardValue(HandCategory.THREE_OF_A_KIND, (trip_rank, *kickers), normalized)

    pair_ranks = sorted((rank for rank, count in rank_counts.items() if count == 2), reverse=True)
    if len(pair_ranks) == 2:
        kicker = max(rank for rank, count in rank_counts.items() if count == 1)
        return FiveCardValue(HandCategory.TWO_PAIR, (pair_ranks[0], pair_ranks[1], kicker), normalized)

    if len(pair_ranks) == 1:
        pair_rank = pair_ranks[0]
        kickers = tuple(sorted((rank for rank, count in rank_counts.items() if count == 1), reverse=True))
        return FiveCardValue(HandCategory.ONE_PAIR, (pair_rank, *kickers), normalized)

    return FiveCardValue(HandCategory.HIGH_CARD, sorted_ranks_desc, normalized)


def compare_row_values(left: ComparableRowValue, right: ComparableRowValue) -> int:
    """Compare two already-evaluated rows."""

    return _compare_keys((int(left.category), left.tiebreak), (int(right.category), right.tiebreak))


def compare_same_size_rows(left_cards: Iterable[Card], right_cards: Iterable[Card], row: RowName) -> int:
    """Compare two rows of the same size and row type."""

    if row is RowName.TOP:
        left_value = evaluate_top_row(left_cards)
        right_value = evaluate_top_row(right_cards)
    else:
        left_value = evaluate_five_card_row(left_cards)
        right_value = evaluate_five_card_row(right_cards)
    return compare_row_values(left_value, right_value)


def compare_cross_rows_for_foul(left_cards: Iterable[Card], right_cards: Iterable[Card]) -> int:
    """Compare rows of different sizes for foul detection.

    Returns ``1`` if the left row is stronger, ``-1`` if the right row is
    stronger, and ``0`` for an exact tie.
    """

    left_tuple = tuple(left_cards)
    right_tuple = tuple(right_cards)
    left_value = evaluate_top_row(left_tuple) if len(left_tuple) == 3 else evaluate_five_card_row(left_tuple)
    right_value = evaluate_top_row(right_tuple) if len(right_tuple) == 3 else evaluate_five_card_row(right_tuple)
    return compare_row_values(left_value, right_value)
