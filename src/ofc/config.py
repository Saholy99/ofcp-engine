"""Variant configuration and constants for heads-up Pineapple OFC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PLAYER_COUNT = 2
TOP_ROW_CAPACITY = 3
MIDDLE_ROW_CAPACITY = 5
BOTTOM_ROW_CAPACITY = 5
INITIAL_DEAL_COUNT = 5
NORMAL_DRAW_COUNT = 3
NORMAL_DRAW_PLACEMENTS = 2
NORMAL_DRAW_DISCARDS = 1
NORMAL_DRAW_TURNS_PER_PLAYER = 4
FANTASYLAND_DEAL_COUNT = 14
FANTASYLAND_PLACEMENTS = 13
FANTASYLAND_DISCARDS = 1


@dataclass(frozen=True)
class VariantConfig:
    """Immutable rule configuration for the supported OFC variant."""

    player_count: int = PLAYER_COUNT
    top_row_capacity: int = TOP_ROW_CAPACITY
    middle_row_capacity: int = MIDDLE_ROW_CAPACITY
    bottom_row_capacity: int = BOTTOM_ROW_CAPACITY
    initial_deal_count: int = INITIAL_DEAL_COUNT
    normal_draw_count: int = NORMAL_DRAW_COUNT
    normal_draw_placements: int = NORMAL_DRAW_PLACEMENTS
    normal_draw_discards: int = NORMAL_DRAW_DISCARDS
    normal_draw_turns_per_player: int = NORMAL_DRAW_TURNS_PER_PLAYER
    fantasyland_deal_count: int = FANTASYLAND_DEAL_COUNT
    fantasyland_placements: int = FANTASYLAND_PLACEMENTS
    fantasyland_discards: int = FANTASYLAND_DISCARDS
    bottom_royalties: Mapping[str, int] | None = None
    middle_royalties: Mapping[str, int] | None = None
    top_pair_royalties: Mapping[int, int] | None = None
    top_trips_royalties: Mapping[int, int] | None = None


DEFAULT_BOTTOM_ROYALTIES = {
    "straight": 2,
    "flush": 4,
    "full_house": 6,
    "four_of_a_kind": 10,
    "straight_flush": 15,
    "royal_flush": 25,
}

DEFAULT_MIDDLE_ROYALTIES = {
    "three_of_a_kind": 2,
    "straight": 4,
    "flush": 8,
    "full_house": 12,
    "four_of_a_kind": 20,
    "straight_flush": 30,
    "royal_flush": 50,
}

DEFAULT_TOP_PAIR_ROYALTIES = {
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 9,
}

DEFAULT_TOP_TRIPS_ROYALTIES = {
    2: 10,
    3: 11,
    4: 12,
    5: 13,
    6: 14,
    7: 15,
    8: 16,
    9: 17,
    10: 18,
    11: 19,
    12: 20,
    13: 21,
    14: 22,
}

DEFAULT_CONFIG = VariantConfig(
    bottom_royalties=DEFAULT_BOTTOM_ROYALTIES,
    middle_royalties=DEFAULT_MIDDLE_ROYALTIES,
    top_pair_royalties=DEFAULT_TOP_PAIR_ROYALTIES,
    top_trips_royalties=DEFAULT_TOP_TRIPS_ROYALTIES,
)
