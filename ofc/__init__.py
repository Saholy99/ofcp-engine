"""Compatibility wrapper for importing the src-layout package without installation."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ofc"
__path__ = [str(_SRC_PACKAGE)]

from ofc.actions import PlaceDrawAction, PlaceInitialFiveAction, Placement, SetFantasylandHandAction
from ofc.board import Board, RowName
from ofc.cards import Card, Rank, Suit, format_card, full_deck, parse_card
from ofc.deck import DeckState, draw_n, make_deck
from ofc.engine import apply, new_hand, new_match, showdown
from ofc.state import GameState, HandPhase, PlayerId, PlayerState

__all__ = [
    "Board",
    "Card",
    "DeckState",
    "GameState",
    "HandPhase",
    "PlaceDrawAction",
    "PlaceInitialFiveAction",
    "Placement",
    "PlayerId",
    "PlayerState",
    "Rank",
    "RowName",
    "SetFantasylandHandAction",
    "Suit",
    "apply",
    "draw_n",
    "format_card",
    "full_deck",
    "make_deck",
    "new_hand",
    "new_match",
    "parse_card",
    "showdown",
]
