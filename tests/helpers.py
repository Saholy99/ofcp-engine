"""Shared helpers for OFC engine tests."""

from __future__ import annotations

from ofc.board import Board, RowName
from ofc.cards import Card, format_card, full_deck, parse_card
from ofc.config import DEFAULT_CONFIG
from ofc.deck import DeckState
from ofc.state import GameState, HandPhase, PlayerId, PlayerState


def cards(tokens: str | list[str] | tuple[str, ...]) -> tuple[Card, ...]:
    """Parse a sequence of card tokens."""

    if isinstance(tokens, str):
        token_list = tokens.split()
    else:
        token_list = list(tokens)
    return tuple(parse_card(token) for token in token_list)


def make_board(*, top: str, middle: str, bottom: str) -> Board:
    """Build a board from row token strings."""

    return Board(top=cards(top), middle=cards(middle), bottom=cards(bottom))


def stacked_deck_tokens(prefix_tokens: list[str]) -> list[str]:
    """Create a full deterministic deck with ``prefix_tokens`` on top."""

    prefix_cards = cards(prefix_tokens)
    if len(prefix_cards) != len(set(prefix_cards)):
        raise ValueError("Deck prefix must not contain duplicate cards")
    remaining = [format_card(card) for card in full_deck() if card not in set(prefix_cards)]
    return prefix_tokens + remaining


def showdown_state(
    left_board: Board,
    right_board: Board,
    *,
    button: PlayerId = PlayerId.PLAYER_0,
    left_fantasyland_active: bool = False,
    right_fantasyland_active: bool = False,
    left_concealed: bool = False,
    right_concealed: bool = False,
    continuation_hand: bool = False,
    next_hand_fantasyland: tuple[bool, bool] = (False, False),
    phase: HandPhase = HandPhase.SHOWDOWN,
) -> GameState:
    """Construct a completed showdown-ready state."""

    left_player = PlayerState(
        player_id=PlayerId.PLAYER_0,
        board=Board() if left_concealed else left_board,
        hidden_discards=(),
        current_private_draw=(),
        fantasyland_active=left_fantasyland_active,
        concealed_fantasyland_board=left_board if left_concealed else None,
        concealed_fantasyland_discard=None,
        initial_placement_done=not left_fantasyland_active,
        normal_draws_taken=4 if not left_fantasyland_active else 0,
        fantasyland_set_done=left_fantasyland_active,
    )
    right_player = PlayerState(
        player_id=PlayerId.PLAYER_1,
        board=Board() if right_concealed else right_board,
        hidden_discards=(),
        current_private_draw=(),
        fantasyland_active=right_fantasyland_active,
        concealed_fantasyland_board=right_board if right_concealed else None,
        concealed_fantasyland_discard=None,
        initial_placement_done=not right_fantasyland_active,
        normal_draws_taken=4 if not right_fantasyland_active else 0,
        fantasyland_set_done=right_fantasyland_active,
    )
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=1,
        button=button,
        acting_player=PlayerId.PLAYER_1,
        phase=phase,
        deck=DeckState(undealt_cards=()),
        players=(left_player, right_player),
        is_continuation_hand=continuation_hand,
        next_hand_fantasyland=next_hand_fantasyland,
    )


def placements(spec: list[tuple[RowName, str]]) -> tuple:
    """Build a tuple of Placement-ready row/card pairs."""

    from ofc.actions import Placement

    return tuple(Placement(row=row, card=parse_card(token)) for row, token in spec)
