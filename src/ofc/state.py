"""Explicit game-state models for a single OFC hand."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ofc.board import Board, visible_cards
from ofc.cards import Card
from ofc.config import DEFAULT_CONFIG, VariantConfig
from ofc.deck import DeckState


class PlayerId(str, Enum):
    """Heads-up player identifiers."""

    PLAYER_0 = "player_0"
    PLAYER_1 = "player_1"


PLAYER_ORDER = (PlayerId.PLAYER_0, PlayerId.PLAYER_1)


class HandPhase(str, Enum):
    """Top-level phase of the current hand."""

    INITIAL_DEAL = "initial_deal"
    DRAW = "draw"
    FANTASYLAND_SET = "fantasyland_set"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class PlayerState:
    """Per-player state, including hidden and public information."""

    player_id: PlayerId
    board: Board = Board()
    hidden_discards: tuple[Card, ...] = ()
    current_private_draw: tuple[Card, ...] = ()
    fantasyland_active: bool = False
    fantasyland_pending: bool = False
    concealed_fantasyland_board: Board | None = None
    concealed_fantasyland_discard: Card | None = None
    initial_placement_done: bool = False
    normal_draws_taken: int = 0
    fantasyland_set_done: bool = False


@dataclass(frozen=True)
class GameState:
    """Complete deterministic hand state."""

    config: VariantConfig
    hand_number: int
    button: PlayerId
    acting_player: PlayerId
    phase: HandPhase
    deck: DeckState
    players: tuple[PlayerState, PlayerState]
    is_continuation_hand: bool = False
    next_hand_fantasyland: tuple[bool, bool] = (False, False)


def player_index(player_id: PlayerId) -> int:
    """Return the stable tuple index for a player."""

    return PLAYER_ORDER.index(player_id)


def other_player(player_id: PlayerId) -> PlayerId:
    """Return the opponent."""

    return PlayerId.PLAYER_1 if player_id == PlayerId.PLAYER_0 else PlayerId.PLAYER_0


def get_player(state: GameState, player_id: PlayerId) -> PlayerState:
    """Return the requested player's state."""

    return state.players[player_index(player_id)]


def replace_player(state: GameState, player_state: PlayerState) -> GameState:
    """Return a new game state with one player's state replaced."""

    players = list(state.players)
    players[player_index(player_state.player_id)] = player_state
    return replace(state, players=tuple(players))


def effective_board(player_state: PlayerState, *, reveal_concealed: bool = False) -> Board:
    """Return the board that should be used for evaluation or public display."""

    if reveal_concealed and player_state.concealed_fantasyland_board is not None:
        return player_state.concealed_fantasyland_board
    return player_state.board


def all_known_cards(state: GameState, *, reveal_concealed: bool = False) -> tuple[Card, ...]:
    """Return all cards currently accounted for in state."""

    cards: list[Card] = []
    for player in state.players:
        cards.extend(visible_cards(player.board))
        cards.extend(player.hidden_discards)
        cards.extend(player.current_private_draw)
        if reveal_concealed and player.concealed_fantasyland_board is not None:
            cards.extend(visible_cards(player.concealed_fantasyland_board))
        if player.concealed_fantasyland_discard is not None:
            cards.append(player.concealed_fantasyland_discard)
    cards.extend(state.deck.undealt_cards)
    return tuple(cards)


def make_player_state(player_id: PlayerId, *, fantasyland_active: bool = False) -> PlayerState:
    """Construct a default player state."""

    return PlayerState(player_id=player_id, fantasyland_active=fantasyland_active)


def make_empty_state(
    deck: DeckState,
    *,
    button: PlayerId,
    acting_player: PlayerId,
    fantasyland_flags: tuple[bool, bool] = (False, False),
    continuation_hand: bool = False,
    hand_number: int = 1,
    config: VariantConfig = DEFAULT_CONFIG,
) -> GameState:
    """Construct a new empty hand state before the first cards are dealt."""

    players = tuple(
        make_player_state(player_id, fantasyland_active=fantasyland_flags[index])
        for index, player_id in enumerate(PLAYER_ORDER)
    )
    return GameState(
        config=config,
        hand_number=hand_number,
        button=button,
        acting_player=acting_player,
        phase=HandPhase.INITIAL_DEAL,
        deck=deck,
        players=players,
        is_continuation_hand=continuation_hand,
    )
