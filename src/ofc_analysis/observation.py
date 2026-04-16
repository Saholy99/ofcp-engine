"""Project exact engine states into observer-facing imperfect-information views."""

from __future__ import annotations

from dataclasses import dataclass

from ofc.board import Board, visible_cards
from ofc.cards import Card, full_deck
from ofc.state import GameState, HandPhase, PLAYER_ORDER, PlayerId, get_player, other_player


@dataclass(frozen=True)
class PlayerObservation:
    """Observer-facing projection of an exact engine state."""

    observer: PlayerId
    acting_player: PlayerId
    phase: HandPhase
    hand_number: int
    button: PlayerId
    is_continuation_hand: bool
    next_hand_fantasyland: tuple[bool, bool]
    public_player_ids: tuple[PlayerId, PlayerId]
    public_boards: tuple[Board, Board]
    own_private_draw: tuple[Card, ...]
    own_hidden_discards: tuple[Card, ...]
    own_fantasyland_active: bool
    own_concealed_fantasyland_board: Board | None
    own_concealed_fantasyland_discard: Card | None
    opponent_fantasyland_active: bool
    opponent_hidden_discard_count: int
    unseen_card_count: int


def project_observation(state: GameState, observer: PlayerId) -> PlayerObservation:
    """Project an exact engine state into a player observation."""

    observer_player = get_player(state, observer)
    opponent_player = get_player(state, other_player(observer))

    public_boards = tuple(state_player.board for state_player in state.players)
    known_cards = list(visible_cards(public_boards[0]) + visible_cards(public_boards[1]))
    known_cards.extend(observer_player.current_private_draw)
    known_cards.extend(observer_player.hidden_discards)
    if observer_player.concealed_fantasyland_board is not None:
        known_cards.extend(visible_cards(observer_player.concealed_fantasyland_board))
    if observer_player.concealed_fantasyland_discard is not None:
        known_cards.append(observer_player.concealed_fantasyland_discard)

    return PlayerObservation(
        observer=observer,
        acting_player=state.acting_player,
        phase=state.phase,
        hand_number=state.hand_number,
        button=state.button,
        is_continuation_hand=state.is_continuation_hand,
        next_hand_fantasyland=state.next_hand_fantasyland,
        public_player_ids=PLAYER_ORDER,
        public_boards=public_boards,
        own_private_draw=observer_player.current_private_draw,
        own_hidden_discards=observer_player.hidden_discards,
        own_fantasyland_active=observer_player.fantasyland_active,
        own_concealed_fantasyland_board=observer_player.concealed_fantasyland_board,
        own_concealed_fantasyland_discard=observer_player.concealed_fantasyland_discard,
        opponent_fantasyland_active=opponent_player.fantasyland_active,
        opponent_hidden_discard_count=len(opponent_player.hidden_discards),
        unseen_card_count=len(full_deck()) - len(set(known_cards)),
    )


__all__ = ["PlayerObservation", "project_observation"]
