"""Shared helpers for OFC engine tests."""

from __future__ import annotations

from collections.abc import Iterable

from ofc.board import Board, RowName, visible_cards
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


def remaining_deck_tokens(excluded_tokens: Iterable[str]) -> list[str]:
    """Return canonical deck tokens excluding the provided cards."""

    excluded_cards = cards(list(excluded_tokens))
    if len(excluded_cards) != len(set(excluded_cards)):
        raise ValueError("Excluded cards must be unique")
    return [format_card(card) for card in full_deck() if card not in set(excluded_cards)]


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


def scenario_payload_from_state(state: GameState) -> dict[str, object]:
    """Serialize a game state into the exact-state scenario payload shape."""

    def board_payload(board: Board) -> dict[str, list[str]]:
        return {
            "top": [format_card(card) for card in board.top],
            "middle": [format_card(card) for card in board.middle],
            "bottom": [format_card(card) for card in board.bottom],
        }

    return {
        "version": "1",
        "state": {
            "hand_number": state.hand_number,
            "button": state.button.value,
            "acting_player": state.acting_player.value,
            "phase": state.phase.value,
            "is_continuation_hand": state.is_continuation_hand,
            "next_hand_fantasyland": list(state.next_hand_fantasyland),
            "deck": {
                "undealt_cards": [format_card(card) for card in state.deck.undealt_cards],
            },
            "players": [
                {
                    "player_id": player.player_id.value,
                    "board": board_payload(player.board),
                    "hidden_discards": [format_card(card) for card in player.hidden_discards],
                    "current_private_draw": [format_card(card) for card in player.current_private_draw],
                    "fantasyland_active": player.fantasyland_active,
                    "concealed_fantasyland_board": None
                    if player.concealed_fantasyland_board is None
                    else board_payload(player.concealed_fantasyland_board),
                    "concealed_fantasyland_discard": None
                    if player.concealed_fantasyland_discard is None
                    else format_card(player.concealed_fantasyland_discard),
                    "initial_placement_done": player.initial_placement_done,
                    "normal_draws_taken": player.normal_draws_taken,
                    "fantasyland_set_done": player.fantasyland_set_done,
                }
                for player in state.players
            ],
        },
    }


def physical_cards_in_state(state: GameState) -> tuple[Card, ...]:
    """Return each physical card location once, treating mirrored FL discards as metadata."""

    collected: list[Card] = []
    for player in state.players:
        collected.extend(visible_cards(player.board))
        collected.extend(player.hidden_discards)
        collected.extend(player.current_private_draw)
        if player.concealed_fantasyland_board is not None:
            collected.extend(visible_cards(player.concealed_fantasyland_board))
    collected.extend(state.deck.undealt_cards)
    return tuple(collected)


def solver_final_draw_state(*, enters_fantasyland: bool) -> GameState:
    """Build a compact final-draw state for solver tests."""

    player_0_top = "Qh Qd" if enters_fantasyland else "Jh Jd"
    player_0 = PlayerState(
        player_id=PlayerId.PLAYER_0,
        board=Board(
            top=cards(player_0_top),
            middle=cards("Kh Kd Ks 4c 3d"),
            bottom=cards("9c Tc Jc Qc"),
        ),
        hidden_discards=cards("6h 7h 8h"),
        current_private_draw=cards("2s Kc 5h"),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=3,
        fantasyland_set_done=False,
    )
    player_1 = PlayerState(
        player_id=PlayerId.PLAYER_1,
        board=Board(
            top=cards("Ts 8d 4s"),
            middle=cards("Ah Ad 7s 6c 3h"),
            bottom=cards("2c 2d 5c 5d 9h"),
        ),
        hidden_discards=cards("6d 7d 8s 9s"),
        current_private_draw=(),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=4,
        fantasyland_set_done=False,
    )
    used_cards = (
        visible_cards(player_0.board)
        + player_0.hidden_discards
        + player_0.current_private_draw
        + visible_cards(player_1.board)
        + player_1.hidden_discards
    )
    deck = DeckState(undealt_cards=cards(remaining_deck_tokens(format_card(card) for card in used_cards)))
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=1,
        button=PlayerId.PLAYER_1,
        acting_player=PlayerId.PLAYER_0,
        phase=HandPhase.DRAW,
        deck=deck,
        players=(player_0, player_1),
        is_continuation_hand=False,
        next_hand_fantasyland=(False, False),
    )


def solver_late_street_exact_search_state() -> GameState:
    """Build a compact late-street state with a small exact-search tree."""

    player_0 = PlayerState(
        player_id=PlayerId.PLAYER_0,
        board=Board(
            top=cards("Qc 4s 3s"),
            middle=cards("7c Ad Td 5s"),
            bottom=cards("9c As 4h 6d"),
        ),
        hidden_discards=cards("9s Ah Ks"),
        current_private_draw=(),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=3,
        fantasyland_set_done=False,
    )
    player_1 = PlayerState(
        player_id=PlayerId.PLAYER_1,
        board=Board(
            top=cards("4d Qd 3h"),
            middle=cards("Tc Ac Jh Th 5c"),
            bottom=cards("2s"),
        ),
        hidden_discards=cards("6s 7h"),
        current_private_draw=cards("Qs 2d 5h"),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=2,
        fantasyland_set_done=False,
    )
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=1,
        button=PlayerId.PLAYER_1,
        acting_player=PlayerId.PLAYER_1,
        phase=HandPhase.DRAW,
        deck=DeckState(
            undealt_cards=cards("3c 8h 8s 5d 7d 3d Kh Ts Js 6c 6h 8d 9h 2h 8c 7s Kd Jc 2c Jd Qh Kc 9d 4c")
        ),
        players=(player_0, player_1),
        is_continuation_hand=False,
        next_hand_fantasyland=(False, False),
    )


def solver_unsupported_top_pair_state() -> GameState:
    """Build an early draw where making QQ on top is structurally fragile."""

    return _solver_survivability_draw_state(
        board=Board(
            top=cards("Qh"),
            middle=cards("7c 4d"),
            bottom=cards("8c 5d"),
        ),
        draw="Qd 2c 3d",
    )


def solver_unsupported_top_trips_state() -> GameState:
    """Build an early draw where making top trips is unsupported below."""

    return _solver_survivability_draw_state(
        board=Board(
            top=cards("Qh Qd"),
            middle=cards("7c 4d"),
            bottom=cards("8c 5d 6h"),
        ),
        hidden_discards="9s",
        draw="Qs 2c 3d",
    )


def solver_middle_over_bottom_pressure_state() -> GameState:
    """Build an early draw where improving middle creates pressure on bottom."""

    return _solver_survivability_draw_state(
        board=Board(
            top=cards("4h"),
            middle=cards("Ah Ad 7c"),
            bottom=cards("9c 5d 6h"),
        ),
        hidden_discards="8s",
        draw="As 2c 3d",
    )


def _solver_survivability_draw_state(*, board: Board, draw: str, hidden_discards: str = "") -> GameState:
    acting_player = PlayerState(
        player_id=PlayerId.PLAYER_0,
        board=board,
        hidden_discards=cards(hidden_discards),
        current_private_draw=cards(draw),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=len(cards(hidden_discards)),
        fantasyland_set_done=False,
    )
    opponent = PlayerState(
        player_id=PlayerId.PLAYER_1,
        board=Board(
            top=cards("9h"),
            middle=cards("Tc 3h"),
            bottom=cards("Jc 4s"),
        ),
        hidden_discards=(),
        current_private_draw=(),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=1,
        fantasyland_set_done=False,
    )
    used_cards = (
        visible_cards(acting_player.board)
        + acting_player.hidden_discards
        + acting_player.current_private_draw
        + visible_cards(opponent.board)
    )
    deck = DeckState(undealt_cards=cards(remaining_deck_tokens(format_card(card) for card in used_cards)))
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=1,
        button=PlayerId.PLAYER_1,
        acting_player=PlayerId.PLAYER_0,
        phase=HandPhase.DRAW,
        deck=deck,
        players=(acting_player, opponent),
        is_continuation_hand=False,
        next_hand_fantasyland=(False, False),
    )
