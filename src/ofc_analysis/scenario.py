"""Load and validate persisted exact-state scenarios for analysis tools.

This module defines the v1 JSON scenario format used by the analysis harness.
Scenarios load into real engine objects while preserving the current engine's
rule semantics and using ``DEFAULT_CONFIG`` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from collections import Counter

from ofc.board import Board, board_full, visible_cards
from ofc.cards import Card, full_deck, parse_card
from ofc.config import DEFAULT_CONFIG
from ofc.deck import DeckState
from ofc.state import GameState, HandPhase, PLAYER_ORDER, PlayerId, PlayerState, get_player, other_player


_ALLOWED_TOP_LEVEL_KEYS = frozenset({"version", "state"})
_ALLOWED_STATE_KEYS = frozenset(
    {
        "hand_number",
        "button",
        "acting_player",
        "phase",
        "is_continuation_hand",
        "next_hand_fantasyland",
        "deck",
        "players",
    }
)
_ALLOWED_DECK_KEYS = frozenset({"undealt_cards"})
_ALLOWED_PLAYER_KEYS = frozenset(
    {
        "player_id",
        "board",
        "hidden_discards",
        "current_private_draw",
        "fantasyland_active",
        "concealed_fantasyland_board",
        "concealed_fantasyland_discard",
        "initial_placement_done",
        "normal_draws_taken",
        "fantasyland_set_done",
    }
)
_ALLOWED_BOARD_KEYS = frozenset({"top", "middle", "bottom"})
_EXPECTED_DRAW_SIZE_BY_PHASE = {
    HandPhase.INITIAL_DEAL: DEFAULT_CONFIG.initial_deal_count,
    HandPhase.DRAW: DEFAULT_CONFIG.normal_draw_count,
    HandPhase.FANTASYLAND_SET: DEFAULT_CONFIG.fantasyland_deal_count,
}


@dataclass(frozen=True)
class ExactStateScenario:
    """Persisted exact engine state loaded from a v1 scenario JSON file."""

    version: str
    state: GameState
    source_path: Path | None = None


def load_scenario(path: str | Path) -> ExactStateScenario:
    """Load an exact-state scenario from disk."""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return load_scenario_data(payload, source_path=resolved_path)


def load_scenario_data(payload: Mapping[str, Any], *, source_path: Path | None = None) -> ExactStateScenario:
    """Load an exact-state scenario from an in-memory JSON-like payload."""

    top_level = _require_mapping(payload, "scenario")
    _require_exact_keys(top_level, _ALLOWED_TOP_LEVEL_KEYS, "scenario")
    version = top_level.get("version")
    if version != "1":
        raise ValueError(f"Unsupported scenario version: {version!r}")
    state = _parse_state(_require_mapping(top_level["state"], "scenario.state"))
    _validate_state(state)
    return ExactStateScenario(version="1", state=state, source_path=source_path)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_keys(mapping: Mapping[str, Any], allowed_keys: frozenset[str], path: str) -> None:
    missing = sorted(allowed_keys - set(mapping))
    unexpected = sorted(set(mapping) - allowed_keys)
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{path} has unexpected keys: {', '.join(unexpected)}")


def _parse_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _parse_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _parse_player_id(value: Any, path: str) -> PlayerId:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a player id string")
    try:
        return PlayerId(value)
    except ValueError as exc:
        raise ValueError(f"{path} must be one of: player_0, player_1") from exc


def _parse_phase(value: Any, path: str) -> HandPhase:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a hand phase string")
    try:
        return HandPhase(value)
    except ValueError as exc:
        allowed = ", ".join(phase.value for phase in HandPhase)
        raise ValueError(f"{path} must be one of: {allowed}") from exc


def _parse_card_list(value: Any, path: str) -> tuple[Card, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of card tokens")
    cards: list[Card] = []
    for index, token in enumerate(value):
        if not isinstance(token, str):
            raise ValueError(f"{path}[{index}] must be a card token string")
        cards.append(parse_card(token))
    return tuple(cards)


def _parse_board(value: Any, path: str) -> Board:
    board_mapping = _require_mapping(value, path)
    _require_exact_keys(board_mapping, _ALLOWED_BOARD_KEYS, path)
    board = Board(
        top=_parse_card_list(board_mapping["top"], f"{path}.top"),
        middle=_parse_card_list(board_mapping["middle"], f"{path}.middle"),
        bottom=_parse_card_list(board_mapping["bottom"], f"{path}.bottom"),
    )
    if len(board.top) > DEFAULT_CONFIG.top_row_capacity:
        raise ValueError(f"{path}.top exceeds row capacity")
    if len(board.middle) > DEFAULT_CONFIG.middle_row_capacity:
        raise ValueError(f"{path}.middle exceeds row capacity")
    if len(board.bottom) > DEFAULT_CONFIG.bottom_row_capacity:
        raise ValueError(f"{path}.bottom exceeds row capacity")
    return board


def _parse_optional_board(value: Any, path: str) -> Board | None:
    if value is None:
        return None
    return _parse_board(value, path)


def _parse_optional_card(value: Any, path: str) -> Card | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a card token string or null")
    return parse_card(value)


def _parse_bool_pair(value: Any, path: str) -> tuple[bool, bool]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be a two-item boolean list")
    return (_parse_bool(value[0], f"{path}[0]"), _parse_bool(value[1], f"{path}[1]"))


def _parse_player_state(value: Mapping[str, Any], path: str) -> PlayerState:
    _require_exact_keys(value, _ALLOWED_PLAYER_KEYS, path)
    return PlayerState(
        player_id=_parse_player_id(value["player_id"], f"{path}.player_id"),
        board=_parse_board(value["board"], f"{path}.board"),
        hidden_discards=_parse_card_list(value["hidden_discards"], f"{path}.hidden_discards"),
        current_private_draw=_parse_card_list(value["current_private_draw"], f"{path}.current_private_draw"),
        fantasyland_active=_parse_bool(value["fantasyland_active"], f"{path}.fantasyland_active"),
        fantasyland_pending=False,
        concealed_fantasyland_board=_parse_optional_board(
            value["concealed_fantasyland_board"], f"{path}.concealed_fantasyland_board"
        ),
        concealed_fantasyland_discard=_parse_optional_card(
            value["concealed_fantasyland_discard"], f"{path}.concealed_fantasyland_discard"
        ),
        initial_placement_done=_parse_bool(value["initial_placement_done"], f"{path}.initial_placement_done"),
        normal_draws_taken=_parse_non_negative_int(value["normal_draws_taken"], f"{path}.normal_draws_taken"),
        fantasyland_set_done=_parse_bool(value["fantasyland_set_done"], f"{path}.fantasyland_set_done"),
    )


def _parse_state(value: Mapping[str, Any]) -> GameState:
    _require_exact_keys(value, _ALLOWED_STATE_KEYS, "scenario.state")
    deck_mapping = _require_mapping(value["deck"], "scenario.state.deck")
    _require_exact_keys(deck_mapping, _ALLOWED_DECK_KEYS, "scenario.state.deck")

    players_value = value["players"]
    if not isinstance(players_value, list) or len(players_value) != 2:
        raise ValueError("scenario.state.players must contain exactly two players")

    parsed_players = [
        _parse_player_state(_require_mapping(player_value, f"scenario.state.players[{index}]"), f"scenario.state.players[{index}]")
        for index, player_value in enumerate(players_value)
    ]

    player_ids = {player.player_id for player in parsed_players}
    if player_ids != set(PLAYER_ORDER):
        raise ValueError("scenario.state.players must contain exactly player_0 and player_1")

    players_by_id = {player.player_id: player for player in parsed_players}
    ordered_players = tuple(players_by_id[player_id] for player_id in PLAYER_ORDER)

    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=_parse_non_negative_int(value["hand_number"], "scenario.state.hand_number"),
        button=_parse_player_id(value["button"], "scenario.state.button"),
        acting_player=_parse_player_id(value["acting_player"], "scenario.state.acting_player"),
        phase=_parse_phase(value["phase"], "scenario.state.phase"),
        deck=DeckState(undealt_cards=_parse_card_list(deck_mapping["undealt_cards"], "scenario.state.deck.undealt_cards")),
        players=ordered_players,
        is_continuation_hand=_parse_bool(value["is_continuation_hand"], "scenario.state.is_continuation_hand"),
        next_hand_fantasyland=_parse_bool_pair(value["next_hand_fantasyland"], "scenario.state.next_hand_fantasyland"),
    )


def _validate_player_state(player: PlayerState) -> None:
    if player.normal_draws_taken > DEFAULT_CONFIG.normal_draw_turns_per_player:
        raise ValueError(f"{player.player_id.value}.normal_draws_taken exceeds normal draw count")
    if (player.concealed_fantasyland_board is None) != (player.concealed_fantasyland_discard is None):
        raise ValueError(f"{player.player_id.value} must provide both concealed Fantasyland board and discard together")
    if player.concealed_fantasyland_board is not None:
        if not player.fantasyland_active:
            raise ValueError(f"{player.player_id.value} cannot have concealed Fantasyland data unless fantasyland_active is true")
        if visible_cards(player.board):
            raise ValueError(f"{player.player_id.value} cannot expose visible board cards while a concealed Fantasyland board exists")
        if not board_full(player.concealed_fantasyland_board):
            raise ValueError(f"{player.player_id.value}.concealed_fantasyland_board must be a full 13-card board")
        if player.concealed_fantasyland_discard not in player.hidden_discards:
            raise ValueError(
                f"{player.player_id.value}.concealed_fantasyland_discard must also appear in hidden_discards"
            )
    if player.fantasyland_set_done and player.concealed_fantasyland_board is None:
        raise ValueError(f"{player.player_id.value}.fantasyland_set_done requires concealed Fantasyland data")


def _validate_actionable_phase_draws(state: GameState) -> None:
    if state.phase in _EXPECTED_DRAW_SIZE_BY_PHASE:
        acting_player = get_player(state, state.acting_player)
        opponent = get_player(state, other_player(state.acting_player))
        expected_draw_size = _EXPECTED_DRAW_SIZE_BY_PHASE[state.phase]
        if len(acting_player.current_private_draw) != expected_draw_size:
            raise ValueError(
                f"acting player current_private_draw must contain exactly {expected_draw_size} cards during {state.phase.value}"
            )
        if opponent.current_private_draw:
            raise ValueError("only the acting player may have a current_private_draw during an action phase")
        if state.phase == HandPhase.INITIAL_DEAL and acting_player.fantasyland_active:
            raise ValueError("a Fantasyland player cannot act during INITIAL_DEAL")
        if state.phase == HandPhase.DRAW:
            if acting_player.fantasyland_active:
                raise ValueError("a Fantasyland player cannot act during DRAW")
            if not acting_player.initial_placement_done:
                raise ValueError("DRAW phase requires the acting player's initial placement to be complete")
        if state.phase == HandPhase.FANTASYLAND_SET:
            if not acting_player.fantasyland_active:
                raise ValueError("FANTASYLAND_SET requires the acting player to be in Fantasyland")
            if acting_player.fantasyland_set_done:
                raise ValueError("FANTASYLAND_SET requires fantasyland_set_done to be false for the acting player")
    elif any(player.current_private_draw for player in state.players):
        raise ValueError(f"{state.phase.value} scenarios may not contain current_private_draw cards")


def _validate_cards(state: GameState) -> None:
    physical_cards = []
    duplicate_check_cards = []
    for player in state.players:
        board_cards = list(visible_cards(player.board))
        hidden_discards = list(player.hidden_discards)
        draw_cards = list(player.current_private_draw)

        physical_cards.extend(board_cards)
        physical_cards.extend(hidden_discards)
        physical_cards.extend(draw_cards)
        duplicate_check_cards.extend(board_cards)
        duplicate_check_cards.extend(hidden_discards)
        duplicate_check_cards.extend(draw_cards)
        if player.concealed_fantasyland_board is not None:
            concealed_board_cards = list(visible_cards(player.concealed_fantasyland_board))
            physical_cards.extend(concealed_board_cards)
            duplicate_check_cards.extend(concealed_board_cards)
    physical_cards.extend(state.deck.undealt_cards)
    duplicate_check_cards.extend(state.deck.undealt_cards)

    duplicate_counter = Counter(duplicate_check_cards)
    duplicate_cards = [
        item
        for item, count in duplicate_counter.items()
        if count > 1 and not isinstance(item, tuple)
    ]
    if duplicate_cards:
        raise ValueError("scenario contains duplicate cards across state locations")

    if len(physical_cards) != 52 or set(physical_cards) != set(full_deck()):
        raise ValueError("scenario must account for all 52 cards exactly once")


def _validate_state(state: GameState) -> None:
    if state.config != DEFAULT_CONFIG:
        raise ValueError("scenario v1 must use DEFAULT_CONFIG only")
    for player in state.players:
        _validate_player_state(player)
    _validate_actionable_phase_draws(state)
    _validate_cards(state)


__all__ = ["ExactStateScenario", "load_scenario", "load_scenario_data"]
