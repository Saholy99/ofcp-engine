"""Canonical deterministic encoding helpers for engine actions.

These helpers keep action rendering stable across CLI output and tests while
reusing the engine's explicit action objects directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ofc.actions import GameAction, PlaceDrawAction, PlaceInitialFiveAction, Placement, SetFantasylandHandAction
from ofc.board import RowName
from ofc.cards import format_card, parse_card
from ofc.state import PlayerId


@dataclass(frozen=True)
class EncodedAction:
    """Stable representation of an engine action for rendering and tests."""

    action_index: int
    action_type: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a plain JSON-friendly dictionary representation."""

        return {
            "action_index": self.action_index,
            "action_type": self.action_type,
            "payload": self.payload,
        }


def encode_action(action_index: int, action: GameAction) -> EncodedAction:
    """Return a stable encoded representation of an engine action."""

    action_type = _action_type(action)
    payload: dict[str, Any] = {
        "player_id": _encode_player_id(action.player_id),
        "placements": _encode_placements(action.placements),
    }
    if isinstance(action, (PlaceDrawAction, SetFantasylandHandAction)):
        payload["discard"] = format_card(action.discard)
    return EncodedAction(action_index=action_index, action_type=action_type, payload=payload)


def encode_actions(actions: Sequence[GameAction]) -> tuple[EncodedAction, ...]:
    """Encode a sequence of actions using stable positional indices."""

    return tuple(encode_action(index, action) for index, action in enumerate(actions))


def decode_action(encoded: EncodedAction | Mapping[str, Any]) -> GameAction:
    """Decode an encoded action back into an engine action object."""

    if isinstance(encoded, EncodedAction):
        action_type = encoded.action_type
        payload = encoded.payload
    else:
        action_type = encoded.get("action_type")
        payload = encoded.get("payload")
    if not isinstance(action_type, str):
        raise ValueError("encoded action must contain an action_type string")
    if not isinstance(payload, Mapping):
        raise ValueError("encoded action must contain a payload object")

    player_id = payload.get("player_id")
    placements_value = payload.get("placements")
    if not isinstance(player_id, str):
        raise ValueError("encoded action payload must contain player_id")
    try:
        player_id = PlayerId(player_id).value
    except ValueError as exc:
        raise ValueError(f"encoded action payload has unsupported player_id: {player_id!r}") from exc
    placements = _decode_placements(placements_value)

    if action_type == "place_initial_five":
        return PlaceInitialFiveAction(player_id=player_id, placements=placements)
    if action_type == "place_draw":
        discard = payload.get("discard")
        if not isinstance(discard, str):
            raise ValueError("place_draw payload must contain discard")
        return PlaceDrawAction(player_id=player_id, placements=placements, discard=parse_card(discard))
    if action_type == "set_fantasyland_hand":
        discard = payload.get("discard")
        if not isinstance(discard, str):
            raise ValueError("set_fantasyland_hand payload must contain discard")
        return SetFantasylandHandAction(player_id=player_id, placements=placements, discard=parse_card(discard))
    raise ValueError(f"unsupported action_type: {action_type!r}")


def _action_type(action: GameAction) -> str:
    if isinstance(action, PlaceInitialFiveAction):
        return "place_initial_five"
    if isinstance(action, PlaceDrawAction):
        return "place_draw"
    if isinstance(action, SetFantasylandHandAction):
        return "set_fantasyland_hand"
    raise TypeError(f"unsupported action type: {type(action)!r}")


def _encode_player_id(player_id: str | PlayerId) -> str:
    try:
        return PlayerId(player_id).value
    except ValueError as exc:
        raise ValueError(f"unsupported player_id: {player_id!r}") from exc


def _encode_placements(placements: tuple[Placement, ...]) -> list[dict[str, str]]:
    return [{"row": placement.row.value, "card": format_card(placement.card)} for placement in placements]


def _decode_placements(value: Any) -> tuple[Placement, ...]:
    if not isinstance(value, list):
        raise ValueError("encoded action placements must be a list")
    placements: list[Placement] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"placement {index} must be an object")
        row_value = item.get("row")
        card_value = item.get("card")
        if not isinstance(row_value, str) or not isinstance(card_value, str):
            raise ValueError(f"placement {index} must contain row and card strings")
        try:
            row = RowName(row_value)
        except ValueError as exc:
            raise ValueError(f"placement {index} has unsupported row: {row_value!r}") from exc
        placements.append(Placement(row=row, card=parse_card(card_value)))
    return tuple(placements)


__all__ = ["EncodedAction", "decode_action", "encode_action", "encode_actions"]
