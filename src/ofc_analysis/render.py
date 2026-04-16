"""Deterministic renderers for exact states, observations, and legal actions."""

from __future__ import annotations

import json
from typing import Any, Sequence

from ofc.board import Board
from ofc.cards import Card, format_card
from ofc.state import GameState
from ofc_analysis.action_codec import EncodedAction
from ofc_analysis.models import RenderedOutput
from ofc_analysis.observation import PlayerObservation


def render_state(state: GameState, *, as_json: bool = False) -> RenderedOutput:
    """Render an exact engine state for CLI or tests."""

    payload = _state_payload(state)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_state_text(payload), payload=payload)


def render_observation(observation: PlayerObservation, *, as_json: bool = False) -> RenderedOutput:
    """Render an observer-facing state projection for CLI or tests."""

    payload = _observation_payload(observation)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_observation_text(payload), payload=payload)


def render_actions(actions: Sequence[EncodedAction], *, as_json: bool = False) -> RenderedOutput:
    """Render a deterministic list of encoded root actions."""

    payload = {
        "action_count": len(actions),
        "actions": [action.as_dict() for action in actions],
    }
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_actions_text(payload), payload=payload)


def _cards_payload(cards: tuple[Card, ...]) -> list[str]:
    return [format_card(card) for card in cards]


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": _cards_payload(board.top),
        "middle": _cards_payload(board.middle),
        "bottom": _cards_payload(board.bottom),
    }


def _state_payload(state: GameState) -> dict[str, Any]:
    return {
        "hand_number": state.hand_number,
        "button": state.button.value,
        "acting_player": state.acting_player.value,
        "phase": state.phase.value,
        "is_continuation_hand": state.is_continuation_hand,
        "next_hand_fantasyland": list(state.next_hand_fantasyland),
        "deck": {
            "cards_remaining": state.deck.cards_remaining,
            "undealt_cards": _cards_payload(state.deck.undealt_cards),
        },
        "players": [
            {
                "player_id": player.player_id.value,
                "board": _board_payload(player.board),
                "hidden_discards": _cards_payload(player.hidden_discards),
                "hidden_discard_count": len(player.hidden_discards),
                "current_private_draw": _cards_payload(player.current_private_draw),
                "fantasyland_active": player.fantasyland_active,
                "concealed_fantasyland_board": None
                if player.concealed_fantasyland_board is None
                else _board_payload(player.concealed_fantasyland_board),
                "concealed_fantasyland_discard": None
                if player.concealed_fantasyland_discard is None
                else format_card(player.concealed_fantasyland_discard),
                "initial_placement_done": player.initial_placement_done,
                "normal_draws_taken": player.normal_draws_taken,
                "fantasyland_set_done": player.fantasyland_set_done,
            }
            for player in state.players
        ],
    }


def _observation_payload(observation: PlayerObservation) -> dict[str, Any]:
    return {
        "observer": observation.observer.value,
        "acting_player": observation.acting_player.value,
        "phase": observation.phase.value,
        "hand_number": observation.hand_number,
        "button": observation.button.value,
        "is_continuation_hand": observation.is_continuation_hand,
        "next_hand_fantasyland": list(observation.next_hand_fantasyland),
        "public_boards": [
            {
                "player_id": player_id.value,
                "board": _board_payload(board),
            }
            for player_id, board in zip(observation.public_player_ids, observation.public_boards, strict=True)
        ],
        "own_private_draw": _cards_payload(observation.own_private_draw),
        "own_hidden_discards": _cards_payload(observation.own_hidden_discards),
        "own_fantasyland_active": observation.own_fantasyland_active,
        "own_concealed_fantasyland_board": None
        if observation.own_concealed_fantasyland_board is None
        else _board_payload(observation.own_concealed_fantasyland_board),
        "own_concealed_fantasyland_discard": None
        if observation.own_concealed_fantasyland_discard is None
        else format_card(observation.own_concealed_fantasyland_discard),
        "opponent_fantasyland_active": observation.opponent_fantasyland_active,
        "opponent_hidden_discard_count": observation.opponent_hidden_discard_count,
        "unseen_card_count": observation.unseen_card_count,
    }


def _state_text(payload: dict[str, Any]) -> str:
    lines = [
        "Exact State",
        f"hand_number: {payload['hand_number']}",
        f"button: {payload['button']}",
        f"acting_player: {payload['acting_player']}",
        f"phase: {payload['phase']}",
        f"is_continuation_hand: {json.dumps(payload['is_continuation_hand'])}",
        f"next_hand_fantasyland: {json.dumps(payload['next_hand_fantasyland'])}",
        f"deck.cards_remaining: {payload['deck']['cards_remaining']}",
        f"deck.undealt_cards: {json.dumps(payload['deck']['undealt_cards'])}",
    ]
    for player in payload["players"]:
        prefix = f"players.{player['player_id']}"
        lines.extend(
            [
                f"{prefix}.board.top: {json.dumps(player['board']['top'])}",
                f"{prefix}.board.middle: {json.dumps(player['board']['middle'])}",
                f"{prefix}.board.bottom: {json.dumps(player['board']['bottom'])}",
                f"{prefix}.hidden_discards: {json.dumps(player['hidden_discards'])}",
                f"{prefix}.current_private_draw: {json.dumps(player['current_private_draw'])}",
                f"{prefix}.fantasyland_active: {json.dumps(player['fantasyland_active'])}",
                f"{prefix}.concealed_fantasyland_board: {json.dumps(player['concealed_fantasyland_board'])}",
                f"{prefix}.concealed_fantasyland_discard: {json.dumps(player['concealed_fantasyland_discard'])}",
                f"{prefix}.initial_placement_done: {json.dumps(player['initial_placement_done'])}",
                f"{prefix}.normal_draws_taken: {player['normal_draws_taken']}",
                f"{prefix}.fantasyland_set_done: {json.dumps(player['fantasyland_set_done'])}",
            ]
        )
    return "\n".join(lines)


def _observation_text(payload: dict[str, Any]) -> str:
    lines = [
        "Player Observation",
        f"observer: {payload['observer']}",
        f"acting_player: {payload['acting_player']}",
        f"phase: {payload['phase']}",
        f"hand_number: {payload['hand_number']}",
        f"button: {payload['button']}",
        f"is_continuation_hand: {json.dumps(payload['is_continuation_hand'])}",
        f"next_hand_fantasyland: {json.dumps(payload['next_hand_fantasyland'])}",
    ]
    for entry in payload["public_boards"]:
        prefix = f"public_boards.{entry['player_id']}"
        lines.extend(
            [
                f"{prefix}.top: {json.dumps(entry['board']['top'])}",
                f"{prefix}.middle: {json.dumps(entry['board']['middle'])}",
                f"{prefix}.bottom: {json.dumps(entry['board']['bottom'])}",
            ]
        )
    lines.extend(
        [
            f"own_private_draw: {json.dumps(payload['own_private_draw'])}",
            f"own_hidden_discards: {json.dumps(payload['own_hidden_discards'])}",
            f"own_fantasyland_active: {json.dumps(payload['own_fantasyland_active'])}",
            f"own_concealed_fantasyland_board: {json.dumps(payload['own_concealed_fantasyland_board'])}",
            f"own_concealed_fantasyland_discard: {json.dumps(payload['own_concealed_fantasyland_discard'])}",
            f"opponent_fantasyland_active: {json.dumps(payload['opponent_fantasyland_active'])}",
            f"opponent_hidden_discard_count: {payload['opponent_hidden_discard_count']}",
            f"unseen_card_count: {payload['unseen_card_count']}",
        ]
    )
    return "\n".join(lines)


def _actions_text(payload: dict[str, Any]) -> str:
    lines = [
        "Legal Actions",
        f"action_count: {payload['action_count']}",
    ]
    for action in payload["actions"]:
        placements = ", ".join(
            f"{placement['row']}:{placement['card']}" for placement in action["payload"]["placements"]
        )
        suffix = f" discard={action['payload']['discard']}" if "discard" in action["payload"] else ""
        lines.append(
            f"[{action['action_index']}] {action['action_type']} {action['payload']['player_id']} "
            f"placements=[{placements}]{suffix}"
        )
    return "\n".join(lines)


__all__ = ["render_actions", "render_observation", "render_state"]
