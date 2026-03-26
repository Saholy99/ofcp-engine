"""Thin public engine orchestration helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ofc.actions import GameAction
from ofc.config import DEFAULT_CONFIG, VariantConfig
from ofc.deck import DeckState, make_deck
from ofc.fantasyland import resolve_next_hand_fantasyland_flags
from ofc.scoring import TerminalResult, score_terminal
from ofc.state import HandPhase, PlayerId, effective_board
from ofc.transitions import advance_after_showdown, apply_action, prepare_hand_state


def new_hand(
    *,
    button: PlayerId = PlayerId.PLAYER_0,
    fantasyland_flags: tuple[bool, bool] = (False, False),
    seed: int | str | None = None,
    preset_order: Iterable[str] | None = None,
    continuation_hand: bool = False,
    hand_number: int = 1,
    config: VariantConfig = DEFAULT_CONFIG,
):
    """Create a new hand with a fresh deterministic deck."""

    deck = make_deck(seed=seed, preset_order=preset_order)
    return prepare_hand_state(
        deck,
        button=button,
        fantasyland_flags=fantasyland_flags,
        continuation_hand=continuation_hand,
        hand_number=hand_number,
        config=config,
    )


def new_match(
    *,
    button: PlayerId = PlayerId.PLAYER_0,
    seed: int | str | None = None,
    preset_order: Iterable[str] | None = None,
    config: VariantConfig = DEFAULT_CONFIG,
):
    """Create the first hand of a match."""

    return new_hand(button=button, fantasyland_flags=(False, False), seed=seed, preset_order=preset_order, config=config)


def apply(state, action: GameAction):
    """Apply a single action."""

    return apply_action(state, action)


def showdown(state) -> tuple:
    """Resolve showdown, reveal any concealed Fantasyland boards, and return terminal state plus result."""

    if state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        raise ValueError("Hand is not ready for showdown")

    left_player, right_player = state.players
    left_board = effective_board(left_player, reveal_concealed=True)
    right_board = effective_board(right_player, reveal_concealed=True)
    result = score_terminal(left_player.player_id, left_board, right_player.player_id, right_board, state.config)
    next_flags = resolve_next_hand_fantasyland_flags(
        tuple(player.fantasyland_active for player in state.players),
        (left_board, right_board),
    )
    terminal_state = replace(state, phase=HandPhase.TERMINAL, next_hand_fantasyland=next_flags)
    return terminal_state, result


__all__ = ["advance_after_showdown", "apply", "new_hand", "new_match", "showdown", "TerminalResult", "DeckState"]
