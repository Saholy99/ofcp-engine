"""Action validation and deterministic state transitions."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from ofc.actions import GameAction, Placement, PlaceDrawAction, PlaceInitialFiveAction, SetFantasylandHandAction
from ofc.board import Board, ROW_ORDER, RowName, place_cards, row_capacity_remaining
from ofc.cards import Card
from ofc.config import DEFAULT_CONFIG, VariantConfig
from ofc.deck import DeckState, draw_n
from ofc.fantasyland import resolve_next_hand_fantasyland_flags
from ofc.scoring import TerminalResult
from ofc.state import (
    HandPhase,
    PlayerId,
    effective_board,
    get_player,
    make_empty_state,
    other_player,
    player_index,
    replace_player,
)


def _pending_action_type(player, config: VariantConfig) -> HandPhase | None:
    if player.fantasyland_active and not player.fantasyland_set_done:
        return HandPhase.FANTASYLAND_SET
    if not player.fantasyland_active and not player.initial_placement_done:
        return HandPhase.INITIAL_DEAL
    if not player.fantasyland_active and player.normal_draws_taken < config.normal_draw_turns_per_player:
        return HandPhase.DRAW
    return None


def _cards_needed_for_phase(phase: HandPhase, config: VariantConfig) -> int:
    if phase == HandPhase.FANTASYLAND_SET:
        return config.fantasyland_deal_count
    if phase == HandPhase.INITIAL_DEAL:
        return config.initial_deal_count
    if phase == HandPhase.DRAW:
        return config.normal_draw_count
    raise ValueError(f"Cannot deal cards for phase {phase}")


def _deal_private_draw(state, player_id: PlayerId, phase: HandPhase):
    player = get_player(state, player_id)
    if player.current_private_draw:
        return replace(state, acting_player=player_id, phase=phase)
    draw_count = _cards_needed_for_phase(phase, state.config)
    cards, new_deck = draw_n(state.deck, draw_count)
    updated_player = replace(player, current_private_draw=cards)
    return replace(
        replace_player(replace(state, acting_player=player_id, phase=phase, deck=new_deck), updated_player),
        acting_player=player_id,
        phase=phase,
    )


def _placement_capacity_check(board: Board, placements: tuple[Placement, ...], config: VariantConfig) -> None:
    per_row_counts = {row: 0 for row in ROW_ORDER}
    for placement in placements:
        per_row_counts[placement.row] += 1
    for row, count in per_row_counts.items():
        if count > row_capacity_remaining(board, row, config):
            raise ValueError(f"Action overfills the {row.value} row")


def _build_board_from_placements(placements: tuple[Placement, ...], config: VariantConfig) -> Board:
    return place_cards(Board(), tuple((placement.row, placement.card) for placement in placements), config)


def _board_after_action(board: Board, placements: tuple[Placement, ...], config: VariantConfig) -> Board:
    return place_cards(board, tuple((placement.row, placement.card) for placement in placements), config)


def _validate_player_turn(state, player_id: str) -> PlayerId:
    try:
        resolved_player = PlayerId(player_id)
    except ValueError as exc:
        raise ValueError(f"Unknown player id: {player_id!r}") from exc
    if resolved_player != state.acting_player:
        raise ValueError("Action must be submitted by the acting player")
    return resolved_player


def _validate_cards_match_draw(draw: tuple[Card, ...], placements: tuple[Placement, ...], discard: Card | None = None) -> None:
    action_cards = tuple(placement.card for placement in placements)
    if discard is not None:
        action_cards += (discard,)
    if set(action_cards) != set(draw):
        raise ValueError("Action cards must exactly match the current private draw")
    if len(action_cards) != len(draw):
        raise ValueError("Action must use every card in the current private draw exactly once")


def validate_action(state, action: GameAction) -> None:
    """Validate an action against the current state.

    Raises ``ValueError`` if the action is invalid.
    """

    player_id = _validate_player_turn(state, action.player_id)
    player = get_player(state, player_id)

    if isinstance(action, PlaceInitialFiveAction):
        if state.phase != HandPhase.INITIAL_DEAL:
            raise ValueError("Initial placement actions are only legal during the initial deal")
        if player.fantasyland_active:
            raise ValueError("Fantasyland players do not take normal initial placement actions")
        if player.initial_placement_done:
            raise ValueError("Initial placement is already complete for this player")
        _validate_cards_match_draw(player.current_private_draw, action.placements)
        _placement_capacity_check(player.board, action.placements, state.config)
        return

    if isinstance(action, PlaceDrawAction):
        if state.phase != HandPhase.DRAW:
            raise ValueError("Draw actions are only legal during draw turns")
        if player.fantasyland_active:
            raise ValueError("Fantasyland players do not take normal draw actions")
        if player.normal_draws_taken >= state.config.normal_draw_turns_per_player:
            raise ValueError("This player has already completed all draw turns")
        _validate_cards_match_draw(player.current_private_draw, action.placements, action.discard)
        _placement_capacity_check(player.board, action.placements, state.config)
        return

    if isinstance(action, SetFantasylandHandAction):
        if state.phase != HandPhase.FANTASYLAND_SET:
            raise ValueError("Fantasyland set actions are only legal during Fantasyland turns")
        if not player.fantasyland_active:
            raise ValueError("Only Fantasyland players may submit Fantasyland set actions")
        if player.fantasyland_set_done:
            raise ValueError("Fantasyland hand is already set for this player")
        _validate_cards_match_draw(player.current_private_draw, action.placements, action.discard)
        _build_board_from_placements(action.placements, state.config)
        return

    raise TypeError(f"Unsupported action type: {type(action)!r}")


def _advance_to_next_actor(state, just_acted: PlayerId):
    candidate_order = (other_player(just_acted), just_acted)
    for candidate in candidate_order:
        player = get_player(state, candidate)
        pending_phase = _pending_action_type(player, state.config)
        if pending_phase is not None:
            return _deal_private_draw(state, candidate, pending_phase)
    return replace(state, phase=HandPhase.SHOWDOWN)


def apply_action(state, action: GameAction):
    """Apply a validated action and advance the state."""

    validate_action(state, action)
    player_id = PlayerId(action.player_id)
    player = get_player(state, player_id)

    if isinstance(action, PlaceInitialFiveAction):
        updated_player = replace(
            player,
            board=_board_after_action(player.board, action.placements, state.config),
            current_private_draw=(),
            initial_placement_done=True,
        )
    elif isinstance(action, PlaceDrawAction):
        updated_player = replace(
            player,
            board=_board_after_action(player.board, action.placements, state.config),
            current_private_draw=(),
            hidden_discards=player.hidden_discards + (action.discard,),
            normal_draws_taken=player.normal_draws_taken + 1,
        )
    else:
        concealed_board = _build_board_from_placements(action.placements, state.config)
        updated_player = replace(
            player,
            current_private_draw=(),
            hidden_discards=player.hidden_discards + (action.discard,),
            concealed_fantasyland_board=concealed_board,
            concealed_fantasyland_discard=action.discard,
            fantasyland_set_done=True,
        )

    updated_state = replace_player(state, updated_player)
    return _advance_to_next_actor(updated_state, player_id)


def _generate_assignments(
    cards: tuple[Card, ...],
    board: Board,
    config: VariantConfig,
    index: int = 0,
    placements: tuple[Placement, ...] = (),
) -> Iterator[tuple[Placement, ...]]:
    if index == len(cards):
        yield placements
        return

    card = cards[index]
    for row in ROW_ORDER:
        if sum(1 for placement in placements if placement.row == row) < row_capacity_remaining(board, row, config):
            yield from _generate_assignments(
                cards,
                board,
                config,
                index + 1,
                placements + (Placement(row=row, card=card),),
            )


def legal_actions(state) -> Iterator[GameAction]:
    """Yield legal actions for the current state in deterministic order."""

    player = get_player(state, state.acting_player)
    draw = player.current_private_draw

    if state.phase == HandPhase.INITIAL_DEAL:
        for placements in _generate_assignments(draw, player.board, state.config):
            yield PlaceInitialFiveAction(player_id=player.player_id, placements=placements)
        return

    if state.phase == HandPhase.DRAW:
        for discard_index, discard in enumerate(draw):
            remaining = tuple(card for index, card in enumerate(draw) if index != discard_index)
            for placements in _generate_assignments(remaining, player.board, state.config):
                yield PlaceDrawAction(player_id=player.player_id, placements=placements, discard=discard)
        return

    if state.phase == HandPhase.FANTASYLAND_SET:
        for discard_index, discard in enumerate(draw):
            remaining = tuple(card for index, card in enumerate(draw) if index != discard_index)
            for placements in _generate_assignments(remaining, Board(), state.config):
                yield SetFantasylandHandAction(player_id=player.player_id, placements=placements, discard=discard)
        return

    return


def prepare_hand_state(
    deck: DeckState,
    *,
    button: PlayerId,
    fantasyland_flags: tuple[bool, bool] = (False, False),
    continuation_hand: bool = False,
    hand_number: int = 1,
    config: VariantConfig = DEFAULT_CONFIG,
):
    """Create a new hand state and deal the acting player's first cards."""

    acting_player = other_player(button)
    state = make_empty_state(
        deck,
        button=button,
        acting_player=acting_player,
        fantasyland_flags=fantasyland_flags,
        continuation_hand=continuation_hand,
        hand_number=hand_number,
        config=config,
    )
    first_player = get_player(state, acting_player)
    first_phase = _pending_action_type(first_player, config)
    if first_phase is None:
        raise ValueError("At least one player must have a pending action at hand start")
    return _deal_private_draw(state, acting_player, first_phase)


def advance_after_showdown(state, result: TerminalResult, next_deck: DeckState):
    """Advance from a completed hand to the next hand."""

    if state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        raise ValueError("Can only advance to the next hand after showdown")

    if state.phase == HandPhase.TERMINAL:
        next_flags = state.next_hand_fantasyland
    else:
        boards = tuple(effective_board(player, reveal_concealed=True) for player in state.players)
        current_flags = tuple(player.fantasyland_active for player in state.players)
        next_flags = resolve_next_hand_fantasyland_flags(current_flags, boards)

    continuation_hand = any(next_flags)
    next_button = state.button if continuation_hand else other_player(state.button)
    return prepare_hand_state(
        next_deck,
        button=next_button,
        fantasyland_flags=next_flags,
        continuation_hand=continuation_hand,
        hand_number=state.hand_number + 1,
        config=state.config,
    )
