"""Interactive single-hand terminal runner for manual OFC debugging."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from ofc.actions import GameAction, Placement, PlaceDrawAction, PlaceInitialFiveAction, SetFantasylandHandAction
from ofc.board import Board, RowName, visible_cards
from ofc.cards import Card, format_card, full_deck, parse_card
from ofc.config import DEFAULT_CONFIG, VariantConfig
from ofc.deck import DeckState
from ofc.engine import new_hand, showdown
from ofc.scoring import is_foul
from ofc.state import GameState, HandPhase, PlayerId, effective_board, get_player, other_player, replace_player
from ofc.transitions import apply_action, legal_actions, validate_action
from ofc_analysis.action_codec import decode_action, encode_action, encode_actions
from ofc_analysis.observation import project_observation
from ofc_solver.models import MoveEstimate, SUPPORTED_ROOT_PHASES
from ofc_solver.monte_carlo import rank_actions_from_observation
from ofc_solver.policy_registry import policy_from_name


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


class SolverSuggestionBackend(Protocol):
    """Backend interface for interactive move suggestions."""

    def top_moves(
        self,
        state: GameState,
        *,
        player_id: PlayerId,
        decision_index: int,
        top_n: int,
    ) -> tuple[MoveEstimate, ...]:
        """Return ranked suggestions for the current decision."""


@dataclass(frozen=True)
class MonteCarloSuggestionBackend:
    """Interactive suggestion backend backed by the current Monte Carlo solver."""

    rollouts_per_action: int
    rng_seed: int | str | None
    policy_name: str = "random"

    def top_moves(
        self,
        state: GameState,
        *,
        player_id: PlayerId,
        decision_index: int,
        top_n: int,
    ) -> tuple[MoveEstimate, ...]:
        """Return top Monte Carlo suggestions, or no suggestions for unsupported phases."""

        if state.phase not in SUPPORTED_ROOT_PHASES:
            return ()
        seed = None if self.rng_seed is None else f"{self.rng_seed}:{decision_index}"
        analysis = rank_actions_from_observation(
            project_observation(state, player_id),
            rollouts_per_action=self.rollouts_per_action,
            rng_seed=seed,
            policy=policy_from_name(self.policy_name),
        )
        return analysis.ranked_actions[:top_n]


@dataclass(frozen=True)
class NoopSuggestionBackend:
    """Suggestion backend used by tests or very fast manual debugging."""

    def top_moves(
        self,
        state: GameState,
        *,
        player_id: PlayerId,
        decision_index: int,
        top_n: int,
    ) -> tuple[MoveEstimate, ...]:
        """Return no suggestions."""

        return ()


def run_play_hand(
    *,
    hero_player: PlayerId,
    button: PlayerId,
    fantasyland_flags: tuple[bool, bool],
    rollouts_per_action: int,
    rng_seed: int | str | None,
    policy_name: str = "random",
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
    backend: SolverSuggestionBackend | None = None,
) -> int:
    """Run one interactive hand and return a process-style exit code."""

    if rollouts_per_action <= 0:
        raise ValueError("rollouts_per_action must be positive")

    suggestion_backend = backend or MonteCarloSuggestionBackend(
        rollouts_per_action=rollouts_per_action,
        rng_seed=rng_seed,
        policy_name=policy_name,
    )
    state = new_hand(
        button=button,
        fantasyland_flags=fantasyland_flags,
        preset_order=[format_card(card) for card in full_deck()],
        continuation_hand=any(fantasyland_flags),
    )
    decision_index = 0

    output_func("Interactive OFC hand")
    output_func(f"hero: {hero_player.value}")
    output_func(f"button: {button.value}")
    output_func(f"fantasyland_active: player_0={fantasyland_flags[0]} player_1={fantasyland_flags[1]}")

    while state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        player_id = state.acting_player
        if player_id == hero_player:
            draw = _prompt_for_hero_draw(state, input_func=input_func, output_func=output_func)
            state = set_current_private_draw(state, player_id, draw, hero_player=hero_player)
            suggestions = suggestion_backend.top_moves(
                state,
                player_id=player_id,
                decision_index=decision_index,
                top_n=3,
            )
            _print_hero_turn_header(state, output_func=output_func)
            _print_suggestions(state, suggestions, output_func=output_func)
            action = _prompt_for_hero_action(state, suggestions, input_func=input_func, output_func=output_func)
        else:
            state, action = _prompt_for_opponent_action(
                state,
                hero_player=hero_player,
                input_func=input_func,
                output_func=output_func,
            )
        state = apply_action(state, action)
        decision_index += 1

    terminal_state, result = showdown(state)
    _print_final_result(terminal_state, result, output_func=output_func)
    return 0


def parse_cards_input(text: str, *, expected_count: int) -> tuple[Card, ...]:
    """Parse a fixed number of unique card tokens from user input."""

    tokens = text.split()
    if len(tokens) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} cards")
    cards = tuple(parse_card(token) for token in tokens)
    if len(set(cards)) != len(cards):
        raise ValueError("Cards must be unique")
    return cards


def set_current_private_draw(
    state: GameState,
    player_id: PlayerId,
    cards: tuple[Card, ...],
    *,
    hero_player: PlayerId | None = None,
) -> GameState:
    """Return state with a user-entered draw installed for the acting player."""

    if player_id != state.acting_player:
        raise ValueError("Can only set the acting player's current private draw")
    expected_count = _draw_count_for_phase(state.phase, state.config)
    if len(cards) != expected_count:
        raise ValueError(f"{state.phase.value} requires exactly {expected_count} cards")
    if len(set(cards)) != len(cards):
        raise ValueError("Current private draw cards must be unique")

    if hero_player is not None:
        state = _reassign_opponent_hidden_conflicts(state, hero_player=hero_player, known_cards=cards)
    committed = _committed_cards_excluding_acting_draw(state, player_id)
    overlap = sorted((format_card(card) for card in set(cards) & set(committed)))
    if overlap:
        raise ValueError(f"Cards already used in this hand: {', '.join(overlap)}")

    player = get_player(state, player_id)
    updated_player = replace(player, current_private_draw=cards)
    excluded = set(committed) | set(cards)
    deck = DeckState(undealt_cards=tuple(card for card in full_deck() if card not in excluded))
    return replace(replace_player(state, updated_player), deck=deck)


def parse_manual_action(state: GameState, text: str) -> GameAction:
    """Parse a row/discard assignment string into an engine action."""

    player = get_player(state, state.acting_player)
    draw = player.current_private_draw
    tokens = text.split()
    if len(tokens) != len(draw):
        raise ValueError(f"Expected {len(draw)} assignments, one for each card in the draw")

    placements: list[Placement] = []
    discard: Card | None = None
    for token, card in zip(tokens, draw, strict=True):
        assignment = _parse_assignment(token)
        if assignment is None:
            if discard is not None:
                raise ValueError("Exactly one card may be discarded")
            discard = card
        else:
            placements.append(Placement(row=assignment, card=card))

    if state.phase == HandPhase.INITIAL_DEAL:
        if discard is not None:
            raise ValueError("Initial placement cannot discard a card")
        action: GameAction = PlaceInitialFiveAction(player_id=player.player_id, placements=tuple(placements))
    elif state.phase == HandPhase.DRAW:
        if discard is None:
            raise ValueError("Draw action must discard exactly one card")
        action = PlaceDrawAction(player_id=player.player_id, placements=tuple(placements), discard=discard)
    elif state.phase == HandPhase.FANTASYLAND_SET:
        if discard is None:
            raise ValueError("Fantasyland set action must discard exactly one card")
        action = SetFantasylandHandAction(player_id=player.player_id, placements=tuple(placements), discard=discard)
    else:
        raise ValueError(f"Cannot choose an action during {state.phase.value}")

    validate_action(state, action)
    return action


def select_action_by_index(state: GameState, action_index: int) -> GameAction:
    """Return the legal normal-phase action with the given 1-indexed UI action number."""

    if state.phase == HandPhase.FANTASYLAND_SET:
        raise ValueError("Fantasyland set actions must be entered manually")
    actions = tuple(legal_actions(state))
    if action_index < 1 or action_index > len(actions):
        raise ValueError(f"Action index must be between 1 and {len(actions)}")
    return actions[action_index - 1]


def describe_action(action: GameAction, *, action_index: int | None = None) -> str:
    """Return a compact human-readable action description."""

    encoded = encode_action(-1 if action_index is None else action_index, action)
    placements = ", ".join(
        f"{placement['row']}:{placement['card']}" for placement in encoded.payload["placements"]
    )
    suffix = f" discard={encoded.payload['discard']}" if "discard" in encoded.payload else ""
    prefix = "" if action_index is None else f"action_index={action_index} "
    return f"{prefix}{encoded.action_type} {encoded.payload['player_id']} placements=[{placements}]{suffix}"


def _prompt_for_hero_draw(
    state: GameState,
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> tuple[Card, ...]:
    player_id = state.acting_player
    expected_count = _draw_count_for_phase(state.phase, state.config)
    while True:
        try:
            text = input_func(
                f"Enter {expected_count} cards for {player_id.value} during {state.phase.value}: "
            )
            return parse_cards_input(text, expected_count=expected_count)
        except ValueError as exc:
            output_func(f"Invalid cards: {exc}")


def _prompt_for_hero_action(
    state: GameState,
    suggestions: tuple[MoveEstimate, ...],
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> GameAction:
    output_func(_assignment_help(state))
    while True:
        text = input_func("Choose suggestion rank, action N, list, or manual assignments: ").strip()
        try:
            if text == "list":
                _print_legal_actions(state, output_func=output_func)
                continue
            return _parse_hero_action_choice(state, suggestions, text)
        except ValueError as exc:
            output_func(f"Invalid action: {exc}")


def _prompt_for_opponent_action(
    state: GameState,
    *,
    hero_player: PlayerId,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> tuple[GameState, GameAction]:
    if state.phase == HandPhase.INITIAL_DEAL:
        visible_cards_entered = _prompt_for_opponent_visible_cards(
            state,
            count=state.config.initial_deal_count,
            input_func=input_func,
            output_func=output_func,
        )
        state = set_current_private_draw(state, state.acting_player, visible_cards_entered, hero_player=hero_player)
        _print_opponent_turn_header(state, visible_cards_entered, output_func=output_func)
        action = _prompt_for_opponent_visible_action(
            state,
            visible_count=5,
            input_func=input_func,
            output_func=output_func,
        )
        return state, action

    if state.phase == HandPhase.DRAW:
        visible_cards_entered = _prompt_for_opponent_visible_cards(
            state,
            count=state.config.normal_draw_placements,
            input_func=input_func,
            output_func=output_func,
        )
        state = _reassign_opponent_hidden_conflicts(state, hero_player=hero_player, known_cards=visible_cards_entered)
        hidden_discard = _choose_unknown_hidden_discard(state, visible_cards_entered=visible_cards_entered)
        draw = visible_cards_entered + (hidden_discard,)
        state = set_current_private_draw(state, state.acting_player, draw, hero_player=hero_player)
        _print_opponent_turn_header(state, visible_cards_entered, output_func=output_func)
        action = _prompt_for_opponent_visible_action(
            state,
            visible_count=2,
            input_func=input_func,
            output_func=output_func,
        )
        return state, action

    if state.phase == HandPhase.FANTASYLAND_SET:
        visible_cards_entered = _prompt_for_opponent_visible_cards(
            state,
            count=state.config.fantasyland_deal_count,
            input_func=input_func,
            output_func=output_func,
        )
        state = set_current_private_draw(state, state.acting_player, visible_cards_entered, hero_player=hero_player)
        _print_opponent_turn_header(state, visible_cards_entered, output_func=output_func)
        action = _prompt_for_opponent_visible_action(
            state,
            visible_count=14,
            input_func=input_func,
            output_func=output_func,
        )
        return state, action

    raise ValueError(f"Cannot act during {state.phase.value}")


def _prompt_for_opponent_visible_cards(
    state: GameState,
    *,
    count: int,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> tuple[Card, ...]:
    while True:
        try:
            text = input_func(
                f"Enter {count} visible cards for opponent {state.acting_player.value} during {state.phase.value}: "
            )
            return parse_cards_input(text, expected_count=count)
        except ValueError as exc:
            output_func(f"Invalid opponent cards: {exc}")


def _prompt_for_opponent_visible_action(
    state: GameState,
    *,
    visible_count: int,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> GameAction:
    if state.phase == HandPhase.FANTASYLAND_SET:
        output_func(_assignment_help(state))
    else:
        output_func(_opponent_assignment_help(state, visible_count=visible_count))
    while True:
        text = input_func("Enter opponent row assignments: ").strip()
        try:
            if state.phase == HandPhase.FANTASYLAND_SET:
                return parse_manual_action(state, text)
            return parse_opponent_visible_action(state, text, visible_count=visible_count)
        except ValueError as exc:
            output_func(f"Invalid opponent action: {exc}")


def parse_opponent_visible_action(state: GameState, text: str, *, visible_count: int) -> GameAction:
    """Parse opponent visible placement assignments without requiring known discards."""

    player = get_player(state, state.acting_player)
    draw = player.current_private_draw
    visible_draw = draw[:visible_count]
    tokens = text.split()
    if len(tokens) != visible_count:
        raise ValueError(f"Expected {visible_count} row assignments")

    placements = tuple(
        Placement(row=_parse_visible_row_assignment(token), card=card)
        for token, card in zip(tokens, visible_draw, strict=True)
    )
    if state.phase == HandPhase.INITIAL_DEAL:
        action: GameAction = PlaceInitialFiveAction(player_id=player.player_id, placements=placements)
    elif state.phase == HandPhase.DRAW:
        action = PlaceDrawAction(player_id=player.player_id, placements=placements, discard=draw[visible_count])
    else:
        raise ValueError(f"Cannot choose an action during {state.phase.value}")

    validate_action(state, action)
    return action


def _action_from_best_choice(text: str, suggestions: tuple[MoveEstimate, ...]) -> GameAction:
    parts = text.split()
    if len(parts) > 2 or parts[0] != "best":
        raise ValueError("Use best or best N")
    rank = 1 if len(parts) == 1 else _parse_positive_int(parts[1], "Suggested rank")
    return _action_from_suggestion_rank(rank, suggestions)


def _action_from_suggestion_rank(rank: int, suggestions: tuple[MoveEstimate, ...]) -> GameAction:
    if rank < 1 or rank > len(suggestions):
        raise ValueError(f"Suggested rank must be between 1 and {len(suggestions)}")
    return decode_action(suggestions[rank - 1].action)


def _parse_hero_action_choice(
    state: GameState,
    suggestions: tuple[MoveEstimate, ...],
    text: str,
) -> GameAction:
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("Enter a suggestion rank, action N, list, or manual assignments")
    if normalized.isdigit():
        value = int(normalized)
        if suggestions:
            return _action_from_suggestion_rank(value, suggestions)
        return select_action_by_index(state, value)
    if normalized.startswith("best"):
        return _action_from_best_choice(normalized, suggestions)
    action_index = _parse_action_index_choice(normalized)
    if action_index is not None:
        return select_action_by_index(state, action_index)
    return parse_manual_action(state, text)


def _parse_action_index_choice(text: str) -> int | None:
    if text.startswith("action_index="):
        return _parse_positive_int(text.removeprefix("action_index="), "Action index")
    parts = text.replace("=", " ").split()
    if len(parts) == 2 and parts[0] in {"action", "action_index"}:
        return _parse_positive_int(parts[1], "Action index")
    return None


def _parse_positive_int(text: str, label: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if value < 1:
        raise ValueError(f"{label} must be at least 1")
    return value


def _print_hero_turn_header(state: GameState, *, output_func: OutputFunc) -> None:
    player = get_player(state, state.acting_player)
    output_func("")
    output_func(f"Turn: {player.player_id.value} ({state.phase.value})")
    output_func(f"Current draw: {' '.join(format_card(card) for card in player.current_private_draw)}")
    for state_player in state.players:
        output_func(f"{state_player.player_id.value} board: {_board_summary(state_player.board)}")


def _print_opponent_turn_header(
    state: GameState,
    visible_cards_entered: tuple[Card, ...],
    *,
    output_func: OutputFunc,
) -> None:
    output_func("")
    output_func(f"Opponent turn: {state.acting_player.value} ({state.phase.value})")
    output_func(f"Visible cards: {' '.join(format_card(card) for card in visible_cards_entered)}")
    output_func("Solver suggestions: skipped for opponent turn.")
    for state_player in state.players:
        output_func(f"{state_player.player_id.value} board: {_board_summary(state_player.board)}")


def _print_suggestions(
    state: GameState,
    suggestions: tuple[MoveEstimate, ...],
    *,
    output_func: OutputFunc,
) -> None:
    if not suggestions:
        if state.phase == HandPhase.FANTASYLAND_SET:
            output_func("Solver suggestions: unavailable for Fantasyland set with the current MVP solver.")
        else:
            output_func("Solver suggestions: unavailable.")
        return

    output_func("Top solver suggestions:")
    for rank, estimate in enumerate(suggestions, start=1):
        action = decode_action(estimate.action)
        output_func(
            f"{rank}. EV={estimate.mean_value:.3f} samples={estimate.sample_count} "
            f"legal_action={estimate.action_index + 1} {describe_action(action)}"
        )


def _print_legal_actions(state: GameState, *, output_func: OutputFunc) -> None:
    if state.phase == HandPhase.FANTASYLAND_SET:
        output_func("Fantasyland set actions are not listed; enter manual assignments.")
        return
    for encoded in encode_actions(tuple(legal_actions(state))):
        output_func(f"action {encoded.action_index + 1}: {describe_action(decode_action(encoded))}")
    output_func("To choose a listed legal action directly, type: action N")


def _print_final_result(state: GameState, result, *, output_func: OutputFunc) -> None:
    output_func("")
    output_func("Final Result")
    for player in state.players:
        board = effective_board(player, reveal_concealed=True)
        output_func(f"{player.player_id.value} board: {_board_summary(board)}")
        output_func(f"{player.player_id.value} fouled: {is_foul(board)}")
    output_func(
        f"{_player_id_text(result.left.player_id)}: total={result.left.total_points} rows={result.left.row_points} "
        f"sweep={result.left.sweep_bonus} royalties={result.left.royalties}"
    )
    output_func(
        f"{_player_id_text(result.right.player_id)}: total={result.right.total_points} rows={result.right.row_points} "
        f"sweep={result.right.sweep_bonus} royalties={result.right.royalties}"
    )
    output_func(
        "next_hand_fantasyland: "
        f"player_0={state.next_hand_fantasyland[0]} player_1={state.next_hand_fantasyland[1]}"
    )


def _assignment_help(state: GameState) -> str:
    draw = get_player(state, state.acting_player).current_private_draw
    cards = " ".join(format_card(card) for card in draw)
    if state.phase == HandPhase.INITIAL_DEAL:
        return f"Manual format: one row per card in order ({cards}); rows: top middle bottom"
    return f"Manual format: one assignment per card in order ({cards}); use top middle bottom discard"


def _opponent_assignment_help(state: GameState, *, visible_count: int) -> str:
    draw = get_player(state, state.acting_player).current_private_draw[:visible_count]
    cards = " ".join(format_card(card) for card in draw)
    return f"Opponent manual format: one row per visible card in order ({cards}); rows: top middle bottom"


def _board_summary(board: Board) -> str:
    return (
        f"top=[{' '.join(format_card(card) for card in board.top)}] "
        f"middle=[{' '.join(format_card(card) for card in board.middle)}] "
        f"bottom=[{' '.join(format_card(card) for card in board.bottom)}]"
    )


def _player_id_text(player_id: str | PlayerId) -> str:
    return player_id.value if isinstance(player_id, PlayerId) else str(player_id)


def _draw_count_for_phase(phase: HandPhase, config: VariantConfig = DEFAULT_CONFIG) -> int:
    if phase == HandPhase.INITIAL_DEAL:
        return config.initial_deal_count
    if phase == HandPhase.DRAW:
        return config.normal_draw_count
    if phase == HandPhase.FANTASYLAND_SET:
        return config.fantasyland_deal_count
    raise ValueError(f"{phase.value} does not require a private draw")


def _parse_assignment(token: str) -> RowName | None:
    normalized = token.strip().lower()
    if normalized in {"top", "front", "t"}:
        return RowName.TOP
    if normalized in {"middle", "mid", "m"}:
        return RowName.MIDDLE
    if normalized in {"bottom", "back", "b"}:
        return RowName.BOTTOM
    if normalized in {"discard", "dead", "d", "x"}:
        return None
    raise ValueError(f"Unknown assignment token: {token!r}")


def _parse_visible_row_assignment(token: str) -> RowName:
    row = _parse_assignment(token)
    if row is None:
        raise ValueError("Visible opponent assignments cannot be discard")
    return row


def _choose_unknown_hidden_discard(
    state: GameState,
    *,
    visible_cards_entered: tuple[Card, ...],
) -> Card:
    committed = set(_committed_cards_excluding_acting_draw(state, state.acting_player))
    blocked = committed | set(visible_cards_entered)
    candidates = [card for card in full_deck() if card not in blocked]
    if not candidates:
        raise ValueError("No available card can represent the opponent's unknown discard")
    return candidates[0]


def _reassign_opponent_hidden_conflicts(
    state: GameState,
    *,
    hero_player: PlayerId,
    known_cards: tuple[Card, ...],
) -> GameState:
    opponent_id = other_player(hero_player)
    opponent = get_player(state, opponent_id)
    conflicts = set(opponent.hidden_discards) & set(known_cards)
    if not conflicts:
        return state

    committed_without_opponent_discards: list[Card] = []
    for player in state.players:
        committed_without_opponent_discards.extend(visible_cards(player.board))
        if player.player_id != state.acting_player:
            committed_without_opponent_discards.extend(player.current_private_draw)
        if player.player_id != opponent_id:
            committed_without_opponent_discards.extend(player.hidden_discards)
        if player.concealed_fantasyland_board is not None:
            committed_without_opponent_discards.extend(visible_cards(player.concealed_fantasyland_board))

    blocked = set(committed_without_opponent_discards) | set(known_cards) | (set(opponent.hidden_discards) - conflicts)
    replacements = [card for card in full_deck() if card not in blocked]
    if len(replacements) < len(conflicts):
        raise ValueError("Cannot reassign opponent unknown discards away from newly visible cards")

    replacement_iter = iter(replacements)
    updated_discards = tuple(next(replacement_iter) if card in conflicts else card for card in opponent.hidden_discards)
    updated_opponent = replace(opponent, hidden_discards=updated_discards)
    return replace_player(state, updated_opponent)


def _committed_cards_excluding_acting_draw(state: GameState, acting_player: PlayerId) -> tuple[Card, ...]:
    cards: list[Card] = []
    for player in state.players:
        cards.extend(visible_cards(player.board))
        cards.extend(player.hidden_discards)
        if player.player_id != acting_player:
            cards.extend(player.current_private_draw)
        if player.concealed_fantasyland_board is not None:
            cards.extend(visible_cards(player.concealed_fantasyland_board))
    return tuple(cards)


__all__ = [
    "MonteCarloSuggestionBackend",
    "NoopSuggestionBackend",
    "SolverSuggestionBackend",
    "describe_action",
    "parse_cards_input",
    "parse_manual_action",
    "parse_opponent_visible_action",
    "run_play_hand",
    "select_action_by_index",
    "set_current_private_draw",
]
