"""Hidden-state sampling for observer-facing solver inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import random

from ofc.board import Board, board_card_count, visible_cards
from ofc.cards import Card, full_deck
from ofc.config import DEFAULT_CONFIG
from ofc.deck import DeckState
from ofc.state import GameState, HandPhase, PLAYER_ORDER, PlayerId, PlayerState
from ofc_analysis.observation import PlayerObservation


@dataclass(frozen=True)
class SampledState:
    """Sampled exact-state result consistent with one observation."""

    state: GameState


def sample_state(observation: PlayerObservation, *, rng: random.Random) -> SampledState:
    """Sample a complete exact state consistent with an observation."""

    _validate_observation_shape(observation)
    known_cards = _known_cards(observation)
    _validate_unique_cards(known_cards, "observation known cards")

    unseen_cards = [card for card in full_deck() if card not in set(known_cards)]
    if len(unseen_cards) != observation.unseen_card_count:
        raise ValueError("observation unseen_card_count does not match known card set")
    rng.shuffle(unseen_cards)

    players = []
    for player_id, public_board in zip(observation.public_player_ids, observation.public_boards, strict=True):
        player = _sample_player_state(player_id, public_board, observation, unseen_cards)
        players.append(player)

    state = GameState(
        config=DEFAULT_CONFIG,
        hand_number=observation.hand_number,
        button=observation.button,
        acting_player=observation.acting_player,
        phase=observation.phase,
        deck=DeckState(undealt_cards=tuple(unseen_cards)),
        players=tuple(players),  # type: ignore[arg-type]
        is_continuation_hand=observation.is_continuation_hand,
        next_hand_fantasyland=observation.next_hand_fantasyland,
    )
    _validate_sampled_card_conservation(state)
    return SampledState(state=state)


def sample_next_deck(*, rng: random.Random) -> DeckState:
    """Sample a fresh full deck for a simulated next hand."""

    cards = list(full_deck())
    rng.shuffle(cards)
    return DeckState(undealt_cards=tuple(cards))


def _validate_observation_shape(observation: PlayerObservation) -> None:
    if observation.public_player_ids != PLAYER_ORDER:
        raise ValueError("observation public_player_ids must use engine player order")
    if len(observation.public_boards) != len(PLAYER_ORDER):
        raise ValueError("observation must contain exactly two public boards")
    if observation.observer not in PLAYER_ORDER:
        raise ValueError("observation observer must be a known player")
    if observation.acting_player not in PLAYER_ORDER:
        raise ValueError("observation acting_player must be a known player")
    if observation.opponent_hidden_discard_count < 0:
        raise ValueError("observation opponent_hidden_discard_count must be non-negative")


def _known_cards(observation: PlayerObservation) -> tuple[Card, ...]:
    known_cards: list[Card] = []
    for board in observation.public_boards:
        known_cards.extend(visible_cards(board))
    known_cards.extend(observation.own_private_draw)
    known_cards.extend(observation.own_hidden_discards)
    if observation.own_concealed_fantasyland_board is not None:
        known_cards.extend(visible_cards(observation.own_concealed_fantasyland_board))
    if (
        observation.own_concealed_fantasyland_discard is not None
        and observation.own_concealed_fantasyland_discard not in observation.own_hidden_discards
    ):
        known_cards.append(observation.own_concealed_fantasyland_discard)
    return tuple(known_cards)


def _sample_player_state(
    player_id: PlayerId,
    public_board: Board,
    observation: PlayerObservation,
    unseen_cards: list[Card],
) -> PlayerState:
    is_observer = player_id == observation.observer
    fantasyland_active = (
        observation.own_fantasyland_active if is_observer else observation.opponent_fantasyland_active
    )
    current_private_draw = _current_private_draw(player_id, observation, unseen_cards)
    hidden_discards = _hidden_discards(player_id, observation, unseen_cards)
    concealed_board, concealed_discard = _concealed_fantasyland_data(
        player_id,
        observation,
        hidden_discards,
        unseen_cards,
    )

    if fantasyland_active:
        initial_placement_done = False
        normal_draws_taken = 0
        fantasyland_set_done = concealed_board is not None
    else:
        normal_draws_taken = len(hidden_discards)
        initial_placement_done = (
            board_card_count(public_board) >= DEFAULT_CONFIG.initial_deal_count
            or normal_draws_taken > 0
            or (player_id == observation.acting_player and observation.phase == HandPhase.DRAW)
        )
        fantasyland_set_done = False

    return PlayerState(
        player_id=player_id,
        board=public_board,
        hidden_discards=hidden_discards,
        current_private_draw=current_private_draw,
        fantasyland_active=fantasyland_active,
        concealed_fantasyland_board=concealed_board,
        concealed_fantasyland_discard=concealed_discard,
        initial_placement_done=initial_placement_done,
        normal_draws_taken=normal_draws_taken,
        fantasyland_set_done=fantasyland_set_done,
    )


def _current_private_draw(
    player_id: PlayerId,
    observation: PlayerObservation,
    unseen_cards: list[Card],
) -> tuple[Card, ...]:
    if player_id == observation.observer:
        if player_id != observation.acting_player and observation.own_private_draw:
            raise ValueError("non-acting observer cannot have a current private draw")
        return observation.own_private_draw

    if player_id != observation.acting_player:
        return ()

    draw_count = _draw_count_for_phase(observation.phase)
    return _take(unseen_cards, draw_count, "opponent current private draw")


def _hidden_discards(
    player_id: PlayerId,
    observation: PlayerObservation,
    unseen_cards: list[Card],
) -> tuple[Card, ...]:
    if player_id == observation.observer:
        return observation.own_hidden_discards
    return _take(unseen_cards, observation.opponent_hidden_discard_count, "opponent hidden discards")


def _concealed_fantasyland_data(
    player_id: PlayerId,
    observation: PlayerObservation,
    hidden_discards: tuple[Card, ...],
    unseen_cards: list[Card],
) -> tuple[Board | None, Card | None]:
    if player_id == observation.observer:
        return observation.own_concealed_fantasyland_board, observation.own_concealed_fantasyland_discard

    if not observation.opponent_fantasyland_active:
        return None, None

    public_board = observation.public_boards[observation.public_player_ids.index(player_id)]
    if visible_cards(public_board):
        return None, None
    if observation.phase == HandPhase.FANTASYLAND_SET and observation.acting_player == player_id:
        return None, None
    if not hidden_discards:
        return None, None

    concealed_discard = hidden_discards[-1]
    board_cards = _take(unseen_cards, DEFAULT_CONFIG.fantasyland_placements, "opponent concealed Fantasyland board")
    return (
        Board(
            top=board_cards[: DEFAULT_CONFIG.top_row_capacity],
            middle=board_cards[
                DEFAULT_CONFIG.top_row_capacity : DEFAULT_CONFIG.top_row_capacity
                + DEFAULT_CONFIG.middle_row_capacity
            ],
            bottom=board_cards[
                DEFAULT_CONFIG.top_row_capacity
                + DEFAULT_CONFIG.middle_row_capacity : DEFAULT_CONFIG.fantasyland_placements
            ],
        ),
        concealed_discard,
    )


def _draw_count_for_phase(phase: HandPhase) -> int:
    if phase == HandPhase.INITIAL_DEAL:
        return DEFAULT_CONFIG.initial_deal_count
    if phase == HandPhase.DRAW:
        return DEFAULT_CONFIG.normal_draw_count
    if phase == HandPhase.FANTASYLAND_SET:
        return DEFAULT_CONFIG.fantasyland_deal_count
    raise ValueError(f"Cannot sample a private draw during {phase.value}")


def _take(cards: list[Card], count: int, context: str) -> tuple[Card, ...]:
    if count < 0:
        raise ValueError(f"Cannot sample a negative number of cards for {context}")
    if count > len(cards):
        raise ValueError(f"Not enough unseen cards to sample {context}")
    taken = tuple(cards[:count])
    del cards[:count]
    return taken


def _validate_unique_cards(cards: Iterable[Card], context: str) -> None:
    cards = tuple(cards)
    if len(cards) != len(set(cards)):
        raise ValueError(f"{context} contain duplicate cards")


def _validate_sampled_card_conservation(state: GameState) -> None:
    physical_cards: list[Card] = []
    for player in state.players:
        physical_cards.extend(visible_cards(player.board))
        physical_cards.extend(player.hidden_discards)
        physical_cards.extend(player.current_private_draw)
        if player.concealed_fantasyland_board is not None:
            physical_cards.extend(visible_cards(player.concealed_fantasyland_board))
    physical_cards.extend(state.deck.undealt_cards)
    if len(physical_cards) != 52 or set(physical_cards) != set(full_deck()):
        raise ValueError("sampled state must account for all 52 physical cards exactly once")


__all__ = ["SampledState", "sample_next_deck", "sample_state"]
