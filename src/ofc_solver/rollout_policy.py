"""Rollout policies used by the baseline Monte Carlo solver."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from ofc.actions import GameAction, Placement, SetFantasylandHandAction
from ofc.board import RowName
from ofc.state import GameState, HandPhase, get_player
from ofc.transitions import legal_actions


class RolloutPolicy(Protocol):
    """Protocol for selecting one legal action during a rollout."""

    def choose_action(self, state: GameState, *, rng: random.Random) -> GameAction:
        """Return one legal action for the supplied engine state."""


@dataclass(frozen=True)
class RandomRolloutPolicy:
    """Uniform random normal-action policy with sampled Fantasyland setting."""

    def choose_action(self, state: GameState, *, rng: random.Random) -> GameAction:
        """Return one legal action for the supplied engine state."""

        if state.phase == HandPhase.FANTASYLAND_SET:
            return sample_fantasyland_set_action(state, rng=rng)

        actions = tuple(legal_actions(state))
        if not actions:
            raise ValueError(f"No rollout actions are available during {state.phase.value}")
        return actions[rng.randrange(len(actions))]


def sample_fantasyland_set_action(state: GameState, *, rng: random.Random) -> SetFantasylandHandAction:
    """Sample one legal-shaped Fantasyland set action without enumerating all sets."""

    if state.phase != HandPhase.FANTASYLAND_SET:
        raise ValueError("Fantasyland set sampling requires FANTASYLAND_SET phase")

    player = get_player(state, state.acting_player)
    cards = list(player.current_private_draw)
    if len(cards) != state.config.fantasyland_deal_count:
        raise ValueError("Fantasyland set sampling requires a 14-card private draw")

    discard = cards.pop(rng.randrange(len(cards)))
    rng.shuffle(cards)
    placements = tuple(
        Placement(row=row, card=card)
        for row, row_cards in (
            (RowName.TOP, cards[: state.config.top_row_capacity]),
            (
                RowName.MIDDLE,
                cards[
                    state.config.top_row_capacity : state.config.top_row_capacity
                    + state.config.middle_row_capacity
                ],
            ),
            (
                RowName.BOTTOM,
                cards[
                    state.config.top_row_capacity
                    + state.config.middle_row_capacity : state.config.fantasyland_placements
                ],
            ),
        )
        for card in row_cards
    )
    return SetFantasylandHandAction(player_id=player.player_id, placements=placements, discard=discard)


__all__ = ["RandomRolloutPolicy", "RolloutPolicy", "sample_fantasyland_set_action"]
