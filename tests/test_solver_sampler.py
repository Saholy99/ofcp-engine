from __future__ import annotations

import random
import unittest

from ofc.board import Board
from ofc.cards import full_deck
from ofc.engine import new_match
from ofc.state import PlayerId, get_player
from ofc_analysis.observation import project_observation
from ofc_solver.sampler import sample_state
from tests.helpers import physical_cards_in_state, solver_final_draw_state


class SolverSamplerTest(unittest.TestCase):
    def test_sample_state_preserves_public_and_observer_private_facts(self) -> None:
        exact_state = solver_final_draw_state(enters_fantasyland=False)
        observation = project_observation(exact_state, PlayerId.PLAYER_0)

        sampled = sample_state(observation, rng=random.Random(11)).state

        self.assertEqual(exact_state.phase, sampled.phase)
        self.assertEqual(exact_state.acting_player, sampled.acting_player)
        self.assertEqual(exact_state.button, sampled.button)
        self.assertEqual(get_player(exact_state, PlayerId.PLAYER_0).board, get_player(sampled, PlayerId.PLAYER_0).board)
        self.assertEqual(
            get_player(exact_state, PlayerId.PLAYER_0).current_private_draw,
            get_player(sampled, PlayerId.PLAYER_0).current_private_draw,
        )
        self.assertEqual(
            get_player(exact_state, PlayerId.PLAYER_0).hidden_discards,
            get_player(sampled, PlayerId.PLAYER_0).hidden_discards,
        )
        self.assertEqual(get_player(exact_state, PlayerId.PLAYER_1).board, get_player(sampled, PlayerId.PLAYER_1).board)
        self.assertEqual(4, len(get_player(sampled, PlayerId.PLAYER_1).hidden_discards))

    def test_sample_state_preserves_card_conservation(self) -> None:
        observation = project_observation(solver_final_draw_state(enters_fantasyland=False), PlayerId.PLAYER_0)

        sampled = sample_state(observation, rng=random.Random(12)).state
        physical_cards = physical_cards_in_state(sampled)

        self.assertEqual(52, len(physical_cards))
        self.assertEqual(set(full_deck()), set(physical_cards))

    def test_sample_state_reconstructs_hidden_opponent_private_draw(self) -> None:
        exact_state = new_match(button=PlayerId.PLAYER_0, seed=7)
        observation = project_observation(exact_state, PlayerId.PLAYER_0)

        sampled = sample_state(observation, rng=random.Random(13)).state

        self.assertEqual(Board(), get_player(sampled, PlayerId.PLAYER_0).board)
        self.assertEqual((), get_player(sampled, PlayerId.PLAYER_0).current_private_draw)
        self.assertEqual(5, len(get_player(sampled, PlayerId.PLAYER_1).current_private_draw))
        self.assertEqual(47, sampled.deck.cards_remaining)


if __name__ == "__main__":
    unittest.main()
