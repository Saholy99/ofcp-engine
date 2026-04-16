from __future__ import annotations

from itertools import islice
import unittest

from ofc.engine import new_match
from ofc.state import PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_actions
from ofc_analysis.observation import project_observation
from ofc_analysis.render import render_actions, render_move_analysis, render_observation, render_state
from ofc_solver.monte_carlo import rank_actions_from_state
from tests.helpers import stacked_deck_tokens


INITIAL_PREFIX = [
    "Ac", "9d", "Kh", "Kd", "Ah",
    "Ks", "8c", "Jh", "Jd", "Qh",
]


class AnalysisRenderTest(unittest.TestCase):
    def _initial_state(self):
        return new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(INITIAL_PREFIX))

    def test_render_state_is_deterministic_in_text_and_json(self) -> None:
        state = self._initial_state()

        text_output = render_state(state)
        json_output = render_state(state, as_json=True)

        self.assertTrue(text_output.text.startswith("Exact State"))
        self.assertIn("phase: initial_deal", text_output.text)
        self.assertIn('players.player_1.current_private_draw: ["Ac", "9d", "Kh", "Kd", "Ah"]', text_output.text)
        self.assertEqual("initial_deal", json_output.payload["phase"])
        self.assertEqual("player_1", json_output.payload["acting_player"])

    def test_render_observation_redacts_opponent_hidden_details(self) -> None:
        observation = project_observation(self._initial_state(), PlayerId.PLAYER_0)

        text_output = render_observation(observation)
        json_output = render_observation(observation, as_json=True)

        self.assertTrue(text_output.text.startswith("Player Observation"))
        self.assertIn("observer: player_0", text_output.text)
        self.assertNotIn("opponent_hidden_discards", json_output.payload)
        self.assertEqual([], json_output.payload["own_private_draw"])
        self.assertEqual(52, json_output.payload["unseen_card_count"])

    def test_render_actions_is_stable(self) -> None:
        state = self._initial_state()
        encoded = encode_actions(tuple(islice(legal_actions(state), 2)))

        text_output = render_actions(encoded)
        json_output = render_actions(encoded, as_json=True)

        self.assertTrue(text_output.text.startswith("Legal Actions"))
        self.assertIn("[0] place_initial_five player_1", text_output.text)
        self.assertEqual(2, json_output.payload["action_count"])
        self.assertEqual("player_1", json_output.payload["actions"][0]["payload"]["player_id"])

    def test_render_move_analysis_is_stable(self) -> None:
        state = self._initial_state()
        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_1, rollouts_per_action=1, rng_seed=2)

        text_output = render_move_analysis(analysis)
        json_output = render_move_analysis(analysis, as_json=True)

        self.assertTrue(text_output.text.startswith("Move Analysis"))
        self.assertIn("observer: player_1", text_output.text)
        self.assertEqual("player_1", json_output.payload["observer"])
        self.assertEqual("initial_deal", json_output.payload["phase"])
        self.assertEqual(232, json_output.payload["action_count"])


if __name__ == "__main__":
    unittest.main()
