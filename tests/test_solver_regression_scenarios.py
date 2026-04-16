from __future__ import annotations

from pathlib import Path
import random
import unittest

from ofc.state import HandPhase, PlayerId, get_player
from ofc_analysis.observation import project_observation
from ofc_analysis.scenario import load_scenario
from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state
from ofc_solver.sampler import sample_state
from tests.helpers import physical_cards_in_state


FIXTURE_DIR = Path("scenarios/regression")


class SolverRegressionScenariosTest(unittest.TestCase):
    def _state(self, name: str):
        return load_scenario(FIXTURE_DIR / name).state

    def test_all_regression_fixtures_load(self) -> None:
        expected = {
            "draw_root.json",
            "fantasyland_continuation_ev.json",
            "immediate_scoring.json",
            "initial_deal.json",
            "opponent_hidden_discards.json",
        }

        loaded = {path.name for path in FIXTURE_DIR.glob("*.json") if load_scenario(path)}

        self.assertEqual(expected, loaded)

    def test_initial_deal_fixture_is_supported_by_solver(self) -> None:
        state = self._state("initial_deal.json")

        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_1, rollouts_per_action=1, rng_seed=301)

        self.assertEqual(HandPhase.INITIAL_DEAL, analysis.phase)
        self.assertEqual(232, len(analysis.ranked_actions))

    def test_obviously_stronger_immediate_actions_rank_above_weaker_actions(self) -> None:
        state = self._state("immediate_scoring.json")

        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=3, rng_seed=302)

        self.assertIn(analysis.ranked_actions[0].action_index, {1, 4})
        self.assertGreater(analysis.ranked_actions[0].mean_value, analysis.ranked_actions[-1].mean_value)

    def test_symmetric_actions_have_equal_estimates_in_immediate_scoring_fixture(self) -> None:
        state = self._state("immediate_scoring.json")

        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=3, rng_seed=303)
        estimates = {estimate.action_index: estimate for estimate in analysis.ranked_actions}

        self.assertEqual(estimates[1].mean_value, estimates[4].mean_value)
        self.assertEqual(estimates[1].stddev, estimates[4].stddev)

    def test_fixed_seed_observation_ranking_is_stable(self) -> None:
        state = self._state("draw_root.json")
        observation = project_observation(state, state.acting_player)

        first = rank_actions_from_observation(observation, rollouts_per_action=2, rng_seed=304)
        second = rank_actions_from_observation(observation, rollouts_per_action=2, rng_seed=304)

        self.assertEqual(first, second)

    def test_fantasyland_continuation_changes_fixed_seed_ev(self) -> None:
        immediate = self._state("immediate_scoring.json")
        fantasyland = self._state("fantasyland_continuation_ev.json")

        immediate_analysis = rank_actions_from_state(
            immediate,
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=2,
            rng_seed=202,
        )
        fantasyland_analysis = rank_actions_from_state(
            fantasyland,
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=2,
            rng_seed=202,
        )

        self.assertGreater(fantasyland_analysis.ranked_actions[0].mean_value, immediate_analysis.ranked_actions[0].mean_value)

    def test_opponent_hidden_discards_affect_sampled_unseen_partition(self) -> None:
        state = self._state("opponent_hidden_discards.json")
        observation = project_observation(state, PlayerId.PLAYER_0)

        sampled = sample_state(observation, rng=random.Random(305)).state

        self.assertEqual(1, observation.opponent_hidden_discard_count)
        self.assertEqual(1, len(get_player(sampled, PlayerId.PLAYER_1).hidden_discards))
        self.assertEqual(52, len(physical_cards_in_state(sampled)))
        self.assertEqual(36, sampled.deck.cards_remaining)


if __name__ == "__main__":
    unittest.main()
