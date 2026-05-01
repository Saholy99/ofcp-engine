from __future__ import annotations

import unittest

from ofc.engine import new_hand, new_match
from ofc.state import HandPhase, PlayerId
from ofc_analysis.observation import project_observation
from ofc_analysis.scenario import load_scenario
from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state
from ofc_solver.models import MoveAnalysis
from ofc_solver.rollout_policy import RandomRolloutPolicy
from tests.helpers import solver_final_draw_state, stacked_deck_tokens


FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
]


class SolverMonteCarloTest(unittest.TestCase):
    def test_rank_actions_from_state_returns_sorted_estimates(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)

        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=3, rng_seed=31)

        self.assertIsInstance(analysis, MoveAnalysis)
        self.assertEqual(PlayerId.PLAYER_0, analysis.observer)
        self.assertGreater(len(analysis.ranked_actions), 0)
        self.assertEqual([estimate.mean_value for estimate in analysis.ranked_actions], sorted(
            (estimate.mean_value for estimate in analysis.ranked_actions),
            reverse=True,
        ))
        self.assertTrue(all(estimate.sample_count == 3 for estimate in analysis.ranked_actions))
        self.assertTrue(all(estimate.action.action_index == estimate.action_index for estimate in analysis.ranked_actions))

    def test_rank_actions_from_state_supports_initial_deal_root_phase(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=1)

        analysis = rank_actions_from_state(state, observer=PlayerId.PLAYER_1, rollouts_per_action=1, rng_seed=2)

        self.assertEqual(HandPhase.INITIAL_DEAL, analysis.phase)
        self.assertEqual(232, len(analysis.ranked_actions))

    def test_rank_actions_fixed_seed_is_repeatable(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)

        first = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=2, rng_seed=32)
        second = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=2, rng_seed=32)

        self.assertEqual(first, second)

    def test_rank_actions_from_observation_matches_exact_state_when_hidden_cards_do_not_affect_value(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)
        observation = project_observation(state, PlayerId.PLAYER_0)

        exact = rank_actions_from_state(state, observer=PlayerId.PLAYER_0, rollouts_per_action=1, rng_seed=33)
        sampled = rank_actions_from_observation(observation, rollouts_per_action=1, rng_seed=33)

        exact_values = {estimate.action_index: estimate.mean_value for estimate in exact.ranked_actions}
        sampled_values = {estimate.action_index: estimate.mean_value for estimate in sampled.ranked_actions}
        self.assertEqual(exact_values, sampled_values)

    def test_rank_actions_can_include_root_action_risk_metadata(self) -> None:
        state = load_scenario("scenarios/regression/draw_root.json").state

        analysis = rank_actions_from_state(
            state,
            observer=PlayerId.PLAYER_1,
            rollouts_per_action=1,
            rng_seed=37,
            policy=RandomRolloutPolicy(),
            root_action_risk=True,
        )

        self.assertTrue(any(estimate.root_risk_score < 0.0 for estimate in analysis.ranked_actions))
        self.assertTrue(any(estimate.root_risk_reasons for estimate in analysis.ranked_actions))
        self.assertTrue(all(estimate.rollout_mean_value is not None for estimate in analysis.ranked_actions))

    def test_rank_actions_rejects_unsupported_root_phase(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )

        with self.assertRaisesRegex(ValueError, "Unsupported root phase"):
            rank_actions_from_state(state, observer=PlayerId.PLAYER_1, rollouts_per_action=1, rng_seed=34)

    def test_rank_actions_requires_acting_player_observer(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)

        with self.assertRaisesRegex(ValueError, "acting player"):
            rank_actions_from_state(state, observer=PlayerId.PLAYER_1, rollouts_per_action=1, rng_seed=35)


if __name__ == "__main__":
    unittest.main()
