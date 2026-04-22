from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import unittest

from ofc.board import RowName
from ofc.cards import format_card
from ofc.engine import new_hand, new_match, showdown
from ofc.state import PlayerId
from ofc.transitions import apply_action, legal_actions, validate_action
from ofc_analysis.scenario import load_scenario
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy
from ofc_solver.policy_registry import policy_from_name
from ofc_solver.rollout_policy import RandomRolloutPolicy
from tests.helpers import (
    physical_cards_in_state,
    solver_late_street_exact_search_state,
    solver_middle_over_bottom_pressure_state,
    solver_unsupported_top_pair_state,
    solver_unsupported_top_trips_state,
    stacked_deck_tokens,
)


FIXTURE_DIR = Path("scenarios/regression")
FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
]


class SolverRolloutPolicyTest(unittest.TestCase):
    def test_random_policy_selects_engine_legal_normal_action(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=1)
        policy = RandomRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(3))

        validate_action(state, action)

    def test_random_policy_samples_fantasyland_set_without_exhaustive_enumeration(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        policy = RandomRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(4))

        validate_action(state, action)
        self.assertEqual(13, len(action.placements))
        self.assertEqual(Counter({"top": 3, "middle": 5, "bottom": 5}), Counter(p.row.value for p in action.placements))
        self.assertEqual(set(physical_cards_in_state(state)[:14]), {p.card for p in action.placements} | {action.discard})

    def test_policy_registry_returns_supported_policies(self) -> None:
        self.assertIsInstance(policy_from_name("random"), RandomRolloutPolicy)
        self.assertIsInstance(policy_from_name("heuristic"), HeuristicRolloutPolicy)

        with self.assertRaisesRegex(ValueError, "Unsupported rollout policy"):
            policy_from_name("not-a-policy")

    def test_heuristic_policy_selects_engine_legal_normal_action(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=1)
        policy = HeuristicRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(3))

        validate_action(state, action)

    def test_heuristic_policy_fixed_seed_is_repeatable(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=1)
        policy = HeuristicRolloutPolicy()

        first = policy.choose_action(state, rng=random.Random(7))
        second = policy.choose_action(state, rng=random.Random(7))

        self.assertEqual(first, second)

    def test_heuristic_policy_prefers_immediate_scoring_oracle_action(self) -> None:
        state = load_scenario(FIXTURE_DIR / "immediate_scoring.json").state
        actions = tuple(legal_actions(state))
        policy = HeuristicRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(11))

        self.assertIn(actions.index(action), {1, 4})

    def test_heuristic_policy_scores_immediate_showdown_by_terminal_value(self) -> None:
        state = load_scenario(FIXTURE_DIR / "immediate_scoring.json").state
        actions = tuple(legal_actions(state))
        policy = HeuristicRolloutPolicy()

        scores_by_action_index = {actions.index(scored.action): scored.score for scored in policy.rank_actions(state)}

        self.assertEqual(29.0, scores_by_action_index[1])
        self.assertEqual(29.0, scores_by_action_index[4])
        self.assertEqual(-6.0, scores_by_action_index[0])

    def test_heuristic_policy_uses_exact_search_for_small_late_street_tree(self) -> None:
        state = solver_late_street_exact_search_state()
        actions = tuple(legal_actions(state))
        policy = HeuristicRolloutPolicy()

        ranked = policy.rank_actions(state)

        self.assertEqual(actions[0], ranked[0].action)
        self.assertEqual(6.0, ranked[0].score)
        self.assertIn("exact-late-street", ranked[0].reasons)

    def test_heuristic_policy_reports_exact_search_decision_diagnostics(self) -> None:
        state = solver_late_street_exact_search_state()
        actions = tuple(legal_actions(state))
        policy = HeuristicRolloutPolicy()

        action, diagnostics = policy.choose_action_with_diagnostics(state, rng=random.Random(12))

        self.assertEqual(actions[0], action)
        self.assertTrue(diagnostics.used_exact_late_search)
        self.assertEqual(76, diagnostics.exact_late_search_node_count)
        self.assertIn("exact-late-street", diagnostics.selected_reasons)

    def test_heuristic_policy_avoids_immediate_completed_board_foul(self) -> None:
        state = load_scenario(FIXTURE_DIR / "immediate_scoring.json").state
        policy = HeuristicRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(11))
        next_state = apply_action(state, action)
        _, result = showdown(next_state)

        self.assertFalse(result.left.fouled)

    def test_heuristic_policy_samples_valid_fantasyland_set(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        policy = HeuristicRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(4))

        validate_action(state, action)
        self.assertEqual(13, len(action.placements))
        self.assertEqual(Counter({"top": 3, "middle": 5, "bottom": 5}), Counter(p.row.value for p in action.placements))
        self.assertEqual(set(physical_cards_in_state(state)[:14]), {p.card for p in action.placements} | {action.discard})

    def test_heuristic_policy_rejects_unsupported_top_pair(self) -> None:
        state = solver_unsupported_top_pair_state()
        policy = HeuristicRolloutPolicy()

        ranked = policy.rank_actions(state)

        self.assertFalse(_action_places_card(ranked[0].action, RowName.TOP, "Qd"))
        unsupported_scores = [score for score in ranked if _action_places_card(score.action, RowName.TOP, "Qd")]
        self.assertTrue(any("unsupported-top-pair" in score.reasons for score in unsupported_scores))

    def test_heuristic_policy_rejects_unsupported_top_trips(self) -> None:
        state = solver_unsupported_top_trips_state()
        policy = HeuristicRolloutPolicy()

        ranked = policy.rank_actions(state)

        self.assertFalse(_action_places_card(ranked[0].action, RowName.TOP, "Qs"))
        unsupported_scores = [score for score in ranked if _action_places_card(score.action, RowName.TOP, "Qs")]
        self.assertTrue(any("unsupported-top-trips" in score.reasons for score in unsupported_scores))

    def test_heuristic_policy_protects_bottom_before_overbuilding_middle(self) -> None:
        state = solver_middle_over_bottom_pressure_state()
        policy = HeuristicRolloutPolicy()

        ranked = policy.rank_actions(state)

        self.assertTrue(_action_places_card(ranked[0].action, RowName.BOTTOM, "As"))
        pressured_scores = [score for score in ranked if _action_places_card(score.action, RowName.MIDDLE, "As")]
        self.assertTrue(any("middle-over-bottom-pressure" in score.reasons for score in pressured_scores))


def _action_places_card(action, row: RowName, token: str) -> bool:
    return any(placement.row is row and format_card(placement.card) == token for placement in action.placements)


if __name__ == "__main__":
    unittest.main()
