from __future__ import annotations

import unittest

from ofc.actions import PlaceInitialFiveAction
from ofc.board import RowName
from ofc.engine import new_match
from ofc.state import PlayerId
from ofc.transitions import legal_actions, validate_action
from ofc_solver.early_search import (
    EarlySearchConfig,
    detect_card_patterns,
    select_early_search_candidates,
)
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy
from tests.helpers import cards, solver_unsupported_top_pair_state, stacked_deck_tokens


class SolverEarlySearchTest(unittest.TestCase):
    def test_candidate_generation_returns_legal_engine_actions_only(self) -> None:
        state = new_match(
            button=PlayerId.PLAYER_0,
            preset_order=stacked_deck_tokens(["Ah", "Ad", "Kh", "Qh", "Jh"]),
        )

        result = select_early_search_candidates(state, config=EarlySearchConfig(beam_size=8))

        self.assertEqual(232, result.total_legal_actions)
        self.assertEqual(8, len(result.candidates))
        for candidate in result.candidates:
            validate_action(state, candidate.action)

    def test_beam_size_is_respected_and_never_exceeds_legal_actions(self) -> None:
        state = solver_unsupported_top_pair_state()

        result = select_early_search_candidates(state, config=EarlySearchConfig(beam_size=3))

        self.assertEqual(3, len(result.candidates))
        self.assertLessEqual(len(result.candidates), result.total_legal_actions)

    def test_candidate_ordering_is_deterministic_for_fixed_state(self) -> None:
        state = new_match(
            button=PlayerId.PLAYER_0,
            preset_order=stacked_deck_tokens(["Ah", "Ad", "Kh", "Qh", "Jh"]),
        )

        first = select_early_search_candidates(state, config=EarlySearchConfig(beam_size=12))
        second = select_early_search_candidates(state, config=EarlySearchConfig(beam_size=12))

        self.assertEqual(
            tuple(candidate.action_index for candidate in first.candidates),
            tuple(candidate.action_index for candidate in second.candidates),
        )
        self.assertEqual(
            tuple(candidate.pattern_score for candidate in first.candidates),
            tuple(candidate.pattern_score for candidate in second.candidates),
        )

    def test_pattern_detection_identifies_made_and_drawing_structures(self) -> None:
        assessment = detect_card_patterns(cards("Ah Ad Ac Kh Kd Qh Jh Th 9h 8h 7c 6d 5s"))

        self.assertIn("full-house-like", assessment.reasons)
        self.assertIn("trips", assessment.reasons)
        self.assertIn("two-pair", assessment.reasons)
        self.assertIn("flush-made", assessment.reasons)
        self.assertIn("straight-made", assessment.reasons)
        self.assertGreater(assessment.score, 0.0)

    def test_pattern_detection_identifies_flush_and_straight_draws(self) -> None:
        assessment = detect_card_patterns(cards("Ah Qh 9h 8c 7d 6s 2c"))

        self.assertIn("flush-draw-3", assessment.reasons)
        self.assertIn("straight-draw-4", assessment.reasons)

    def test_initial_deal_prioritizes_strong_structure_on_bottom_without_fixing_one_move(self) -> None:
        state = new_match(
            button=PlayerId.PLAYER_0,
            preset_order=stacked_deck_tokens(["Ah", "Ad", "Ac", "Kh", "Kd"]),
        )

        result = select_early_search_candidates(state, config=EarlySearchConfig(beam_size=10))

        self.assertGreater(len(result.candidates), 1)
        top_action = result.candidates[0].action
        self.assertIsInstance(top_action, PlaceInitialFiveAction)
        bottom_cards = {placement.card for placement in top_action.placements if placement.row is RowName.BOTTOM}
        self.assertTrue(set(cards("Ah Ad Ac")).issubset(bottom_cards))
        self.assertIn("bottom:full-house-like", result.candidates[0].reasons)

    def test_initial_deal_candidate_order_is_unchanged_by_draw_safe_config(self) -> None:
        state = new_match(
            button=PlayerId.PLAYER_0,
            preset_order=stacked_deck_tokens(["Ah", "Ad", "Ac", "Kh", "Kd"]),
        )

        old_style = select_early_search_candidates(
            state,
            config=EarlySearchConfig(beam_size=10, draw_safe_candidates=False),
        )
        safe_draw_enabled = select_early_search_candidates(
            state,
            config=EarlySearchConfig(beam_size=10, draw_safe_candidates=True),
        )

        self.assertEqual(
            tuple(candidate.action_index for candidate in old_style.candidates),
            tuple(candidate.action_index for candidate in safe_draw_enabled.candidates),
        )

    def test_safe_draw_candidates_include_baseline_heuristic_top_actions(self) -> None:
        state = solver_unsupported_top_pair_state()
        legal = tuple(legal_actions(state))
        index_by_action = {action: index for index, action in enumerate(legal)}
        expected_indices = tuple(
            index_by_action[scored.action]
            for scored in HeuristicRolloutPolicy().rank_actions(state)[:3]
        )

        result = select_early_search_candidates(
            state,
            config=EarlySearchConfig(
                beam_size=8,
                draw_safe_candidates=True,
                draw_baseline_keep=3,
                draw_safety_keep=2,
            ),
        )

        selected = {candidate.action_index for candidate in result.candidates}
        self.assertTrue(set(expected_indices).issubset(selected))
        for index in expected_indices:
            candidate = next(candidate for candidate in result.candidates if candidate.action_index == index)
            self.assertIn("baseline-keep", candidate.selection_reasons)

    def test_safe_draw_candidates_include_safety_candidates(self) -> None:
        state = solver_unsupported_top_pair_state()

        result = select_early_search_candidates(
            state,
            config=EarlySearchConfig(
                beam_size=8,
                draw_safe_candidates=True,
                draw_baseline_keep=1,
                draw_safety_keep=3,
            ),
        )

        self.assertTrue(any("safety-keep" in candidate.selection_reasons for candidate in result.candidates))
        self.assertTrue(any("top-flexibility-keep" in candidate.selection_reasons for candidate in result.candidates))

    def test_safe_draw_candidates_have_no_duplicate_actions(self) -> None:
        state = solver_unsupported_top_pair_state()

        result = select_early_search_candidates(
            state,
            config=EarlySearchConfig(
                beam_size=12,
                draw_safe_candidates=True,
                draw_baseline_keep=6,
                draw_safety_keep=6,
            ),
        )

        action_indices = tuple(candidate.action_index for candidate in result.candidates)
        self.assertEqual(len(action_indices), len(set(action_indices)))

    def test_safe_draw_candidate_ordering_is_deterministic(self) -> None:
        state = solver_unsupported_top_pair_state()
        config = EarlySearchConfig(
            beam_size=12,
            draw_safe_candidates=True,
            draw_baseline_keep=4,
            draw_safety_keep=4,
        )

        first = select_early_search_candidates(state, config=config)
        second = select_early_search_candidates(state, config=config)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
