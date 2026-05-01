from __future__ import annotations

import unittest

from ofc.board import RowName
from ofc.cards import format_card
from ofc.transitions import legal_actions
from ofc_analysis.scenario import load_scenario
from ofc_solver.root_action_risk import score_root_action
from tests.helpers import (
    solver_final_draw_state,
    solver_middle_over_bottom_pressure_state,
    solver_unsupported_top_pair_state,
    solver_unsupported_top_trips_state,
)


class SolverRootActionRiskTest(unittest.TestCase):
    def test_penalizes_unsupported_top_qq_pair_at_root(self) -> None:
        state = solver_unsupported_top_pair_state()
        bad_action = _first_action_placing(state, RowName.TOP, "Qd")
        safer_action = _first_action_not_placing(state, RowName.TOP, "Qd")

        bad = score_root_action(state, bad_action)
        safer = score_root_action(state, safer_action)

        self.assertLess(bad.contribution, 0.0)
        self.assertIn("unsupported-top-pair", bad.reasons)
        self.assertLess(bad.contribution, safer.contribution)

    def test_penalizes_unsupported_top_trips_more_than_pair_pressure(self) -> None:
        state = solver_unsupported_top_trips_state()
        bad_action = _first_action_placing(state, RowName.TOP, "Qs")

        assessment = score_root_action(state, bad_action)

        self.assertLessEqual(assessment.contribution, -4.0)
        self.assertIn("unsupported-top-trips", assessment.reasons)

    def test_penalizes_middle_outpacing_bottom_support(self) -> None:
        state = solver_middle_over_bottom_pressure_state()
        pressured_action = _first_action_placing(state, RowName.MIDDLE, "As")

        assessment = score_root_action(state, pressured_action)

        self.assertLess(assessment.contribution, 0.0)
        self.assertIn("middle-over-bottom-pressure", assessment.reasons)

    def test_scores_initial_deal_top_slot_pressure(self) -> None:
        state = load_scenario("scenarios/regression/initial_deal.json").state
        top_full_action = tuple(legal_actions(state))[0]

        assessment = score_root_action(state, top_full_action)

        self.assertLess(assessment.contribution, 0.0)
        self.assertIn("top-slots-closed", assessment.reasons)

    def test_skips_final_draw_roots(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)
        action = tuple(legal_actions(state))[0]

        assessment = score_root_action(state, action)

        self.assertEqual(0.0, assessment.contribution)
        self.assertEqual((), assessment.reasons)


def _first_action_placing(state, row: RowName, token: str):
    return next(action for action in legal_actions(state) if _action_places_card(action, row, token))


def _first_action_not_placing(state, row: RowName, token: str):
    return next(action for action in legal_actions(state) if not _action_places_card(action, row, token))


def _action_places_card(action, row: RowName, token: str) -> bool:
    return any(placement.row is row and format_card(placement.card) == token for placement in action.placements)


if __name__ == "__main__":
    unittest.main()
