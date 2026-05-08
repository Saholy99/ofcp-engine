from __future__ import annotations

import random
import unittest

from ofc.engine import showdown
from ofc.state import HandPhase, PlayerId
from ofc.transitions import apply_action, legal_actions
from ofc_solver.late_search import (
    LateSearchConfig,
    evaluate_late_root_action,
    rank_late_root_actions,
)
from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state
from ofc_analysis.observation import project_observation
from tests.helpers import solver_final_draw_state, solver_late_street_exact_search_state


def _terminal_value(state, perspective: PlayerId) -> float:
    _, result = showdown(state)
    if result.left.player_id == perspective.value:
        return float(result.left.total_points)
    return float(result.right.total_points)


def _brute_force_value(state, perspective: PlayerId) -> float:
    if state.phase in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        return _terminal_value(state, perspective)
    values = tuple(_brute_force_value(apply_action(state, action), perspective) for action in legal_actions(state))
    if state.acting_player == perspective:
        return max(values)
    return min(values)


class SolverLateSearchTest(unittest.TestCase):
    def test_exact_late_search_matches_bruteforce_on_small_tree(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_late_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(51),
            config=LateSearchConfig(mode="exact", max_depth=4, max_nodes=500),
        )

        self.assertTrue(result.activated)
        self.assertEqual("exact", result.mode)
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(_brute_force_value(apply_action(state, action), PlayerId.PLAYER_1), result.value)
        self.assertGreater(result.nodes_searched, 0)
        self.assertGreater(result.terminal_evaluations, 0)

    def test_node_limit_falls_back_to_heuristic(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_late_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(52),
            config=LateSearchConfig(mode="exact", max_depth=4, max_nodes=1),
        )

        self.assertFalse(result.activated)
        self.assertEqual("fallback", result.mode)
        self.assertEqual("exact-budget-exceeded", result.fallback_reason)
        self.assertLessEqual(result.nodes_searched, 1)

    def test_auto_mode_uses_beam_when_exact_budget_is_exceeded(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_late_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(53),
            config=LateSearchConfig(mode="auto", max_depth=4, max_nodes=5, beam_size=2),
        )

        self.assertTrue(result.activated)
        self.assertEqual("beam", result.mode)
        self.assertGreater(result.nodes_searched, 0)
        self.assertGreaterEqual(result.candidate_count, 1)

    def test_beam_limit_can_fall_back_to_heuristic(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_late_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(54),
            config=LateSearchConfig(mode="beam", max_depth=0, max_nodes=50, beam_size=2),
        )

        self.assertFalse(result.activated)
        self.assertEqual("fallback", result.mode)
        self.assertEqual("beam-depth-exceeded", result.fallback_reason)

    def test_rank_late_root_actions_is_deterministic(self) -> None:
        state = solver_late_street_exact_search_state()
        config = LateSearchConfig(mode="auto", max_depth=4, max_nodes=500, beam_size=2)

        first = rank_late_root_actions(state, perspective=PlayerId.PLAYER_1, rng=random.Random(55), config=config)
        second = rank_late_root_actions(state, perspective=PlayerId.PLAYER_1, rng=random.Random(55), config=config)

        self.assertEqual(first, second)

    def test_rank_actions_from_state_exposes_late_search_metadata(self) -> None:
        state = solver_late_street_exact_search_state()

        analysis = rank_actions_from_state(
            state,
            observer=PlayerId.PLAYER_1,
            rollouts_per_action=1,
            rng_seed=56,
            late_search=True,
            late_search_config=LateSearchConfig(mode="auto", max_depth=4, max_nodes=500),
        )

        self.assertTrue(analysis.late_search_enabled)
        self.assertTrue(any(estimate.late_search_activated for estimate in analysis.ranked_actions))
        self.assertTrue(all(estimate.late_search_nodes >= 0 for estimate in analysis.ranked_actions))

    def test_observation_late_search_uses_sampled_state_without_leaking_hidden_output(self) -> None:
        state = solver_late_street_exact_search_state()
        observation = project_observation(state, PlayerId.PLAYER_1)

        analysis = rank_actions_from_observation(
            observation,
            rollouts_per_action=1,
            rng_seed=57,
            late_search=True,
            late_search_config=LateSearchConfig(mode="auto", max_depth=4, max_nodes=500),
        )

        self.assertTrue(analysis.late_search_enabled)
        self.assertEqual(len(tuple(legal_actions(state))), analysis.total_legal_actions)
        rendered_actions = [estimate.action.as_dict() for estimate in analysis.ranked_actions]
        self.assertTrue(all("hidden_discards" not in str(action) for action in rendered_actions))


if __name__ == "__main__":
    unittest.main()
