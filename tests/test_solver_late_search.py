from __future__ import annotations

import random
import unittest

from ofc.engine import showdown
from ofc.state import HandPhase, PlayerId
from ofc.transitions import apply_action, legal_actions
from ofc_solver.late_search import (
    FinalDrawAutoSearchConfig,
    LateSearchConfig,
    assess_final_draw_auto_search,
    evaluate_final_draw_auto_root_action,
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

    def test_final_draw_auto_gate_activates_on_terminal_root_draw(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)
        action = tuple(legal_actions(state))[0]

        assessment = assess_final_draw_auto_search(
            state,
            action,
            config=FinalDrawAutoSearchConfig(max_depth=1, max_nodes=16),
        )
        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(58),
            config=FinalDrawAutoSearchConfig(max_depth=1, max_nodes=16),
        )

        self.assertTrue(assessment.eligible)
        self.assertEqual("eligible", assessment.reason)
        self.assertTrue(result.activated)
        self.assertTrue(result.phase_auto_search_activated)
        self.assertEqual("exact", result.mode)
        self.assertEqual(assessment.tree_nodes, result.phase_auto_search_tree_nodes)

    def test_final_draw_auto_gate_does_not_activate_on_mid_draw(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(59),
            config=FinalDrawAutoSearchConfig(max_depth=1, max_nodes=16),
        )

        self.assertFalse(result.activated)
        self.assertFalse(result.phase_auto_search_activated)
        self.assertIn(result.phase_auto_search_reason, {"tree-depth-exceeded", "tree-budget-exceeded"})

    def test_final_draw_auto_gate_falls_back_when_budget_is_exceeded(self) -> None:
        state = solver_late_street_exact_search_state()
        action = tuple(legal_actions(state))[0]

        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_1,
            rng=random.Random(60),
            config=FinalDrawAutoSearchConfig(max_depth=4, max_nodes=1),
        )

        self.assertFalse(result.activated)
        self.assertEqual("fallback", result.mode)
        self.assertFalse(result.phase_auto_search_activated)
        self.assertEqual("tree-budget-exceeded", result.phase_auto_search_reason)

    def test_rank_actions_from_state_exposes_final_draw_auto_metadata(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)

        analysis = rank_actions_from_state(
            state,
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=1,
            rng_seed=61,
            final_draw_auto_search=True,
            final_draw_auto_search_config=FinalDrawAutoSearchConfig(max_depth=1, max_nodes=16),
        )

        self.assertTrue(analysis.final_draw_auto_search_enabled)
        self.assertTrue(any(estimate.phase_auto_search_activated for estimate in analysis.ranked_actions))
        self.assertTrue(all(estimate.phase_auto_search_tree_nodes >= 0 for estimate in analysis.ranked_actions))

    def test_final_draw_auto_current_hand_only_remains_available(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)
        action = tuple(legal_actions(state))[1]

        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(62),
            config=FinalDrawAutoSearchConfig(max_depth=1, max_nodes=16),
        )

        self.assertTrue(result.activated)
        self.assertTrue(result.rollout_result.root_player_next_fantasyland)
        self.assertFalse(result.continuation_aware)
        self.assertEqual(0.0, result.continuation_value)
        self.assertEqual(result.current_hand_value, result.value)
        self.assertEqual(0, result.rollout_result.continuation_hands_simulated)

    def test_final_draw_auto_continuation_adds_one_immediate_fantasyland_hand(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)
        action = tuple(legal_actions(state))[1]

        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(63),
            config=FinalDrawAutoSearchConfig(
                max_depth=1,
                max_nodes=16,
                include_continuation=True,
                continuation_rollouts=1,
            ),
        )

        self.assertTrue(result.activated)
        self.assertTrue(result.continuation_aware)
        self.assertTrue(result.continuation_triggered)
        self.assertEqual(1, result.continuation_rollouts)
        self.assertEqual(1, result.rollout_result.continuation_hands_simulated)
        self.assertEqual(result.current_hand_value + result.continuation_value, result.value)
        self.assertEqual(result.value, result.rollout_result.total_value)

    def test_final_draw_auto_continuation_skips_when_fantasyland_is_not_triggered(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)
        action = tuple(legal_actions(state))[1]

        result = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(64),
            config=FinalDrawAutoSearchConfig(
                max_depth=1,
                max_nodes=16,
                include_continuation=True,
                continuation_rollouts=1,
            ),
        )

        self.assertTrue(result.activated)
        self.assertTrue(result.continuation_aware)
        self.assertFalse(result.continuation_triggered)
        self.assertEqual("no-fantasyland-continuation", result.continuation_reason)
        self.assertEqual(0.0, result.continuation_value)
        self.assertEqual(result.current_hand_value, result.value)
        self.assertEqual(0, result.rollout_result.continuation_hands_simulated)

    def test_final_draw_auto_continuation_is_deterministic_under_fixed_seed(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)
        action = tuple(legal_actions(state))[1]
        config = FinalDrawAutoSearchConfig(
            max_depth=1,
            max_nodes=16,
            include_continuation=True,
            continuation_rollouts=2,
        )

        first = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(65),
            config=config,
        )
        second = evaluate_final_draw_auto_root_action(
            state,
            action,
            perspective=PlayerId.PLAYER_0,
            rng=random.Random(65),
            config=config,
        )

        self.assertEqual(first.value, second.value)
        self.assertEqual(first.continuation_value, second.continuation_value)
        self.assertEqual(first.continuation_rollouts, second.continuation_rollouts)

    def test_observation_final_draw_auto_continuation_does_not_leak_hidden_output(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)
        observation = project_observation(state, PlayerId.PLAYER_0)

        analysis = rank_actions_from_observation(
            observation,
            rollouts_per_action=1,
            rng_seed=66,
            final_draw_auto_search=True,
            final_draw_auto_search_config=FinalDrawAutoSearchConfig(
                max_depth=1,
                max_nodes=16,
                include_continuation=True,
                continuation_rollouts=1,
            ),
        )

        self.assertTrue(analysis.final_draw_auto_search_enabled)
        self.assertTrue(analysis.final_draw_auto_include_continuation)
        rendered_actions = [estimate.action.as_dict() for estimate in analysis.ranked_actions]
        self.assertTrue(all("hidden_discards" not in str(action) for action in rendered_actions))


if __name__ == "__main__":
    unittest.main()
