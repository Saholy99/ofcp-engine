from __future__ import annotations

import unittest

from ofc.actions import PlaceDrawAction, SetFantasylandHandAction
from ofc.board import RowName
from ofc.engine import new_hand, new_match
from ofc.state import HandPhase, PlayerId
from ofc.transitions import apply_action, legal_actions, validate_action
from ofc_analysis.action_codec import encode_action
from ofc_analysis.play import (
    NoopSuggestionBackend,
    _parse_hero_action_choice,
    parse_cards_input,
    parse_manual_action,
    parse_opponent_visible_action,
    run_play_hand,
    select_action_by_index,
    set_current_private_draw,
)
from ofc_solver.models import MoveEstimate
from tests.helpers import cards, stacked_deck_tokens


FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
]


class AnalysisPlayTest(unittest.TestCase):
    def test_parse_cards_input_validates_count_and_uniqueness(self) -> None:
        self.assertEqual(cards("Ac Kd 2s"), parse_cards_input("Ac Kd 2s", expected_count=3))
        with self.assertRaisesRegex(ValueError, "Expected exactly 3 cards"):
            parse_cards_input("Ac Kd", expected_count=3)
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_cards_input("Ac Ac 2s", expected_count=3)

    def test_set_current_private_draw_rejects_already_used_cards(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(["Ac", "Kd", "2s", "3h", "4c"]))
        state = set_current_private_draw(state, PlayerId.PLAYER_1, cards("Ac Kd 2s 3h 4c"))
        state = apply_action(state, parse_manual_action(state, "top top middle middle bottom"))

        with self.assertRaisesRegex(ValueError, "already used"):
            set_current_private_draw(state, PlayerId.PLAYER_0, cards("Ac Qd Jh Ts 9c"))

    def test_parse_manual_draw_action_maps_assignments_to_current_draw_order(self) -> None:
        state = new_match(button=PlayerId.PLAYER_1, seed=1)
        state = set_current_private_draw(state, PlayerId.PLAYER_0, cards("Ac Kd 2s 3h 4c"))
        initial_action = parse_manual_action(state, "top top middle middle bottom")
        state = apply_action(state, initial_action)
        state = set_current_private_draw(state, PlayerId.PLAYER_1, cards("Qs Jd Th 9c 8s"))
        other_initial_action = parse_manual_action(state, "top top middle middle bottom")
        state = apply_action(state, other_initial_action)
        state = set_current_private_draw(state, PlayerId.PLAYER_0, cards("7c 6d 5h"))

        action = parse_manual_action(state, "bottom top discard")

        self.assertIsInstance(action, PlaceDrawAction)
        self.assertEqual(RowName.BOTTOM, action.placements[0].row)
        self.assertEqual(cards("5h")[0], action.discard)
        validate_action(state, action)

    def test_parse_opponent_visible_draw_action_uses_unknown_hidden_discard(self) -> None:
        state = new_match(button=PlayerId.PLAYER_1, seed=1)
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_0,
            cards("Ac Kd 2s 3h 4c"),
            hero_player=PlayerId.PLAYER_0,
        )
        state = apply_action(state, parse_manual_action(state, "top top middle middle bottom"))
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_1,
            cards("Qs Jd Th 9c 8s"),
            hero_player=PlayerId.PLAYER_0,
        )
        state = apply_action(
            state,
            parse_opponent_visible_action(state, "top top middle middle bottom", visible_count=5),
        )
        state = set_current_private_draw(state, PlayerId.PLAYER_0, cards("7c 6d 5h"), hero_player=PlayerId.PLAYER_0)
        state = apply_action(state, parse_manual_action(state, "bottom top discard"))
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_1,
            cards("9d 8d 2c"),
            hero_player=PlayerId.PLAYER_0,
        )

        action = parse_opponent_visible_action(state, "middle bottom", visible_count=2)

        self.assertIsInstance(action, PlaceDrawAction)
        self.assertEqual(cards("2c")[0], action.discard)
        validate_action(state, action)

    def test_hero_can_later_enter_card_previously_used_as_opponent_unknown_discard(self) -> None:
        state = new_match(button=PlayerId.PLAYER_1, seed=1)
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_0,
            cards("Ac Kd 2s 3h 4c"),
            hero_player=PlayerId.PLAYER_0,
        )
        state = apply_action(state, parse_manual_action(state, "top top middle middle bottom"))
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_1,
            cards("Qs Jd Th 9c 8s"),
            hero_player=PlayerId.PLAYER_0,
        )
        state = apply_action(
            state,
            parse_opponent_visible_action(state, "top top middle middle bottom", visible_count=5),
        )
        state = set_current_private_draw(state, PlayerId.PLAYER_0, cards("7c 6d 5h"), hero_player=PlayerId.PLAYER_0)
        state = apply_action(state, parse_manual_action(state, "bottom top discard"))
        state = set_current_private_draw(
            state,
            PlayerId.PLAYER_1,
            cards("9d 8d 2c"),
            hero_player=PlayerId.PLAYER_0,
        )
        state = apply_action(state, parse_opponent_visible_action(state, "middle bottom", visible_count=2))

        state = set_current_private_draw(state, PlayerId.PLAYER_0, cards("2c 7d 8h"), hero_player=PlayerId.PLAYER_0)

        self.assertEqual(cards("2c 7d 8h"), state.players[0].current_private_draw)

    def test_select_action_by_index_returns_engine_legal_action(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=2)
        action = select_action_by_index(state, 1)

        validate_action(state, action)

    def test_select_action_by_index_rejects_zero_based_input(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=2)

        with self.assertRaisesRegex(ValueError, "between 1 and"):
            select_action_by_index(state, 0)

    def test_bare_number_chooses_displayed_solver_suggestion_rank(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=2)
        actions = tuple(legal_actions(state))
        suggested_index = 2
        suggestion = MoveEstimate(
            action_index=suggested_index,
            action=encode_action(suggested_index, actions[suggested_index]),
            mean_value=1.0,
            stddev=0.0,
            sample_count=1,
            min_value=1.0,
            max_value=1.0,
        )

        action = _parse_hero_action_choice(state, (suggestion,), "1")

        self.assertEqual(actions[suggested_index], action)

    def test_action_prefix_chooses_legal_action_index_when_suggestions_exist(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=2)
        actions = tuple(legal_actions(state))
        suggested_index = 2
        suggestion = MoveEstimate(
            action_index=suggested_index,
            action=encode_action(suggested_index, actions[suggested_index]),
            mean_value=1.0,
            stddev=0.0,
            sample_count=1,
            min_value=1.0,
            max_value=1.0,
        )

        action = _parse_hero_action_choice(state, (suggestion,), "action 1")

        self.assertEqual(actions[0], action)

    def test_bare_number_rejects_out_of_range_suggestion_when_suggestions_exist(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=2)
        actions = tuple(legal_actions(state))
        suggested_index = 2
        suggestion = MoveEstimate(
            action_index=suggested_index,
            action=encode_action(suggested_index, actions[suggested_index]),
            mean_value=1.0,
            stddev=0.0,
            sample_count=1,
            min_value=1.0,
            max_value=1.0,
        )

        with self.assertRaisesRegex(ValueError, "Suggested rank"):
            _parse_hero_action_choice(state, (suggestion,), "2")

    def test_fantasyland_start_manual_set_action_is_supported_at_helper_level(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        state = set_current_private_draw(state, PlayerId.PLAYER_1, cards(" ".join(FANTASYLAND_PREFIX)))

        action = parse_manual_action(
            state,
            "top top top middle middle middle middle middle bottom bottom bottom bottom bottom discard",
        )

        self.assertEqual(HandPhase.FANTASYLAND_SET, state.phase)
        self.assertIsInstance(action, SetFantasylandHandAction)
        validate_action(state, action)

    def test_scripted_normal_hand_progresses_to_final_result(self) -> None:
        inputs = iter(
            [
                "Ac 9d Kh Kd Ah",
                "top top middle middle bottom",
                "Ks 8c Jh Jd Qh",
                "top top middle middle bottom",
                "Ad 4s",
                "bottom top",
                "Qc 3s 6h",
                "bottom top discard",
                "8s 6c",
                "middle middle",
                "9h 5d Td",
                "middle middle discard",
                "Qs Tc",
                "bottom bottom",
                "As 7c 5c",
                "bottom bottom discard",
                "2d 3h",
                "middle bottom",
                "2c 4d 6d",
                "middle bottom discard",
            ]
        )
        output: list[str] = []

        exit_code = run_play_hand(
            hero_player=PlayerId.PLAYER_0,
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, False),
            rollouts_per_action=1,
            rng_seed="scripted-test",
            input_func=lambda prompt: next(inputs),
            output_func=output.append,
            backend=NoopSuggestionBackend(),
        )

        self.assertEqual(0, exit_code)
        self.assertIn("Final Result", output)
        self.assertTrue(any("player_1: total=6" in line for line in output))
        self.assertTrue(any("next_hand_fantasyland: player_0=False player_1=False" in line for line in output))


if __name__ == "__main__":
    unittest.main()
