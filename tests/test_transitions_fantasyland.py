from __future__ import annotations

import unittest

from ofc.actions import PlaceInitialFiveAction, SetFantasylandHandAction
from ofc.board import RowName
from ofc.deck import make_deck
from ofc.engine import new_hand, showdown
from ofc.state import HandPhase, PlayerId, get_player
from ofc.transitions import advance_after_showdown, apply_action
from tests.helpers import make_board, placements, showdown_state, stacked_deck_tokens


PLAYER_1_FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
    "Qc", "9d", "Jh", "Jd", "Qh",
    "9s", "8d", "7c",
]

DOUBLE_FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
    "Tc", "9c", "6s", "Jh", "Jd", "8d", "5c", "2d", "Qh", "Qd", "Ts", "4s", "3d", "7c",
]


class FantasylandTransitionsTest(unittest.TestCase):
    def test_one_player_fantasyland_uses_standard_turn_order_and_concealment(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(PLAYER_1_FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        self.assertEqual(PlayerId.PLAYER_1, state.acting_player)
        self.assertEqual(HandPhase.FANTASYLAND_SET, state.phase)
        self.assertEqual(14, len(get_player(state, PlayerId.PLAYER_1).current_private_draw))

        fantasyland_action = SetFantasylandHandAction(
            player_id=PlayerId.PLAYER_1,
            placements=placements(
                [
                    (RowName.TOP, "7h"),
                    (RowName.TOP, "7d"),
                    (RowName.TOP, "7s"),
                    (RowName.MIDDLE, "Kh"),
                    (RowName.MIDDLE, "Kd"),
                    (RowName.MIDDLE, "Ks"),
                    (RowName.MIDDLE, "2h"),
                    (RowName.MIDDLE, "2s"),
                    (RowName.BOTTOM, "Ah"),
                    (RowName.BOTTOM, "Ad"),
                    (RowName.BOTTOM, "Ac"),
                    (RowName.BOTTOM, "As"),
                    (RowName.BOTTOM, "3c"),
                ]
            ),
            discard=placements([(RowName.TOP, "4c")])[0].card,
        )
        state = apply_action(state, fantasyland_action)
        player_1 = get_player(state, PlayerId.PLAYER_1)
        self.assertEqual(0, len(player_1.board.top))
        self.assertIsNotNone(player_1.concealed_fantasyland_board)
        self.assertEqual(PlayerId.PLAYER_0, state.acting_player)
        self.assertEqual(HandPhase.INITIAL_DEAL, state.phase)
        self.assertEqual(5, len(get_player(state, PlayerId.PLAYER_0).current_private_draw))

        state = apply_action(
            state,
            PlaceInitialFiveAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements(
                    [
                        (RowName.TOP, "Qc"),
                        (RowName.TOP, "9d"),
                        (RowName.MIDDLE, "Jh"),
                        (RowName.MIDDLE, "Jd"),
                        (RowName.BOTTOM, "Qh"),
                    ]
                ),
            ),
        )
        self.assertEqual(HandPhase.DRAW, state.phase)
        self.assertEqual(PlayerId.PLAYER_0, state.acting_player)

    def test_both_players_fantasyland_go_directly_to_showdown_after_two_set_actions(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(True, True),
            preset_order=stacked_deck_tokens(DOUBLE_FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        state = apply_action(
            state,
            SetFantasylandHandAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements(
                    [
                        (RowName.TOP, "7h"),
                        (RowName.TOP, "7d"),
                        (RowName.TOP, "7s"),
                        (RowName.MIDDLE, "Kh"),
                        (RowName.MIDDLE, "Kd"),
                        (RowName.MIDDLE, "Ks"),
                        (RowName.MIDDLE, "2h"),
                        (RowName.MIDDLE, "2s"),
                        (RowName.BOTTOM, "Ah"),
                        (RowName.BOTTOM, "Ad"),
                        (RowName.BOTTOM, "Ac"),
                        (RowName.BOTTOM, "As"),
                        (RowName.BOTTOM, "3c"),
                    ]
                ),
                discard=placements([(RowName.TOP, "4c")])[0].card,
            ),
        )
        self.assertEqual(HandPhase.FANTASYLAND_SET, state.phase)
        self.assertEqual(PlayerId.PLAYER_0, state.acting_player)
        state = apply_action(
            state,
            SetFantasylandHandAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements(
                    [
                        (RowName.TOP, "Tc"),
                        (RowName.TOP, "9c"),
                        (RowName.TOP, "6s"),
                        (RowName.MIDDLE, "Jh"),
                        (RowName.MIDDLE, "Jd"),
                        (RowName.MIDDLE, "8d"),
                        (RowName.MIDDLE, "5c"),
                        (RowName.MIDDLE, "2d"),
                        (RowName.BOTTOM, "Qh"),
                        (RowName.BOTTOM, "Qd"),
                        (RowName.BOTTOM, "Ts"),
                        (RowName.BOTTOM, "7c"),
                        (RowName.BOTTOM, "4s"),
                    ]
                ),
                discard=placements([(RowName.TOP, "3d")])[0].card,
            ),
        )
        self.assertEqual(HandPhase.SHOWDOWN, state.phase)

    def test_button_freezes_during_continuation_and_rotates_after_exit(self) -> None:
        normal_state = showdown_state(
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="Qh Qd 2s", middle="Kh Kd 9h 5d 2c", bottom="Ah As 8h 6d 4c"),
            button=PlayerId.PLAYER_0,
            continuation_hand=False,
        )
        terminal_state, result = showdown(normal_state)
        next_state = advance_after_showdown(terminal_state, result, make_deck(seed=1))
        self.assertEqual(PlayerId.PLAYER_0, next_state.button)
        self.assertTrue(next_state.is_continuation_hand)

        continuation_state = showdown_state(
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="As Kc 7h", middle="Jh Jd 9h 5d 2c", bottom="Qh Qd 8h 6d 4c"),
            button=PlayerId.PLAYER_0,
            right_fantasyland_active=True,
            right_concealed=True,
            continuation_hand=True,
        )
        terminal_state, result = showdown(continuation_state)
        next_state = advance_after_showdown(terminal_state, result, make_deck(seed=2))
        self.assertEqual(PlayerId.PLAYER_1, next_state.button)
        self.assertFalse(next_state.is_continuation_hand)

    def test_showdown_reveals_concealed_fantasyland_board_for_scoring(self) -> None:
        state = showdown_state(
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="7h 7d 7s", middle="Kh Kd Ks 2c 2d", bottom="Ah Ad Ac As 3c"),
            button=PlayerId.PLAYER_0,
            right_fantasyland_active=True,
            right_concealed=True,
            continuation_hand=True,
        )
        terminal_state, result = showdown(state)
        self.assertEqual(HandPhase.TERMINAL, terminal_state.phase)
        self.assertGreater(result.right.royalties, 0)


if __name__ == "__main__":
    unittest.main()
