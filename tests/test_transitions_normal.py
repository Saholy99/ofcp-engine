from __future__ import annotations

import unittest

from ofc.actions import PlaceDrawAction, PlaceInitialFiveAction
from ofc.board import RowName
from ofc.engine import new_match, showdown
from ofc.state import HandPhase, PlayerId, all_known_cards, get_player
from ofc.transitions import apply_action, validate_action
from tests.helpers import placements, stacked_deck_tokens


NORMAL_HAND_PREFIX = [
    "Ac", "9d", "Kh", "Kd", "Ah",
    "Ks", "8c", "Jh", "Jd", "Qh",
    "Ad", "4s", "5s",
    "Qc", "3s", "6h",
    "8s", "6c", "7d",
    "9h", "5d", "Td",
    "Qs", "Tc", "Jc",
    "As", "7c", "5c",
    "2d", "3h", "4h",
    "2c", "4d", "6d",
]


class NormalTransitionsTest(unittest.TestCase):
    def test_new_match_deals_first_five_to_non_button_player(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(NORMAL_HAND_PREFIX))
        self.assertEqual(PlayerId.PLAYER_1, state.acting_player)
        self.assertEqual(HandPhase.INITIAL_DEAL, state.phase)
        self.assertEqual(5, len(get_player(state, PlayerId.PLAYER_1).current_private_draw))
        self.assertEqual(0, len(get_player(state, PlayerId.PLAYER_0).current_private_draw))

    def test_validate_action_rejects_wrong_player(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(NORMAL_HAND_PREFIX))
        bad_action = PlaceInitialFiveAction(
            player_id=PlayerId.PLAYER_0,
            placements=placements(
                [
                    (RowName.TOP, "Ac"),
                    (RowName.TOP, "9d"),
                    (RowName.MIDDLE, "Kh"),
                    (RowName.MIDDLE, "Kd"),
                    (RowName.BOTTOM, "Ah"),
                ]
            ),
        )
        with self.assertRaises(ValueError):
            validate_action(state, bad_action)

    def test_full_normal_hand_reaches_showdown(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(NORMAL_HAND_PREFIX))

        actions = [
            PlaceInitialFiveAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements(
                    [
                        (RowName.TOP, "Ac"),
                        (RowName.TOP, "9d"),
                        (RowName.MIDDLE, "Kh"),
                        (RowName.MIDDLE, "Kd"),
                        (RowName.BOTTOM, "Ah"),
                    ]
                ),
            ),
            PlaceInitialFiveAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements(
                    [
                        (RowName.TOP, "Ks"),
                        (RowName.TOP, "8c"),
                        (RowName.MIDDLE, "Jh"),
                        (RowName.MIDDLE, "Jd"),
                        (RowName.BOTTOM, "Qh"),
                    ]
                ),
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements([(RowName.BOTTOM, "Ad"), (RowName.TOP, "4s")]),
                discard=placements([(RowName.TOP, "5s")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements([(RowName.BOTTOM, "Qc"), (RowName.TOP, "3s")]),
                discard=placements([(RowName.TOP, "6h")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements([(RowName.MIDDLE, "8s"), (RowName.MIDDLE, "6c")]),
                discard=placements([(RowName.TOP, "7d")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements([(RowName.MIDDLE, "9h"), (RowName.MIDDLE, "5d")]),
                discard=placements([(RowName.TOP, "Td")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements([(RowName.BOTTOM, "Qs"), (RowName.BOTTOM, "Tc")]),
                discard=placements([(RowName.TOP, "Jc")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements([(RowName.BOTTOM, "As"), (RowName.BOTTOM, "7c")]),
                discard=placements([(RowName.TOP, "5c")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_1,
                placements=placements([(RowName.MIDDLE, "2d"), (RowName.BOTTOM, "3h")]),
                discard=placements([(RowName.TOP, "4h")])[0].card,
            ),
            PlaceDrawAction(
                player_id=PlayerId.PLAYER_0,
                placements=placements([(RowName.MIDDLE, "2c"), (RowName.BOTTOM, "4d")]),
                discard=placements([(RowName.TOP, "6d")])[0].card,
            ),
        ]

        for action in actions:
            state = apply_action(state, action)

        self.assertEqual(HandPhase.SHOWDOWN, state.phase)
        self.assertEqual(52, len(set(all_known_cards(state))))
        terminal_state, result = showdown(state)
        self.assertEqual(HandPhase.TERMINAL, terminal_state.phase)
        self.assertEqual(6, result.right.total_points)
        self.assertEqual(-6, result.left.total_points)


if __name__ == "__main__":
    unittest.main()
