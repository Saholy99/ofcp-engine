from __future__ import annotations

import unittest

from ofc.actions import PlaceDrawAction, PlaceInitialFiveAction, SetFantasylandHandAction
from ofc.board import RowName
from ofc.cards import parse_card
from tests.helpers import placements


class ActionsTest(unittest.TestCase):
    def test_place_initial_five_requires_five_unique_cards(self) -> None:
        with self.assertRaises(ValueError):
            PlaceInitialFiveAction(
                player_id="player_0",
                placements=placements(
                    [
                        (RowName.TOP, "Ah"),
                        (RowName.TOP, "Ah"),
                        (RowName.MIDDLE, "Kd"),
                        (RowName.MIDDLE, "Qc"),
                        (RowName.BOTTOM, "2s"),
                    ]
                ),
            )

    def test_place_draw_action_requires_three_unique_cards_and_two_placements(self) -> None:
        with self.assertRaises(ValueError):
            PlaceDrawAction(
                player_id="player_0",
                placements=(placements([(RowName.TOP, "Ah")])[0],),
                discard=parse_card("Kd"),
            )

    def test_fantasyland_set_requires_fourteen_unique_cards(self) -> None:
        with self.assertRaises(ValueError):
            SetFantasylandHandAction(
                player_id="player_1",
                placements=placements(
                    [(RowName.TOP, "Ah")] * 13
                ),
                discard=parse_card("Kd"),
            )


if __name__ == "__main__":
    unittest.main()
