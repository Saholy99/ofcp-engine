from __future__ import annotations

import unittest

from ofc.fantasyland import (
    qualifies_for_fantasyland,
    qualifies_to_stay_in_fantasyland,
    resolve_next_hand_fantasyland_flags,
)
from tests.helpers import make_board


class FantasylandTest(unittest.TestCase):
    def test_legal_qq_plus_enters_fantasyland(self) -> None:
        board = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd 8s 6c 2d",
            bottom="Ah Ad Qs Tc 3h",
        )
        self.assertTrue(qualifies_for_fantasyland(board))

    def test_fouled_hand_cannot_enter_fantasyland(self) -> None:
        board = make_board(
            top="Qh Qd As",
            middle="Qs Qc Jh 9d 3c",
            bottom="Ah Ad Ks 7c 4d",
        )
        self.assertFalse(qualifies_for_fantasyland(board))

    def test_stay_conditions_work_for_legal_fantasyland_hand(self) -> None:
        board = make_board(
            top="7h 7d 7s",
            middle="Kh Kd Ks 2c 2d",
            bottom="Ah Ad Ac As 3c",
        )
        self.assertTrue(qualifies_to_stay_in_fantasyland(board))

    def test_resolve_next_hand_flags_uses_entry_or_stay_rules(self) -> None:
        normal_entry = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd 8s 6c 2d",
            bottom="Ah Ad Qs Tc 3h",
        )
        fantasyland_exit = make_board(
            top="Ac Kd 7s",
            middle="Jh Jd 9s 5c 2d",
            bottom="Qh Qd 8s 6c 4d",
        )
        self.assertEqual((True, False), resolve_next_hand_fantasyland_flags((False, True), (normal_entry, fantasyland_exit)))


if __name__ == "__main__":
    unittest.main()
