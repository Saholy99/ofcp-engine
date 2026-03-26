from __future__ import annotations

import unittest

from ofc.board import Board, RowName, place_cards
from ofc.scoring import is_foul, royalties_for_board
from tests.helpers import cards, make_board


class BoardLegalityTest(unittest.TestCase):
    def test_place_cards_respects_capacity_and_duplicates(self) -> None:
        board = place_cards(Board(), ((RowName.TOP, cards("Ah")[0]), (RowName.TOP, cards("Kd")[0])))
        self.assertEqual(2, len(board.top))
        with self.assertRaises(ValueError):
            place_cards(board, ((RowName.TOP, cards("Qs")[0]), (RowName.TOP, cards("Jc")[0])))
        with self.assertRaises(ValueError):
            place_cards(board, ((RowName.MIDDLE, cards("Ah")[0]),))

    def test_is_foul_false_for_legal_board(self) -> None:
        board = make_board(
            top="Ac 9d 4s",
            middle="Kh Kd 8s 6c 2d",
            bottom="Ah Ad Qs Tc 3h",
        )
        self.assertFalse(is_foul(board))

    def test_is_foul_true_when_top_beats_middle(self) -> None:
        board = make_board(
            top="Kh Kd Ac",
            middle="Ks Kc Qh 7d 3s",
            bottom="Ah Ad Js 9c 2d",
        )
        self.assertTrue(is_foul(board))

    def test_fouled_board_receives_no_royalties(self) -> None:
        board = make_board(
            top="Qh Qd As",
            middle="Qs Qc Jh 9d 3c",
            bottom="Ah Ad As Ks 2h",
        )
        self.assertTrue(is_foul(board))
        self.assertEqual(0, royalties_for_board(board))


if __name__ == "__main__":
    unittest.main()
