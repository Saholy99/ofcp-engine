from __future__ import annotations

import unittest

from ofc.scoring import royalties_for_board, score_rows, score_terminal
from tests.helpers import make_board


class ScoringTest(unittest.TestCase):
    def test_royalties_lookup_across_rows(self) -> None:
        board = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd Ks 4c 3d",
            bottom="9c Tc Jc Qc Kc",
        )
        self.assertEqual(24, royalties_for_board(board))

    def test_row_ties_score_zero(self) -> None:
        left = make_board(
            top="Ac Kd 7s",
            middle="Jh Jd 9s 5c 2d",
            bottom="Qh Qd 8s 6c 4d",
        )
        right = make_board(
            top="As Kh 7c",
            middle="Js Jc 9h 5d 2c",
            bottom="Qs Qc 8h 6d 4c",
        )
        outcome = score_rows(left, right)
        self.assertEqual((0, 0, 0), (outcome.top, outcome.middle, outcome.bottom))

    def test_sweep_scoring_with_royalties(self) -> None:
        left = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd Ks 4c 3d",
            bottom="9c Tc Jc Qc Kc",
        )
        right = make_board(
            top="Ks 8c 3s",
            middle="Jh Jd 9c 5d 2c",
            bottom="Qs Qc As 7c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(30, result.left.total_points)
        self.assertEqual(-30, result.right.total_points)

    def test_one_player_foul_is_zero_sum_and_keeps_winner_royalties(self) -> None:
        legal = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd Ks 4c 3d",
            bottom="9c Tc Jc Qc Kc",
        )
        foul = make_board(
            top="Ah Ad Kc",
            middle="As Ac Qd 7s 3c",
            bottom="Js Jc 9d 5h 2c",
        )
        result = score_terminal("player_0", legal, "player_1", foul)
        self.assertEqual(30, result.left.total_points)
        self.assertEqual(-30, result.right.total_points)

    def test_both_players_foul_score_zero(self) -> None:
        left = make_board(
            top="Ah Ad Kc",
            middle="As Ac Qd 7s 3c",
            bottom="Js Jc 9d 5h 2c",
        )
        right = make_board(
            top="Kh Kd Ac",
            middle="Ks Kc Qh 7d 3s",
            bottom="Jh Jd 9h 6c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(0, result.left.total_points)
        self.assertEqual(0, result.right.total_points)


if __name__ == "__main__":
    unittest.main()
