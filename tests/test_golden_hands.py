from __future__ import annotations

import unittest

from ofc.engine import showdown
from ofc.fantasyland import resolve_next_hand_fantasyland_flags
from ofc.scoring import score_terminal
from tests.helpers import make_board, showdown_state


class GoldenHandsTest(unittest.TestCase):
    def test_legal_hand_with_no_royalties(self) -> None:
        left = make_board(
            top="Ac 9d 4s",
            middle="Kh Kd 8s 6c 2d",
            bottom="Ah Ad Qs Tc 3h",
        )
        right = make_board(
            top="Ks 8c 3s",
            middle="Jh Jd 9h 5d 2c",
            bottom="Qh Qc As 7c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(6, result.left.total_points)

    def test_legal_hand_with_royalties(self) -> None:
        left = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd Ks 4c 3d",
            bottom="9c Tc Jc Qc Kc",
        )
        right = make_board(
            top="Ks 8c 3s",
            middle="Jh Jd 9h 5d 2c",
            bottom="Qh Qc As 7c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(30, result.left.total_points)

    def test_row_ties(self) -> None:
        left = make_board(
            top="Ac Kd 7s",
            middle="Jh Jd 9s 5c 2d",
            bottom="Qh Qd 8s 6c 4d",
        )
        right = make_board(
            top="As Kh 7c",
            middle="Th Td 9h 5d 2c",
            bottom="Ah Ad 8h 6d 4c",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(0, result.left.total_points)
        self.assertEqual(0, result.right.total_points)

    def test_sweep_without_royalties(self) -> None:
        left = make_board(
            top="Ac 9d 4s",
            middle="Kh Kd 8s 6c 2d",
            bottom="Ah Ad Qs Tc 3h",
        )
        right = make_board(
            top="Ks 8c 3s",
            middle="Jh Jd 9h 5d 2c",
            bottom="Qh Qc As 7c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(6, result.left.total_points)

    def test_sweep_with_royalties(self) -> None:
        left = make_board(
            top="Qh Qd 2s",
            middle="Kh Kd Ks 4c 3d",
            bottom="9c Tc Jc Qc Kc",
        )
        right = make_board(
            top="Ks 8c 3s",
            middle="Jh Jd 9h 5d 2c",
            bottom="Qh Qc As 7c 4d",
        )
        result = score_terminal("player_0", left, "player_1", right)
        self.assertEqual(30, result.left.total_points)

    def test_one_player_foul(self) -> None:
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

    def test_both_player_foul(self) -> None:
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

    def test_fantasyland_entry(self) -> None:
        boards = (
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="Qh Qd 2s", middle="Kh Kd 9h 5d 2c", bottom="Ah As 8h 6d 4c"),
        )
        self.assertEqual((False, True), resolve_next_hand_fantasyland_flags((False, False), boards))

    def test_fantasyland_stay(self) -> None:
        boards = (
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="7h 7d 7s", middle="Kh Kd Ks 2c 2d", bottom="Ah Ad Ac As 3c"),
        )
        self.assertEqual((False, True), resolve_next_hand_fantasyland_flags((False, True), boards))

    def test_fantasyland_exit(self) -> None:
        boards = (
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="As Kc 7h", middle="Jh Jd 9h 5d 2c", bottom="Qh Qd 8h 6d 4c"),
        )
        self.assertEqual((False, False), resolve_next_hand_fantasyland_flags((False, True), boards))

    def test_concealed_fantasyland_reveals_at_showdown(self) -> None:
        state = showdown_state(
            make_board(top="Ac 9d 4s", middle="Kh Kd 8s 6c 2d", bottom="Ah Ad Qs Tc 3h"),
            make_board(top="7h 7d 7s", middle="Kh Kd Ks 2c 2d", bottom="Ah Ad Ac As 3c"),
            right_fantasyland_active=True,
            right_concealed=True,
            continuation_hand=True,
        )
        _, result = showdown(state)
        self.assertGreater(result.right.total_points, 0)


if __name__ == "__main__":
    unittest.main()
