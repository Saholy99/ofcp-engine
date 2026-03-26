from __future__ import annotations

import unittest

from ofc.board import RowName
from ofc.evaluator import HandCategory, compare_same_size_rows, evaluate_five_card_row
from tests.helpers import cards


class FiveCardEvaluatorTest(unittest.TestCase):
    def test_all_five_card_categories(self) -> None:
        cases = [
            ("As Ks Qs Js Ts", HandCategory.ROYAL_FLUSH),
            ("9s 8s 7s 6s 5s", HandCategory.STRAIGHT_FLUSH),
            ("Ah Ad Ac As 2d", HandCategory.FOUR_OF_A_KIND),
            ("Kh Kd Ks 2c 2d", HandCategory.FULL_HOUSE),
            ("As Js 8s 5s 2s", HandCategory.FLUSH),
            ("5s 4d 3h 2c Ad", HandCategory.STRAIGHT),
            ("Qh Qd Qs 9c 2d", HandCategory.THREE_OF_A_KIND),
            ("Jh Jd 4s 4c 9d", HandCategory.TWO_PAIR),
            ("8h 8d Ks 4c 2d", HandCategory.ONE_PAIR),
            ("As Kd 9h 5c 2d", HandCategory.HIGH_CARD),
        ]
        for tokens, category in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(category, evaluate_five_card_row(cards(tokens)).category)

    def test_wheel_straight_uses_five_high_tiebreak(self) -> None:
        value = evaluate_five_card_row(cards("5s 4d 3h 2c Ad"))
        self.assertEqual(HandCategory.STRAIGHT, value.category)
        self.assertEqual((5,), value.tiebreak)

    def test_pair_tiebreaker_uses_kickers(self) -> None:
        stronger = cards("Kh Kd As 7c 3d")
        weaker = cards("Ks Kc Qh 7d 3s")
        self.assertEqual(1, compare_same_size_rows(stronger, weaker, RowName.MIDDLE))


if __name__ == "__main__":
    unittest.main()
