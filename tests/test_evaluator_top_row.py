from __future__ import annotations

import unittest

from ofc.evaluator import HandCategory, compare_cross_rows_for_foul, evaluate_top_row
from tests.helpers import cards


class TopRowEvaluatorTest(unittest.TestCase):
    def test_top_row_only_supports_high_card_pair_and_trips(self) -> None:
        self.assertEqual(HandCategory.HIGH_CARD, evaluate_top_row(cards("As Ks Qs")).category)
        self.assertEqual(HandCategory.ONE_PAIR, evaluate_top_row(cards("Qh Qd 2s")).category)
        self.assertEqual(HandCategory.THREE_OF_A_KIND, evaluate_top_row(cards("7h 7d 7s")).category)

    def test_cross_row_foul_pair_kicker_example(self) -> None:
        top = cards("Kh Kd Ac")
        middle = cards("Ks Kc Qh 7d 3s")
        self.assertEqual(1, compare_cross_rows_for_foul(top, middle))

    def test_five_card_row_wins_when_equal_prefix_extends_past_top_row(self) -> None:
        top = cards("Kh Kd Qc")
        middle = cards("Ks Kc Qh 7d 3s")
        self.assertEqual(-1, compare_cross_rows_for_foul(top, middle))


if __name__ == "__main__":
    unittest.main()
