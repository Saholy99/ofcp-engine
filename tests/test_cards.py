from __future__ import annotations

import unittest

from ofc.cards import format_card, full_deck, parse_card


class CardsTest(unittest.TestCase):
    def test_parse_and_format_round_trip(self) -> None:
        card = parse_card("Ah")
        self.assertEqual("Ah", format_card(card))

    def test_full_deck_has_52_unique_cards_in_canonical_order(self) -> None:
        deck = full_deck()
        self.assertEqual(52, len(deck))
        self.assertEqual(52, len(set(deck)))
        self.assertEqual("2c", format_card(deck[0]))
        self.assertEqual("As", format_card(deck[-1]))

    def test_parse_card_rejects_invalid_tokens(self) -> None:
        with self.assertRaises(ValueError):
            parse_card("10h")
        with self.assertRaises(ValueError):
            parse_card("1x")


if __name__ == "__main__":
    unittest.main()
