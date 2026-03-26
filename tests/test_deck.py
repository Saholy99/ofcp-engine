from __future__ import annotations

import unittest

from ofc.deck import draw_n, make_deck
from tests.helpers import stacked_deck_tokens


class DeckTest(unittest.TestCase):
    def test_make_deck_with_seed_is_reproducible(self) -> None:
        first = make_deck(seed=7)
        second = make_deck(seed=7)
        self.assertEqual(first.undealt_cards, second.undealt_cards)

    def test_make_deck_with_preset_order_uses_exact_order(self) -> None:
        preset = stacked_deck_tokens(["Ah", "Kd", "2c"])
        deck = make_deck(preset_order=preset)
        self.assertEqual(["Ah", "Kd", "2c"], [str(card) for card in deck.undealt_cards[:3]])

    def test_draw_n_removes_cards_without_duplication(self) -> None:
        deck = make_deck(seed=11)
        drawn, remaining = draw_n(deck, 7)
        self.assertEqual(7, len(drawn))
        self.assertEqual(45, remaining.cards_remaining)
        self.assertEqual(52, len(set(drawn + remaining.undealt_cards)))

    def test_make_deck_rejects_duplicate_preset_order(self) -> None:
        bad_deck = ["Ah", "Ah"] + stacked_deck_tokens(["Kd"])[:50]
        with self.assertRaises(ValueError):
            make_deck(preset_order=bad_deck)


if __name__ == "__main__":
    unittest.main()
