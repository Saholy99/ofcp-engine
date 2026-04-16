from __future__ import annotations

from itertools import islice
import unittest

from ofc.actions import PlaceDrawAction
from ofc.board import RowName
from ofc.engine import new_match
from ofc.state import PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import decode_action, encode_action, encode_actions
from tests.helpers import placements, stacked_deck_tokens


INITIAL_PREFIX = [
    "Ac", "9d", "Kh", "Kd", "Ah",
    "Ks", "8c", "Jh", "Jd", "Qh",
]


class AnalysisActionCodecTest(unittest.TestCase):
    def test_encode_actions_uses_stable_indices_and_raw_player_ids(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(INITIAL_PREFIX))

        encoded = encode_actions(tuple(islice(legal_actions(state), 2)))

        self.assertEqual((0, 1), (encoded[0].action_index, encoded[1].action_index))
        self.assertEqual("place_initial_five", encoded[0].action_type)
        self.assertEqual("player_1", encoded[0].payload["player_id"])
        self.assertEqual(
            [
                {"row": "top", "card": "Ac"},
                {"row": "top", "card": "9d"},
                {"row": "top", "card": "Kh"},
                {"row": "middle", "card": "Kd"},
                {"row": "middle", "card": "Ah"},
            ],
            encoded[0].payload["placements"],
        )

    def test_decode_action_round_trips_draw_action(self) -> None:
        action = PlaceDrawAction(
            player_id=PlayerId.PLAYER_0,
            placements=placements([(RowName.TOP, "Ac"), (RowName.MIDDLE, "Kd")]),
            discard=placements([(RowName.TOP, "Qh")])[0].card,
        )

        encoded = encode_action(7, action)
        decoded = decode_action(encoded)

        self.assertEqual(7, encoded.action_index)
        self.assertEqual("place_draw", encoded.action_type)
        self.assertEqual("player_0", encoded.payload["player_id"])
        self.assertEqual("player_0", decoded.player_id)
        self.assertEqual(action.placements, decoded.placements)
        self.assertEqual(action.discard, decoded.discard)


if __name__ == "__main__":
    unittest.main()
