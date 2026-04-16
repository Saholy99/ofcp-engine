from __future__ import annotations

from collections import Counter
import random
import unittest

from ofc.engine import new_hand, new_match
from ofc.state import PlayerId
from ofc.transitions import validate_action
from ofc_solver.rollout_policy import RandomRolloutPolicy
from tests.helpers import physical_cards_in_state, stacked_deck_tokens


FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
]


class SolverRolloutPolicyTest(unittest.TestCase):
    def test_random_policy_selects_engine_legal_normal_action(self) -> None:
        state = new_match(button=PlayerId.PLAYER_0, seed=1)
        policy = RandomRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(3))

        validate_action(state, action)

    def test_random_policy_samples_fantasyland_set_without_exhaustive_enumeration(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )
        policy = RandomRolloutPolicy()

        action = policy.choose_action(state, rng=random.Random(4))

        validate_action(state, action)
        self.assertEqual(13, len(action.placements))
        self.assertEqual(Counter({"top": 3, "middle": 5, "bottom": 5}), Counter(p.row.value for p in action.placements))
        self.assertEqual(set(physical_cards_in_state(state)[:14]), {p.card for p in action.placements} | {action.discard})


if __name__ == "__main__":
    unittest.main()
