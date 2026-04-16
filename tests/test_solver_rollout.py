from __future__ import annotations

import random
import unittest

from ofc.actions import PlaceDrawAction
from ofc.board import RowName
from ofc.engine import new_hand
from ofc.state import PlayerId
from ofc_solver.rollout import run_rollout
from ofc_solver.rollout_policy import RandomRolloutPolicy
from tests.helpers import placements, solver_final_draw_state, stacked_deck_tokens


FANTASYLAND_PREFIX = [
    "7h", "7d", "7s", "Kh", "Kd", "Ks", "2h", "2s", "Ah", "Ad", "Ac", "As", "3c", "4c",
]


def _final_draw_action() -> PlaceDrawAction:
    return PlaceDrawAction(
        player_id=PlayerId.PLAYER_0,
        placements=placements([(RowName.TOP, "2s"), (RowName.BOTTOM, "Kc")]),
        discard=placements([(RowName.TOP, "5h")])[0].card,
    )


class SolverRolloutTest(unittest.TestCase):
    def test_run_rollout_stops_without_continuation_when_no_fantasyland_is_triggered(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=False)

        result = run_rollout(
            state,
            root_action=_final_draw_action(),
            root_player=PlayerId.PLAYER_0,
            rng=random.Random(21),
            policy=RandomRolloutPolicy(),
        )

        self.assertEqual(0, result.continuation_hands_simulated)
        self.assertEqual(result.current_hand_value, result.total_value)

    def test_run_rollout_includes_exactly_one_immediate_continuation_hand(self) -> None:
        state = solver_final_draw_state(enters_fantasyland=True)

        result = run_rollout(
            state,
            root_action=_final_draw_action(),
            root_player=PlayerId.PLAYER_0,
            rng=random.Random(22),
            policy=RandomRolloutPolicy(),
        )

        self.assertEqual(1, result.continuation_hands_simulated)
        self.assertEqual(result.current_hand_value + result.continuation_value, result.total_value)

    def test_run_rollout_rejects_fantasyland_root_phase(self) -> None:
        state = new_hand(
            button=PlayerId.PLAYER_0,
            fantasyland_flags=(False, True),
            preset_order=stacked_deck_tokens(FANTASYLAND_PREFIX),
            continuation_hand=True,
        )

        with self.assertRaisesRegex(ValueError, "Unsupported root phase"):
            run_rollout(
                state,
                root_action=_final_draw_action(),
                root_player=PlayerId.PLAYER_1,
                rng=random.Random(23),
                policy=RandomRolloutPolicy(),
            )


if __name__ == "__main__":
    unittest.main()
