from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc.engine import new_match
from ofc.state import PlayerId
from ofc_analysis.scenario import load_scenario, load_scenario_data
from tests.helpers import scenario_payload_from_state, stacked_deck_tokens


INITIAL_PREFIX = [
    "Ac", "9d", "Kh", "Kd", "Ah",
    "Ks", "8c", "Jh", "Jd", "Qh",
]


class AnalysisScenarioTest(unittest.TestCase):
    def _initial_state(self):
        return new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(INITIAL_PREFIX))

    def test_load_scenario_data_round_trips_engine_state(self) -> None:
        state = self._initial_state()

        scenario = load_scenario_data(scenario_payload_from_state(state))

        self.assertEqual("1", scenario.version)
        self.assertEqual(state, scenario.state)

    def test_load_scenario_reads_from_disk(self) -> None:
        state = self._initial_state()
        payload = scenario_payload_from_state(state)

        with TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "scenario.json"
            scenario_path.write_text(json.dumps(payload), encoding="utf-8")

            scenario = load_scenario(scenario_path)

        self.assertEqual(scenario_path, scenario.source_path)
        self.assertEqual(state, scenario.state)

    def test_duplicate_card_error_is_rejected(self) -> None:
        payload = scenario_payload_from_state(self._initial_state())
        draw_card = payload["state"]["players"][1]["current_private_draw"][0]
        payload["state"]["deck"]["undealt_cards"][0] = draw_card

        with self.assertRaisesRegex(ValueError, "duplicate cards"):
            load_scenario_data(payload)

    def test_missing_required_field_is_rejected(self) -> None:
        payload = scenario_payload_from_state(self._initial_state())
        del payload["state"]["players"][0]["board"]["top"]

        with self.assertRaisesRegex(ValueError, "missing required keys: top"):
            load_scenario_data(payload)

    def test_action_phase_draw_size_must_match_phase(self) -> None:
        payload = scenario_payload_from_state(self._initial_state())
        removed = payload["state"]["players"][1]["current_private_draw"].pop()
        payload["state"]["deck"]["undealt_cards"].insert(0, removed)

        with self.assertRaisesRegex(ValueError, "exactly 5 cards during initial_deal"):
            load_scenario_data(payload)


if __name__ == "__main__":
    unittest.main()
