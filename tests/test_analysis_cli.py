from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc.engine import new_match
from ofc.state import PlayerId
from ofc_analysis.cli import main
from tests.helpers import scenario_payload_from_state, stacked_deck_tokens


INITIAL_PREFIX = [
    "Ac", "9d", "Kh", "Kd", "Ah",
    "Ks", "8c", "Jh", "Jd", "Qh",
]


class AnalysisCliTest(unittest.TestCase):
    def _initial_payload(self) -> dict[str, object]:
        state = new_match(button=PlayerId.PLAYER_0, preset_order=stacked_deck_tokens(INITIAL_PREFIX))
        return scenario_payload_from_state(state)

    def _write_scenario(self, temp_dir: str, payload: dict[str, object]) -> Path:
        scenario_path = Path(temp_dir) / "scenario.json"
        scenario_path.write_text(json.dumps(payload), encoding="utf-8")
        return scenario_path

    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_show_state_text_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            scenario_path = self._write_scenario(temp_dir, self._initial_payload())

            exit_code, stdout, stderr = self._run_cli(["show-state", str(scenario_path)])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        self.assertIn("Exact State", stdout)
        self.assertIn("phase: initial_deal", stdout)

    def test_show_state_observer_json_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            scenario_path = self._write_scenario(temp_dir, self._initial_payload())

            exit_code, stdout, stderr = self._run_cli(
                ["show-state", str(scenario_path), "--observer", "player_1", "--json"]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        payload = json.loads(stdout)
        self.assertEqual("player_1", payload["observer"])
        self.assertEqual("initial_deal", payload["phase"])

    def test_list_actions_json_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            scenario_path = self._write_scenario(temp_dir, self._initial_payload())

            exit_code, stdout, stderr = self._run_cli(["list-actions", str(scenario_path), "--json"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        payload = json.loads(stdout)
        self.assertGreater(payload["action_count"], 0)
        self.assertEqual(0, payload["actions"][0]["action_index"])
        self.assertEqual("place_initial_five", payload["actions"][0]["action_type"])

    def test_list_actions_rejects_unsupported_phase(self) -> None:
        payload = self._initial_payload()
        draw_cards = payload["state"]["players"][1]["current_private_draw"]
        payload["state"]["players"][1]["current_private_draw"] = []
        payload["state"]["deck"]["undealt_cards"] = draw_cards + payload["state"]["deck"]["undealt_cards"]
        payload["state"]["phase"] = "showdown"

        with TemporaryDirectory() as temp_dir:
            scenario_path = self._write_scenario(temp_dir, payload)

            exit_code, stdout, stderr = self._run_cli(["list-actions", str(scenario_path)])

        self.assertEqual(2, exit_code)
        self.assertEqual("", stdout)
        self.assertIn("Unsupported phase for list-actions", stderr)


if __name__ == "__main__":
    unittest.main()
