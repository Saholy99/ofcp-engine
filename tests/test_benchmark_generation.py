from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc.engine import showdown
from ofc.state import HandPhase, player_index
from ofc.transitions import apply_action, legal_actions
from ofc_analysis.benchmark_generation import (
    generate_final_draw_fantasyland_benchmark,
    generate_late_final_benchmark,
)
from ofc_solver.benchmark import load_benchmark_manifest
from ofc_analysis.scenario import load_scenario


class BenchmarkGenerationTest(unittest.TestCase):
    def test_late_final_benchmark_generation_is_deterministic_and_loadable(self) -> None:
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first_manifest = Path(first_dir) / "manifest.json"
            second_manifest = Path(second_dir) / "manifest.json"

            first = generate_late_final_benchmark(
                manifest_path=first_manifest,
                scenario_dir=Path(first_dir) / "cases",
                seed="unit-large",
                final_count=4,
                late_count=4,
                mid_count=2,
                rollouts=1,
            )
            second = generate_late_final_benchmark(
                manifest_path=second_manifest,
                scenario_dir=Path(second_dir) / "cases",
                seed="unit-large",
                final_count=4,
                late_count=4,
                mid_count=2,
                rollouts=1,
            )

            first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
            second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))

            self.assertEqual(first_payload, second_payload)
            self.assertEqual(10, len(first_payload["cases"]))
            self.assertEqual(4, first.tag_counts["final_draw"])
            self.assertEqual(4, first.tag_counts["late_draw"])
            self.assertEqual(2, first.tag_counts["mid_draw"])
            self.assertEqual(first.tag_counts, second.tag_counts)
            manifest = load_benchmark_manifest(first_manifest)
            self.assertEqual(10, len(manifest.cases))
            self.assertTrue(all(case.scenario_path.exists() for case in manifest.cases))

    def test_final_draw_fantasyland_generation_is_deterministic_and_loadable(self) -> None:
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first_manifest = Path(first_dir) / "manifest.json"
            second_manifest = Path(second_dir) / "manifest.json"

            first = generate_final_draw_fantasyland_benchmark(
                manifest_path=first_manifest,
                scenario_dir=Path(first_dir) / "cases",
                seed="unit-final-fl",
                count=12,
                rollouts=1,
            )
            second = generate_final_draw_fantasyland_benchmark(
                manifest_path=second_manifest,
                scenario_dir=Path(second_dir) / "cases",
                seed="unit-final-fl",
                count=12,
                rollouts=1,
            )

            first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
            second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))

            self.assertEqual(first_payload, second_payload)
            self.assertEqual(12, first.case_count)
            self.assertEqual(first.tag_counts, second.tag_counts)
            self.assertGreater(first.fantasyland_trigger_case_count, 0)
            self.assertGreater(first.fantasyland_trigger_action_count, first.fantasyland_trigger_case_count)
            manifest = load_benchmark_manifest(first_manifest)
            self.assertEqual(12, len(manifest.cases))
            self.assertTrue(all("final_draw" in case.tags for case in manifest.cases))
            self.assertTrue(all("fantasyland_targeted" in case.tags for case in manifest.cases))

    def test_final_draw_fantasyland_cases_are_terminal_draws_with_triggering_actions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            summary = generate_final_draw_fantasyland_benchmark(
                manifest_path=manifest_path,
                scenario_dir=Path(temp_dir) / "cases",
                seed="unit-final-fl-valid",
                count=10,
                rollouts=1,
            )
            manifest = load_benchmark_manifest(manifest_path)

            trigger_cases = 0
            for case in manifest.cases:
                state = load_scenario(case.scenario_path).state
                self.assertEqual(HandPhase.DRAW, state.phase)
                triggered = False
                for action in legal_actions(state):
                    next_state = apply_action(state, action)
                    self.assertIn(next_state.phase, {HandPhase.SHOWDOWN, HandPhase.TERMINAL})
                    terminal_state, _ = showdown(next_state)
                    if terminal_state.next_hand_fantasyland[player_index(state.acting_player)]:
                        triggered = True
                trigger_cases += int(triggered)

            self.assertEqual(summary.fantasyland_trigger_case_count, trigger_cases)
            self.assertGreaterEqual(trigger_cases, 8)


if __name__ == "__main__":
    unittest.main()
