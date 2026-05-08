from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc_analysis.benchmark_generation import generate_late_final_benchmark
from ofc_solver.benchmark import load_benchmark_manifest


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


if __name__ == "__main__":
    unittest.main()
