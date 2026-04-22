from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc_solver.benchmark import load_benchmark_manifest
from ofc_solver.benchmark_corpus import write_expansive_benchmark_corpus


class SolverBenchmarkCorpusTest(unittest.TestCase):
    def test_write_expansive_benchmark_corpus_creates_broad_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = write_expansive_benchmark_corpus(Path(temp_dir) / "benchmarks")

            manifest = load_benchmark_manifest(manifest_path)

        self.assertGreaterEqual(len(manifest.cases), 50)
        tags = {tag for case in manifest.cases for tag in case.tags}
        self.assertTrue(
            {
                "initial_deal",
                "early_draw",
                "mid_draw",
                "late_draw",
                "final_draw",
                "fantasyland",
                "hidden_discards",
                "oracle",
            }.issubset(tags)
        )
        self.assertTrue(any(case.expected_top_action_indices for case in manifest.cases))
        self.assertTrue(all(case.rollouts_per_action >= 1 for case in manifest.cases))
        self.assertTrue(any({"strategy", "survivability"}.issubset(set(case.tags)) for case in manifest.cases))

    def test_write_expansive_benchmark_corpus_strategy_cases_are_deterministic(self) -> None:
        with TemporaryDirectory() as left_dir, TemporaryDirectory() as right_dir:
            left_manifest_path = write_expansive_benchmark_corpus(Path(left_dir) / "benchmarks")
            right_manifest_path = write_expansive_benchmark_corpus(Path(right_dir) / "benchmarks")

            left_manifest = load_benchmark_manifest(left_manifest_path)
            right_manifest = load_benchmark_manifest(right_manifest_path)

        left_cases = [
            (case.name, case.observer, case.rng_seed, case.tags)
            for case in left_manifest.cases
            if "strategy" in case.tags
        ]
        right_cases = [
            (case.name, case.observer, case.rng_seed, case.tags)
            for case in right_manifest.cases
            if "strategy" in case.tags
        ]
        self.assertEqual(left_cases, right_cases)
        self.assertGreaterEqual(len(left_cases), 3)


if __name__ == "__main__":
    unittest.main()
