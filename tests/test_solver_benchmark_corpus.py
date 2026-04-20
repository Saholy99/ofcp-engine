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


if __name__ == "__main__":
    unittest.main()
