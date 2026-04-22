from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc_analysis.cli import main
from ofc_analysis.render import render_benchmark_run
from ofc_solver.benchmark import compare_benchmark_runs, load_benchmark_manifest, run_benchmark_manifest


FIXTURE_DIR = Path("scenarios/regression")
BENCHMARK_MANIFEST = Path("scenarios/benchmarks/solver_diagnostics.json")


class SolverBenchmarkTest(unittest.TestCase):
    def test_load_benchmark_manifest_resolves_case_fields(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        self.assertEqual("1", manifest.version)
        self.assertGreaterEqual(len(manifest.cases), 3)
        self.assertEqual("late-draw-immediate-scoring", manifest.cases[1].name)
        self.assertEqual(FIXTURE_DIR / "immediate_scoring.json", manifest.cases[1].scenario_path)
        self.assertEqual("player_0", manifest.cases[1].observer.value)
        self.assertEqual(3, manifest.cases[1].rollouts_per_action)
        self.assertEqual(302, manifest.cases[1].rng_seed)
        self.assertEqual((1, 4), manifest.cases[1].expected_top_action_indices)
        self.assertIn("oracle", manifest.cases[1].tags)

    def test_run_benchmark_manifest_collects_unlabeled_diagnostics(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        run = run_benchmark_manifest(manifest, policy_name="random")
        initial_case = run.case_results[0]

        self.assertEqual("random", run.policy_name)
        self.assertEqual(len(manifest.cases), run.case_count)
        self.assertEqual("initial-deal-baseline", initial_case.name)
        self.assertIsNone(initial_case.top1_agreement)
        self.assertIsNone(initial_case.top3_agreement)
        self.assertEqual(232, initial_case.action_count)
        self.assertEqual(1, initial_case.rollouts_per_action)
        self.assertEqual(232, len(initial_case.action_diagnostics))
        self.assertTrue(all(diagnostic.sample_count == 1 for diagnostic in initial_case.action_diagnostics))
        self.assertTrue(all(0.0 <= diagnostic.continuation_frequency <= 1.0 for diagnostic in initial_case.action_diagnostics))
        self.assertTrue(all(diagnostic.mean_policy_decisions >= 0.0 for diagnostic in initial_case.action_diagnostics))
        self.assertTrue(
            all(diagnostic.exact_late_search_rollout_frequency == 0.0 for diagnostic in initial_case.action_diagnostics)
        )
        self.assertGreaterEqual(initial_case.elapsed_seconds, 0.0)

    def test_run_benchmark_manifest_scores_labeled_expected_actions(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        run = run_benchmark_manifest(manifest, policy_name="random")
        labeled_case = run.case_results[1]

        self.assertEqual("late-draw-immediate-scoring", labeled_case.name)
        self.assertEqual((1, 4), labeled_case.expected_top_action_indices)
        self.assertIn(labeled_case.top_action_index, {1, 4})
        self.assertTrue(labeled_case.top1_agreement)
        self.assertTrue(labeled_case.top3_agreement)
        self.assertEqual(6, labeled_case.action_count)
        self.assertEqual(3, labeled_case.rollouts_per_action)
        self.assertEqual(6, len(labeled_case.action_diagnostics))
        top_diagnostic = labeled_case.action_diagnostics[labeled_case.top_action_index]
        self.assertEqual(3, top_diagnostic.sample_count)
        self.assertEqual(0.0, top_diagnostic.mean_continuation_value)
        self.assertEqual(0.0, top_diagnostic.continuation_frequency)
        self.assertEqual(0.0, top_diagnostic.exact_late_search_rollout_frequency)

    def test_benchmark_solver_cli_outputs_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["benchmark-solver", str(BENCHMARK_MANIFEST), "--policy", "random", "--json"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("random", payload["policy_name"])
        self.assertEqual(3, payload["case_count"])
        self.assertEqual("initial-deal-baseline", payload["cases"][0]["name"])
        self.assertEqual(232, payload["cases"][0]["action_count"])
        self.assertIn("action_diagnostics", payload["cases"][0])
        self.assertEqual(1, payload["cases"][0]["action_diagnostics"][0]["sample_count"])
        self.assertIn("mean_policy_decisions", payload["cases"][0]["action_diagnostics"][0])
        self.assertIn("exact_late_search_rollout_frequency", payload["cases"][0]["action_diagnostics"][0])

    def test_run_benchmark_manifest_accepts_heuristic_policy(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        run = run_benchmark_manifest(manifest, policy_name="heuristic")

        self.assertEqual("heuristic", run.policy_name)
        self.assertEqual(len(manifest.cases), run.case_count)

    def test_benchmark_solver_cli_accepts_heuristic_policy(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["benchmark-solver", str(BENCHMARK_MANIFEST), "--policy", "heuristic", "--json"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic", payload["policy_name"])

    def test_manifest_rejects_invalid_policy_name(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        with self.assertRaisesRegex(ValueError, "Unsupported rollout policy"):
            run_benchmark_manifest(manifest, policy_name="not-a-policy")

    def test_compare_benchmark_runs_reports_aggregate_deltas(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)
        random_run = run_benchmark_manifest(manifest, policy_name="random")
        heuristic_run = run_benchmark_manifest(manifest, policy_name="heuristic")

        comparison = compare_benchmark_runs(random_run, heuristic_run)

        self.assertEqual("random", comparison.left_policy_name)
        self.assertEqual("heuristic", comparison.right_policy_name)
        self.assertEqual(random_run.case_count, comparison.case_count)
        self.assertIn("root_foul_rate", comparison.deltas)
        self.assertIn("both_foul_rate", comparison.deltas)
        self.assertIn("top_action_root_foul_rate", comparison.deltas)
        self.assertIn("top_action_both_foul_rate", comparison.deltas)
        self.assertIn("exact_late_search_rollout_frequency", comparison.deltas)
        self.assertIn("top_action_exact_late_search_rollout_frequency", comparison.deltas)
        self.assertGreaterEqual(comparison.left.top_action_root_foul_rate, 0.0)
        self.assertGreaterEqual(comparison.right.top_action_root_foul_rate, 0.0)
        self.assertEqual(tuple(sorted(slice.tag for slice in comparison.tag_slices)), tuple(slice.tag for slice in comparison.tag_slices))
        initial_slice = next(slice for slice in comparison.tag_slices if slice.tag == "initial_deal")
        self.assertGreaterEqual(initial_slice.case_count, 1)
        self.assertIn("top_action_root_foul_rate", initial_slice.deltas)

    def test_compare_benchmarks_cli_outputs_json(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)
        random_run = run_benchmark_manifest(manifest, policy_name="random")
        heuristic_run = run_benchmark_manifest(manifest, policy_name="heuristic")

        with TemporaryDirectory() as temp_dir:
            left_path = Path(temp_dir) / "random.json"
            right_path = Path(temp_dir) / "heuristic.json"
            left_path.write_text(json.dumps(render_benchmark_run(random_run, as_json=True).payload), encoding="utf-8")
            right_path.write_text(json.dumps(render_benchmark_run(heuristic_run, as_json=True).payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["compare-benchmarks", str(left_path), str(right_path), "--json"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("random", payload["left_policy_name"])
        self.assertEqual("heuristic", payload["right_policy_name"])
        self.assertIn("root_foul_rate", payload["deltas"])
        self.assertIn("top_action_root_foul_rate", payload["deltas"])
        self.assertIn("exact_late_search_rollout_frequency", payload["deltas"])
        self.assertIn("top_action_exact_late_search_rollout_frequency", payload["deltas"])
        self.assertIn("top_action_root_foul_rate", payload["left"])
        self.assertIn("exact_late_search_rollout_frequency", payload["left"])
        self.assertIn("top_action_changes", payload)
        self.assertIn("tag_slices", payload)
        self.assertEqual(sorted(slice["tag"] for slice in payload["tag_slices"]), [slice["tag"] for slice in payload["tag_slices"]])
        initial_slice = next(slice for slice in payload["tag_slices"] if slice["tag"] == "initial_deal")
        self.assertIn("top_action_root_foul_rate", initial_slice["left"])
        self.assertIn("top_action_root_foul_rate", initial_slice["deltas"])

    def test_manifest_paths_are_resolved_relative_to_manifest_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "cases": [
                            {
                                "name": "relative-case",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "draw_root.json"),
                                "observer": "player_1",
                                "rollouts": 1,
                                "seed": "relative",
                                "tags": ["draw"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_benchmark_manifest(manifest_path)

        self.assertEqual("relative-case", manifest.cases[0].name)
        self.assertEqual(FIXTURE_DIR / "draw_root.json", manifest.cases[0].scenario_path)


if __name__ == "__main__":
    unittest.main()
