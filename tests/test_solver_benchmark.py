from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ofc_analysis.cli import main
from ofc_analysis.benchmark_generation import generate_final_draw_fantasyland_benchmark
from ofc_analysis.render import render_benchmark_run
from ofc_solver.benchmark import (
    compare_benchmark_runs,
    load_benchmark_manifest,
    run_early_search_benchmark,
    run_benchmark_manifest,
    run_late_search_benchmark,
    run_root_action_risk_ablation_benchmark,
)


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

    def test_run_benchmark_manifest_can_use_early_search_beam(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        run = run_benchmark_manifest(
            manifest,
            policy_name="heuristic",
            early_search=True,
            beam_size=4,
            candidate_extra_rollouts=1,
        )
        initial_case = run.case_results[0]

        self.assertEqual("heuristic+early-search[beam=4,+1]", run.policy_name)
        self.assertTrue(run.early_search_enabled)
        self.assertEqual(232, initial_case.action_count)
        self.assertEqual(4, initial_case.candidate_count)
        self.assertEqual(4, len(initial_case.ranked_actions))
        self.assertEqual(4, len(initial_case.action_diagnostics))
        self.assertEqual(2, initial_case.ranked_actions[0].sample_count)
        self.assertNotEqual((), initial_case.ranked_actions[0].pattern_reasons)

    def test_benchmark_solver_cli_accepts_early_search_options(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--early-search",
                    "--beam-size",
                    "4",
                    "--candidate-extra-rollouts",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic+early-search[beam=4,+1]", payload["policy_name"])
        self.assertTrue(payload["early_search_enabled"])
        self.assertEqual(4, payload["cases"][0]["candidate_count"])
        self.assertIn("candidate_pruning_ratio", payload["cases"][0])
        self.assertIn("pattern_score", payload["cases"][0]["ranked_actions"][0])

    def test_benchmark_solver_cli_accepts_safe_draw_candidate_options(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--early-search",
                    "--beam-size",
                    "8",
                    "--draw-safe-candidates",
                    "--draw-baseline-keep",
                    "3",
                    "--draw-safety-keep",
                    "3",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["draw_safe_candidates"])
        self.assertEqual(3, payload["draw_baseline_keep"])
        self.assertEqual(3, payload["draw_safety_keep"])
        self.assertIn("selection_reasons", payload["cases"][2]["ranked_actions"][0])

    def test_benchmark_solver_cli_accepts_late_search_options(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--late-search",
                    "--late-search-mode",
                    "auto",
                    "--late-search-max-depth",
                    "4",
                    "--late-search-max-nodes",
                    "500",
                    "--late-search-beam-size",
                    "2",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["late_search_enabled"])
        self.assertEqual("auto", payload["late_search_mode"])
        self.assertIn("late_search_activation_rate", payload["cases"][1]["action_diagnostics"][0])
        self.assertIn("late_search_nodes", payload["cases"][1]["ranked_actions"][0])
        self.assertIn("top_action_late_search_activation_rate", payload["aggregate"])

    def test_benchmark_solver_cli_accepts_final_draw_auto_search_options(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--final-draw-auto-search",
                    "--final-draw-auto-max-depth",
                    "1",
                    "--final-draw-auto-max-nodes",
                    "16",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["final_draw_auto_search_enabled"])
        self.assertIn("phase_auto_search_activation_rate", payload["cases"][1]["action_diagnostics"][0])
        self.assertIn("top_action_phase_auto_search_activation_rate", payload["aggregate"])

    def test_benchmark_solver_cli_accepts_final_draw_auto_continuation_options(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--final-draw-auto-search",
                    "--final-draw-auto-max-depth",
                    "1",
                    "--final-draw-auto-max-nodes",
                    "16",
                    "--final-draw-auto-continuation",
                    "--final-draw-continuation-rollouts",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["final_draw_auto_search_enabled"])
        self.assertTrue(payload["final_draw_auto_include_continuation"])
        self.assertEqual(1, payload["final_draw_continuation_rollouts"])
        self.assertIn("final_draw_continuation_trigger_rate", payload["cases"][1]["action_diagnostics"][0])
        self.assertIn("top_action_final_draw_continuation_trigger_rate", payload["aggregate"])

    def test_benchmark_solver_cli_runs_targeted_final_draw_fantasyland_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            generate_final_draw_fantasyland_benchmark(
                manifest_path=manifest_path,
                scenario_dir=Path(temp_dir) / "cases",
                seed="unit-final-fl-benchmark",
                count=8,
                rollouts=1,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "benchmark-solver",
                        str(manifest_path),
                        "--policy",
                        "heuristic",
                        "--final-draw-auto-search",
                        "--final-draw-auto-continuation",
                        "--final-draw-continuation-rollouts",
                        "1",
                        "--json",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(8, payload["case_count"])
        self.assertTrue(payload["final_draw_auto_include_continuation"])
        self.assertGreater(payload["aggregate"]["final_draw_continuation_trigger_rate"], 0.0)
        self.assertIn("mean_final_draw_continuation_value", payload["aggregate"])

    def test_benchmark_solver_cli_can_filter_by_tag(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-solver",
                    str(BENCHMARK_MANIFEST),
                    "--policy",
                    "heuristic",
                    "--include-tag",
                    "late_draw",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(2, payload["case_count"])
        self.assertTrue(all("late_draw" in case["tags"] for case in payload["cases"]))

    def test_run_late_search_benchmark_builds_comparison(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        benchmark = run_late_search_benchmark(
            manifest,
            policy_name="heuristic",
            include_tags=("late_draw",),
            exclude_tags=(),
        )

        self.assertEqual(2, benchmark.comparison.case_count)
        self.assertEqual("heuristic", benchmark.left_run.policy_name)
        self.assertIn("+late-search[auto", benchmark.right_run.policy_name)
        self.assertTrue(benchmark.right_run.late_search_enabled)
        self.assertIn("top_action_late_search_activation_rate", benchmark.comparison.deltas)

    def test_benchmark_late_search_cli_outputs_comparison_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-late-search",
                    str(BENCHMARK_MANIFEST),
                    "--include-tag",
                    "late_draw",
                    "--late-search-mode",
                    "auto",
                    "--late-search-max-depth",
                    "4",
                    "--late-search-max-nodes",
                    "500",
                    "--late-search-beam-size",
                    "2",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic", payload["left_policy_name"])
        self.assertIn("+late-search[auto", payload["right_policy_name"])
        self.assertTrue(payload["late_search"]["enabled_on_right"])
        self.assertEqual(2, payload["case_count"])
        self.assertIn("top_action_late_search_activation_rate", payload["deltas"])
        self.assertIn("late_search_nodes", payload["cases"][0]["right_ranked_actions"][0])

    def test_run_early_search_benchmark_builds_comparison(self) -> None:
        manifest = load_benchmark_manifest(BENCHMARK_MANIFEST)

        benchmark = run_early_search_benchmark(
            manifest,
            policy_name="heuristic",
            include_tags=("initial_deal",),
            exclude_tags=(),
            beam_size=4,
            candidate_extra_rollouts=1,
        )

        self.assertEqual(1, benchmark.comparison.case_count)
        self.assertEqual("heuristic", benchmark.left_run.policy_name)
        self.assertEqual("heuristic+early-search[beam=4,+1]", benchmark.right_run.policy_name)
        self.assertEqual(4, benchmark.right_run.case_results[0].candidate_count)
        self.assertIn("top_action_root_foul_rate", benchmark.comparison.deltas)

    def test_benchmark_early_search_cli_outputs_comparison_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "benchmark-early-search",
                    str(BENCHMARK_MANIFEST),
                    "--include-tag",
                    "initial_deal",
                    "--beam-size",
                    "4",
                    "--candidate-extra-rollouts",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic", payload["left_policy_name"])
        self.assertEqual("heuristic+early-search[beam=4,+1]", payload["right_policy_name"])
        self.assertTrue(payload["early_search"]["enabled_on_right"])
        self.assertEqual(1, payload["case_count"])
        self.assertEqual(4, payload["cases"][0]["right_candidate_count"])
        self.assertIn("pattern_score", payload["cases"][0]["right_ranked_actions"][0])

    def test_benchmark_root_action_risk_cli_outputs_comparison_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "root_risk_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "cases": [
                            {
                                "name": "early-root-risk",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "draw_root.json"),
                                "observer": "player_1",
                                "rollouts": 1,
                                "seed": "root-risk-cli",
                                "tags": ["early_draw", "root_risk"],
                            },
                            {
                                "name": "final-draw-excluded",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "immediate_scoring.json"),
                                "observer": "player_0",
                                "rollouts": 1,
                                "seed": "root-risk-final",
                                "tags": ["final_draw"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "benchmark-root-action-risk",
                        str(manifest_path),
                        "--include-tag",
                        "early_draw",
                        "--exclude-tag",
                        "final_draw",
                        "--json",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic", payload["left_policy_name"])
        self.assertEqual("heuristic+root-risk", payload["right_policy_name"])
        self.assertEqual("default", payload["root_action_risk"]["right_config_label"])
        self.assertEqual(1, payload["case_count"])
        self.assertIn("top_action_root_foul_rate", payload["deltas"])
        self.assertIn("top_action_continuation_frequency", payload["deltas"])
        self.assertEqual("early-root-risk", payload["cases"][0]["name"])
        self.assertIn("root_risk_score", payload["cases"][0]["right_ranked_actions"][0])
        self.assertTrue(payload["root_action_risk"]["enabled_on_right"])

    def test_benchmark_root_action_risk_cli_can_enable_full_component_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "root_risk_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "cases": [
                            {
                                "name": "early-root-risk",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "draw_root.json"),
                                "observer": "player_1",
                                "rollouts": 1,
                                "seed": "root-risk-cli-full",
                                "tags": ["early_draw", "root_risk"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "benchmark-root-action-risk",
                        str(manifest_path),
                        "--include-tag",
                        "early_draw",
                        "--root-action-risk-config",
                        "full",
                        "--json",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic+root-risk[full]", payload["right_policy_name"])
        self.assertEqual("full", payload["root_action_risk"]["right_config_label"])

    def test_run_root_action_risk_ablation_benchmark_builds_expected_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "ablation_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "cases": [
                            {
                                "name": "early-root-risk",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "draw_root.json"),
                                "observer": "player_1",
                                "rollouts": 1,
                                "seed": "root-risk-ablation",
                                "tags": ["early_draw", "root_risk"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_benchmark_manifest(manifest_path)

        benchmark = run_root_action_risk_ablation_benchmark(
            manifest,
            include_tags=("early_draw",),
            exclude_tags=(),
        )

        self.assertEqual(1, benchmark.baseline_run.case_count)
        self.assertEqual(1, benchmark.full_run.case_count)
        self.assertEqual(10, len(benchmark.ablations))
        self.assertEqual(
            "unsupported_top_pair",
            benchmark.ablations[0].component,
        )
        self.assertIn("top_action_root_foul_rate", benchmark.ablations[0].comparison_vs_baseline.deltas)
        self.assertIn("top_action_continuation_frequency", benchmark.ablations[0].comparison_vs_full.deltas)

    def test_benchmark_root_action_risk_ablation_cli_outputs_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "ablation_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "cases": [
                            {
                                "name": "early-root-risk",
                                "scenario": str(Path.cwd() / FIXTURE_DIR / "draw_root.json"),
                                "observer": "player_1",
                                "rollouts": 1,
                                "seed": "root-risk-ablation-cli",
                                "tags": ["early_draw", "root_risk"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "benchmark-root-action-risk-ablation",
                        str(manifest_path),
                        "--include-tag",
                        "early_draw",
                        "--json",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual("heuristic", payload["baseline"]["policy_name"])
        self.assertEqual("heuristic+root-risk[full]", payload["full"]["policy_name"])
        self.assertEqual(10, len(payload["ablations"]))
        self.assertEqual(
            [
                "unsupported_top_pair",
                "unsupported_top_trips",
                "middle_over_bottom_pressure",
                "bottom_underbuilt",
                "top_slots_closed",
            ],
            payload["component_order"],
        )
        first = payload["ablations"][0]
        self.assertIn("top_action_root_foul_rate", first["aggregate"])
        self.assertIn("top_action_both_foul_rate", first["aggregate"])
        self.assertIn("top_action_continuation_frequency", first["aggregate"])
        self.assertIn("top_action_changes_vs_baseline", first)
        self.assertIn("top_action_changes_vs_full", first)

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
