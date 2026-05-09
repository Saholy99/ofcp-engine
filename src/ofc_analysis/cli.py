"""CLI for reproducible inspection of exact states and legal root actions."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import json
import sys

from ofc.state import HandPhase, PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_actions
from ofc_analysis.benchmark_generation import (
    generate_final_draw_fantasyland_benchmark,
    generate_late_final_benchmark,
)
from ofc_analysis.models import RenderedOutput
from ofc_analysis.observation import project_observation
from ofc_analysis.play import run_play_hand
from ofc_analysis.render import (
    render_actions,
    render_benchmark_comparison,
    render_early_search_benchmark,
    render_benchmark_run,
    render_late_search_benchmark,
    render_move_analysis,
    render_observation,
    render_root_action_risk_ablation_benchmark,
    render_root_action_risk_benchmark,
    render_state,
)
from ofc_analysis.scenario import load_scenario
from ofc_solver.benchmark import (
    compare_benchmark_payloads,
    load_benchmark_manifest,
    run_early_search_benchmark,
    run_benchmark_manifest,
    run_late_search_benchmark,
    run_root_action_risk_ablation_benchmark,
    run_root_action_risk_benchmark,
)
from ofc_solver.monte_carlo import rank_actions_from_observation
from ofc_solver.policy_registry import POLICY_NAMES, policy_from_name
from ofc_solver.early_search import EarlySearchConfig
from ofc_solver.late_search import FinalDrawAutoSearchConfig, LateSearchConfig
from ofc_solver.root_action_risk import RootRiskConfig


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis CLI entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "show-state":
            scenario = load_scenario(args.scenario)
            if args.observer is None:
                output = render_state(scenario.state, as_json=args.as_json)
            else:
                observation = project_observation(scenario.state, PlayerId(args.observer))
                output = render_observation(observation, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "list-actions":
            scenario = load_scenario(args.scenario)
            if scenario.state.phase not in {HandPhase.INITIAL_DEAL, HandPhase.DRAW}:
                print(
                    (
                        "Unsupported phase for list-actions: "
                        f"{scenario.state.phase.value}. Supported phases: initial_deal, draw."
                    ),
                    file=sys.stderr,
                )
                return 2
            actions = tuple(legal_actions(scenario.state))
            output = render_actions(encode_actions(actions), as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "solve-move":
            scenario = load_scenario(args.scenario)
            observer = PlayerId(args.observer)
            if observer != scenario.state.acting_player:
                print("solve-move v1 requires observer to be the acting player", file=sys.stderr)
                return 2
            observation = project_observation(scenario.state, observer)
            analysis = rank_actions_from_observation(
                observation,
                rollouts_per_action=args.rollouts,
                rng_seed=args.seed,
                policy=policy_from_name(args.policy),
                root_action_risk=args.root_action_risk,
                root_action_risk_config=(
                    _root_action_risk_config_from_name(args.root_action_risk_config)
                    if args.root_action_risk
                    else None
                ),
                early_search=args.early_search,
                early_search_config=EarlySearchConfig(
                    beam_size=args.beam_size,
                    candidate_extra_rollouts=args.candidate_extra_rollouts,
                    draw_safe_candidates=args.draw_safe_candidates,
                    draw_baseline_keep=args.draw_baseline_keep,
                    draw_safety_keep=args.draw_safety_keep,
                ),
                late_search=args.late_search,
                late_search_config=LateSearchConfig(
                    mode=args.late_search_mode,
                    max_depth=args.late_search_max_depth,
                    max_nodes=args.late_search_max_nodes,
                    beam_size=args.late_search_beam_size,
                ),
                final_draw_auto_search=args.final_draw_auto_search,
                final_draw_auto_search_config=FinalDrawAutoSearchConfig(
                    max_depth=args.final_draw_auto_max_depth,
                    max_nodes=args.final_draw_auto_max_nodes,
                    include_continuation=args.final_draw_auto_continuation,
                    continuation_rollouts=args.final_draw_continuation_rollouts,
                ),
            )
            output = render_move_analysis(analysis, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-solver":
            manifest = load_benchmark_manifest(args.manifest)
            if args.include_tag is not None or args.exclude_tag is not None or args.phase is not None:
                from ofc_solver.benchmark import filter_benchmark_manifest

                manifest = filter_benchmark_manifest(
                    manifest,
                    include_tags=tuple(args.include_tag or ()),
                    exclude_tags=tuple(args.exclude_tag or ()),
                    phases=tuple(HandPhase(phase) for phase in (args.phase or ())),
                )
            run = run_benchmark_manifest(
                manifest,
                policy_name=args.policy,
                root_action_risk=args.root_action_risk,
                root_action_risk_config=(
                    _root_action_risk_config_from_name(args.root_action_risk_config)
                    if args.root_action_risk
                    else None
                ),
                early_search=args.early_search,
                beam_size=args.beam_size,
                candidate_extra_rollouts=args.candidate_extra_rollouts,
                draw_safe_candidates=args.draw_safe_candidates,
                draw_baseline_keep=args.draw_baseline_keep,
                draw_safety_keep=args.draw_safety_keep,
                late_search=args.late_search,
                late_search_config=LateSearchConfig(
                    mode=args.late_search_mode,
                    max_depth=args.late_search_max_depth,
                    max_nodes=args.late_search_max_nodes,
                    beam_size=args.late_search_beam_size,
                ),
                final_draw_auto_search=args.final_draw_auto_search,
                final_draw_auto_search_config=FinalDrawAutoSearchConfig(
                    max_depth=args.final_draw_auto_max_depth,
                    max_nodes=args.final_draw_auto_max_nodes,
                    include_continuation=args.final_draw_auto_continuation,
                    continuation_rollouts=args.final_draw_continuation_rollouts,
                ),
            )
            output = render_benchmark_run(run, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-early-search":
            manifest = load_benchmark_manifest(args.manifest)
            include_tags = tuple(args.include_tag) if args.include_tag is not None else ("initial_deal", "early_draw")
            exclude_tags = tuple(args.exclude_tag) if args.exclude_tag is not None else ("final_draw",)
            if args.non_final and "final_draw" not in exclude_tags:
                exclude_tags = exclude_tags + ("final_draw",)
            if args.exclude_strategy and "strategy" not in exclude_tags:
                exclude_tags = exclude_tags + ("strategy",)
            benchmark = run_early_search_benchmark(
                manifest,
                policy_name=args.policy,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                phases=tuple(HandPhase(phase) for phase in (args.phase or ())),
                beam_size=args.beam_size,
                candidate_extra_rollouts=args.candidate_extra_rollouts,
                draw_safe_candidates=args.draw_safe_candidates,
                draw_baseline_keep=args.draw_baseline_keep,
                draw_safety_keep=args.draw_safety_keep,
                root_action_risk=args.root_action_risk,
                root_action_risk_config=(
                    _root_action_risk_config_from_name(args.root_action_risk_config)
                    if args.root_action_risk
                    else None
                ),
            )
            output = render_early_search_benchmark(benchmark, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-late-search":
            manifest = load_benchmark_manifest(args.manifest)
            include_tags = tuple(args.include_tag) if args.include_tag is not None else ("late_draw", "final_draw")
            exclude_tags = tuple(args.exclude_tag) if args.exclude_tag is not None else ()
            if args.exclude_strategy and "strategy" not in exclude_tags:
                exclude_tags = exclude_tags + ("strategy",)
            benchmark = run_late_search_benchmark(
                manifest,
                policy_name=args.policy,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                phases=tuple(HandPhase(phase) for phase in (args.phase or (HandPhase.DRAW.value,))),
                late_search_config=LateSearchConfig(
                    mode=args.late_search_mode,
                    max_depth=args.late_search_max_depth,
                    max_nodes=args.late_search_max_nodes,
                    beam_size=args.late_search_beam_size,
                ),
                root_action_risk=args.root_action_risk,
                root_action_risk_config=(
                    _root_action_risk_config_from_name(args.root_action_risk_config)
                    if args.root_action_risk
                    else None
                ),
            )
            output = render_late_search_benchmark(benchmark, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-root-action-risk":
            manifest = load_benchmark_manifest(args.manifest)
            include_tags = tuple(args.include_tag) if args.include_tag is not None else ("initial_deal", "early_draw")
            exclude_tags = tuple(args.exclude_tag) if args.exclude_tag is not None else ("final_draw",)
            if args.non_final and "final_draw" not in exclude_tags:
                exclude_tags = exclude_tags + ("final_draw",)
            if args.exclude_strategy and "strategy" not in exclude_tags:
                exclude_tags = exclude_tags + ("strategy",)
            benchmark = run_root_action_risk_benchmark(
                manifest,
                policy_name=args.policy,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                phases=tuple(HandPhase(phase) for phase in (args.phase or ())),
                root_action_risk_config=_root_action_risk_config_from_name(args.root_action_risk_config),
            )
            output = render_root_action_risk_benchmark(benchmark, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-root-action-risk-ablation":
            manifest = load_benchmark_manifest(args.manifest)
            include_tags = tuple(args.include_tag) if args.include_tag is not None else ("initial_deal", "early_draw")
            exclude_tags = tuple(args.exclude_tag) if args.exclude_tag is not None else ("final_draw",)
            if args.non_final and "final_draw" not in exclude_tags:
                exclude_tags = exclude_tags + ("final_draw",)
            if args.exclude_strategy and "strategy" not in exclude_tags:
                exclude_tags = exclude_tags + ("strategy",)
            benchmark = run_root_action_risk_ablation_benchmark(
                manifest,
                policy_name=args.policy,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                phases=tuple(HandPhase(phase) for phase in (args.phase or ())),
            )
            output = render_root_action_risk_ablation_benchmark(benchmark, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "compare-benchmarks":
            left_payload = _load_json_payload(args.left)
            right_payload = _load_json_payload(args.right)
            comparison = compare_benchmark_payloads(left_payload, right_payload)
            output = render_benchmark_comparison(comparison, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "generate-late-final-benchmark":
            summary = generate_late_final_benchmark(
                manifest_path=args.manifest,
                scenario_dir=args.scenario_dir,
                seed=args.seed,
                final_count=args.final_count,
                late_count=args.late_count,
                mid_count=args.mid_count,
                rollouts=args.rollouts,
            )
            payload = {
                "manifest_path": str(summary.manifest_path),
                "scenario_dir": str(summary.scenario_dir),
                "case_count": summary.case_count,
                "tag_counts": summary.tag_counts,
            }
            _emit(RenderedOutput(text=json.dumps(payload, indent=2), payload=payload), as_json=args.as_json)
            return 0

        if args.command == "generate-final-draw-fantasyland-benchmark":
            summary = generate_final_draw_fantasyland_benchmark(
                manifest_path=args.manifest,
                scenario_dir=args.scenario_dir,
                seed=args.seed,
                count=args.count,
                rollouts=args.rollouts,
            )
            payload = {
                "manifest_path": str(summary.manifest_path),
                "scenario_dir": str(summary.scenario_dir),
                "case_count": summary.case_count,
                "tag_counts": summary.tag_counts,
                "legal_action_count": summary.legal_action_count,
                "fantasyland_trigger_case_count": summary.fantasyland_trigger_case_count,
                "fantasyland_trigger_action_count": summary.fantasyland_trigger_action_count,
            }
            _emit(RenderedOutput(text=json.dumps(payload, indent=2), payload=payload), as_json=args.as_json)
            return 0

        if args.command == "play-hand":
            hero_player = _resolve_play_hero(args)
            button = _resolve_play_button(args)
            fantasyland_flags = _resolve_play_fantasyland_flags(args)
            return run_play_hand(
                hero_player=hero_player,
                button=button,
                fantasyland_flags=fantasyland_flags,
                rollouts_per_action=args.rollouts,
                rng_seed=args.seed,
                policy_name=args.policy,
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofc-analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_state = subparsers.add_parser("show-state")
    show_state.add_argument("scenario")
    show_state.add_argument("--observer", choices=[player.value for player in PlayerId])
    show_state.add_argument("--json", action="store_true", dest="as_json")

    list_actions = subparsers.add_parser("list-actions")
    list_actions.add_argument("scenario")
    list_actions.add_argument("--json", action="store_true", dest="as_json")

    solve_move = subparsers.add_parser("solve-move")
    solve_move.add_argument("scenario")
    solve_move.add_argument("--observer", choices=[player.value for player in PlayerId], required=True)
    solve_move.add_argument("--rollouts", type=int, required=True)
    solve_move.add_argument("--seed", required=True)
    solve_move.add_argument("--policy", choices=POLICY_NAMES, default="random")
    solve_move.add_argument("--root-action-risk", action="store_true", help="Apply root-only risk scoring.")
    solve_move.add_argument(
        "--root-action-risk-config",
        choices=("default", "full"),
        default="default",
        help="Root-risk component set to use when --root-action-risk is enabled.",
    )
    solve_move.add_argument("--early-search", action="store_true", help="Use early-game candidate beam pruning.")
    solve_move.add_argument("--beam-size", type=int, default=48, help="Candidate beam size for --early-search.")
    solve_move.add_argument(
        "--candidate-extra-rollouts",
        type=int,
        default=0,
        help="Additional rollouts per kept candidate in --early-search mode.",
    )
    _add_draw_safe_candidate_args(solve_move)
    _add_late_search_args(solve_move)
    _add_final_draw_auto_search_args(solve_move)
    solve_move.add_argument("--json", action="store_true", dest="as_json")

    benchmark_solver = subparsers.add_parser("benchmark-solver")
    benchmark_solver.add_argument("manifest")
    benchmark_solver.add_argument("--policy", choices=POLICY_NAMES, default="random")
    _add_benchmark_filter_args(benchmark_solver)
    benchmark_solver.add_argument("--root-action-risk", action="store_true", help="Apply root-only risk scoring.")
    benchmark_solver.add_argument(
        "--root-action-risk-config",
        choices=("default", "full"),
        default="default",
        help="Root-risk component set to use when --root-action-risk is enabled.",
    )
    benchmark_solver.add_argument("--early-search", action="store_true", help="Use early-game candidate beam pruning.")
    benchmark_solver.add_argument("--beam-size", type=int, default=48, help="Candidate beam size for --early-search.")
    benchmark_solver.add_argument(
        "--candidate-extra-rollouts",
        type=int,
        default=0,
        help="Additional rollouts per kept candidate in --early-search mode.",
    )
    _add_draw_safe_candidate_args(benchmark_solver)
    _add_late_search_args(benchmark_solver)
    _add_final_draw_auto_search_args(benchmark_solver)
    benchmark_solver.add_argument("--json", action="store_true", dest="as_json")

    benchmark_early_search = subparsers.add_parser("benchmark-early-search")
    benchmark_early_search.add_argument("manifest")
    benchmark_early_search.add_argument("--policy", choices=POLICY_NAMES, default="heuristic")
    benchmark_early_search.add_argument(
        "--include-tag",
        action="append",
        help="Include cases with this tag. Repeat to allow multiple tags.",
    )
    benchmark_early_search.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude cases with this tag. Repeat to exclude multiple tags.",
    )
    benchmark_early_search.add_argument(
        "--phase",
        action="append",
        choices=[HandPhase.INITIAL_DEAL.value, HandPhase.DRAW.value],
        help="Restrict to a root engine phase. Repeat to allow multiple phases.",
    )
    benchmark_early_search.add_argument("--beam-size", type=int, default=48)
    benchmark_early_search.add_argument("--candidate-extra-rollouts", type=int, default=0)
    _add_draw_safe_candidate_args(benchmark_early_search)
    benchmark_early_search.add_argument("--root-action-risk", action="store_true", help="Apply root-only risk scoring.")
    benchmark_early_search.add_argument(
        "--root-action-risk-config",
        choices=("default", "full"),
        default="default",
        help="Root-risk component set to use when --root-action-risk is enabled.",
    )
    benchmark_early_search.add_argument("--non-final", action="store_true", help="Exclude final_draw-tagged cases.")
    benchmark_early_search.add_argument("--exclude-strategy", action="store_true", help="Exclude strategy stress cases.")
    benchmark_early_search.add_argument("--json", action="store_true", dest="as_json")

    benchmark_late_search = subparsers.add_parser("benchmark-late-search")
    benchmark_late_search.add_argument("manifest")
    benchmark_late_search.add_argument("--policy", choices=POLICY_NAMES, default="heuristic")
    benchmark_late_search.add_argument(
        "--include-tag",
        action="append",
        help="Include cases with this tag. Repeat to allow multiple tags.",
    )
    benchmark_late_search.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude cases with this tag. Repeat to exclude multiple tags.",
    )
    benchmark_late_search.add_argument(
        "--phase",
        action="append",
        choices=[HandPhase.INITIAL_DEAL.value, HandPhase.DRAW.value],
        help="Restrict to a root engine phase. Repeat to allow multiple phases.",
    )
    _add_late_search_config_args(benchmark_late_search)
    benchmark_late_search.add_argument("--root-action-risk", action="store_true", help="Apply root-only risk scoring.")
    benchmark_late_search.add_argument(
        "--root-action-risk-config",
        choices=("default", "full"),
        default="default",
        help="Root-risk component set to use when --root-action-risk is enabled.",
    )
    benchmark_late_search.add_argument("--exclude-strategy", action="store_true", help="Exclude strategy stress cases.")
    benchmark_late_search.add_argument("--json", action="store_true", dest="as_json")

    benchmark_root_risk = subparsers.add_parser("benchmark-root-action-risk")
    benchmark_root_risk.add_argument("manifest")
    benchmark_root_risk.add_argument("--policy", choices=POLICY_NAMES, default="heuristic")
    benchmark_root_risk.add_argument(
        "--include-tag",
        action="append",
        help="Include cases with this tag. Repeat to allow multiple tags.",
    )
    benchmark_root_risk.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude cases with this tag. Repeat to exclude multiple tags.",
    )
    benchmark_root_risk.add_argument(
        "--phase",
        action="append",
        choices=[HandPhase.INITIAL_DEAL.value, HandPhase.DRAW.value],
        help="Restrict to a root engine phase. Repeat to allow multiple phases.",
    )
    benchmark_root_risk.add_argument(
        "--root-action-risk-config",
        choices=("default", "full"),
        default="default",
        help="Compare against the safer default root-risk set or the full experimental set.",
    )
    benchmark_root_risk.add_argument("--non-final", action="store_true", help="Exclude final_draw-tagged cases.")
    benchmark_root_risk.add_argument("--exclude-strategy", action="store_true", help="Exclude strategy stress cases.")
    benchmark_root_risk.add_argument("--json", action="store_true", dest="as_json")

    benchmark_root_risk_ablation = subparsers.add_parser("benchmark-root-action-risk-ablation")
    benchmark_root_risk_ablation.add_argument("manifest")
    benchmark_root_risk_ablation.add_argument("--policy", choices=POLICY_NAMES, default="heuristic")
    benchmark_root_risk_ablation.add_argument(
        "--include-tag",
        action="append",
        help="Include cases with this tag. Repeat to allow multiple tags.",
    )
    benchmark_root_risk_ablation.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude cases with this tag. Repeat to exclude multiple tags.",
    )
    benchmark_root_risk_ablation.add_argument(
        "--phase",
        action="append",
        choices=[HandPhase.INITIAL_DEAL.value, HandPhase.DRAW.value],
        help="Restrict to a root engine phase. Repeat to allow multiple phases.",
    )
    benchmark_root_risk_ablation.add_argument(
        "--non-final",
        action="store_true",
        help="Exclude final_draw-tagged cases.",
    )
    benchmark_root_risk_ablation.add_argument(
        "--exclude-strategy",
        action="store_true",
        help="Exclude strategy stress cases.",
    )
    benchmark_root_risk_ablation.add_argument("--json", action="store_true", dest="as_json")

    compare_benchmarks = subparsers.add_parser("compare-benchmarks")
    compare_benchmarks.add_argument("left")
    compare_benchmarks.add_argument("right")
    compare_benchmarks.add_argument("--json", action="store_true", dest="as_json")

    generate_benchmark = subparsers.add_parser("generate-late-final-benchmark")
    generate_benchmark.add_argument("manifest")
    generate_benchmark.add_argument("--scenario-dir", required=True)
    generate_benchmark.add_argument("--seed", default="late-final-large")
    generate_benchmark.add_argument("--final-count", type=int, default=100)
    generate_benchmark.add_argument("--late-count", type=int, default=100)
    generate_benchmark.add_argument("--mid-count", type=int, default=50)
    generate_benchmark.add_argument("--rollouts", type=int, default=1)
    generate_benchmark.add_argument("--json", action="store_true", dest="as_json")

    generate_final_fl_benchmark = subparsers.add_parser("generate-final-draw-fantasyland-benchmark")
    generate_final_fl_benchmark.add_argument("manifest")
    generate_final_fl_benchmark.add_argument("--scenario-dir", required=True)
    generate_final_fl_benchmark.add_argument("--seed", default="final-draw-fantasyland-targeted")
    generate_final_fl_benchmark.add_argument("--count", type=int, default=150)
    generate_final_fl_benchmark.add_argument("--rollouts", type=int, default=2)
    generate_final_fl_benchmark.add_argument("--json", action="store_true", dest="as_json")

    play_hand = subparsers.add_parser(
        "play-hand",
        description=(
            "Play one hand from a single hero seat. Hero turns show solver suggestions; "
            "opponent turns ask only for visible placements."
        ),
    )
    play_hand.add_argument("--hero", choices=[player.value for player in PlayerId], help="Seat controlled by you.")
    play_hand.add_argument("--button", choices=[player.value for player in PlayerId], help="Button for this hand.")
    play_hand.add_argument(
        "--fantasyland",
        choices=[player.value for player in PlayerId],
        action="append",
        help="Mark a player as already in Fantasyland. Repeat for both players.",
    )
    play_hand.add_argument("--no-fantasyland", action="store_true", help="Start with neither player in Fantasyland.")
    play_hand.add_argument("--rollouts", type=int, default=1, help="Monte Carlo rollouts per legal hero action.")
    play_hand.add_argument("--seed", default="play-hand", help="Seed for reproducible hero solver suggestions.")
    play_hand.add_argument("--policy", choices=POLICY_NAMES, default="random", help="Rollout policy for hero suggestions.")

    return parser


def _add_draw_safe_candidate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--draw-safe-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use conservative DRAW candidate selection; use --no-draw-safe-candidates for old pattern-only draw beams.",
    )
    parser.add_argument(
        "--draw-baseline-keep",
        type=int,
        default=8,
        help="Number of heuristic-policy DRAW candidates to preserve before pattern fill.",
    )
    parser.add_argument(
        "--draw-safety-keep",
        type=int,
        default=8,
        help="Number of low-risk/survivability DRAW candidates to preserve before pattern fill.",
    )


def _add_benchmark_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-tag",
        action="append",
        help="Include cases with this tag. Repeat to allow multiple tags.",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude cases with this tag. Repeat to exclude multiple tags.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=[HandPhase.INITIAL_DEAL.value, HandPhase.DRAW.value],
        help="Restrict to a root engine phase. Repeat to allow multiple phases.",
    )


def _add_late_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--late-search", action="store_true", help="Use bounded late-street DRAW search.")
    _add_late_search_config_args(parser)


def _add_late_search_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--late-search-mode",
        choices=("auto", "exact", "beam"),
        default="auto",
        help="Late-search mode: exact, beam, or auto exact-then-beam-then-fallback.",
    )
    parser.add_argument("--late-search-max-depth", type=int, default=4)
    parser.add_argument("--late-search-max-nodes", type=int, default=500)
    parser.add_argument("--late-search-beam-size", type=int, default=8)


def _add_final_draw_auto_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--final-draw-auto-search",
        action="store_true",
        help="Use exact late-search only when a DRAW root has a tiny remaining tree.",
    )
    parser.add_argument("--final-draw-auto-max-depth", type=int, default=0)
    parser.add_argument("--final-draw-auto-max-nodes", type=int, default=64)
    parser.add_argument(
        "--final-draw-auto-continuation",
        action="store_true",
        help="Include one immediate Fantasyland continuation convention in final-draw auto exact evaluation.",
    )
    parser.add_argument(
        "--final-draw-continuation-rollouts",
        type=int,
        default=1,
        help="Number of sampled immediate Fantasyland continuation hands for final-draw auto exact evaluation.",
    )


def _emit(output, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(output.payload or {}, indent=2, sort_keys=True))
    else:
        print(output.text or "")


def _load_json_payload(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _root_action_risk_config_from_name(name: str) -> RootRiskConfig:
    if name == "default":
        return RootRiskConfig.default()
    if name == "full":
        return RootRiskConfig.all_on()
    raise ValueError(f"Unsupported root action risk config: {name!r}")


def _resolve_play_button(args) -> PlayerId:
    if args.button is not None:
        return PlayerId(args.button)
    while True:
        raw_value = input("Button player [player_1]: ").strip() or "player_1"
        try:
            return PlayerId(raw_value)
        except ValueError:
            print("Button must be player_0 or player_1", file=sys.stderr)


def _resolve_play_hero(args) -> PlayerId:
    if args.hero is not None:
        return PlayerId(args.hero)
    while True:
        raw_value = input("Hero player [player_0]: ").strip() or "player_0"
        try:
            return PlayerId(raw_value)
        except ValueError:
            print("Hero must be player_0 or player_1", file=sys.stderr)


def _resolve_play_fantasyland_flags(args) -> tuple[bool, bool]:
    if args.no_fantasyland:
        return (False, False)
    if args.fantasyland is not None:
        selected = {PlayerId(value) for value in args.fantasyland}
        return (PlayerId.PLAYER_0 in selected, PlayerId.PLAYER_1 in selected)
    return (
        _prompt_bool("Is player_0 already in Fantasyland? [y/N]: "),
        _prompt_bool("Is player_1 already in Fantasyland? [y/N]: "),
    )


def _prompt_bool(prompt: str) -> bool:
    while True:
        value = input(prompt).strip().lower()
        if value in {"", "n", "no"}:
            return False
        if value in {"y", "yes"}:
            return True
        print("Please answer y or n", file=sys.stderr)


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
