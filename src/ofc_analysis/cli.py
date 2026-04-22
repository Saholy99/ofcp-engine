"""CLI for reproducible inspection of exact states and legal root actions."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import json
import sys

from ofc.state import HandPhase, PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_actions
from ofc_analysis.observation import project_observation
from ofc_analysis.play import run_play_hand
from ofc_analysis.render import (
    render_actions,
    render_benchmark_comparison,
    render_benchmark_run,
    render_move_analysis,
    render_observation,
    render_state,
)
from ofc_analysis.scenario import load_scenario
from ofc_solver.benchmark import compare_benchmark_payloads, load_benchmark_manifest, run_benchmark_manifest
from ofc_solver.monte_carlo import rank_actions_from_observation
from ofc_solver.policy_registry import POLICY_NAMES, policy_from_name


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
            )
            output = render_move_analysis(analysis, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "benchmark-solver":
            manifest = load_benchmark_manifest(args.manifest)
            run = run_benchmark_manifest(manifest, policy_name=args.policy)
            output = render_benchmark_run(run, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0

        if args.command == "compare-benchmarks":
            left_payload = _load_json_payload(args.left)
            right_payload = _load_json_payload(args.right)
            comparison = compare_benchmark_payloads(left_payload, right_payload)
            output = render_benchmark_comparison(comparison, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
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
    solve_move.add_argument("--json", action="store_true", dest="as_json")

    benchmark_solver = subparsers.add_parser("benchmark-solver")
    benchmark_solver.add_argument("manifest")
    benchmark_solver.add_argument("--policy", choices=POLICY_NAMES, default="random")
    benchmark_solver.add_argument("--json", action="store_true", dest="as_json")

    compare_benchmarks = subparsers.add_parser("compare-benchmarks")
    compare_benchmarks.add_argument("left")
    compare_benchmarks.add_argument("right")
    compare_benchmarks.add_argument("--json", action="store_true", dest="as_json")

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


def _emit(output, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(output.payload or {}, indent=2, sort_keys=True))
    else:
        print(output.text or "")


def _load_json_payload(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
