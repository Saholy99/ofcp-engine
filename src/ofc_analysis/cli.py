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
from ofc_analysis.render import render_actions, render_move_analysis, render_observation, render_state
from ofc_analysis.scenario import load_scenario
from ofc_solver.monte_carlo import rank_actions_from_observation


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
            )
            output = render_move_analysis(analysis, as_json=args.as_json)
            _emit(output, as_json=args.as_json)
            return 0
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
    solve_move.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _emit(output, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(output.payload or {}, indent=2, sort_keys=True))
    else:
        print(output.text or "")


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
