"""Deterministic generation for the broad solver benchmark corpus."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any

from ofc.actions import GameAction
from ofc.board import Board
from ofc.cards import format_card
from ofc.engine import new_hand, new_match
from ofc.state import GameState, PlayerId, PlayerState
from ofc.transitions import apply_action, legal_actions
from ofc_solver.models import SUPPORTED_ROOT_PHASES
from ofc_solver.rollout_policy import sample_fantasyland_set_action


@dataclass(frozen=True)
class CorpusCase:
    """One generated or referenced benchmark case."""

    name: str
    scenario_path: Path
    observer: PlayerId
    rollouts_per_action: int
    rng_seed: int | str | None
    tags: tuple[str, ...]
    state: GameState | None = None
    expected_top_action_indices: tuple[int, ...] = ()


def write_expansive_benchmark_corpus(
    output_dir: Path = Path("scenarios/benchmarks"),
    *,
    manifest_name: str = "solver_expansive.json",
    generated_dir_name: str = "generated",
) -> Path:
    """Write a broad deterministic solver benchmark corpus and return its manifest path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / generated_dir_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    for existing in generated_dir.glob("*.json"):
        existing.unlink()

    cases = _build_cases(output_dir=output_dir, generated_dir=generated_dir)
    manifest_cases = []
    for case in cases:
        if case.state is not None:
            case.scenario_path.write_text(
                json.dumps(_scenario_payload(case.state), indent=2),
                encoding="utf-8",
            )
        payload: dict[str, Any] = {
            "name": case.name,
            "scenario": _relative_path(case.scenario_path, output_dir),
            "observer": case.observer.value,
            "rollouts": case.rollouts_per_action,
            "seed": case.rng_seed,
            "tags": list(case.tags),
        }
        if case.expected_top_action_indices:
            payload["expected_top_action_indices"] = list(case.expected_top_action_indices)
        manifest_cases.append(payload)

    manifest_path = output_dir / manifest_name
    manifest_path.write_text(
        json.dumps({"version": "1", "cases": manifest_cases}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _build_cases(*, output_dir: Path, generated_dir: Path) -> tuple[CorpusCase, ...]:
    cases: list[CorpusCase] = []
    cases.extend(_regression_cases())

    for index in range(8):
        state = new_match(button=_button_for_index(index), seed=f"benchmark-initial-{index}")
        cases.append(_generated_case(generated_dir, f"generated-initial-{index:02d}", state, ("initial_deal", "generated")))

    for index in range(8):
        state = _normal_random_walk_state(f"benchmark-early-{index}", _button_for_index(index), actions_to_apply=2)
        cases.append(_generated_case(generated_dir, f"generated-early-draw-{index:02d}", state, ("early_draw", "generated")))

    for index in range(8):
        state = _normal_random_walk_state(f"benchmark-mid-{index}", _button_for_index(index), actions_to_apply=5)
        cases.append(
            _generated_case(
                generated_dir,
                f"generated-mid-draw-{index:02d}",
                state,
                ("mid_draw", "hidden_discards", "generated"),
            )
        )

    for index in range(8):
        state = _normal_random_walk_state(f"benchmark-late-{index}", _button_for_index(index), actions_to_apply=7)
        cases.append(
            _generated_case(
                generated_dir,
                f"generated-late-draw-{index:02d}",
                state,
                ("late_draw", "hidden_discards", "generated"),
            )
        )

    for index in range(8):
        actions_to_apply = 8 if index % 2 == 0 else 9
        state = _normal_random_walk_state(f"benchmark-final-{index}", _button_for_index(index), actions_to_apply=actions_to_apply)
        cases.append(
            _generated_case(
                generated_dir,
                f"generated-final-draw-{index:02d}",
                state,
                ("final_draw", "hidden_discards", "generated"),
            )
        )

    for index in range(8):
        normal_actions_to_apply = (0, 1, 2, 3, 4, 1, 3, 4)[index]
        state = _fantasyland_continuation_state(f"benchmark-fl-{index}", index, normal_actions_to_apply)
        tags = _fantasyland_tags(normal_actions_to_apply)
        cases.append(_generated_case(generated_dir, f"generated-fantasyland-{index:02d}", state, tags))

    return tuple(cases)


def _regression_cases() -> tuple[CorpusCase, ...]:
    regression_dir = Path("scenarios/regression")
    return (
        CorpusCase(
            name="regression-initial-deal",
            scenario_path=regression_dir / "initial_deal.json",
            observer=PlayerId.PLAYER_1,
            rollouts_per_action=1,
            rng_seed=301,
            tags=("initial_deal", "regression"),
        ),
        CorpusCase(
            name="regression-draw-root",
            scenario_path=regression_dir / "draw_root.json",
            observer=PlayerId.PLAYER_1,
            rollouts_per_action=1,
            rng_seed=304,
            tags=("early_draw", "regression"),
        ),
        CorpusCase(
            name="regression-opponent-hidden-discards",
            scenario_path=regression_dir / "opponent_hidden_discards.json",
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=1,
            rng_seed=305,
            tags=("early_draw", "hidden_discards", "regression"),
        ),
        CorpusCase(
            name="regression-immediate-scoring",
            scenario_path=regression_dir / "immediate_scoring.json",
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=3,
            rng_seed=302,
            tags=("final_draw", "oracle", "regression"),
            expected_top_action_indices=(1, 4),
        ),
        CorpusCase(
            name="regression-fantasyland-continuation",
            scenario_path=regression_dir / "fantasyland_continuation_ev.json",
            observer=PlayerId.PLAYER_0,
            rollouts_per_action=2,
            rng_seed=202,
            tags=("final_draw", "fantasyland", "regression"),
        ),
    )


def _generated_case(
    generated_dir: Path,
    name: str,
    state: GameState,
    tags: tuple[str, ...],
) -> CorpusCase:
    if state.phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(f"Generated benchmark state {name} has unsupported phase {state.phase.value}")
    return CorpusCase(
        name=name,
        scenario_path=generated_dir / f"{name}.json",
        observer=state.acting_player,
        rollouts_per_action=3,
        rng_seed=name,
        tags=tags,
        state=state,
    )


def _normal_random_walk_state(seed: str, button: PlayerId, *, actions_to_apply: int) -> GameState:
    rng = random.Random(seed)
    state = new_match(button=button, seed=seed)
    for _ in range(actions_to_apply):
        state = apply_action(state, _choose_random_action(state, rng))
    if state.phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(f"Random walk reached unsupported benchmark phase {state.phase.value}")
    return state


def _fantasyland_continuation_state(seed: str, index: int, normal_actions_to_apply: int) -> GameState:
    rng = random.Random(seed)
    fantasyland_player = PlayerId.PLAYER_1 if index % 2 == 0 else PlayerId.PLAYER_0
    button = PlayerId.PLAYER_0 if fantasyland_player == PlayerId.PLAYER_1 else PlayerId.PLAYER_1
    flags = (
        fantasyland_player == PlayerId.PLAYER_0,
        fantasyland_player == PlayerId.PLAYER_1,
    )
    state = new_hand(
        button=button,
        fantasyland_flags=flags,
        seed=seed,
        continuation_hand=True,
    )
    state = apply_action(state, sample_fantasyland_set_action(state, rng=rng))
    for _ in range(normal_actions_to_apply):
        state = apply_action(state, _choose_random_action(state, rng))
    if state.phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(f"Fantasyland corpus state reached unsupported phase {state.phase.value}")
    return state


def _choose_random_action(state: GameState, rng: random.Random) -> GameAction:
    actions = tuple(legal_actions(state))
    if not actions:
        raise ValueError(f"No legal actions available during {state.phase.value}")
    return actions[rng.randrange(len(actions))]


def _fantasyland_tags(normal_actions_to_apply: int) -> tuple[str, ...]:
    if normal_actions_to_apply == 0:
        phase_tag = "initial_deal"
    elif normal_actions_to_apply == 1:
        phase_tag = "early_draw"
    elif normal_actions_to_apply in {2, 3}:
        phase_tag = "mid_draw"
    else:
        phase_tag = "final_draw"
    tags = [phase_tag, "fantasyland", "continuation", "generated"]
    if normal_actions_to_apply > 1:
        tags.append("hidden_discards")
    return tuple(tags)


def _button_for_index(index: int) -> PlayerId:
    return PlayerId.PLAYER_0 if index % 2 == 0 else PlayerId.PLAYER_1


def _scenario_payload(state: GameState) -> dict[str, Any]:
    return {
        "version": "1",
        "state": {
            "hand_number": state.hand_number,
            "button": state.button.value,
            "acting_player": state.acting_player.value,
            "phase": state.phase.value,
            "is_continuation_hand": state.is_continuation_hand,
            "next_hand_fantasyland": list(state.next_hand_fantasyland),
            "deck": {
                "undealt_cards": _cards_payload(state.deck.undealt_cards),
            },
            "players": [_player_payload(player) for player in state.players],
        },
    }


def _player_payload(player: PlayerState) -> dict[str, Any]:
    return {
        "player_id": player.player_id.value,
        "board": _board_payload(player.board),
        "hidden_discards": _cards_payload(player.hidden_discards),
        "current_private_draw": _cards_payload(player.current_private_draw),
        "fantasyland_active": player.fantasyland_active,
        "concealed_fantasyland_board": None
        if player.concealed_fantasyland_board is None
        else _board_payload(player.concealed_fantasyland_board),
        "concealed_fantasyland_discard": None
        if player.concealed_fantasyland_discard is None
        else format_card(player.concealed_fantasyland_discard),
        "initial_placement_done": player.initial_placement_done,
        "normal_draws_taken": player.normal_draws_taken,
        "fantasyland_set_done": player.fantasyland_set_done,
    }


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": _cards_payload(board.top),
        "middle": _cards_payload(board.middle),
        "bottom": _cards_payload(board.bottom),
    }


def _cards_payload(cards) -> list[str]:
    return [format_card(card) for card in cards]


def _relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=Path(start).resolve())).as_posix()


def main() -> int:
    """Generate the default expansive benchmark corpus."""

    manifest_path = write_expansive_benchmark_corpus()
    print(manifest_path)
    return 0


__all__ = ["CorpusCase", "write_expansive_benchmark_corpus"]


if __name__ == "__main__":
    raise SystemExit(main())
