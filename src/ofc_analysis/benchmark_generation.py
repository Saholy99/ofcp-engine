"""Deterministic benchmark corpus generation for solver analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from ofc.board import Board, board_full, visible_cards
from ofc.cards import format_card
from ofc.engine import new_match, showdown
from ofc.scoring import is_foul
from ofc.state import GameState, HandPhase, PlayerId, get_player
from ofc.transitions import apply_action, legal_actions
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy


@dataclass(frozen=True)
class GeneratedBenchmarkSummary:
    """Summary of a generated benchmark corpus."""

    manifest_path: Path
    scenario_dir: Path
    case_count: int
    tag_counts: dict[str, int]


def generate_late_final_benchmark(
    *,
    manifest_path: str | Path,
    scenario_dir: str | Path,
    seed: int | str = "late-final-large",
    final_count: int = 100,
    late_count: int = 100,
    mid_count: int = 50,
    rollouts: int = 1,
) -> GeneratedBenchmarkSummary:
    """Generate a deterministic final/late draw benchmark manifest and scenarios."""

    manifest = Path(manifest_path)
    cases_dir = Path(scenario_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()
    targets = (
        ("final_draw", final_count, 1),
        ("late_draw", late_count, 2),
        ("mid_draw", mid_count, 3),
    )
    for tag, count, remaining_decisions in targets:
        for index, state in enumerate(
            _generate_states(
                tag=tag,
                count=count,
                remaining_decisions=remaining_decisions,
                seed=seed,
            )
        ):
            name = f"generated-{tag.replace('_', '-')}-{index:03d}"
            scenario_path = cases_dir / f"{name}.json"
            scenario_path.write_text(
                json.dumps(_scenario_payload_from_state(state), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tags = _case_tags(state, primary_tag=tag)
            tag_counter.update(tags)
            cases.append(
                {
                    "name": name,
                    "scenario": _relative_path(scenario_path, manifest.parent),
                    "observer": state.acting_player.value,
                    "rollouts": rollouts,
                    "seed": name,
                    "tags": tags,
                }
            )

    manifest.write_text(
        json.dumps({"version": "1", "cases": cases}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return GeneratedBenchmarkSummary(
        manifest_path=manifest,
        scenario_dir=cases_dir,
        case_count=len(cases),
        tag_counts=dict(tag_counter),
    )


def _generate_states(
    *,
    tag: str,
    count: int,
    remaining_decisions: int,
    seed: int | str,
) -> tuple[GameState, ...]:
    states: list[GameState] = []
    seen: set[str] = set()
    attempts = 0
    policy = HeuristicRolloutPolicy()
    max_attempts = max(1000, count * 200)
    while len(states) < count and attempts < max_attempts:
        button = PlayerId.PLAYER_0 if attempts % 2 == 0 else PlayerId.PLAYER_1
        state = new_match(button=button, seed=f"{seed}:{tag}:{attempts}")
        rng = random.Random(f"{seed}:{tag}:policy:{attempts}")
        for _ in range(16):
            if state.phase == HandPhase.DRAW and _remaining_normal_draw_decisions(state) == remaining_decisions:
                signature = _state_signature(state)
                if signature not in seen:
                    seen.add(signature)
                    states.append(state)
                    break
            if state.phase not in {HandPhase.INITIAL_DEAL, HandPhase.DRAW}:
                break
            actions = tuple(legal_actions(state))
            if not actions:
                break
            state = apply_action(state, policy.choose_action(state, rng=rng))
        attempts += 1
    if len(states) < count:
        raise ValueError(f"Could only generate {len(states)} {tag} cases after {attempts} attempts")
    return tuple(states)


def _remaining_normal_draw_decisions(state: GameState) -> int:
    return sum(
        state.config.normal_draw_turns_per_player - player.normal_draws_taken
        for player in state.players
        if not player.fantasyland_active
    )


def _case_tags(state: GameState, *, primary_tag: str) -> list[str]:
    tags = [primary_tag, "generated", state.acting_player.value]
    if any(player.hidden_discards for player in state.players):
        tags.append("hidden_discards")
    if _has_fantasyland_potential(state):
        tags.append("fantasyland")
    tags.append("doomed" if _is_doomed_final_draw(state) else "survivable")
    return tags


def _has_fantasyland_potential(state: GameState) -> bool:
    player = get_player(state, state.acting_player)
    top_cards = player.board.top + player.current_private_draw
    ranks = [card.rank for card in top_cards]
    return any(rank >= 12 and ranks.count(rank) >= 2 for rank in set(ranks))


def _is_doomed_final_draw(state: GameState) -> bool:
    if _remaining_normal_draw_decisions(state) != 1:
        return False
    outcomes = []
    for action in legal_actions(state):
        next_state = apply_action(state, action)
        if next_state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
            return False
        terminal_state, _ = showdown(next_state)
        board = get_player(terminal_state, state.acting_player).board
        outcomes.append(board_full(board) and is_foul(board))
    return bool(outcomes) and all(outcomes)


def _scenario_payload_from_state(state: GameState) -> dict[str, Any]:
    return {
        "version": "1",
        "state": {
            "hand_number": state.hand_number,
            "button": state.button.value,
            "acting_player": state.acting_player.value,
            "phase": state.phase.value,
            "is_continuation_hand": state.is_continuation_hand,
            "next_hand_fantasyland": list(state.next_hand_fantasyland),
            "deck": {"undealt_cards": [format_card(card) for card in state.deck.undealt_cards]},
            "players": [
                {
                    "player_id": player.player_id.value,
                    "board": _board_payload(player.board),
                    "hidden_discards": [format_card(card) for card in player.hidden_discards],
                    "current_private_draw": [format_card(card) for card in player.current_private_draw],
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
                for player in state.players
            ],
        },
    }


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": [format_card(card) for card in board.top],
        "middle": [format_card(card) for card in board.middle],
        "bottom": [format_card(card) for card in board.bottom],
    }


def _state_signature(state: GameState) -> str:
    pieces: list[str] = [state.acting_player.value]
    for player in state.players:
        pieces.extend(format_card(card) for card in visible_cards(player.board))
        pieces.extend(format_card(card) for card in player.hidden_discards)
        pieces.extend(format_card(card) for card in player.current_private_draw)
    return "|".join(pieces)


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


__all__ = ["GeneratedBenchmarkSummary", "generate_late_final_benchmark"]
