"""Deterministic benchmark corpus generation for solver analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from ofc.board import Board, board_full, visible_cards
from ofc.cards import Card, format_card, full_deck, parse_card
from ofc.config import DEFAULT_CONFIG
from ofc.deck import DeckState
from ofc.engine import new_match, showdown
from ofc.scoring import is_foul
from ofc.state import GameState, HandPhase, PlayerId, PlayerState, get_player, player_index
from ofc.transitions import apply_action, legal_actions
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy


@dataclass(frozen=True)
class GeneratedBenchmarkSummary:
    """Summary of a generated benchmark corpus."""

    manifest_path: Path
    scenario_dir: Path
    case_count: int
    tag_counts: dict[str, int]
    legal_action_count: int = 0
    fantasyland_trigger_case_count: int = 0
    fantasyland_trigger_action_count: int = 0


@dataclass(frozen=True)
class _TargetedTemplate:
    name: str
    mode: str
    fantasy_rank: str
    middle_rank: str
    middle_pair_rank: str
    bottom_rank: str
    top_kicker: str
    bottom_kicker: str
    draw_kicker: str
    opponent_style: str


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


def generate_final_draw_fantasyland_benchmark(
    *,
    manifest_path: str | Path,
    scenario_dir: str | Path,
    seed: int | str = "final-draw-fantasyland-targeted",
    count: int = 150,
    rollouts: int = 2,
) -> GeneratedBenchmarkSummary:
    """Generate deterministic terminal final-draw cases with Fantasyland choices."""

    manifest = Path(manifest_path)
    cases_dir = Path(scenario_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    templates = _targeted_templates(seed=seed, count=count)
    cases: list[dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()
    legal_action_count = 0
    trigger_case_count = 0
    trigger_action_count = 0

    for index, template in enumerate(templates):
        acting_player = PlayerId.PLAYER_0 if index % 2 == 0 else PlayerId.PLAYER_1
        state = _targeted_final_draw_state(template, acting_player=acting_player, hand_number=index + 1)
        trigger_summary = _fantasyland_trigger_summary(state)
        if trigger_summary["trigger_actions"] <= 0:
            raise ValueError(f"targeted template {template.name} did not produce a Fantasyland-triggering action")

        name = f"generated-final-draw-fl-{index:03d}"
        scenario_path = cases_dir / f"{name}.json"
        scenario_path.write_text(
            json.dumps(_scenario_payload_from_state(state), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tags = _targeted_case_tags(state, template=template, trigger_summary=trigger_summary)
        tag_counter.update(tags)
        legal_action_count += trigger_summary["legal_actions"]
        trigger_action_count += trigger_summary["trigger_actions"]
        trigger_case_count += int(trigger_summary["trigger_actions"] > 0)
        cases.append(
            {
                "name": name,
                "scenario": _relative_path(scenario_path, manifest.parent),
                "observer": state.acting_player.value,
                "rollouts": rollouts,
                "seed": f"{seed}:{name}",
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
        legal_action_count=legal_action_count,
        fantasyland_trigger_case_count=trigger_case_count,
        fantasyland_trigger_action_count=trigger_action_count,
    )


def _targeted_templates(*, seed: int | str, count: int) -> tuple[_TargetedTemplate, ...]:
    rng = random.Random(seed)
    fantasy_ranks = ("Q", "K", "A")
    middle_ranks = ("J", "T", "9", "8")
    bottom_ranks = ("T", "9", "8", "7")
    pair_ranks = ("6", "5", "4", "3")
    low_ranks = ("2", "3", "4", "5", "6", "7", "8", "9")
    styles = ("balanced", "opponent_strong", "opponent_foul", "opponent_weak")
    modes = ("pair", "pair", "pair", "trips", "trips")
    templates: list[_TargetedTemplate] = []
    attempts = 0
    while len(templates) < count:
        mode = modes[attempts % len(modes)]
        fantasy_rank = fantasy_ranks[(attempts + rng.randrange(len(fantasy_ranks))) % len(fantasy_ranks)]
        middle_rank = _first_distinct(middle_ranks, {fantasy_rank}, offset=attempts)
        bottom_rank = _first_distinct(bottom_ranks, {fantasy_rank, middle_rank}, offset=attempts + 1)
        middle_pair_rank = _first_distinct(pair_ranks, {fantasy_rank, middle_rank, bottom_rank}, offset=attempts + 2)
        used_ranks = {fantasy_rank, middle_rank, bottom_rank, middle_pair_rank}
        low_choices = [rank for rank in low_ranks if rank not in used_ranks]
        if len(low_choices) < 3:
            attempts += 1
            continue
        rng.shuffle(low_choices)
        templates.append(
            _TargetedTemplate(
                name=f"target-{len(templates):03d}",
                mode=mode,
                fantasy_rank=fantasy_rank,
                middle_rank=middle_rank,
                middle_pair_rank=middle_pair_rank,
                bottom_rank=bottom_rank,
                top_kicker=low_choices[0],
                bottom_kicker=low_choices[1],
                draw_kicker=low_choices[2],
                opponent_style=styles[(attempts + rng.randrange(len(styles))) % len(styles)],
            )
        )
        attempts += 1
    return tuple(templates)


def _first_distinct(candidates: tuple[str, ...], disallowed: set[str], *, offset: int) -> str:
    for step in range(len(candidates)):
        candidate = candidates[(offset + step) % len(candidates)]
        if candidate not in disallowed:
            return candidate
    raise ValueError("could not select a distinct rank")


def _targeted_final_draw_state(
    template: _TargetedTemplate,
    *,
    acting_player: PlayerId,
    hand_number: int,
) -> GameState:
    actor = _targeted_actor(template, acting_player)
    opponent_id = PlayerId.PLAYER_1 if acting_player == PlayerId.PLAYER_0 else PlayerId.PLAYER_0
    opponent = _targeted_opponent(
        opponent_id,
        style=template.opponent_style,
        excluded=set(_physical_player_cards(actor)),
    )
    players = (actor, opponent) if acting_player == PlayerId.PLAYER_0 else (opponent, actor)
    used_cards = tuple(card for player in players for card in _physical_player_cards(player))
    deck = DeckState(undealt_cards=tuple(card for card in full_deck() if card not in set(used_cards)))
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=hand_number,
        button=PlayerId.PLAYER_1 if acting_player == PlayerId.PLAYER_0 else PlayerId.PLAYER_0,
        acting_player=acting_player,
        phase=HandPhase.DRAW,
        deck=deck,
        players=players,
        is_continuation_hand=False,
        next_hand_fantasyland=(False, False),
    )


def _targeted_actor(template: _TargetedTemplate, player_id: PlayerId) -> PlayerState:
    fantasy_cards = _rank_cards(template.fantasy_rank)
    middle_cards = _rank_cards(template.middle_rank)
    middle_pair_cards = _rank_cards(template.middle_pair_rank)
    bottom_cards = _rank_cards(template.bottom_rank)
    if template.mode == "trips":
        top = (fantasy_cards[0], fantasy_cards[1])
        middle = middle_cards[:3] + middle_pair_cards[:2]
    else:
        top = (fantasy_cards[0], _card(template.top_kicker, "c"))
        middle = middle_cards[:3] + (_card("A", "c"), _card("K", "c"))
    bottom = bottom_cards[:3] + (_card(template.bottom_kicker, "d"),)
    draw = (fantasy_cards[2], bottom_cards[3], _card(template.draw_kicker, "h"))
    hidden_discards = _hidden_discards(excluded=set(top + middle + bottom + draw), count=3)
    return PlayerState(
        player_id=player_id,
        board=Board(top=top, middle=middle, bottom=bottom),
        hidden_discards=hidden_discards,
        current_private_draw=draw,
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=3,
        fantasyland_set_done=False,
    )


def _targeted_opponent(player_id: PlayerId, *, style: str, excluded: set[Card]) -> PlayerState:
    local_excluded = set(excluded)
    if style == "opponent_strong":
        top_rank, middle_rank, middle_pair, bottom_rank, kicker = _available_ranks(
            local_excluded,
            counts=(2, 3, 2, 4, 1),
            preferred=("T", "K", "8", "A", "2", "Q", "J", "9", "7", "6", "5", "4", "3"),
        )
        top = _take_rank_cards(top_rank, 2, local_excluded) + _take_any_cards(1, local_excluded)
        middle = _take_rank_cards(middle_rank, 3, local_excluded) + _take_rank_cards(middle_pair, 2, local_excluded)
        bottom = _take_rank_cards(bottom_rank, 4, local_excluded) + _take_rank_cards(kicker, 1, local_excluded)
    elif style == "opponent_foul":
        top_rank, middle_rank, bottom_rank, kicker_a, kicker_b, kicker_c, kicker_d = _available_ranks(
            local_excluded,
            counts=(2, 2, 2, 1, 1, 1, 1),
            preferred=("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"),
        )
        top = _take_rank_cards(top_rank, 2, local_excluded) + _take_rank_cards(kicker_a, 1, local_excluded)
        middle = _take_rank_cards(middle_rank, 2, local_excluded) + _take_rank_cards(kicker_b, 1, local_excluded) + _take_rank_cards(kicker_c, 1, local_excluded) + _take_any_cards(1, local_excluded)
        bottom = _take_rank_cards(bottom_rank, 2, local_excluded) + _take_rank_cards(kicker_d, 1, local_excluded) + _take_any_cards(2, local_excluded)
    elif style == "opponent_weak":
        top_rank, middle_rank, bottom_rank = _available_ranks(
            local_excluded,
            counts=(1, 2, 2),
            preferred=("8", "9", "T", "7", "6", "5", "4", "3", "2", "J", "Q", "K", "A"),
        )
        top_made = _take_rank_cards(top_rank, 1, local_excluded)
        middle_made = _take_rank_cards(middle_rank, 2, local_excluded)
        bottom_made = _take_rank_cards(bottom_rank, 2, local_excluded)
        top = top_made + _take_any_cards(2, local_excluded)
        middle = middle_made + _take_any_cards(3, local_excluded)
        bottom = bottom_made + _take_any_cards(3, local_excluded)
    else:
        top_rank, middle_rank, bottom_rank = _available_ranks(
            local_excluded,
            counts=(2, 3, 3),
            preferred=("9", "J", "Q", "T", "8", "7", "6", "5", "4", "3", "2", "K", "A"),
        )
        top_made = _take_rank_cards(top_rank, 2, local_excluded)
        middle_made = _take_rank_cards(middle_rank, 3, local_excluded)
        bottom_made = _take_rank_cards(bottom_rank, 3, local_excluded)
        top = top_made + _take_any_cards(1, local_excluded)
        middle = middle_made + _take_any_cards(2, local_excluded)
        bottom = bottom_made + _take_any_cards(2, local_excluded)
    board = Board(top=top, middle=middle, bottom=bottom)
    hidden_discards = _take_any_cards(4, local_excluded)
    return PlayerState(
        player_id=player_id,
        board=board,
        hidden_discards=hidden_discards,
        current_private_draw=(),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=4,
        fantasyland_set_done=False,
    )


def _available_ranks(
    excluded: set[Card],
    *,
    counts: tuple[int, ...],
    preferred: tuple[str, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    excluded_ranks: set[str] = set()
    for count in counts:
        for rank in preferred:
            if rank in excluded_ranks:
                continue
            available = sum(1 for suit in ("h", "d", "s", "c") if _card(rank, suit) not in excluded)
            if available >= count:
                selected.append(rank)
                excluded_ranks.add(rank)
                break
        else:
            raise ValueError("not enough rank availability for targeted opponent")
    return tuple(selected)


def _take_rank_cards(rank: str, count: int, excluded: set[Card]) -> tuple[Card, ...]:
    cards: list[Card] = []
    for suit in ("h", "d", "s", "c"):
        card = _card(rank, suit)
        if card in excluded:
            continue
        cards.append(card)
        excluded.add(card)
        if len(cards) == count:
            return tuple(cards)
    raise ValueError(f"not enough cards for rank {rank}")


def _take_any_cards(count: int, excluded: set[Card]) -> tuple[Card, ...]:
    cards: list[Card] = []
    for card in full_deck():
        if card in excluded:
            continue
        cards.append(card)
        excluded.add(card)
        if len(cards) == count:
            return tuple(cards)
    raise ValueError("not enough available cards")


def _targeted_case_tags(
    state: GameState,
    *,
    template: _TargetedTemplate,
    trigger_summary: dict[str, int],
) -> list[str]:
    tags = [
        "final_draw",
        "fantasyland_targeted",
        "fantasyland_entry",
        "generated",
        state.acting_player.value,
    ]
    if trigger_summary["trigger_actions"] < trigger_summary["legal_actions"]:
        tags.append("fantasyland_miss")
    if trigger_summary["foul_actions"] > 0:
        tags.append("foul_risk")
    if trigger_summary["legal_actions"] > trigger_summary["trigger_actions"] + trigger_summary["foul_actions"]:
        tags.append("safe_value")
    if any(player.hidden_discards for player in state.players):
        tags.append("hidden_discards")
    if template.mode == "trips":
        tags.append("top_trips")
    if trigger_summary["foul_actions"] > 0:
        tags.append("both_foul_pressure" if trigger_summary["both_foul_actions"] > 0 else "foul_pressure")
    return tags


def _fantasyland_trigger_summary(state: GameState) -> dict[str, int]:
    legal_count = 0
    trigger_count = 0
    foul_count = 0
    both_foul_count = 0
    for action in legal_actions(state):
        legal_count += 1
        next_state = apply_action(state, action)
        if next_state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
            raise ValueError("targeted final-draw action did not reach terminal showdown")
        terminal_state, result = showdown(next_state)
        root_breakdown = result.left if result.left.player_id == state.acting_player.value else result.right
        opponent_breakdown = result.right if result.left.player_id == state.acting_player.value else result.left
        trigger_count += int(terminal_state.next_hand_fantasyland[player_index(state.acting_player)])
        foul_count += int(root_breakdown.fouled)
        both_foul_count += int(root_breakdown.fouled and opponent_breakdown.fouled)
    return {
        "legal_actions": legal_count,
        "trigger_actions": trigger_count,
        "foul_actions": foul_count,
        "both_foul_actions": both_foul_count,
    }


def _rank_cards(rank: str) -> tuple[Card, Card, Card, Card]:
    return (_card(rank, "h"), _card(rank, "d"), _card(rank, "s"), _card(rank, "c"))


def _card(rank: str, suit: str) -> Card:
    return parse_card(f"{rank}{suit}")


def _cards(tokens: str) -> tuple[Card, ...]:
    return tuple(parse_card(token) for token in tokens.split())


def _hidden_discards(*, excluded: set[Card], count: int) -> tuple[Card, ...]:
    discards: list[Card] = []
    for card in full_deck():
        if card in excluded:
            continue
        discards.append(card)
        if len(discards) == count:
            return tuple(discards)
    raise ValueError("not enough cards for hidden discards")


def _physical_player_cards(player: PlayerState) -> tuple[Card, ...]:
    cards: list[Card] = []
    cards.extend(visible_cards(player.board))
    cards.extend(player.hidden_discards)
    cards.extend(player.current_private_draw)
    if player.concealed_fantasyland_board is not None:
        cards.extend(visible_cards(player.concealed_fantasyland_board))
    return tuple(cards)


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
