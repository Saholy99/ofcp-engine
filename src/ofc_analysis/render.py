"""Deterministic renderers for exact states, observations, and legal actions."""

from __future__ import annotations

import json
from typing import Any, Sequence, TYPE_CHECKING

from ofc.board import Board
from ofc.cards import Card, format_card
from ofc.state import GameState
from ofc_analysis.action_codec import EncodedAction
from ofc_analysis.models import RenderedOutput
from ofc_analysis.observation import PlayerObservation
from ofc_solver.models import MoveAnalysis

if TYPE_CHECKING:
    from ofc_solver.benchmark import BenchmarkAggregate, BenchmarkComparison, BenchmarkRun, RootActionRiskBenchmark


def render_state(state: GameState, *, as_json: bool = False) -> RenderedOutput:
    """Render an exact engine state for CLI or tests."""

    payload = _state_payload(state)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_state_text(payload), payload=payload)


def render_observation(observation: PlayerObservation, *, as_json: bool = False) -> RenderedOutput:
    """Render an observer-facing state projection for CLI or tests."""

    payload = _observation_payload(observation)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_observation_text(payload), payload=payload)


def render_actions(actions: Sequence[EncodedAction], *, as_json: bool = False) -> RenderedOutput:
    """Render a deterministic list of encoded root actions."""

    payload = {
        "action_count": len(actions),
        "actions": [action.as_dict() for action in actions],
    }
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_actions_text(payload), payload=payload)


def render_move_analysis(analysis: MoveAnalysis, *, as_json: bool = False) -> RenderedOutput:
    """Render Monte Carlo move-ranking output for CLI or tests."""

    payload = _move_analysis_payload(analysis)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_move_analysis_text(payload), payload=payload)


def render_benchmark_run(run: BenchmarkRun, *, as_json: bool = False) -> RenderedOutput:
    """Render solver benchmark output for CLI or tests."""

    payload = _benchmark_run_payload(run)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_benchmark_run_text(payload), payload=payload)


def render_benchmark_comparison(comparison: BenchmarkComparison, *, as_json: bool = False) -> RenderedOutput:
    """Render a side-by-side solver benchmark comparison."""

    payload = _benchmark_comparison_payload(comparison)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_benchmark_comparison_text(payload), payload=payload)


def render_root_action_risk_benchmark(
    benchmark: RootActionRiskBenchmark,
    *,
    as_json: bool = False,
) -> RenderedOutput:
    """Render the focused root-action-risk benchmark comparison."""

    payload = _root_action_risk_benchmark_payload(benchmark)
    if as_json:
        return RenderedOutput(payload=payload)
    return RenderedOutput(text=_root_action_risk_benchmark_text(payload), payload=payload)


def _cards_payload(cards: tuple[Card, ...]) -> list[str]:
    return [format_card(card) for card in cards]


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": _cards_payload(board.top),
        "middle": _cards_payload(board.middle),
        "bottom": _cards_payload(board.bottom),
    }


def _state_payload(state: GameState) -> dict[str, Any]:
    return {
        "hand_number": state.hand_number,
        "button": state.button.value,
        "acting_player": state.acting_player.value,
        "phase": state.phase.value,
        "is_continuation_hand": state.is_continuation_hand,
        "next_hand_fantasyland": list(state.next_hand_fantasyland),
        "deck": {
            "cards_remaining": state.deck.cards_remaining,
            "undealt_cards": _cards_payload(state.deck.undealt_cards),
        },
        "players": [
            {
                "player_id": player.player_id.value,
                "board": _board_payload(player.board),
                "hidden_discards": _cards_payload(player.hidden_discards),
                "hidden_discard_count": len(player.hidden_discards),
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
            for player in state.players
        ],
    }


def _observation_payload(observation: PlayerObservation) -> dict[str, Any]:
    return {
        "observer": observation.observer.value,
        "acting_player": observation.acting_player.value,
        "phase": observation.phase.value,
        "hand_number": observation.hand_number,
        "button": observation.button.value,
        "is_continuation_hand": observation.is_continuation_hand,
        "next_hand_fantasyland": list(observation.next_hand_fantasyland),
        "public_boards": [
            {
                "player_id": player_id.value,
                "board": _board_payload(board),
            }
            for player_id, board in zip(observation.public_player_ids, observation.public_boards, strict=True)
        ],
        "own_private_draw": _cards_payload(observation.own_private_draw),
        "own_hidden_discards": _cards_payload(observation.own_hidden_discards),
        "own_fantasyland_active": observation.own_fantasyland_active,
        "own_concealed_fantasyland_board": None
        if observation.own_concealed_fantasyland_board is None
        else _board_payload(observation.own_concealed_fantasyland_board),
        "own_concealed_fantasyland_discard": None
        if observation.own_concealed_fantasyland_discard is None
        else format_card(observation.own_concealed_fantasyland_discard),
        "opponent_fantasyland_active": observation.opponent_fantasyland_active,
        "opponent_hidden_discard_count": observation.opponent_hidden_discard_count,
        "unseen_card_count": observation.unseen_card_count,
    }


def _move_analysis_payload(analysis: MoveAnalysis) -> dict[str, Any]:
    return {
        "observer": analysis.observer.value,
        "phase": analysis.phase.value,
        "rollouts_per_action": analysis.rollouts_per_action,
        "rng_seed": analysis.rng_seed,
        "action_count": len(analysis.ranked_actions),
        "ranked_actions": [
            {
                "rank": rank,
                "action_index": estimate.action_index,
                "action": estimate.action.as_dict(),
                "mean_value": estimate.mean_value,
                "rollout_mean_value": estimate.rollout_mean_value,
                "stddev": estimate.stddev,
                "sample_count": estimate.sample_count,
                "min_value": estimate.min_value,
                "max_value": estimate.max_value,
                "root_risk_score": estimate.root_risk_score,
                "root_risk_reasons": list(estimate.root_risk_reasons),
            }
            for rank, estimate in enumerate(analysis.ranked_actions, start=1)
        ],
    }


def _benchmark_run_payload(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "policy_name": run.policy_name,
        "root_action_risk_enabled": run.root_action_risk_enabled,
        "case_count": run.case_count,
        "elapsed_seconds": run.elapsed_seconds,
        "cases": [
            {
                "name": case.name,
                "scenario_path": str(case.scenario_path),
                "tags": list(case.tags),
                "observer": case.observer.value,
                "phase": case.phase.value,
                "rollouts_per_action": case.rollouts_per_action,
                "rng_seed": case.rng_seed,
                "expected_top_action_indices": list(case.expected_top_action_indices),
                "top_action_index": case.top_action_index,
                "top1_agreement": case.top1_agreement,
                "top3_agreement": case.top3_agreement,
                "action_count": case.action_count,
                "elapsed_seconds": case.elapsed_seconds,
                "ranked_actions": [
                    {
                        "rank": rank,
                        "action_index": estimate.action_index,
                        "mean_value": estimate.mean_value,
                        "rollout_mean_value": estimate.rollout_mean_value,
                        "stddev": estimate.stddev,
                        "sample_count": estimate.sample_count,
                        "min_value": estimate.min_value,
                        "max_value": estimate.max_value,
                        "root_risk_score": estimate.root_risk_score,
                        "root_risk_reasons": list(estimate.root_risk_reasons),
                        "action": estimate.action.as_dict(),
                    }
                    for rank, estimate in enumerate(case.ranked_actions, start=1)
                ],
                "action_diagnostics": [
                    {
                        "action_index": diagnostic.action_index,
                        "sample_count": diagnostic.sample_count,
                        "mean_total_value": diagnostic.mean_total_value,
                        "mean_current_hand_value": diagnostic.mean_current_hand_value,
                        "mean_continuation_value": diagnostic.mean_continuation_value,
                        "continuation_frequency": diagnostic.continuation_frequency,
                        "root_foul_rate": diagnostic.root_foul_rate,
                        "opponent_foul_rate": diagnostic.opponent_foul_rate,
                        "both_foul_rate": diagnostic.both_foul_rate,
                        "root_fantasyland_frequency": diagnostic.root_fantasyland_frequency,
                        "opponent_fantasyland_frequency": diagnostic.opponent_fantasyland_frequency,
                        "mean_policy_decisions": diagnostic.mean_policy_decisions,
                        "exact_late_search_rollout_frequency": diagnostic.exact_late_search_rollout_frequency,
                        "mean_exact_late_search_decisions": diagnostic.mean_exact_late_search_decisions,
                        "mean_exact_late_search_nodes": diagnostic.mean_exact_late_search_nodes,
                    }
                    for diagnostic in case.action_diagnostics
                ],
            }
            for case in run.case_results
        ],
    }


def _benchmark_aggregate_payload(aggregate: BenchmarkAggregate) -> dict[str, Any]:
    return {
        "policy_name": aggregate.policy_name,
        "case_count": aggregate.case_count,
        "action_count": aggregate.action_count,
        "sample_count": aggregate.sample_count,
        "root_foul_rate": aggregate.root_foul_rate,
        "opponent_foul_rate": aggregate.opponent_foul_rate,
        "both_foul_rate": aggregate.both_foul_rate,
        "continuation_frequency": aggregate.continuation_frequency,
        "root_fantasyland_frequency": aggregate.root_fantasyland_frequency,
        "opponent_fantasyland_frequency": aggregate.opponent_fantasyland_frequency,
        "mean_policy_decisions": aggregate.mean_policy_decisions,
        "exact_late_search_rollout_frequency": aggregate.exact_late_search_rollout_frequency,
        "mean_exact_late_search_decisions": aggregate.mean_exact_late_search_decisions,
        "mean_exact_late_search_nodes": aggregate.mean_exact_late_search_nodes,
        "top_action_root_foul_rate": aggregate.top_action_root_foul_rate,
        "top_action_opponent_foul_rate": aggregate.top_action_opponent_foul_rate,
        "top_action_both_foul_rate": aggregate.top_action_both_foul_rate,
        "top_action_continuation_frequency": aggregate.top_action_continuation_frequency,
        "top_action_root_fantasyland_frequency": aggregate.top_action_root_fantasyland_frequency,
        "top_action_opponent_fantasyland_frequency": aggregate.top_action_opponent_fantasyland_frequency,
        "top_action_mean_policy_decisions": aggregate.top_action_mean_policy_decisions,
        "top_action_exact_late_search_rollout_frequency": (
            aggregate.top_action_exact_late_search_rollout_frequency
        ),
        "top_action_mean_exact_late_search_decisions": aggregate.top_action_mean_exact_late_search_decisions,
        "top_action_mean_exact_late_search_nodes": aggregate.top_action_mean_exact_late_search_nodes,
        "labeled_top1_rate": aggregate.labeled_top1_rate,
        "labeled_top3_rate": aggregate.labeled_top3_rate,
        "elapsed_seconds": aggregate.elapsed_seconds,
    }


def _benchmark_tag_slice_aggregate_payload(aggregate) -> dict[str, Any]:
    return {
        "case_count": aggregate.case_count,
        "action_count": aggregate.action_count,
        "sample_count": aggregate.sample_count,
        "root_foul_rate": aggregate.root_foul_rate,
        "both_foul_rate": aggregate.both_foul_rate,
        "continuation_frequency": aggregate.continuation_frequency,
        "root_fantasyland_frequency": aggregate.root_fantasyland_frequency,
        "exact_late_search_rollout_frequency": aggregate.exact_late_search_rollout_frequency,
        "top_action_root_foul_rate": aggregate.top_action_root_foul_rate,
        "top_action_both_foul_rate": aggregate.top_action_both_foul_rate,
        "top_action_continuation_frequency": aggregate.top_action_continuation_frequency,
        "top_action_root_fantasyland_frequency": aggregate.top_action_root_fantasyland_frequency,
        "top_action_exact_late_search_rollout_frequency": (
            aggregate.top_action_exact_late_search_rollout_frequency
        ),
        "labeled_top1_rate": aggregate.labeled_top1_rate,
        "labeled_top3_rate": aggregate.labeled_top3_rate,
    }


def _benchmark_comparison_payload(comparison: BenchmarkComparison) -> dict[str, Any]:
    return {
        "left_policy_name": comparison.left_policy_name,
        "right_policy_name": comparison.right_policy_name,
        "case_count": comparison.case_count,
        "left": _benchmark_aggregate_payload(comparison.left),
        "right": _benchmark_aggregate_payload(comparison.right),
        "deltas": dict(comparison.deltas),
        "top_action_changes": [
            {
                "case_name": change.case_name,
                "left_top_action_index": change.left_top_action_index,
                "right_top_action_index": change.right_top_action_index,
                "left_top_mean_value": change.left_top_mean_value,
                "right_top_mean_value": change.right_top_mean_value,
            }
            for change in comparison.top_action_changes
        ],
        "tag_slices": [
            {
                "tag": tag_slice.tag,
                "case_count": tag_slice.case_count,
                "left": _benchmark_tag_slice_aggregate_payload(tag_slice.left),
                "right": _benchmark_tag_slice_aggregate_payload(tag_slice.right),
                "deltas": dict(tag_slice.deltas),
            }
            for tag_slice in comparison.tag_slices
        ],
    }


def _root_action_risk_benchmark_payload(benchmark) -> dict[str, Any]:
    comparison_payload = _benchmark_comparison_payload(benchmark.comparison)
    comparison_payload["root_action_risk"] = {
        "enabled_on_left": benchmark.left_run.root_action_risk_enabled,
        "enabled_on_right": benchmark.right_run.root_action_risk_enabled,
        "include_tags": list(benchmark.include_tags),
        "exclude_tags": list(benchmark.exclude_tags),
        "phases": [phase.value for phase in benchmark.phases],
    }
    comparison_payload["cases"] = [
        {
            "name": left_case.name,
            "scenario_path": str(left_case.scenario_path),
            "tags": list(left_case.tags),
            "phase": left_case.phase.value,
            "left_top_action_index": left_case.top_action_index,
            "right_top_action_index": right_case.top_action_index,
            "left_elapsed_seconds": left_case.elapsed_seconds,
            "right_elapsed_seconds": right_case.elapsed_seconds,
            "left_ranked_actions": _ranked_action_payloads(left_case.ranked_actions[:5]),
            "right_ranked_actions": _ranked_action_payloads(right_case.ranked_actions[:5]),
        }
        for left_case, right_case in zip(
            benchmark.left_run.case_results,
            benchmark.right_run.case_results,
            strict=True,
        )
    ]
    return comparison_payload


def _ranked_action_payloads(ranked_actions) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "action_index": estimate.action_index,
            "mean_value": estimate.mean_value,
            "rollout_mean_value": estimate.rollout_mean_value,
            "root_risk_score": estimate.root_risk_score,
            "root_risk_reasons": list(estimate.root_risk_reasons),
            "sample_count": estimate.sample_count,
            "action": estimate.action.as_dict(),
        }
        for rank, estimate in enumerate(ranked_actions, start=1)
    ]


def _state_text(payload: dict[str, Any]) -> str:
    lines = [
        "Exact State",
        f"hand_number: {payload['hand_number']}",
        f"button: {payload['button']}",
        f"acting_player: {payload['acting_player']}",
        f"phase: {payload['phase']}",
        f"is_continuation_hand: {json.dumps(payload['is_continuation_hand'])}",
        f"next_hand_fantasyland: {json.dumps(payload['next_hand_fantasyland'])}",
        f"deck.cards_remaining: {payload['deck']['cards_remaining']}",
        f"deck.undealt_cards: {json.dumps(payload['deck']['undealt_cards'])}",
    ]
    for player in payload["players"]:
        prefix = f"players.{player['player_id']}"
        lines.extend(
            [
                f"{prefix}.board.top: {json.dumps(player['board']['top'])}",
                f"{prefix}.board.middle: {json.dumps(player['board']['middle'])}",
                f"{prefix}.board.bottom: {json.dumps(player['board']['bottom'])}",
                f"{prefix}.hidden_discards: {json.dumps(player['hidden_discards'])}",
                f"{prefix}.current_private_draw: {json.dumps(player['current_private_draw'])}",
                f"{prefix}.fantasyland_active: {json.dumps(player['fantasyland_active'])}",
                f"{prefix}.concealed_fantasyland_board: {json.dumps(player['concealed_fantasyland_board'])}",
                f"{prefix}.concealed_fantasyland_discard: {json.dumps(player['concealed_fantasyland_discard'])}",
                f"{prefix}.initial_placement_done: {json.dumps(player['initial_placement_done'])}",
                f"{prefix}.normal_draws_taken: {player['normal_draws_taken']}",
                f"{prefix}.fantasyland_set_done: {json.dumps(player['fantasyland_set_done'])}",
            ]
        )
    return "\n".join(lines)


def _observation_text(payload: dict[str, Any]) -> str:
    lines = [
        "Player Observation",
        f"observer: {payload['observer']}",
        f"acting_player: {payload['acting_player']}",
        f"phase: {payload['phase']}",
        f"hand_number: {payload['hand_number']}",
        f"button: {payload['button']}",
        f"is_continuation_hand: {json.dumps(payload['is_continuation_hand'])}",
        f"next_hand_fantasyland: {json.dumps(payload['next_hand_fantasyland'])}",
    ]
    for entry in payload["public_boards"]:
        prefix = f"public_boards.{entry['player_id']}"
        lines.extend(
            [
                f"{prefix}.top: {json.dumps(entry['board']['top'])}",
                f"{prefix}.middle: {json.dumps(entry['board']['middle'])}",
                f"{prefix}.bottom: {json.dumps(entry['board']['bottom'])}",
            ]
        )
    lines.extend(
        [
            f"own_private_draw: {json.dumps(payload['own_private_draw'])}",
            f"own_hidden_discards: {json.dumps(payload['own_hidden_discards'])}",
            f"own_fantasyland_active: {json.dumps(payload['own_fantasyland_active'])}",
            f"own_concealed_fantasyland_board: {json.dumps(payload['own_concealed_fantasyland_board'])}",
            f"own_concealed_fantasyland_discard: {json.dumps(payload['own_concealed_fantasyland_discard'])}",
            f"opponent_fantasyland_active: {json.dumps(payload['opponent_fantasyland_active'])}",
            f"opponent_hidden_discard_count: {payload['opponent_hidden_discard_count']}",
            f"unseen_card_count: {payload['unseen_card_count']}",
        ]
    )
    return "\n".join(lines)


def _actions_text(payload: dict[str, Any]) -> str:
    lines = [
        "Legal Actions",
        f"action_count: {payload['action_count']}",
    ]
    for action in payload["actions"]:
        placements = ", ".join(
            f"{placement['row']}:{placement['card']}" for placement in action["payload"]["placements"]
        )
        suffix = f" discard={action['payload']['discard']}" if "discard" in action["payload"] else ""
        lines.append(
            f"[{action['action_index']}] {action['action_type']} {action['payload']['player_id']} "
            f"placements=[{placements}]{suffix}"
        )
    return "\n".join(lines)


def _move_analysis_text(payload: dict[str, Any]) -> str:
    lines = [
        "Move Analysis",
        f"observer: {payload['observer']}",
        f"phase: {payload['phase']}",
        f"rollouts_per_action: {payload['rollouts_per_action']}",
        f"rng_seed: {json.dumps(payload['rng_seed'])}",
        f"action_count: {payload['action_count']}",
    ]
    for estimate in payload["ranked_actions"]:
        action = estimate["action"]
        placements = ", ".join(
            f"{placement['row']}:{placement['card']}" for placement in action["payload"]["placements"]
        )
        suffix = f" discard={action['payload']['discard']}" if "discard" in action["payload"] else ""
        lines.append(
            f"[{estimate['rank']}] action_index={estimate['action_index']} "
            f"mean={estimate['mean_value']:.6f} rollout_mean={_format_optional_float(estimate['rollout_mean_value'])} "
            f"root_risk={estimate['root_risk_score']:.6f} stddev={estimate['stddev']:.6f} "
            f"samples={estimate['sample_count']} min={estimate['min_value']:.6f} "
            f"max={estimate['max_value']:.6f} {action['action_type']} "
            f"{action['payload']['player_id']} placements=[{placements}]{suffix}"
        )
    return "\n".join(lines)


def _benchmark_run_text(payload: dict[str, Any]) -> str:
    lines = [
        "Solver Benchmark",
        f"policy_name: {payload['policy_name']}",
        f"case_count: {payload['case_count']}",
        f"elapsed_seconds: {payload['elapsed_seconds']:.6f}",
    ]
    for case in payload["cases"]:
        top1 = "n/a" if case["top1_agreement"] is None else json.dumps(case["top1_agreement"])
        top3 = "n/a" if case["top3_agreement"] is None else json.dumps(case["top3_agreement"])
        lines.extend(
            [
                f"case: {case['name']}",
                f"  scenario_path: {case['scenario_path']}",
                f"  observer: {case['observer']}",
                f"  phase: {case['phase']}",
                f"  tags: {json.dumps(case['tags'])}",
                f"  action_count: {case['action_count']}",
                f"  rollouts_per_action: {case['rollouts_per_action']}",
                f"  top_action_index: {case['top_action_index']}",
                f"  expected_top_action_indices: {json.dumps(case['expected_top_action_indices'])}",
                f"  top1_agreement: {top1}",
                f"  top3_agreement: {top3}",
                f"  elapsed_seconds: {case['elapsed_seconds']:.6f}",
            ]
        )
        for estimate in case["ranked_actions"][:3]:
            lines.append(
                f"  rank {estimate['rank']}: action_index={estimate['action_index']} "
                f"mean={estimate['mean_value']:.6f} "
                f"root_risk={estimate['root_risk_score']:.6f} stddev={estimate['stddev']:.6f} "
                f"samples={estimate['sample_count']}"
            )
    return "\n".join(lines)


def _benchmark_comparison_text(payload: dict[str, Any]) -> str:
    lines = [
        "Benchmark Comparison",
        f"left_policy_name: {payload['left_policy_name']}",
        f"right_policy_name: {payload['right_policy_name']}",
        f"case_count: {payload['case_count']}",
    ]
    for field in (
        "root_foul_rate",
        "opponent_foul_rate",
        "both_foul_rate",
        "continuation_frequency",
        "root_fantasyland_frequency",
        "opponent_fantasyland_frequency",
        "mean_policy_decisions",
        "exact_late_search_rollout_frequency",
        "mean_exact_late_search_decisions",
        "mean_exact_late_search_nodes",
        "top_action_root_foul_rate",
        "top_action_opponent_foul_rate",
        "top_action_both_foul_rate",
        "top_action_continuation_frequency",
        "top_action_root_fantasyland_frequency",
        "top_action_opponent_fantasyland_frequency",
        "top_action_mean_policy_decisions",
        "top_action_exact_late_search_rollout_frequency",
        "top_action_mean_exact_late_search_decisions",
        "top_action_mean_exact_late_search_nodes",
        "labeled_top1_rate",
        "labeled_top3_rate",
        "elapsed_seconds",
    ):
        left_value = _format_optional_float(payload["left"][field])
        right_value = _format_optional_float(payload["right"][field])
        delta_value = _format_optional_float(payload["deltas"][field], signed=True)
        lines.append(f"{field}: left={left_value} right={right_value} delta={delta_value}")
    if payload["top_action_changes"]:
        lines.append("top_action_changes:")
        for change in payload["top_action_changes"]:
            lines.append(
                f"  {change['case_name']}: "
                f"{change['left_top_action_index']} -> {change['right_top_action_index']} "
                f"(means {change['left_top_mean_value']:.6f} -> {change['right_top_mean_value']:.6f})"
            )
    else:
        lines.append("top_action_changes: none")
    if payload["tag_slices"]:
        lines.append("tag_slices:")
        for tag_slice in payload["tag_slices"]:
            lines.append(
                f"  {tag_slice['tag']}: cases={tag_slice['case_count']} "
                f"top_root={tag_slice['left']['top_action_root_foul_rate']:.6f}"
                f"->{tag_slice['right']['top_action_root_foul_rate']:.6f} "
                f"delta={tag_slice['deltas']['top_action_root_foul_rate']:+.6f}"
            )
    else:
        lines.append("tag_slices: none")
    return "\n".join(lines)


def _root_action_risk_benchmark_text(payload: dict[str, Any]) -> str:
    lines = [
        "Root Action Risk Benchmark",
        f"left_policy_name: {payload['left_policy_name']}",
        f"right_policy_name: {payload['right_policy_name']}",
        f"case_count: {payload['case_count']}",
        f"include_tags: {json.dumps(payload['root_action_risk']['include_tags'])}",
        f"exclude_tags: {json.dumps(payload['root_action_risk']['exclude_tags'])}",
    ]
    for field in (
        "top_action_root_foul_rate",
        "top_action_both_foul_rate",
        "top_action_continuation_frequency",
        "labeled_top1_rate",
        "labeled_top3_rate",
        "elapsed_seconds",
    ):
        left_value = _format_optional_float(payload["left"][field])
        right_value = _format_optional_float(payload["right"][field])
        delta_value = _format_optional_float(payload["deltas"][field], signed=True)
        lines.append(f"{field}: left={left_value} right={right_value} delta={delta_value}")
    for case in payload["cases"]:
        lines.append(
            f"case: {case['name']} left_top={case['left_top_action_index']} "
            f"right_top={case['right_top_action_index']}"
        )
        for estimate in case["right_ranked_actions"][:3]:
            reasons = ",".join(estimate["root_risk_reasons"]) or "none"
            lines.append(
                f"  right rank {estimate['rank']}: action_index={estimate['action_index']} "
                f"mean={estimate['mean_value']:.6f} root_risk={estimate['root_risk_score']:.6f} "
                f"reasons={reasons}"
            )
    return "\n".join(lines)


def _format_optional_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.6f}"
    return f"{value:.6f}"


__all__ = [
    "render_actions",
    "render_benchmark_comparison",
    "render_benchmark_run",
    "render_move_analysis",
    "render_observation",
    "render_root_action_risk_benchmark",
    "render_state",
]
