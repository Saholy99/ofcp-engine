"""Deterministic full-hand policy benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

from ofc.actions import GameAction
from ofc.engine import new_match, showdown
from ofc.scoring import is_foul
from ofc.state import HandPhase, PlayerId, effective_board, player_index
from ofc.transitions import apply_action, legal_actions
from ofc_analysis.observation import project_observation
from ofc_solver.monte_carlo import rank_actions_from_observation
from ofc_solver.policy_registry import policy_from_name


@dataclass(frozen=True)
class FullHandResult:
    """One deterministic full-hand benchmark result."""

    seed: str
    player_0_policy: str
    player_1_policy: str
    player_0_points: int
    player_1_points: int
    player_0_fouled: bool
    player_1_fouled: bool
    player_0_fantasyland: bool
    player_1_fantasyland: bool
    player_0_royalties: int
    player_1_royalties: int
    decision_count: int
    elapsed_seconds: float
    initial_early_search_activations: int = 0
    final_draw_auto_activations: int = 0
    continuation_aware_final_draw_activations: int = 0

    @property
    def both_fouled(self) -> bool:
        return self.player_0_fouled and self.player_1_fouled


@dataclass(frozen=True)
class FullHandBenchmark:
    """Aggregate deterministic full-hand benchmark output."""

    player_0_policy: str
    player_1_policy: str
    hand_count: int
    seed: int | str
    rollouts_per_action: int
    requested_hand_count: int
    hands: tuple[FullHandResult, ...]
    elapsed_seconds: float
    jobs: int = 1
    max_seconds: float | None = None
    stopped_early: bool = False

    @property
    def total_player_0_points(self) -> int:
        return sum(hand.player_0_points for hand in self.hands)

    @property
    def total_player_1_points(self) -> int:
        return sum(hand.player_1_points for hand in self.hands)

    @property
    def average_player_0_points(self) -> float:
        return self.total_player_0_points / self.hand_count

    @property
    def average_player_1_points(self) -> float:
        return self.total_player_1_points / self.hand_count

    @property
    def standard_deviation_player_0_points(self) -> float:
        return _sample_stdev(hand.player_0_points for hand in self.hands)

    @property
    def standard_error_player_0_points(self) -> float:
        if self.hand_count <= 1:
            return 0.0
        return self.standard_deviation_player_0_points / math.sqrt(self.hand_count)

    @property
    def player_0_foul_rate(self) -> float:
        return _rate(hand.player_0_fouled for hand in self.hands)

    @property
    def player_1_foul_rate(self) -> float:
        return _rate(hand.player_1_fouled for hand in self.hands)

    @property
    def both_foul_rate(self) -> float:
        return _rate(hand.both_fouled for hand in self.hands)

    @property
    def player_0_fantasyland_rate(self) -> float:
        return _rate(hand.player_0_fantasyland for hand in self.hands)

    @property
    def player_1_fantasyland_rate(self) -> float:
        return _rate(hand.player_1_fantasyland for hand in self.hands)

    @property
    def average_player_0_royalties(self) -> float:
        return sum(hand.player_0_royalties for hand in self.hands) / self.hand_count

    @property
    def average_player_1_royalties(self) -> float:
        return sum(hand.player_1_royalties for hand in self.hands) / self.hand_count

    @property
    def average_runtime_per_hand(self) -> float:
        return self.elapsed_seconds / self.hand_count

    @property
    def zero_sum_consistent(self) -> bool:
        return all(hand.player_0_points + hand.player_1_points == 0 for hand in self.hands)

    @property
    def initial_early_search_activations(self) -> int:
        return sum(hand.initial_early_search_activations for hand in self.hands)

    @property
    def final_draw_auto_activations(self) -> int:
        return sum(hand.final_draw_auto_activations for hand in self.hands)

    @property
    def continuation_aware_final_draw_activations(self) -> int:
        return sum(hand.continuation_aware_final_draw_activations for hand in self.hands)

    def as_dict(self, *, include_hands: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "player_0_policy": self.player_0_policy,
            "player_1_policy": self.player_1_policy,
            "hand_count": self.hand_count,
            "requested_hand_count": self.requested_hand_count,
            "seed": self.seed,
            "rollouts_per_action": self.rollouts_per_action,
            "jobs": self.jobs,
            "max_seconds": self.max_seconds,
            "stopped_early": self.stopped_early,
            "total_player_0_points": self.total_player_0_points,
            "total_player_1_points": self.total_player_1_points,
            "average_player_0_points": self.average_player_0_points,
            "average_player_1_points": self.average_player_1_points,
            "standard_deviation_player_0_points": self.standard_deviation_player_0_points,
            "standard_error_player_0_points": self.standard_error_player_0_points,
            "player_0_foul_rate": self.player_0_foul_rate,
            "player_1_foul_rate": self.player_1_foul_rate,
            "both_foul_rate": self.both_foul_rate,
            "player_0_fantasyland_rate": self.player_0_fantasyland_rate,
            "player_1_fantasyland_rate": self.player_1_fantasyland_rate,
            "average_player_0_royalties": self.average_player_0_royalties,
            "average_player_1_royalties": self.average_player_1_royalties,
            "average_runtime_per_hand": self.average_runtime_per_hand,
            "zero_sum_consistent": self.zero_sum_consistent,
            "initial_early_search_activations": self.initial_early_search_activations,
            "final_draw_auto_activations": self.final_draw_auto_activations,
            "continuation_aware_final_draw_activations": self.continuation_aware_final_draw_activations,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_hands:
            payload["hands"] = [self._hand_payload(hand) for hand in self.hands]
        return payload

    def _hand_payload(self, hand: FullHandResult) -> dict[str, object]:
        return {
            "seed": hand.seed,
            "player_0_points": hand.player_0_points,
            "player_1_points": hand.player_1_points,
            "player_0_fouled": hand.player_0_fouled,
            "player_1_fouled": hand.player_1_fouled,
            "both_fouled": hand.both_fouled,
            "player_0_fantasyland": hand.player_0_fantasyland,
            "player_1_fantasyland": hand.player_1_fantasyland,
            "player_0_royalties": hand.player_0_royalties,
            "player_1_royalties": hand.player_1_royalties,
            "decision_count": hand.decision_count,
            "initial_early_search_activations": hand.initial_early_search_activations,
            "final_draw_auto_activations": hand.final_draw_auto_activations,
            "continuation_aware_final_draw_activations": hand.continuation_aware_final_draw_activations,
            "elapsed_seconds": hand.elapsed_seconds,
        }

    def deterministic_payload(self) -> dict[str, object]:
        payload = self.as_dict()
        payload.pop("elapsed_seconds", None)
        payload.pop("average_runtime_per_hand", None)
        payload.pop("jobs", None)
        for hand in payload["hands"]:  # type: ignore[index]
            hand.pop("elapsed_seconds", None)
        return payload


def run_full_hand_benchmark(
    *,
    player_0_policy: str,
    player_1_policy: str,
    hand_count: int,
    seed: int | str,
    rollouts_per_action: int,
    jobs: int = 1,
    save_traces: bool = False,
    trace_dir: Path | str | None = None,
    max_seconds: float | None = None,
) -> FullHandBenchmark:
    """Run deterministic complete normal hands for two solver policies."""

    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if rollouts_per_action <= 0:
        raise ValueError("rollouts_per_action must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    _validate_policy_name(player_0_policy)
    _validate_policy_name(player_1_policy)

    start = time.perf_counter()
    tasks = tuple(
        _FullHandTask(
            hand_index=index,
            seed=f"{seed}:{index}",
            player_0_policy=player_0_policy,
            player_1_policy=player_1_policy,
            rollouts_per_action=rollouts_per_action,
        )
        for index in range(hand_count)
    )
    if jobs == 1 or max_seconds is not None:
        hands_list: list[FullHandResult] = []
        stopped_early = False
        for task in tasks:
            if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                stopped_early = True
                break
            hands_list.append(_run_one_hand_task(task))
        hands = tuple(hands_list)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            hands = tuple(executor.map(_run_one_hand_task, tasks))
        stopped_early = False
    if not hands:
        raise ValueError("No full-hand benchmark hands completed")
    benchmark = FullHandBenchmark(
        player_0_policy=player_0_policy,
        player_1_policy=player_1_policy,
        hand_count=len(hands),
        requested_hand_count=hand_count,
        seed=seed,
        rollouts_per_action=rollouts_per_action,
        jobs=jobs,
        max_seconds=max_seconds,
        stopped_early=stopped_early,
        hands=hands,
        elapsed_seconds=time.perf_counter() - start,
    )
    if save_traces:
        _write_traces(benchmark, trace_dir)
    return benchmark


@dataclass(frozen=True)
class _FullHandTask:
    hand_index: int
    seed: str
    player_0_policy: str
    player_1_policy: str
    rollouts_per_action: int


@dataclass(frozen=True)
class _SelectedAction:
    action: GameAction
    initial_early_search_activated: bool = False
    final_draw_auto_activated: bool = False
    continuation_aware_final_draw_activated: bool = False


def _run_one_hand_task(task: _FullHandTask) -> FullHandResult:
    return _run_one_hand(
        hand_index=task.hand_index,
        seed=task.seed,
        player_0_policy=task.player_0_policy,
        player_1_policy=task.player_1_policy,
        rollouts_per_action=task.rollouts_per_action,
    )


def _run_one_hand(
    *,
    hand_index: int,
    seed: str,
    player_0_policy: str,
    player_1_policy: str,
    rollouts_per_action: int,
) -> FullHandResult:
    button = PlayerId.PLAYER_0 if hand_index % 2 == 0 else PlayerId.PLAYER_1
    state = new_match(button=button, seed=seed)
    decision_count = 0
    initial_early_search_activations = 0
    final_draw_auto_activations = 0
    continuation_aware_final_draw_activations = 0
    start = time.perf_counter()

    while state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        policy_name = player_0_policy if state.acting_player == PlayerId.PLAYER_0 else player_1_policy
        selected = _select_action(
            state,
            policy_name=policy_name,
            decision_seed=f"{seed}:decision:{decision_count}",
            rollouts_per_action=rollouts_per_action,
        )
        initial_early_search_activations += int(selected.initial_early_search_activated)
        final_draw_auto_activations += int(selected.final_draw_auto_activated)
        continuation_aware_final_draw_activations += int(selected.continuation_aware_final_draw_activated)
        state = apply_action(state, selected.action)
        decision_count += 1

    terminal_state, result = showdown(state)
    return FullHandResult(
        seed=seed,
        player_0_policy=player_0_policy,
        player_1_policy=player_1_policy,
        player_0_points=result.left.total_points,
        player_1_points=result.right.total_points,
        player_0_fouled=is_foul(effective_board(terminal_state.players[0], reveal_concealed=True)),
        player_1_fouled=is_foul(effective_board(terminal_state.players[1], reveal_concealed=True)),
        player_0_fantasyland=terminal_state.next_hand_fantasyland[player_index(PlayerId.PLAYER_0)],
        player_1_fantasyland=terminal_state.next_hand_fantasyland[player_index(PlayerId.PLAYER_1)],
        player_0_royalties=result.left.royalties,
        player_1_royalties=result.right.royalties,
        decision_count=decision_count,
        initial_early_search_activations=initial_early_search_activations,
        final_draw_auto_activations=final_draw_auto_activations,
        continuation_aware_final_draw_activations=continuation_aware_final_draw_activations,
        elapsed_seconds=time.perf_counter() - start,
    )


def _select_action(
    state,
    *,
    policy_name: str,
    decision_seed: str,
    rollouts_per_action: int,
) -> _SelectedAction:
    legal = tuple(legal_actions(state))
    analysis = rank_actions_from_observation(
        project_observation(state, state.acting_player),
        rollouts_per_action=rollouts_per_action,
        rng_seed=decision_seed,
        policy=policy_from_name("heuristic"),
        solver_mode="recommended" if policy_name == "recommended" else "manual",
    )
    top = analysis.ranked_actions[0]
    return _SelectedAction(
        action=legal[top.action_index],
        initial_early_search_activated=(
            state.phase == HandPhase.INITIAL_DEAL and analysis.early_search_enabled
        ),
        final_draw_auto_activated=top.phase_auto_search_activated,
        continuation_aware_final_draw_activated=top.final_draw_continuation_aware,
    )


def _write_traces(benchmark: FullHandBenchmark, trace_dir: Path | str | None) -> None:
    target_dir = Path(trace_dir) if trace_dir is not None else Path("reports/benchmark_outputs/full_hand_traces")
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, hand in enumerate(benchmark.hands):
        trace_path = target_dir / f"hand_{index:04d}.json"
        trace_path.write_text(
            json.dumps(benchmark._hand_payload(hand), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _validate_policy_name(policy_name: str) -> None:
    if policy_name not in {"baseline", "recommended"}:
        raise ValueError("full-hand policy must be one of: baseline, recommended")


def _rate(values) -> float:
    values = tuple(bool(value) for value in values)
    return sum(1 for value in values if value) / len(values)


def _sample_stdev(values) -> float:
    values = tuple(float(value) for value in values)
    if len(values) <= 1:
        return 0.0
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


__all__ = ["FullHandBenchmark", "FullHandResult", "run_full_hand_benchmark"]
