"""Deterministic full-hand policy benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
import time

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
    hands: tuple[FullHandResult, ...]
    elapsed_seconds: float

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

    def as_dict(self) -> dict[str, object]:
        return {
            "player_0_policy": self.player_0_policy,
            "player_1_policy": self.player_1_policy,
            "hand_count": self.hand_count,
            "seed": self.seed,
            "rollouts_per_action": self.rollouts_per_action,
            "total_player_0_points": self.total_player_0_points,
            "total_player_1_points": self.total_player_1_points,
            "average_player_0_points": self.average_player_0_points,
            "average_player_1_points": self.average_player_1_points,
            "player_0_foul_rate": self.player_0_foul_rate,
            "player_1_foul_rate": self.player_1_foul_rate,
            "both_foul_rate": self.both_foul_rate,
            "player_0_fantasyland_rate": self.player_0_fantasyland_rate,
            "player_1_fantasyland_rate": self.player_1_fantasyland_rate,
            "average_player_0_royalties": self.average_player_0_royalties,
            "average_player_1_royalties": self.average_player_1_royalties,
            "average_runtime_per_hand": self.average_runtime_per_hand,
            "zero_sum_consistent": self.zero_sum_consistent,
            "elapsed_seconds": self.elapsed_seconds,
            "hands": [
                {
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
                    "elapsed_seconds": hand.elapsed_seconds,
                }
                for hand in self.hands
            ],
        }

    def deterministic_payload(self) -> dict[str, object]:
        payload = self.as_dict()
        payload.pop("elapsed_seconds", None)
        payload.pop("average_runtime_per_hand", None)
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
) -> FullHandBenchmark:
    """Run deterministic complete normal hands for two solver policies."""

    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if rollouts_per_action <= 0:
        raise ValueError("rollouts_per_action must be positive")
    _validate_policy_name(player_0_policy)
    _validate_policy_name(player_1_policy)

    start = time.perf_counter()
    hands = tuple(
        _run_one_hand(
            hand_index=index,
            seed=f"{seed}:{index}",
            player_0_policy=player_0_policy,
            player_1_policy=player_1_policy,
            rollouts_per_action=rollouts_per_action,
        )
        for index in range(hand_count)
    )
    return FullHandBenchmark(
        player_0_policy=player_0_policy,
        player_1_policy=player_1_policy,
        hand_count=hand_count,
        seed=seed,
        rollouts_per_action=rollouts_per_action,
        hands=hands,
        elapsed_seconds=time.perf_counter() - start,
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
    start = time.perf_counter()

    while state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        policy_name = player_0_policy if state.acting_player == PlayerId.PLAYER_0 else player_1_policy
        action = _select_action(
            state,
            policy_name=policy_name,
            decision_seed=f"{seed}:decision:{decision_count}",
            rollouts_per_action=rollouts_per_action,
        )
        state = apply_action(state, action)
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
        elapsed_seconds=time.perf_counter() - start,
    )


def _select_action(
    state,
    *,
    policy_name: str,
    decision_seed: str,
    rollouts_per_action: int,
):
    legal = tuple(legal_actions(state))
    analysis = rank_actions_from_observation(
        project_observation(state, state.acting_player),
        rollouts_per_action=rollouts_per_action,
        rng_seed=decision_seed,
        policy=policy_from_name("heuristic"),
        solver_mode="recommended" if policy_name == "recommended" else "manual",
    )
    return legal[analysis.ranked_actions[0].action_index]


def _validate_policy_name(policy_name: str) -> None:
    if policy_name not in {"baseline", "recommended"}:
        raise ValueError("full-hand policy must be one of: baseline, recommended")


def _rate(values) -> float:
    values = tuple(bool(value) for value in values)
    return sum(1 for value in values if value) / len(values)


__all__ = ["FullHandBenchmark", "FullHandResult", "run_full_hand_benchmark"]
