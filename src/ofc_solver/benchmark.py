"""Benchmark and diagnostic harness for solver move ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from ofc.actions import GameAction
from ofc.state import HandPhase, PlayerId
from ofc.transitions import legal_actions
from ofc_analysis.action_codec import encode_action
from ofc_analysis.observation import project_observation
from ofc_analysis.scenario import load_scenario
from ofc_solver.models import MoveEstimate, SUPPORTED_ROOT_PHASES
from ofc_solver.rollout import RolloutResult, run_rollout
from ofc_solver.rollout_policy import RandomRolloutPolicy, RolloutPolicy
from ofc_solver.sampler import sample_state


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark scenario and solver configuration."""

    name: str
    scenario_path: Path
    observer: PlayerId
    rollouts_per_action: int
    rng_seed: int | str | None
    expected_top_action_indices: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkManifest:
    """A versioned collection of benchmark cases."""

    version: str
    cases: tuple[BenchmarkCase, ...]
    source_path: Path | None = None


@dataclass(frozen=True)
class BenchmarkActionDiagnostics:
    """Aggregated rollout diagnostics for one root action."""

    action_index: int
    sample_count: int
    mean_total_value: float
    mean_current_hand_value: float
    mean_continuation_value: float
    continuation_frequency: float
    root_foul_rate: float
    opponent_foul_rate: float
    both_foul_rate: float
    root_fantasyland_frequency: float
    opponent_fantasyland_frequency: float


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Benchmark output for one case."""

    name: str
    scenario_path: Path
    tags: tuple[str, ...]
    observer: PlayerId
    phase: HandPhase
    rollouts_per_action: int
    rng_seed: int | str | None
    expected_top_action_indices: tuple[int, ...]
    top_action_index: int
    top1_agreement: bool | None
    top3_agreement: bool | None
    action_count: int
    elapsed_seconds: float
    ranked_actions: tuple[MoveEstimate, ...]
    action_diagnostics: tuple[BenchmarkActionDiagnostics, ...]


@dataclass(frozen=True)
class BenchmarkRun:
    """Benchmark output for a full manifest run."""

    policy_name: str
    case_results: tuple[BenchmarkCaseResult, ...]
    elapsed_seconds: float

    @property
    def case_count(self) -> int:
        return len(self.case_results)


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    """Load a solver benchmark manifest from disk."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return load_benchmark_manifest_data(payload, source_path=manifest_path)


def load_benchmark_manifest_data(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> BenchmarkManifest:
    """Load a solver benchmark manifest from an in-memory payload."""

    manifest = _require_mapping(payload, "benchmark manifest")
    _require_exact_keys(manifest, {"version", "cases"}, "benchmark manifest")
    version = manifest["version"]
    if version != "1":
        raise ValueError(f"Unsupported benchmark manifest version: {version!r}")
    cases_value = manifest["cases"]
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("benchmark manifest cases must be a non-empty list")

    base_path = source_path.parent if source_path is not None else Path.cwd()
    cases = tuple(
        _parse_case(_require_mapping(case_value, f"benchmark manifest.cases[{index}]"), base_path)
        for index, case_value in enumerate(cases_value)
    )
    return BenchmarkManifest(version="1", cases=cases, source_path=source_path)


def run_benchmark_manifest(manifest: BenchmarkManifest, *, policy_name: str = "random") -> BenchmarkRun:
    """Run all cases in a loaded benchmark manifest."""

    policy = _policy_from_name(policy_name)
    start = time.perf_counter()
    case_results = tuple(run_benchmark_case(case, policy=policy) for case in manifest.cases)
    return BenchmarkRun(
        policy_name=policy_name,
        case_results=case_results,
        elapsed_seconds=time.perf_counter() - start,
    )


def run_benchmark_case(case: BenchmarkCase, *, policy: RolloutPolicy) -> BenchmarkCaseResult:
    """Run one benchmark case and collect ranked actions plus diagnostics."""

    exact_state = load_scenario(case.scenario_path).state
    if exact_state.phase not in SUPPORTED_ROOT_PHASES:
        raise ValueError(
            "Unsupported benchmark root phase: "
            f"{exact_state.phase.value}. Supported phases: initial_deal, draw."
        )
    if case.observer != exact_state.acting_player:
        raise ValueError("benchmark observer must be the acting player")

    observation = project_observation(exact_state, case.observer)
    enumeration_state = sample_state(observation, rng=random.Random(case.rng_seed)).state
    root_actions = tuple(legal_actions(enumeration_state))

    rng = random.Random(case.rng_seed)
    estimates: list[MoveEstimate] = []
    diagnostics: list[BenchmarkActionDiagnostics] = []
    start = time.perf_counter()

    for action_index, action in enumerate(root_actions):
        rollout_results = tuple(
            run_rollout(
                sample_state(observation, rng=rng).state,
                root_action=action,
                root_player=case.observer,
                rng=rng,
                policy=policy,
            )
            for _ in range(case.rollouts_per_action)
        )
        estimates.append(_estimate(action_index, action, rollout_results))
        diagnostics.append(_diagnostics(action_index, rollout_results))

    ranked_actions = tuple(sorted(estimates, key=lambda estimate: (-estimate.mean_value, estimate.action_index)))
    top_action_index = ranked_actions[0].action_index
    top3_action_indices = tuple(estimate.action_index for estimate in ranked_actions[:3])
    expected = set(case.expected_top_action_indices)
    top1_agreement = None if not expected else top_action_index in expected
    top3_agreement = None if not expected else bool(expected & set(top3_action_indices))

    return BenchmarkCaseResult(
        name=case.name,
        scenario_path=case.scenario_path,
        tags=case.tags,
        observer=case.observer,
        phase=exact_state.phase,
        rollouts_per_action=case.rollouts_per_action,
        rng_seed=case.rng_seed,
        expected_top_action_indices=case.expected_top_action_indices,
        top_action_index=top_action_index,
        top1_agreement=top1_agreement,
        top3_agreement=top3_agreement,
        action_count=len(root_actions),
        elapsed_seconds=time.perf_counter() - start,
        ranked_actions=ranked_actions,
        action_diagnostics=tuple(diagnostics),
    )


def _parse_case(value: Mapping[str, Any], base_path: Path) -> BenchmarkCase:
    allowed_keys = {
        "name",
        "scenario",
        "observer",
        "rollouts",
        "seed",
        "expected_top_action_indices",
        "tags",
    }
    _require_subset_keys(value, allowed_keys, "benchmark case")
    name = _parse_non_empty_string(value.get("name"), "benchmark case.name")
    scenario_path = _normalize_path(_parse_non_empty_string(value.get("scenario"), "benchmark case.scenario"), base_path)
    observer = _parse_player_id(value.get("observer"), "benchmark case.observer")
    rollouts = _parse_positive_int(value.get("rollouts"), "benchmark case.rollouts")
    seed = _parse_seed(value.get("seed"))
    expected = _parse_int_tuple(value.get("expected_top_action_indices", []), "benchmark case.expected_top_action_indices")
    tags = _parse_str_tuple(value.get("tags", []), "benchmark case.tags")
    return BenchmarkCase(
        name=name,
        scenario_path=scenario_path,
        observer=observer,
        rollouts_per_action=rollouts,
        rng_seed=seed,
        expected_top_action_indices=expected,
        tags=tags,
    )


def _policy_from_name(policy_name: str) -> RolloutPolicy:
    if policy_name == "random":
        return RandomRolloutPolicy()
    raise ValueError(f"Unsupported benchmark policy: {policy_name!r}. Supported policies: random.")


def _estimate(action_index: int, action: GameAction, rollout_results: tuple[RolloutResult, ...]) -> MoveEstimate:
    values = tuple(result.total_value for result in rollout_results)
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return MoveEstimate(
        action_index=action_index,
        action=encode_action(action_index, action),
        mean_value=mean_value,
        stddev=math.sqrt(variance),
        sample_count=len(values),
        min_value=min(values),
        max_value=max(values),
    )


def _diagnostics(
    action_index: int,
    rollout_results: tuple[RolloutResult, ...],
) -> BenchmarkActionDiagnostics:
    sample_count = len(rollout_results)
    return BenchmarkActionDiagnostics(
        action_index=action_index,
        sample_count=sample_count,
        mean_total_value=_mean(result.total_value for result in rollout_results),
        mean_current_hand_value=_mean(result.current_hand_value for result in rollout_results),
        mean_continuation_value=_mean(result.continuation_value for result in rollout_results),
        continuation_frequency=_rate(result.continuation_hands_simulated > 0 for result in rollout_results),
        root_foul_rate=_rate(result.root_player_fouled for result in rollout_results),
        opponent_foul_rate=_rate(result.opponent_fouled for result in rollout_results),
        both_foul_rate=_rate(result.both_players_fouled for result in rollout_results),
        root_fantasyland_frequency=_rate(result.root_player_next_fantasyland for result in rollout_results),
        opponent_fantasyland_frequency=_rate(result.opponent_next_fantasyland for result in rollout_results),
    )


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _rate(values) -> float:
    values = tuple(bool(value) for value in values)
    return sum(1 for value in values if value) / len(values)


def _normalize_path(value: str, base_path: Path) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else base_path / path
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return resolved


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_keys(mapping: Mapping[str, Any], allowed_keys: set[str], path: str) -> None:
    missing = sorted(allowed_keys - set(mapping))
    unexpected = sorted(set(mapping) - allowed_keys)
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{path} has unexpected keys: {', '.join(unexpected)}")


def _require_subset_keys(mapping: Mapping[str, Any], allowed_keys: set[str], path: str) -> None:
    unexpected = sorted(set(mapping) - allowed_keys)
    if unexpected:
        raise ValueError(f"{path} has unexpected keys: {', '.join(unexpected)}")


def _parse_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _parse_player_id(value: Any, path: str) -> PlayerId:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a player id string")
    try:
        return PlayerId(value)
    except ValueError as exc:
        raise ValueError(f"{path} must be one of: player_0, player_1") from exc


def _parse_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _parse_seed(value: Any) -> int | str | None:
    if value is None or isinstance(value, (int, str)):
        return value
    raise ValueError("benchmark case.seed must be an integer, string, or null")


def _parse_int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ValueError(f"{path} must contain non-negative integer action indices")
    return tuple(value)


def _parse_str_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{path} must contain non-empty strings")
    return tuple(value)


__all__ = [
    "BenchmarkActionDiagnostics",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkManifest",
    "BenchmarkRun",
    "load_benchmark_manifest",
    "load_benchmark_manifest_data",
    "run_benchmark_case",
    "run_benchmark_manifest",
]
