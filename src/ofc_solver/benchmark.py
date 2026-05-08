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
from ofc_solver.early_search import EarlySearchCandidate, EarlySearchConfig, select_early_search_candidates
from ofc_solver.late_search import (
    FinalDrawAutoSearchConfig,
    LateSearchConfig,
    LateSearchResult,
    evaluate_final_draw_auto_root_action,
    evaluate_late_root_action,
)
from ofc_solver.models import MoveEstimate, SUPPORTED_ROOT_PHASES
from ofc_solver.policy_registry import policy_from_name
from ofc_solver.root_action_risk import (
    ROOT_RISK_COMPONENT_KEYS,
    RootActionRiskAssessment,
    RootRiskConfig,
    score_root_action,
)
from ofc_solver.rollout import RolloutResult, run_rollout
from ofc_solver.rollout_policy import RolloutPolicy
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
    mean_policy_decisions: float
    exact_late_search_rollout_frequency: float
    mean_exact_late_search_decisions: float
    mean_exact_late_search_nodes: float
    late_search_activation_rate: float
    late_search_exact_rate: float
    late_search_beam_rate: float
    late_search_fallback_rate: float
    mean_late_search_nodes: float
    mean_late_search_depth: float
    mean_late_search_terminal_evaluations: float
    phase_auto_search_activation_rate: float
    mean_phase_auto_search_tree_nodes: float
    mean_phase_auto_search_depth: float


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
    candidate_count: int
    candidate_pruning_ratio: float
    elapsed_seconds: float
    ranked_actions: tuple[MoveEstimate, ...]
    action_diagnostics: tuple[BenchmarkActionDiagnostics, ...]


@dataclass(frozen=True)
class BenchmarkRun:
    """Benchmark output for a full manifest run."""

    policy_name: str
    case_results: tuple[BenchmarkCaseResult, ...]
    elapsed_seconds: float
    root_action_risk_enabled: bool = False
    root_action_risk_config_label: str | None = None
    early_search_enabled: bool = False
    beam_size: int | None = None
    candidate_extra_rollouts: int = 0
    draw_safe_candidates: bool = True
    draw_baseline_keep: int = 0
    draw_safety_keep: int = 0
    late_search_enabled: bool = False
    late_search_mode: str | None = None
    late_search_max_depth: int | None = None
    late_search_max_nodes: int | None = None
    late_search_beam_size: int | None = None
    final_draw_auto_search_enabled: bool = False
    final_draw_auto_max_depth: int | None = None
    final_draw_auto_max_nodes: int | None = None

    @property
    def case_count(self) -> int:
        return len(self.case_results)


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Aggregate diagnostics for one benchmark run."""

    policy_name: str
    case_count: int
    action_count: int
    sample_count: int
    root_foul_rate: float
    opponent_foul_rate: float
    both_foul_rate: float
    continuation_frequency: float
    root_fantasyland_frequency: float
    opponent_fantasyland_frequency: float
    mean_policy_decisions: float
    exact_late_search_rollout_frequency: float
    mean_exact_late_search_decisions: float
    mean_exact_late_search_nodes: float
    late_search_activation_rate: float
    late_search_exact_rate: float
    late_search_beam_rate: float
    late_search_fallback_rate: float
    mean_late_search_nodes: float
    mean_late_search_depth: float
    mean_late_search_terminal_evaluations: float
    phase_auto_search_activation_rate: float
    mean_phase_auto_search_tree_nodes: float
    mean_phase_auto_search_depth: float
    top_action_root_foul_rate: float
    top_action_opponent_foul_rate: float
    top_action_both_foul_rate: float
    top_action_continuation_frequency: float
    top_action_root_fantasyland_frequency: float
    top_action_opponent_fantasyland_frequency: float
    top_action_mean_policy_decisions: float
    top_action_exact_late_search_rollout_frequency: float
    top_action_mean_exact_late_search_decisions: float
    top_action_mean_exact_late_search_nodes: float
    top_action_late_search_activation_rate: float
    top_action_late_search_exact_rate: float
    top_action_late_search_beam_rate: float
    top_action_late_search_fallback_rate: float
    top_action_mean_late_search_nodes: float
    top_action_mean_late_search_depth: float
    top_action_mean_late_search_terminal_evaluations: float
    top_action_phase_auto_search_activation_rate: float
    top_action_mean_phase_auto_search_tree_nodes: float
    top_action_mean_phase_auto_search_depth: float
    labeled_top1_rate: float | None
    labeled_top3_rate: float | None
    elapsed_seconds: float


@dataclass(frozen=True)
class BenchmarkTagSliceAggregate:
    """Aggregate diagnostics for cases sharing one benchmark tag."""

    case_count: int
    action_count: int
    sample_count: int
    root_foul_rate: float
    both_foul_rate: float
    continuation_frequency: float
    root_fantasyland_frequency: float
    exact_late_search_rollout_frequency: float
    late_search_activation_rate: float
    late_search_exact_rate: float
    late_search_beam_rate: float
    late_search_fallback_rate: float
    mean_late_search_nodes: float
    phase_auto_search_activation_rate: float
    mean_phase_auto_search_tree_nodes: float
    top_action_root_foul_rate: float
    top_action_both_foul_rate: float
    top_action_continuation_frequency: float
    top_action_root_fantasyland_frequency: float
    top_action_exact_late_search_rollout_frequency: float
    top_action_late_search_activation_rate: float
    top_action_late_search_exact_rate: float
    top_action_late_search_beam_rate: float
    top_action_late_search_fallback_rate: float
    top_action_mean_late_search_nodes: float
    top_action_phase_auto_search_activation_rate: float
    top_action_mean_phase_auto_search_tree_nodes: float
    labeled_top1_rate: float | None
    labeled_top3_rate: float | None


@dataclass(frozen=True)
class BenchmarkTagSliceComparison:
    """Side-by-side comparison for one benchmark tag slice."""

    tag: str
    case_count: int
    left: BenchmarkTagSliceAggregate
    right: BenchmarkTagSliceAggregate
    deltas: dict[str, float | None]


@dataclass(frozen=True)
class TopActionChange:
    """One benchmark case whose top-ranked action changed between runs."""

    case_name: str
    left_top_action_index: int
    right_top_action_index: int
    left_top_mean_value: float
    right_top_mean_value: float


@dataclass(frozen=True)
class BenchmarkComparison:
    """Side-by-side aggregate comparison of two benchmark runs."""

    left_policy_name: str
    right_policy_name: str
    case_count: int
    left: BenchmarkAggregate
    right: BenchmarkAggregate
    deltas: dict[str, float | None]
    top_action_changes: tuple[TopActionChange, ...]
    tag_slices: tuple[BenchmarkTagSliceComparison, ...]


@dataclass(frozen=True)
class RootActionRiskBenchmark:
    """Focused comparison of heuristic root ranking with and without root risk."""

    comparison: BenchmarkComparison
    left_run: BenchmarkRun
    right_run: BenchmarkRun
    include_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    phases: tuple[HandPhase, ...]


@dataclass(frozen=True)
class RootActionRiskAblationRun:
    """One ablation run plus comparisons against baseline and full root risk."""

    name: str
    mode: str
    component: str | None
    config: RootRiskConfig
    run: BenchmarkRun
    aggregate: BenchmarkAggregate
    comparison_vs_baseline: BenchmarkComparison
    comparison_vs_full: BenchmarkComparison


@dataclass(frozen=True)
class RootActionRiskAblationBenchmark:
    """Full ablation pass for the current root-risk component set."""

    baseline_run: BenchmarkRun
    baseline_aggregate: BenchmarkAggregate
    full_run: BenchmarkRun
    full_aggregate: BenchmarkAggregate
    ablations: tuple[RootActionRiskAblationRun, ...]
    include_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    phases: tuple[HandPhase, ...]
    component_order: tuple[str, ...]


@dataclass(frozen=True)
class EarlySearchBenchmark:
    """Focused comparison of baseline ranking against early-search pruning."""

    comparison: BenchmarkComparison
    left_run: BenchmarkRun
    right_run: BenchmarkRun
    include_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    phases: tuple[HandPhase, ...]


@dataclass(frozen=True)
class LateSearchBenchmark:
    """Focused comparison of baseline ranking against bounded late-street search."""

    comparison: BenchmarkComparison
    left_run: BenchmarkRun
    right_run: BenchmarkRun
    include_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    phases: tuple[HandPhase, ...]


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


def run_benchmark_manifest(
    manifest: BenchmarkManifest,
    *,
    policy_name: str = "random",
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
    early_search: bool = False,
    beam_size: int = 48,
    candidate_extra_rollouts: int = 0,
    draw_safe_candidates: bool = True,
    draw_baseline_keep: int = 8,
    draw_safety_keep: int = 8,
    late_search: bool = False,
    late_search_config: LateSearchConfig | None = None,
    final_draw_auto_search: bool = False,
    final_draw_auto_search_config: FinalDrawAutoSearchConfig | None = None,
) -> BenchmarkRun:
    """Run all cases in a loaded benchmark manifest."""

    effective_root_risk_config = root_action_risk_config or (
        RootRiskConfig.default() if root_action_risk else RootRiskConfig.all_off()
    )
    risk_enabled = bool(effective_root_risk_config.enabled_components)
    early_search_config = EarlySearchConfig(
        beam_size=beam_size,
        candidate_extra_rollouts=candidate_extra_rollouts,
        draw_safe_candidates=draw_safe_candidates,
        draw_baseline_keep=draw_baseline_keep,
        draw_safety_keep=draw_safety_keep,
    )
    policy = policy_from_name(policy_name)
    effective_late_search_config = late_search_config or LateSearchConfig()
    effective_final_draw_auto_config = final_draw_auto_search_config or FinalDrawAutoSearchConfig()
    start = time.perf_counter()
    case_results = tuple(
        run_benchmark_case(
            case,
            policy=policy,
            root_action_risk=risk_enabled,
            root_action_risk_config=effective_root_risk_config,
            early_search=early_search,
            early_search_config=early_search_config,
            late_search=late_search,
            late_search_config=effective_late_search_config,
            final_draw_auto_search=final_draw_auto_search,
            final_draw_auto_search_config=effective_final_draw_auto_config,
        )
        for case in manifest.cases
    )
    effective_name = _policy_name_for_root_risk(policy_name, risk_enabled, effective_root_risk_config)
    effective_name = _policy_name_for_early_search(
        effective_name,
        early_search,
        early_search_config,
    )
    effective_name = _policy_name_for_late_search(
        effective_name,
        late_search,
        effective_late_search_config,
    )
    effective_name = _policy_name_for_final_draw_auto_search(
        effective_name,
        final_draw_auto_search,
        effective_final_draw_auto_config,
    )
    return BenchmarkRun(
        policy_name=effective_name,
        case_results=case_results,
        elapsed_seconds=time.perf_counter() - start,
        root_action_risk_enabled=risk_enabled,
        root_action_risk_config_label=effective_root_risk_config.label if risk_enabled else "off",
        early_search_enabled=early_search,
        beam_size=beam_size if early_search else None,
        candidate_extra_rollouts=candidate_extra_rollouts if early_search else 0,
        draw_safe_candidates=draw_safe_candidates if early_search else False,
        draw_baseline_keep=draw_baseline_keep if early_search else 0,
        draw_safety_keep=draw_safety_keep if early_search else 0,
        late_search_enabled=late_search,
        late_search_mode=effective_late_search_config.mode if late_search else None,
        late_search_max_depth=effective_late_search_config.max_depth if late_search else None,
        late_search_max_nodes=effective_late_search_config.max_nodes if late_search else None,
        late_search_beam_size=effective_late_search_config.beam_size if late_search else None,
        final_draw_auto_search_enabled=final_draw_auto_search,
        final_draw_auto_max_depth=effective_final_draw_auto_config.max_depth if final_draw_auto_search else None,
        final_draw_auto_max_nodes=effective_final_draw_auto_config.max_nodes if final_draw_auto_search else None,
    )


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    policy: RolloutPolicy,
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
    early_search: bool = False,
    early_search_config: EarlySearchConfig | None = None,
    late_search: bool = False,
    late_search_config: LateSearchConfig | None = None,
    final_draw_auto_search: bool = False,
    final_draw_auto_search_config: FinalDrawAutoSearchConfig | None = None,
) -> BenchmarkCaseResult:
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
    candidate_set = (
        select_early_search_candidates(enumeration_state, config=early_search_config)
        if early_search
        else None
    )
    root_actions = tuple(candidate.action for candidate in candidate_set.candidates) if candidate_set else tuple(legal_actions(enumeration_state))
    action_indices = tuple(candidate.action_index for candidate in candidate_set.candidates) if candidate_set else tuple(range(len(root_actions)))
    candidate_by_index = {} if candidate_set is None else {candidate.action_index: candidate for candidate in candidate_set.candidates}
    rollouts_per_candidate = case.rollouts_per_action + (
        early_search_config.candidate_extra_rollouts if early_search and early_search_config is not None else 0
    )
    effective_late_search_config = late_search_config or LateSearchConfig()
    effective_final_draw_auto_config = final_draw_auto_search_config or FinalDrawAutoSearchConfig()

    rng = random.Random(case.rng_seed)
    estimates: list[MoveEstimate] = []
    diagnostics: list[BenchmarkActionDiagnostics] = []
    start = time.perf_counter()

    for action_index, action in zip(action_indices, root_actions, strict=True):
        late_results: tuple[LateSearchResult, ...] = ()
        if late_search:
            late_results = tuple(
                evaluate_late_root_action(
                    sample_state(observation, rng=rng).state,
                    action,
                    perspective=case.observer,
                    rng=rng,
                    config=effective_late_search_config,
                    policy=policy,
                )
                for _ in range(rollouts_per_candidate)
            )
            rollout_results = tuple(result.rollout_result for result in late_results)
        elif final_draw_auto_search:
            late_results = tuple(
                evaluate_final_draw_auto_root_action(
                    sample_state(observation, rng=rng).state,
                    action,
                    perspective=case.observer,
                    rng=rng,
                    config=effective_final_draw_auto_config,
                    policy=policy,
                )
                for _ in range(rollouts_per_candidate)
            )
            rollout_results = tuple(result.rollout_result for result in late_results)
        else:
            rollout_results = tuple(
                run_rollout(
                    sample_state(observation, rng=rng).state,
                    root_action=action,
                    root_player=case.observer,
                    rng=rng,
                    policy=policy,
                )
                for _ in range(rollouts_per_candidate)
            )
        risk = (
            score_root_action(
                enumeration_state,
                action,
                config=root_action_risk_config,
            )
            if root_action_risk
            else None
        )
        estimates.append(
            _estimate(
                action_index,
                action,
                rollout_results,
                root_risk=risk,
                candidate=candidate_by_index.get(action_index),
                late_search_results=late_results,
            )
        )
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
        rollouts_per_action=rollouts_per_candidate,
        rng_seed=case.rng_seed,
        expected_top_action_indices=case.expected_top_action_indices,
        top_action_index=top_action_index,
        top1_agreement=top1_agreement,
        top3_agreement=top3_agreement,
        action_count=candidate_set.total_legal_actions if candidate_set else len(root_actions),
        candidate_count=len(root_actions),
        candidate_pruning_ratio=candidate_set.pruning_ratio if candidate_set else 0.0,
        elapsed_seconds=time.perf_counter() - start,
        ranked_actions=ranked_actions,
        action_diagnostics=tuple(diagnostics),
    )


def compare_benchmark_runs(left: BenchmarkRun, right: BenchmarkRun) -> BenchmarkComparison:
    """Compare two in-memory benchmark runs."""

    _validate_comparable_case_names(
        tuple(case.name for case in left.case_results),
        tuple(case.name for case in right.case_results),
    )
    left_aggregate = _aggregate_benchmark_run(left)
    right_aggregate = _aggregate_benchmark_run(right)
    return BenchmarkComparison(
        left_policy_name=left.policy_name,
        right_policy_name=right.policy_name,
        case_count=left.case_count,
        left=left_aggregate,
        right=right_aggregate,
        deltas=_aggregate_deltas(left_aggregate, right_aggregate),
        top_action_changes=_top_action_changes_from_runs(left, right),
        tag_slices=_tag_slice_comparisons_from_runs(left, right),
    )


def filter_benchmark_manifest(
    manifest: BenchmarkManifest,
    *,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    phases: tuple[HandPhase, ...] = (),
) -> BenchmarkManifest:
    """Return a manifest subset matching tag and root phase filters."""

    include = set(include_tags)
    exclude = set(exclude_tags)
    phase_set = set(phases)
    selected: list[BenchmarkCase] = []
    for case in manifest.cases:
        case_tags = set(case.tags)
        if include and not case_tags.intersection(include):
            continue
        if exclude and case_tags.intersection(exclude):
            continue
        if phase_set:
            state = load_scenario(case.scenario_path).state
            if state.phase not in phase_set:
                continue
        selected.append(case)
    if not selected:
        raise ValueError("No benchmark cases matched the requested root-action-risk filters")
    return BenchmarkManifest(version=manifest.version, cases=tuple(selected), source_path=manifest.source_path)


def run_root_action_risk_benchmark(
    manifest: BenchmarkManifest,
    *,
    policy_name: str = "heuristic",
    include_tags: tuple[str, ...] = ("initial_deal", "early_draw"),
    exclude_tags: tuple[str, ...] = ("final_draw",),
    phases: tuple[HandPhase, ...] = (),
    root_action_risk_config: RootRiskConfig | None = None,
) -> RootActionRiskBenchmark:
    """Run a focused heuristic-vs-root-risk benchmark comparison."""

    filtered = filter_benchmark_manifest(
        manifest,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )
    left = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=False,
        root_action_risk_config=RootRiskConfig.all_off(),
    )
    comparison_config = root_action_risk_config or RootRiskConfig.default()
    right = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=True,
        root_action_risk_config=comparison_config,
    )
    return RootActionRiskBenchmark(
        comparison=compare_benchmark_runs(left, right),
        left_run=left,
        right_run=right,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )


def run_early_search_benchmark(
    manifest: BenchmarkManifest,
    *,
    policy_name: str = "heuristic",
    include_tags: tuple[str, ...] = ("initial_deal", "early_draw"),
    exclude_tags: tuple[str, ...] = ("final_draw",),
    phases: tuple[HandPhase, ...] = (),
    beam_size: int = 48,
    candidate_extra_rollouts: int = 0,
    draw_safe_candidates: bool = True,
    draw_baseline_keep: int = 8,
    draw_safety_keep: int = 8,
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
) -> EarlySearchBenchmark:
    """Run a focused baseline-vs-early-search benchmark comparison."""

    filtered = filter_benchmark_manifest(
        manifest,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )
    left = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=root_action_risk,
        root_action_risk_config=root_action_risk_config,
        early_search=False,
    )
    right = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=root_action_risk,
        root_action_risk_config=root_action_risk_config,
        early_search=True,
        beam_size=beam_size,
        candidate_extra_rollouts=candidate_extra_rollouts,
        draw_safe_candidates=draw_safe_candidates,
        draw_baseline_keep=draw_baseline_keep,
        draw_safety_keep=draw_safety_keep,
    )
    return EarlySearchBenchmark(
        comparison=compare_benchmark_runs(left, right),
        left_run=left,
        right_run=right,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )


def run_late_search_benchmark(
    manifest: BenchmarkManifest,
    *,
    policy_name: str = "heuristic",
    include_tags: tuple[str, ...] = ("late_draw", "final_draw"),
    exclude_tags: tuple[str, ...] = (),
    phases: tuple[HandPhase, ...] = (HandPhase.DRAW,),
    late_search_config: LateSearchConfig | None = None,
    root_action_risk: bool = False,
    root_action_risk_config: RootRiskConfig | None = None,
) -> LateSearchBenchmark:
    """Run a focused baseline-vs-late-search benchmark comparison."""

    filtered = filter_benchmark_manifest(
        manifest,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )
    left = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=root_action_risk,
        root_action_risk_config=root_action_risk_config,
        late_search=False,
    )
    effective_late_search_config = late_search_config or LateSearchConfig()
    right = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=root_action_risk,
        root_action_risk_config=root_action_risk_config,
        late_search=True,
        late_search_config=effective_late_search_config,
    )
    return LateSearchBenchmark(
        comparison=compare_benchmark_runs(left, right),
        left_run=left,
        right_run=right,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )


def run_root_action_risk_ablation_benchmark(
    manifest: BenchmarkManifest,
    *,
    policy_name: str = "heuristic",
    include_tags: tuple[str, ...] = ("initial_deal", "early_draw"),
    exclude_tags: tuple[str, ...] = ("final_draw",),
    phases: tuple[HandPhase, ...] = (),
) -> RootActionRiskAblationBenchmark:
    """Run baseline/full plus single-component and leave-one-out ablations."""

    filtered = filter_benchmark_manifest(
        manifest,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
    )
    baseline_run = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=False,
        root_action_risk_config=RootRiskConfig.all_off(),
    )
    full_run = run_benchmark_manifest(
        filtered,
        policy_name=policy_name,
        root_action_risk=True,
        root_action_risk_config=RootRiskConfig.all_on(),
    )
    baseline_aggregate = _aggregate_benchmark_run(baseline_run)
    full_aggregate = _aggregate_benchmark_run(full_run)

    ablations: list[RootActionRiskAblationRun] = []
    for component in ROOT_RISK_COMPONENT_KEYS:
        single_config = RootRiskConfig.only(component)
        single_run = run_benchmark_manifest(
            filtered,
            policy_name=policy_name,
            root_action_risk=True,
            root_action_risk_config=single_config,
        )
        ablations.append(
            RootActionRiskAblationRun(
                name=f"only:{component}",
                mode="single_component_only",
                component=component,
                config=single_config,
                run=single_run,
                aggregate=_aggregate_benchmark_run(single_run),
                comparison_vs_baseline=compare_benchmark_runs(baseline_run, single_run),
                comparison_vs_full=compare_benchmark_runs(full_run, single_run),
            )
        )

    for component in ROOT_RISK_COMPONENT_KEYS:
        leave_one_out_config = RootRiskConfig.leave_one_out(component)
        leave_one_out_run = run_benchmark_manifest(
            filtered,
            policy_name=policy_name,
            root_action_risk=True,
            root_action_risk_config=leave_one_out_config,
        )
        ablations.append(
            RootActionRiskAblationRun(
                name=f"without:{component}",
                mode="leave_one_out",
                component=component,
                config=leave_one_out_config,
                run=leave_one_out_run,
                aggregate=_aggregate_benchmark_run(leave_one_out_run),
                comparison_vs_baseline=compare_benchmark_runs(baseline_run, leave_one_out_run),
                comparison_vs_full=compare_benchmark_runs(full_run, leave_one_out_run),
            )
        )

    return RootActionRiskAblationBenchmark(
        baseline_run=baseline_run,
        baseline_aggregate=baseline_aggregate,
        full_run=full_run,
        full_aggregate=full_aggregate,
        ablations=tuple(ablations),
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        phases=phases,
        component_order=ROOT_RISK_COMPONENT_KEYS,
    )


def compare_benchmark_payloads(
    left_payload: Mapping[str, Any],
    right_payload: Mapping[str, Any],
) -> BenchmarkComparison:
    """Compare two rendered benchmark JSON payloads."""

    left_cases = _require_case_payloads(left_payload, "left benchmark")
    right_cases = _require_case_payloads(right_payload, "right benchmark")
    _validate_comparable_case_names(
        tuple(str(case["name"]) for case in left_cases),
        tuple(str(case["name"]) for case in right_cases),
    )
    left_aggregate = _aggregate_benchmark_payload(left_payload)
    right_aggregate = _aggregate_benchmark_payload(right_payload)
    return BenchmarkComparison(
        left_policy_name=str(left_payload["policy_name"]),
        right_policy_name=str(right_payload["policy_name"]),
        case_count=len(left_cases),
        left=left_aggregate,
        right=right_aggregate,
        deltas=_aggregate_deltas(left_aggregate, right_aggregate),
        top_action_changes=_top_action_changes_from_payloads(left_cases, right_cases),
        tag_slices=_tag_slice_comparisons_from_payloads(left_cases, right_cases),
    )


def _policy_name_for_root_risk(
    policy_name: str,
    root_action_risk_enabled: bool,
    root_action_risk_config: RootRiskConfig,
) -> str:
    if not root_action_risk_enabled:
        return policy_name
    if root_action_risk_config == RootRiskConfig.default():
        return f"{policy_name}+root-risk"
    if root_action_risk_config == RootRiskConfig.all_on():
        return f"{policy_name}+root-risk[full]"
    return f"{policy_name}+root-risk[{root_action_risk_config.label}]"


def _policy_name_for_early_search(
    policy_name: str,
    enabled: bool,
    config: EarlySearchConfig,
) -> str:
    if not enabled:
        return policy_name
    extra = f",+{config.candidate_extra_rollouts}" if config.candidate_extra_rollouts else ""
    return f"{policy_name}+early-search[beam={config.beam_size}{extra}]"


def _policy_name_for_late_search(
    policy_name: str,
    enabled: bool,
    config: LateSearchConfig,
) -> str:
    if not enabled:
        return policy_name
    return (
        f"{policy_name}+late-search[{config.mode},d={config.max_depth},"
        f"n={config.max_nodes},b={config.beam_size}]"
    )


def _policy_name_for_final_draw_auto_search(
    policy_name: str,
    enabled: bool,
    config: FinalDrawAutoSearchConfig,
) -> str:
    if not enabled:
        return policy_name
    return f"{policy_name}+final-draw-auto[d={config.max_depth},n={config.max_nodes}]"


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


def _estimate(
    action_index: int,
    action: GameAction,
    rollout_results: tuple[RolloutResult, ...],
    *,
    root_risk: RootActionRiskAssessment | None = None,
    candidate: EarlySearchCandidate | None = None,
    late_search_results: tuple[LateSearchResult, ...] = (),
) -> MoveEstimate:
    values = tuple(result.total_value for result in rollout_results)
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    risk_score = 0.0 if root_risk is None else root_risk.contribution
    pattern_score = 0.0 if candidate is None else candidate.pattern_score
    final_score = mean_value + risk_score
    late_modes = {result.mode for result in late_search_results}
    fallback_reasons = tuple(
        reason
        for reason in (result.fallback_reason for result in late_search_results)
        if reason is not None
    )
    phase_auto_reasons = tuple(
        reason
        for reason in (result.phase_auto_search_reason for result in late_search_results)
        if reason is not None
    )
    return MoveEstimate(
        action_index=action_index,
        action=encode_action(action_index, action),
        mean_value=final_score,
        stddev=math.sqrt(variance),
        sample_count=len(values),
        min_value=min(values),
        max_value=max(values),
        rollout_mean_value=mean_value,
        root_risk_score=risk_score,
        root_risk_reasons=() if root_risk is None else root_risk.reasons,
        pattern_score=pattern_score,
        pattern_reasons=() if candidate is None else candidate.reasons,
        selection_reasons=() if candidate is None else candidate.selection_reasons,
        candidate_rank=None if candidate is None else candidate.candidate_rank,
        final_score=final_score,
        late_search_activated=any(result.activated for result in late_search_results),
        late_search_mode=late_modes.pop() if len(late_modes) == 1 else "mixed" if late_modes else None,
        late_search_nodes=sum(result.nodes_searched for result in late_search_results),
        late_search_depth=max((result.depth_reached for result in late_search_results), default=0),
        late_search_candidate_count=sum(result.candidate_count for result in late_search_results),
        late_search_terminal_evaluations=sum(result.terminal_evaluations for result in late_search_results),
        late_search_fallback_reason=";".join(dict.fromkeys(fallback_reasons)) if fallback_reasons else None,
        phase_auto_search_activated=any(result.phase_auto_search_activated for result in late_search_results),
        phase_auto_search_reason=";".join(dict.fromkeys(phase_auto_reasons)) if phase_auto_reasons else None,
        phase_auto_search_tree_nodes=sum(result.phase_auto_search_tree_nodes for result in late_search_results),
        phase_auto_search_depth=max((result.phase_auto_search_depth for result in late_search_results), default=0),
        late_search_runtime_seconds=sum(result.runtime_seconds for result in late_search_results),
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
        mean_policy_decisions=_mean(result.policy_decision_count for result in rollout_results),
        exact_late_search_rollout_frequency=_rate(
            result.exact_late_search_decision_count > 0 for result in rollout_results
        ),
        mean_exact_late_search_decisions=_mean(
            result.exact_late_search_decision_count for result in rollout_results
        ),
        mean_exact_late_search_nodes=_mean(result.exact_late_search_node_count for result in rollout_results),
        late_search_activation_rate=_rate(result.late_search_activated for result in rollout_results),
        late_search_exact_rate=_rate(result.late_search_mode == "exact" for result in rollout_results),
        late_search_beam_rate=_rate(result.late_search_mode == "beam" for result in rollout_results),
        late_search_fallback_rate=_rate(result.late_search_mode == "fallback" for result in rollout_results),
        mean_late_search_nodes=_mean(result.late_search_nodes for result in rollout_results),
        mean_late_search_depth=_mean(result.late_search_depth for result in rollout_results),
        mean_late_search_terminal_evaluations=_mean(
            result.late_search_terminal_evaluations for result in rollout_results
        ),
        phase_auto_search_activation_rate=_rate(result.phase_auto_search_activated for result in rollout_results),
        mean_phase_auto_search_tree_nodes=_mean(result.phase_auto_search_tree_nodes for result in rollout_results),
        mean_phase_auto_search_depth=_mean(result.phase_auto_search_depth for result in rollout_results),
    )


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _rate(values) -> float:
    values = tuple(bool(value) for value in values)
    return sum(1 for value in values if value) / len(values)


def _aggregate_benchmark_run(run: BenchmarkRun) -> BenchmarkAggregate:
    diagnostics = tuple(diagnostic for case in run.case_results for diagnostic in case.action_diagnostics)
    top_diagnostics = _top_action_diagnostics_from_run(run)
    sample_count = sum(diagnostic.sample_count for diagnostic in diagnostics)
    labeled_cases = tuple(case for case in run.case_results if case.top1_agreement is not None)
    return BenchmarkAggregate(
        policy_name=run.policy_name,
        case_count=run.case_count,
        action_count=sum(case.action_count for case in run.case_results),
        sample_count=sample_count,
        root_foul_rate=_weighted_diagnostic_rate(diagnostics, "root_foul_rate"),
        opponent_foul_rate=_weighted_diagnostic_rate(diagnostics, "opponent_foul_rate"),
        both_foul_rate=_weighted_diagnostic_rate(diagnostics, "both_foul_rate"),
        continuation_frequency=_weighted_diagnostic_rate(diagnostics, "continuation_frequency"),
        root_fantasyland_frequency=_weighted_diagnostic_rate(diagnostics, "root_fantasyland_frequency"),
        opponent_fantasyland_frequency=_weighted_diagnostic_rate(diagnostics, "opponent_fantasyland_frequency"),
        mean_policy_decisions=_weighted_diagnostic_rate(diagnostics, "mean_policy_decisions"),
        exact_late_search_rollout_frequency=_weighted_diagnostic_rate(
            diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        mean_exact_late_search_decisions=_weighted_diagnostic_rate(
            diagnostics,
            "mean_exact_late_search_decisions",
        ),
        mean_exact_late_search_nodes=_weighted_diagnostic_rate(diagnostics, "mean_exact_late_search_nodes"),
        late_search_activation_rate=_weighted_diagnostic_rate(diagnostics, "late_search_activation_rate"),
        late_search_exact_rate=_weighted_diagnostic_rate(diagnostics, "late_search_exact_rate"),
        late_search_beam_rate=_weighted_diagnostic_rate(diagnostics, "late_search_beam_rate"),
        late_search_fallback_rate=_weighted_diagnostic_rate(diagnostics, "late_search_fallback_rate"),
        mean_late_search_nodes=_weighted_diagnostic_rate(diagnostics, "mean_late_search_nodes"),
        mean_late_search_depth=_weighted_diagnostic_rate(diagnostics, "mean_late_search_depth"),
        mean_late_search_terminal_evaluations=_weighted_diagnostic_rate(
            diagnostics,
            "mean_late_search_terminal_evaluations",
        ),
        phase_auto_search_activation_rate=_weighted_diagnostic_rate(
            diagnostics,
            "phase_auto_search_activation_rate",
        ),
        mean_phase_auto_search_tree_nodes=_weighted_diagnostic_rate(
            diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        mean_phase_auto_search_depth=_weighted_diagnostic_rate(diagnostics, "mean_phase_auto_search_depth"),
        top_action_root_foul_rate=_weighted_diagnostic_rate(top_diagnostics, "root_foul_rate"),
        top_action_opponent_foul_rate=_weighted_diagnostic_rate(top_diagnostics, "opponent_foul_rate"),
        top_action_both_foul_rate=_weighted_diagnostic_rate(top_diagnostics, "both_foul_rate"),
        top_action_continuation_frequency=_weighted_diagnostic_rate(top_diagnostics, "continuation_frequency"),
        top_action_root_fantasyland_frequency=_weighted_diagnostic_rate(
            top_diagnostics,
            "root_fantasyland_frequency",
        ),
        top_action_opponent_fantasyland_frequency=_weighted_diagnostic_rate(
            top_diagnostics,
            "opponent_fantasyland_frequency",
        ),
        top_action_mean_policy_decisions=_weighted_diagnostic_rate(top_diagnostics, "mean_policy_decisions"),
        top_action_exact_late_search_rollout_frequency=_weighted_diagnostic_rate(
            top_diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        top_action_mean_exact_late_search_decisions=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_exact_late_search_decisions",
        ),
        top_action_mean_exact_late_search_nodes=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_exact_late_search_nodes",
        ),
        top_action_late_search_activation_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "late_search_activation_rate",
        ),
        top_action_late_search_exact_rate=_weighted_diagnostic_rate(top_diagnostics, "late_search_exact_rate"),
        top_action_late_search_beam_rate=_weighted_diagnostic_rate(top_diagnostics, "late_search_beam_rate"),
        top_action_late_search_fallback_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "late_search_fallback_rate",
        ),
        top_action_mean_late_search_nodes=_weighted_diagnostic_rate(top_diagnostics, "mean_late_search_nodes"),
        top_action_mean_late_search_depth=_weighted_diagnostic_rate(top_diagnostics, "mean_late_search_depth"),
        top_action_mean_late_search_terminal_evaluations=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_late_search_terminal_evaluations",
        ),
        top_action_phase_auto_search_activation_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "phase_auto_search_activation_rate",
        ),
        top_action_mean_phase_auto_search_tree_nodes=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        top_action_mean_phase_auto_search_depth=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_phase_auto_search_depth",
        ),
        labeled_top1_rate=_optional_boolean_rate(case.top1_agreement for case in labeled_cases),
        labeled_top3_rate=_optional_boolean_rate(case.top3_agreement for case in labeled_cases),
        elapsed_seconds=run.elapsed_seconds,
    )


def _aggregate_benchmark_payload(payload: Mapping[str, Any]) -> BenchmarkAggregate:
    cases = _require_case_payloads(payload, "benchmark")
    diagnostics = tuple(diagnostic for case in cases for diagnostic in case["action_diagnostics"])
    top_diagnostics = _top_action_diagnostics_from_payloads(cases)
    sample_count = sum(int(diagnostic["sample_count"]) for diagnostic in diagnostics)
    labeled_cases = tuple(case for case in cases if case["top1_agreement"] is not None)
    return BenchmarkAggregate(
        policy_name=str(payload["policy_name"]),
        case_count=int(payload["case_count"]),
        action_count=sum(int(case["action_count"]) for case in cases),
        sample_count=sample_count,
        root_foul_rate=_weighted_payload_rate(diagnostics, "root_foul_rate"),
        opponent_foul_rate=_weighted_payload_rate(diagnostics, "opponent_foul_rate"),
        both_foul_rate=_weighted_payload_rate(diagnostics, "both_foul_rate"),
        continuation_frequency=_weighted_payload_rate(diagnostics, "continuation_frequency"),
        root_fantasyland_frequency=_weighted_payload_rate(diagnostics, "root_fantasyland_frequency"),
        opponent_fantasyland_frequency=_weighted_payload_rate(diagnostics, "opponent_fantasyland_frequency"),
        mean_policy_decisions=_weighted_payload_rate(diagnostics, "mean_policy_decisions"),
        exact_late_search_rollout_frequency=_weighted_payload_rate(
            diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        mean_exact_late_search_decisions=_weighted_payload_rate(
            diagnostics,
            "mean_exact_late_search_decisions",
        ),
        mean_exact_late_search_nodes=_weighted_payload_rate(diagnostics, "mean_exact_late_search_nodes"),
        late_search_activation_rate=_weighted_payload_rate(diagnostics, "late_search_activation_rate"),
        late_search_exact_rate=_weighted_payload_rate(diagnostics, "late_search_exact_rate"),
        late_search_beam_rate=_weighted_payload_rate(diagnostics, "late_search_beam_rate"),
        late_search_fallback_rate=_weighted_payload_rate(diagnostics, "late_search_fallback_rate"),
        mean_late_search_nodes=_weighted_payload_rate(diagnostics, "mean_late_search_nodes"),
        mean_late_search_depth=_weighted_payload_rate(diagnostics, "mean_late_search_depth"),
        mean_late_search_terminal_evaluations=_weighted_payload_rate(
            diagnostics,
            "mean_late_search_terminal_evaluations",
        ),
        phase_auto_search_activation_rate=_weighted_payload_rate(diagnostics, "phase_auto_search_activation_rate"),
        mean_phase_auto_search_tree_nodes=_weighted_payload_rate(diagnostics, "mean_phase_auto_search_tree_nodes"),
        mean_phase_auto_search_depth=_weighted_payload_rate(diagnostics, "mean_phase_auto_search_depth"),
        top_action_root_foul_rate=_weighted_payload_rate(top_diagnostics, "root_foul_rate"),
        top_action_opponent_foul_rate=_weighted_payload_rate(top_diagnostics, "opponent_foul_rate"),
        top_action_both_foul_rate=_weighted_payload_rate(top_diagnostics, "both_foul_rate"),
        top_action_continuation_frequency=_weighted_payload_rate(top_diagnostics, "continuation_frequency"),
        top_action_root_fantasyland_frequency=_weighted_payload_rate(top_diagnostics, "root_fantasyland_frequency"),
        top_action_opponent_fantasyland_frequency=_weighted_payload_rate(
            top_diagnostics,
            "opponent_fantasyland_frequency",
        ),
        top_action_mean_policy_decisions=_weighted_payload_rate(top_diagnostics, "mean_policy_decisions"),
        top_action_exact_late_search_rollout_frequency=_weighted_payload_rate(
            top_diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        top_action_mean_exact_late_search_decisions=_weighted_payload_rate(
            top_diagnostics,
            "mean_exact_late_search_decisions",
        ),
        top_action_mean_exact_late_search_nodes=_weighted_payload_rate(
            top_diagnostics,
            "mean_exact_late_search_nodes",
        ),
        top_action_late_search_activation_rate=_weighted_payload_rate(
            top_diagnostics,
            "late_search_activation_rate",
        ),
        top_action_late_search_exact_rate=_weighted_payload_rate(top_diagnostics, "late_search_exact_rate"),
        top_action_late_search_beam_rate=_weighted_payload_rate(top_diagnostics, "late_search_beam_rate"),
        top_action_late_search_fallback_rate=_weighted_payload_rate(
            top_diagnostics,
            "late_search_fallback_rate",
        ),
        top_action_mean_late_search_nodes=_weighted_payload_rate(top_diagnostics, "mean_late_search_nodes"),
        top_action_mean_late_search_depth=_weighted_payload_rate(top_diagnostics, "mean_late_search_depth"),
        top_action_mean_late_search_terminal_evaluations=_weighted_payload_rate(
            top_diagnostics,
            "mean_late_search_terminal_evaluations",
        ),
        top_action_phase_auto_search_activation_rate=_weighted_payload_rate(
            top_diagnostics,
            "phase_auto_search_activation_rate",
        ),
        top_action_mean_phase_auto_search_tree_nodes=_weighted_payload_rate(
            top_diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        top_action_mean_phase_auto_search_depth=_weighted_payload_rate(
            top_diagnostics,
            "mean_phase_auto_search_depth",
        ),
        labeled_top1_rate=_optional_boolean_rate(bool(case["top1_agreement"]) for case in labeled_cases),
        labeled_top3_rate=_optional_boolean_rate(bool(case["top3_agreement"]) for case in labeled_cases),
        elapsed_seconds=float(payload["elapsed_seconds"]),
    )


def _aggregate_deltas(left: BenchmarkAggregate, right: BenchmarkAggregate) -> dict[str, float | None]:
    fields = (
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
        "late_search_activation_rate",
        "late_search_exact_rate",
        "late_search_beam_rate",
        "late_search_fallback_rate",
        "mean_late_search_nodes",
        "mean_late_search_depth",
        "mean_late_search_terminal_evaluations",
        "phase_auto_search_activation_rate",
        "mean_phase_auto_search_tree_nodes",
        "mean_phase_auto_search_depth",
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
        "top_action_late_search_activation_rate",
        "top_action_late_search_exact_rate",
        "top_action_late_search_beam_rate",
        "top_action_late_search_fallback_rate",
        "top_action_mean_late_search_nodes",
        "top_action_mean_late_search_depth",
        "top_action_mean_late_search_terminal_evaluations",
        "top_action_phase_auto_search_activation_rate",
        "top_action_mean_phase_auto_search_tree_nodes",
        "top_action_mean_phase_auto_search_depth",
        "labeled_top1_rate",
        "labeled_top3_rate",
        "elapsed_seconds",
    )
    return {field: _delta(getattr(left, field), getattr(right, field)) for field in fields}


def _tag_slice_deltas(
    left: BenchmarkTagSliceAggregate,
    right: BenchmarkTagSliceAggregate,
) -> dict[str, float | None]:
    fields = (
        "root_foul_rate",
        "both_foul_rate",
        "continuation_frequency",
        "root_fantasyland_frequency",
        "exact_late_search_rollout_frequency",
        "late_search_activation_rate",
        "late_search_exact_rate",
        "late_search_beam_rate",
        "late_search_fallback_rate",
        "mean_late_search_nodes",
        "phase_auto_search_activation_rate",
        "mean_phase_auto_search_tree_nodes",
        "top_action_root_foul_rate",
        "top_action_both_foul_rate",
        "top_action_continuation_frequency",
        "top_action_root_fantasyland_frequency",
        "top_action_exact_late_search_rollout_frequency",
        "top_action_late_search_activation_rate",
        "top_action_late_search_exact_rate",
        "top_action_late_search_beam_rate",
        "top_action_late_search_fallback_rate",
        "top_action_mean_late_search_nodes",
        "top_action_phase_auto_search_activation_rate",
        "top_action_mean_phase_auto_search_tree_nodes",
        "labeled_top1_rate",
        "labeled_top3_rate",
    )
    return {field: _delta(getattr(left, field), getattr(right, field)) for field in fields}


def _tag_slice_comparisons_from_runs(
    left: BenchmarkRun,
    right: BenchmarkRun,
) -> tuple[BenchmarkTagSliceComparison, ...]:
    tags = sorted({tag for case in left.case_results for tag in case.tags})
    comparisons: list[BenchmarkTagSliceComparison] = []
    for tag in tags:
        left_cases = tuple(case for case in left.case_results if tag in case.tags)
        right_cases = tuple(case for case in right.case_results if tag in case.tags)
        left_aggregate = _aggregate_tag_slice_cases(left_cases)
        right_aggregate = _aggregate_tag_slice_cases(right_cases)
        comparisons.append(
            BenchmarkTagSliceComparison(
                tag=tag,
                case_count=len(left_cases),
                left=left_aggregate,
                right=right_aggregate,
                deltas=_tag_slice_deltas(left_aggregate, right_aggregate),
            )
        )
    return tuple(comparisons)


def _aggregate_tag_slice_cases(cases: tuple[BenchmarkCaseResult, ...]) -> BenchmarkTagSliceAggregate:
    diagnostics = tuple(diagnostic for case in cases for diagnostic in case.action_diagnostics)
    top_diagnostics = tuple(
        _diagnostic_for_action_index(case.action_diagnostics, case.top_action_index)
        for case in cases
    )
    labeled_cases = tuple(case for case in cases if case.top1_agreement is not None)
    return BenchmarkTagSliceAggregate(
        case_count=len(cases),
        action_count=sum(case.action_count for case in cases),
        sample_count=sum(diagnostic.sample_count for diagnostic in diagnostics),
        root_foul_rate=_weighted_diagnostic_rate(diagnostics, "root_foul_rate"),
        both_foul_rate=_weighted_diagnostic_rate(diagnostics, "both_foul_rate"),
        continuation_frequency=_weighted_diagnostic_rate(diagnostics, "continuation_frequency"),
        root_fantasyland_frequency=_weighted_diagnostic_rate(diagnostics, "root_fantasyland_frequency"),
        exact_late_search_rollout_frequency=_weighted_diagnostic_rate(
            diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        late_search_activation_rate=_weighted_diagnostic_rate(diagnostics, "late_search_activation_rate"),
        late_search_exact_rate=_weighted_diagnostic_rate(diagnostics, "late_search_exact_rate"),
        late_search_beam_rate=_weighted_diagnostic_rate(diagnostics, "late_search_beam_rate"),
        late_search_fallback_rate=_weighted_diagnostic_rate(diagnostics, "late_search_fallback_rate"),
        mean_late_search_nodes=_weighted_diagnostic_rate(diagnostics, "mean_late_search_nodes"),
        phase_auto_search_activation_rate=_weighted_diagnostic_rate(
            diagnostics,
            "phase_auto_search_activation_rate",
        ),
        mean_phase_auto_search_tree_nodes=_weighted_diagnostic_rate(
            diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        top_action_root_foul_rate=_weighted_diagnostic_rate(top_diagnostics, "root_foul_rate"),
        top_action_both_foul_rate=_weighted_diagnostic_rate(top_diagnostics, "both_foul_rate"),
        top_action_continuation_frequency=_weighted_diagnostic_rate(top_diagnostics, "continuation_frequency"),
        top_action_root_fantasyland_frequency=_weighted_diagnostic_rate(
            top_diagnostics,
            "root_fantasyland_frequency",
        ),
        top_action_exact_late_search_rollout_frequency=_weighted_diagnostic_rate(
            top_diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        top_action_late_search_activation_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "late_search_activation_rate",
        ),
        top_action_late_search_exact_rate=_weighted_diagnostic_rate(top_diagnostics, "late_search_exact_rate"),
        top_action_late_search_beam_rate=_weighted_diagnostic_rate(top_diagnostics, "late_search_beam_rate"),
        top_action_late_search_fallback_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "late_search_fallback_rate",
        ),
        top_action_mean_late_search_nodes=_weighted_diagnostic_rate(top_diagnostics, "mean_late_search_nodes"),
        top_action_phase_auto_search_activation_rate=_weighted_diagnostic_rate(
            top_diagnostics,
            "phase_auto_search_activation_rate",
        ),
        top_action_mean_phase_auto_search_tree_nodes=_weighted_diagnostic_rate(
            top_diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        labeled_top1_rate=_optional_boolean_rate(case.top1_agreement for case in labeled_cases),
        labeled_top3_rate=_optional_boolean_rate(case.top3_agreement for case in labeled_cases),
    )


def _tag_slice_comparisons_from_payloads(
    left_cases: tuple[Mapping[str, Any], ...],
    right_cases: tuple[Mapping[str, Any], ...],
) -> tuple[BenchmarkTagSliceComparison, ...]:
    tags = sorted({str(tag) for case in left_cases for tag in case["tags"]})
    comparisons: list[BenchmarkTagSliceComparison] = []
    for tag in tags:
        left_slice_cases = tuple(case for case in left_cases if tag in case["tags"])
        right_slice_cases = tuple(case for case in right_cases if tag in case["tags"])
        left_aggregate = _aggregate_tag_slice_payload_cases(left_slice_cases)
        right_aggregate = _aggregate_tag_slice_payload_cases(right_slice_cases)
        comparisons.append(
            BenchmarkTagSliceComparison(
                tag=tag,
                case_count=len(left_slice_cases),
                left=left_aggregate,
                right=right_aggregate,
                deltas=_tag_slice_deltas(left_aggregate, right_aggregate),
            )
        )
    return tuple(comparisons)


def _aggregate_tag_slice_payload_cases(cases: tuple[Mapping[str, Any], ...]) -> BenchmarkTagSliceAggregate:
    diagnostics = tuple(diagnostic for case in cases for diagnostic in case["action_diagnostics"])
    top_diagnostics = _top_action_diagnostics_from_payloads(cases)
    labeled_cases = tuple(case for case in cases if case["top1_agreement"] is not None)
    return BenchmarkTagSliceAggregate(
        case_count=len(cases),
        action_count=sum(int(case["action_count"]) for case in cases),
        sample_count=sum(int(diagnostic["sample_count"]) for diagnostic in diagnostics),
        root_foul_rate=_weighted_payload_rate(diagnostics, "root_foul_rate"),
        both_foul_rate=_weighted_payload_rate(diagnostics, "both_foul_rate"),
        continuation_frequency=_weighted_payload_rate(diagnostics, "continuation_frequency"),
        root_fantasyland_frequency=_weighted_payload_rate(diagnostics, "root_fantasyland_frequency"),
        exact_late_search_rollout_frequency=_weighted_payload_rate(
            diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        late_search_activation_rate=_weighted_payload_rate(diagnostics, "late_search_activation_rate"),
        late_search_exact_rate=_weighted_payload_rate(diagnostics, "late_search_exact_rate"),
        late_search_beam_rate=_weighted_payload_rate(diagnostics, "late_search_beam_rate"),
        late_search_fallback_rate=_weighted_payload_rate(diagnostics, "late_search_fallback_rate"),
        mean_late_search_nodes=_weighted_payload_rate(diagnostics, "mean_late_search_nodes"),
        phase_auto_search_activation_rate=_weighted_payload_rate(diagnostics, "phase_auto_search_activation_rate"),
        mean_phase_auto_search_tree_nodes=_weighted_payload_rate(diagnostics, "mean_phase_auto_search_tree_nodes"),
        top_action_root_foul_rate=_weighted_payload_rate(top_diagnostics, "root_foul_rate"),
        top_action_both_foul_rate=_weighted_payload_rate(top_diagnostics, "both_foul_rate"),
        top_action_continuation_frequency=_weighted_payload_rate(top_diagnostics, "continuation_frequency"),
        top_action_root_fantasyland_frequency=_weighted_payload_rate(top_diagnostics, "root_fantasyland_frequency"),
        top_action_exact_late_search_rollout_frequency=_weighted_payload_rate(
            top_diagnostics,
            "exact_late_search_rollout_frequency",
        ),
        top_action_late_search_activation_rate=_weighted_payload_rate(
            top_diagnostics,
            "late_search_activation_rate",
        ),
        top_action_late_search_exact_rate=_weighted_payload_rate(top_diagnostics, "late_search_exact_rate"),
        top_action_late_search_beam_rate=_weighted_payload_rate(top_diagnostics, "late_search_beam_rate"),
        top_action_late_search_fallback_rate=_weighted_payload_rate(
            top_diagnostics,
            "late_search_fallback_rate",
        ),
        top_action_mean_late_search_nodes=_weighted_payload_rate(top_diagnostics, "mean_late_search_nodes"),
        top_action_phase_auto_search_activation_rate=_weighted_payload_rate(
            top_diagnostics,
            "phase_auto_search_activation_rate",
        ),
        top_action_mean_phase_auto_search_tree_nodes=_weighted_payload_rate(
            top_diagnostics,
            "mean_phase_auto_search_tree_nodes",
        ),
        labeled_top1_rate=_optional_boolean_rate(bool(case["top1_agreement"]) for case in labeled_cases),
        labeled_top3_rate=_optional_boolean_rate(bool(case["top3_agreement"]) for case in labeled_cases),
    )


def _top_action_changes_from_runs(left: BenchmarkRun, right: BenchmarkRun) -> tuple[TopActionChange, ...]:
    changes = []
    for left_case, right_case in zip(left.case_results, right.case_results, strict=True):
        if left_case.top_action_index == right_case.top_action_index:
            continue
        changes.append(
            TopActionChange(
                case_name=left_case.name,
                left_top_action_index=left_case.top_action_index,
                right_top_action_index=right_case.top_action_index,
                left_top_mean_value=left_case.ranked_actions[0].mean_value,
                right_top_mean_value=right_case.ranked_actions[0].mean_value,
            )
        )
    return tuple(changes)


def _top_action_changes_from_payloads(
    left_cases: tuple[Mapping[str, Any], ...],
    right_cases: tuple[Mapping[str, Any], ...],
) -> tuple[TopActionChange, ...]:
    changes = []
    for left_case, right_case in zip(left_cases, right_cases, strict=True):
        if left_case["top_action_index"] == right_case["top_action_index"]:
            continue
        changes.append(
            TopActionChange(
                case_name=str(left_case["name"]),
                left_top_action_index=int(left_case["top_action_index"]),
                right_top_action_index=int(right_case["top_action_index"]),
                left_top_mean_value=float(left_case["ranked_actions"][0]["mean_value"]),
                right_top_mean_value=float(right_case["ranked_actions"][0]["mean_value"]),
            )
        )
    return tuple(changes)


def _top_action_diagnostics_from_run(run: BenchmarkRun) -> tuple[BenchmarkActionDiagnostics, ...]:
    return tuple(
        _diagnostic_for_action_index(case.action_diagnostics, case.top_action_index)
        for case in run.case_results
    )


def _diagnostic_for_action_index(
    diagnostics: tuple[BenchmarkActionDiagnostics, ...],
    action_index: int,
) -> BenchmarkActionDiagnostics:
    for diagnostic in diagnostics:
        if diagnostic.action_index == action_index:
            return diagnostic
    raise ValueError(f"Benchmark case is missing diagnostics for top action index {action_index}")


def _top_action_diagnostics_from_payloads(
    cases: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _diagnostic_payload_for_action_index(case["action_diagnostics"], int(case["top_action_index"]))
        for case in cases
    )


def _diagnostic_payload_for_action_index(diagnostics: Any, action_index: int) -> Mapping[str, Any]:
    if not isinstance(diagnostics, list):
        raise ValueError("benchmark case action_diagnostics must be a list")
    for diagnostic in diagnostics:
        normalized = _require_mapping(diagnostic, "benchmark case.action_diagnostics[]")
        if int(normalized["action_index"]) == action_index:
            return normalized
    raise ValueError(f"benchmark case is missing diagnostics for top action index {action_index}")


def _weighted_diagnostic_rate(diagnostics: tuple[BenchmarkActionDiagnostics, ...], field: str) -> float:
    sample_count = sum(diagnostic.sample_count for diagnostic in diagnostics)
    if sample_count == 0:
        return 0.0
    return sum(diagnostic.sample_count * float(getattr(diagnostic, field)) for diagnostic in diagnostics) / sample_count


def _weighted_payload_rate(diagnostics: tuple[Mapping[str, Any], ...], field: str) -> float:
    sample_count = sum(int(diagnostic["sample_count"]) for diagnostic in diagnostics)
    if sample_count == 0:
        return 0.0
    return sum(int(diagnostic["sample_count"]) * float(diagnostic.get(field, 0.0)) for diagnostic in diagnostics) / sample_count


def _optional_boolean_rate(values) -> float | None:
    normalized = tuple(values)
    if not normalized:
        return None
    return sum(1 for value in normalized if value) / len(normalized)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def _require_case_payloads(payload: Mapping[str, Any], path: str) -> tuple[Mapping[str, Any], ...]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a cases list")
    return tuple(_require_mapping(case, f"{path}.cases[{index}]") for index, case in enumerate(cases))


def _validate_comparable_case_names(left_names: tuple[str, ...], right_names: tuple[str, ...]) -> None:
    if left_names != right_names:
        raise ValueError("Benchmark runs must contain the same cases in the same order")


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
    "BenchmarkAggregate",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkComparison",
    "BenchmarkManifest",
    "BenchmarkRun",
    "BenchmarkTagSliceAggregate",
    "BenchmarkTagSliceComparison",
    "EarlySearchBenchmark",
    "LateSearchBenchmark",
    "RootActionRiskAblationBenchmark",
    "RootActionRiskAblationRun",
    "RootActionRiskBenchmark",
    "TopActionChange",
    "compare_benchmark_payloads",
    "compare_benchmark_runs",
    "filter_benchmark_manifest",
    "load_benchmark_manifest",
    "load_benchmark_manifest_data",
    "run_early_search_benchmark",
    "run_late_search_benchmark",
    "run_benchmark_case",
    "run_benchmark_manifest",
    "run_root_action_risk_ablation_benchmark",
    "run_root_action_risk_benchmark",
]
