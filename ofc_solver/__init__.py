"""Compatibility wrapper for importing the src-layout solver package."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ofc_solver"
__path__ = [str(_SRC_PACKAGE)]

__all__ = [
    "BenchmarkRun",
    "EarlySearchConfig",
    "LateSearchConfig",
    "MoveAnalysis",
    "MoveEstimate",
    "load_benchmark_manifest",
    "rank_actions_from_observation",
    "rank_actions_from_state",
    "run_early_search_benchmark",
    "run_late_search_benchmark",
    "run_benchmark_manifest",
    "select_early_search_candidates",
]


def __getattr__(name: str):
    """Lazily expose public solver helpers without creating import cycles."""

    if name in {"MoveAnalysis", "MoveEstimate"}:
        from ofc_solver.models import MoveAnalysis, MoveEstimate

        return {"MoveAnalysis": MoveAnalysis, "MoveEstimate": MoveEstimate}[name]
    if name in {"rank_actions_from_observation", "rank_actions_from_state"}:
        from ofc_solver.monte_carlo import rank_actions_from_observation, rank_actions_from_state

        return {
            "rank_actions_from_observation": rank_actions_from_observation,
            "rank_actions_from_state": rank_actions_from_state,
        }[name]
    if name in {"EarlySearchConfig", "select_early_search_candidates"}:
        from ofc_solver.early_search import EarlySearchConfig, select_early_search_candidates

        return {
            "EarlySearchConfig": EarlySearchConfig,
            "select_early_search_candidates": select_early_search_candidates,
        }[name]
    if name in {"LateSearchConfig"}:
        from ofc_solver.late_search import LateSearchConfig

        return {"LateSearchConfig": LateSearchConfig}[name]
    if name in {
        "BenchmarkRun",
        "load_benchmark_manifest",
        "run_early_search_benchmark",
        "run_late_search_benchmark",
        "run_benchmark_manifest",
    }:
        from ofc_solver.benchmark import (
            BenchmarkRun,
            load_benchmark_manifest,
            run_early_search_benchmark,
            run_late_search_benchmark,
            run_benchmark_manifest,
        )

        return {
            "BenchmarkRun": BenchmarkRun,
            "load_benchmark_manifest": load_benchmark_manifest,
            "run_early_search_benchmark": run_early_search_benchmark,
            "run_late_search_benchmark": run_late_search_benchmark,
            "run_benchmark_manifest": run_benchmark_manifest,
        }[name]
    raise AttributeError(f"module 'ofc_solver' has no attribute {name!r}")
