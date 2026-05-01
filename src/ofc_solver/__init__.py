"""Public scaffolding exports for the solver layer."""

__all__ = [
    "BenchmarkRun",
    "MoveAnalysis",
    "MoveEstimate",
    "RootActionRiskAssessment",
    "load_benchmark_manifest",
    "rank_actions_from_observation",
    "rank_actions_from_state",
    "run_benchmark_manifest",
    "run_root_action_risk_benchmark",
    "score_root_action",
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
    if name in {"RootActionRiskAssessment", "score_root_action"}:
        from ofc_solver.root_action_risk import RootActionRiskAssessment, score_root_action

        return {
            "RootActionRiskAssessment": RootActionRiskAssessment,
            "score_root_action": score_root_action,
        }[name]
    if name in {
        "BenchmarkRun",
        "load_benchmark_manifest",
        "run_benchmark_manifest",
        "run_root_action_risk_benchmark",
    }:
        from ofc_solver.benchmark import (
            BenchmarkRun,
            load_benchmark_manifest,
            run_benchmark_manifest,
            run_root_action_risk_benchmark,
        )

        return {
            "BenchmarkRun": BenchmarkRun,
            "load_benchmark_manifest": load_benchmark_manifest,
            "run_benchmark_manifest": run_benchmark_manifest,
            "run_root_action_risk_benchmark": run_root_action_risk_benchmark,
        }[name]
    raise AttributeError(f"module 'ofc_solver' has no attribute {name!r}")
