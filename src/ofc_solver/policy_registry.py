"""Shared rollout policy lookup for solver and benchmark entrypoints."""

from __future__ import annotations

from ofc_solver.heuristic_policy import HeuristicRolloutPolicy
from ofc_solver.rollout_policy import RandomRolloutPolicy, RolloutPolicy


POLICY_NAMES = ("random", "heuristic")


def policy_from_name(policy_name: str) -> RolloutPolicy:
    """Return a rollout policy by stable CLI/config name."""

    if policy_name == "random":
        return RandomRolloutPolicy()
    if policy_name == "heuristic":
        return HeuristicRolloutPolicy()
    supported = ", ".join(POLICY_NAMES)
    raise ValueError(f"Unsupported rollout policy: {policy_name!r}. Supported policies: {supported}.")


__all__ = ["POLICY_NAMES", "policy_from_name"]
