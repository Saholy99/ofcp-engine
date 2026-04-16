"""Public exports for the analysis harness layer.

This package contains exact-state scenario loading, observer-facing state
projection, deterministic rendering, and CLI helpers. It must remain separate
from both the engine package and the future solver package.
"""

from ofc_analysis.action_codec import EncodedAction, decode_action, encode_action, encode_actions
from ofc_analysis.observation import PlayerObservation, project_observation
from ofc_analysis.render import render_actions, render_move_analysis, render_observation, render_state
from ofc_analysis.scenario import ExactStateScenario, load_scenario, load_scenario_data

__all__ = [
    "EncodedAction",
    "ExactStateScenario",
    "PlayerObservation",
    "decode_action",
    "encode_action",
    "encode_actions",
    "load_scenario",
    "load_scenario_data",
    "project_observation",
    "render_actions",
    "render_move_analysis",
    "render_observation",
    "render_state",
]
