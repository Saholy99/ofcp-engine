"""Compatibility wrapper for importing the src-layout analysis package."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "ofc_analysis"
__path__ = [str(_SRC_PACKAGE)]

from ofc_analysis.action_codec import EncodedAction, decode_action, encode_action, encode_actions
from ofc_analysis.observation import PlayerObservation, project_observation
from ofc_analysis.play import MonteCarloSuggestionBackend, NoopSuggestionBackend, run_play_hand
from ofc_analysis.render import render_actions, render_move_analysis, render_observation, render_state
from ofc_analysis.scenario import ExactStateScenario, load_scenario, load_scenario_data

__all__ = [
    "EncodedAction",
    "ExactStateScenario",
    "PlayerObservation",
    "MonteCarloSuggestionBackend",
    "NoopSuggestionBackend",
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
    "run_play_hand",
]
