"""Shared lightweight analysis-layer model scaffolding.

This module currently holds only generic render-output structure so the
analysis package can share a stable type without adding behavior yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderedOutput:
    """Placeholder container for text or JSON render results."""

    text: str | None = None
    payload: dict[str, Any] | None = None
