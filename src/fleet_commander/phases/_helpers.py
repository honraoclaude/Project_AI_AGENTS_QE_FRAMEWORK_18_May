"""
Shared helpers used by all three phase orchestrators.

Extracted to avoid the same two functions being copied verbatim across
development.py, testing.py, and release.py.
"""

from __future__ import annotations

from typing import Any

from src.core.schemas import StoryState


def _merge_results(
    state: StoryState,
    agent_ids: list[int],
    results: list[Any],
) -> StoryState:
    """
    Merge batch dispatch results into state. Failed agents (exceptions)
    are logged but do not prevent other agents from running — gates
    enforce the fail-closed logic.
    """
    for agent_id, result in zip(agent_ids, results):
        if isinstance(result, Exception):
            state["agent_results"][str(agent_id)] = {
                "error": str(result),
                "data": {},
            }
        else:
            state["agent_results"][str(agent_id)] = result
    return state


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    if not isinstance(result, dict):
        return None
    return result.get("data")
