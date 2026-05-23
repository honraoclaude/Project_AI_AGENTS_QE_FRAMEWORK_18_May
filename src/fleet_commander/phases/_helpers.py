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


def _get_agent_data_required(
    state: StoryState,
    agent_id: str,
    caller: str = "",
) -> dict | None:
    """
    Like _get_agent_data() but appends a warning to state["phase_errors"] when the
    upstream agent result is absent.  Use this in gates and agents that MUST have
    upstream data to produce a meaningful output — the absence is then visible in
    the pipeline report rather than silently defaulting.
    """
    data = _get_agent_data(state, agent_id)
    if data is None:
        label = f" (caller: {caller})" if caller else ""
        state["phase_errors"].append(
            f"Agent {agent_id} result missing{label} — downstream logic will use defaults; "
            "upstream agent may have failed or not yet run"
        )
    return data


def _collect_escalated_agents(state: StoryState, agent_ids: list[int]) -> list[str]:
    """
    Return warning strings for agents whose confidence score was escalated
    (i.e. fell below the escalation threshold).  Append the result to
    state["phase_errors"] at the call site so the information is visible
    to the fleet commander without blocking any gate.
    """
    warnings: list[str] = []
    for aid in agent_ids:
        result = state["agent_results"].get(str(aid))
        if not isinstance(result, dict):
            continue
        confidence = result.get("confidence", {})
        if confidence.get("escalated", False):
            name = result.get("agent_name", f"Agent {aid}")
            score = confidence.get("final_score", "?")
            warnings.append(
                f"Agent {aid} ({name}): confidence escalated (score={score}) — "
                "low-confidence decision; human review recommended"
            )
    return warnings
