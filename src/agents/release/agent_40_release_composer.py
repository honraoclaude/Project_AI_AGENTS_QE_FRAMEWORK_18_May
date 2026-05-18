"""
Agent 40 — Release Composer
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs sequentially after Gate G7.
Has access to Agents 13, 17, 18.

Purpose:
  Assembles the release package metadata — change set name, component list,
  component count, and release type classification (MAJOR/MINOR/PATCH).
  Provides the structured release package that Change Set Integrity (41) validates.

  Stub: in production reads the actual Copado change set via MCP.

Output data keys consumed by downstream:
  release_name       → str   (Copado change set name)
  component_count    → int
  release_type       → str   (MAJOR / MINOR / PATCH)
  components_summary → dict  (component_type → count)
  composer_verdict   → str   (COMPOSED / PARTIAL / FAILED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 40
AGENT_NAME = "Release Composer"

_REGULATED_COMPONENT_TYPES = {"ApexClass", "ApexTrigger", "CustomObject", "CustomField", "ValidationRule"}

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_composer_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "composer_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the release package. "
                "State the release name, component count, release type, "
                "and any composition concerns."
            ),
        },
        "composer_concern": {
            "type": "string",
            "enum": ["none", "no_components", "regulated_components_present",
                     "large_change_set", "metadata_missing"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce FSC release composition step.
You will receive metadata about the components being released, their types, and the inferred
release type. Write a clear 2–3 sentence narrative describing the release package and
any concerns the QE team should be aware of before proceeding.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent13_data = _get_agent_data(state, "13")
    agent17_data = _get_agent_data(state, "17")
    agent18_data = _get_agent_data(state, "18")

    release_name, component_count, release_type, components_summary, verdict = _compose_release(
        story_id, agent13_data, agent17_data, agent18_data,
    )

    trace_msg = _build_trace_message(
        story_id, release_name, component_count, release_type, components_summary, verdict,
    )
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent13_data, agent18_data, component_count)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Release composition for {story_id}: {component_count} component(s), "
        f"type={release_type} — verdict={verdict}"
    )
    why = trace.get("narrative", "Release Composer assembled the release package.")

    data = {
        "release_name": release_name,
        "component_count": component_count,
        "release_type": release_type,
        "components_summary": components_summary,
        "composer_verdict": verdict,
        "composer_concern": trace.get("composer_concern", "none"),
        "narrative": trace.get("narrative", ""),
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=why,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.fast_model,
    )


# ── Deterministic composition ─────────────────────────────────────────────────

def _compose_release(
    story_id: str,
    agent13_data: dict | None,
    agent17_data: dict | None,
    agent18_data: dict | None,
) -> tuple[str, int, str, dict, str]:
    """Returns (release_name, component_count, release_type, components_summary, verdict)."""
    changed_files_count = (agent13_data or {}).get("changed_files_count", 0)
    detected_objects    = (agent13_data or {}).get("detected_objects", [])
    component_types     = (agent18_data or {}).get("component_types", {})

    # Stub: derive component count from metadata signals
    component_count = max(changed_files_count, len(component_types))
    if component_count == 0:
        component_count = len(detected_objects)

    # Build components summary from attribution data
    components_summary: dict = dict(component_types) if component_types else {}

    # Infer release type from component profile
    has_regulated   = any(ct in _REGULATED_COMPONENT_TYPES for ct in components_summary)
    apex_count      = components_summary.get("ApexClass", 0) + components_summary.get("ApexTrigger", 0)
    obj_count       = components_summary.get("CustomObject", 0)

    if obj_count > 0:
        release_type = "MAJOR"
    elif apex_count >= 1 or has_regulated:
        release_type = "MINOR"
    else:
        release_type = "PATCH"

    release_name = f"{story_id}-{release_type.lower()}-release"

    if component_count == 0:
        verdict = "PARTIAL"
    else:
        verdict = "COMPOSED"

    return release_name, component_count, release_type, components_summary, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent13_data: dict | None,
    agent18_data: dict | None,
    component_count: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    if agent13_data:
        scorer.add("metadata_available", True, +8)
    else:
        scorer.add("no_metadata", 0, -10)

    if agent18_data:
        scorer.add("component_attribution_available", True, +6)

    if component_count == 0:
        scorer.add("no_components_found", 0, -8)
    elif component_count >= 5:
        scorer.add("large_change_set", component_count, -3)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a release composition narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    release_name: str,
    component_count: int,
    release_type: str,
    components_summary: dict,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Release name: {release_name}\n"
        f"Component count: {component_count}\n"
        f"Release type: {release_type}\n"
        f"Components: {components_summary or '(none identified)'}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
