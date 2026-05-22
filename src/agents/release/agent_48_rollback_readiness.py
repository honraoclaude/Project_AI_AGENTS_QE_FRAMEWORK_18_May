"""
Agent 48 — Rollback Readiness
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=55)

Runs in Release Final Batch (parallel with Agents 49, 50) — after Gate G12.
Has access to Agents 13, 41.

Purpose:
  Assesses whether the deployed change can be rolled back if a critical
  post-release defect is discovered. Destructive changes, schema migrations,
  and cross-org data writes increase rollback complexity.

  This agent is scaffolded for Phase 2 production integration. In v1, it
  produces a risk assessment from the change set composition signals.

Output data keys consumed by downstream:
  rollback_feasible  → bool   (True if rollback is practical)
  rollback_risk      → str    (LOW / MEDIUM / HIGH)
  rollback_steps     → list   (stub steps for the ops team)
  rollback_verdict   → str    (FEASIBLE / RISKY / NOT_FEASIBLE)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 48
AGENT_NAME = "Rollback Readiness"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_rollback_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "rollback_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising rollback feasibility. "
                "State the risk level, any factors that complicate rollback, "
                "and key steps the ops team should prepare."
            ),
        },
        "rollback_concern": {
            "type": "string",
            "enum": ["none", "destructive_changes", "schema_migration",
                     "data_writes", "high_complexity"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a rollback readiness assessment
in a Salesforce FSC Wealth Management deployment pipeline.
You will receive the change set composition including any destructive changes.
Write a clear 2–3 sentence narrative on rollback feasibility, key risk factors,
and recommended rollback steps for the operations team.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent8_data  = _get_agent_data(state, "8")
    agent13_data = _get_agent_data(state, "13")
    agent40_data = _get_agent_data(state, "40")
    agent41_data = _get_agent_data(state, "41")

    feasible, risk, steps, verdict = _assess_rollback(agent8_data, agent13_data, agent40_data, agent41_data)

    trace_msg = _build_trace_message(story_id, feasible, risk, steps, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent13_data, agent41_data, agent8_data, agent40_data)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = f"Rollback readiness for {story_id}: feasible={feasible}, risk={risk} — verdict={verdict}"
    why = trace.get("narrative", "Rollback Readiness assessed change set reversibility.")

    data = {
        "rollback_feasible": feasible,
        "rollback_risk": risk,
        "rollback_steps": steps,
        "rollback_verdict": verdict,
        "rollback_concern": trace.get("rollback_concern", "none"),
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


# ── Deterministic rollback assessment ────────────────────────────────────────

def _assess_rollback(
    agent8_data: dict | None,
    agent13_data: dict | None,
    agent40_data: dict | None,
    agent41_data: dict | None,
) -> tuple[bool, str, list[str], str]:
    """Returns (feasible, risk_level, steps, verdict)."""
    destructive       = (agent13_data or {}).get("has_destructive_changes", False)
    dep_depth         = (agent13_data or {}).get("dependency_depth", 0)
    has_external_deps = (agent8_data or {}).get("has_external_dependencies", False)
    release_type      = (agent40_data or {}).get("release_type", "PATCH")
    integrity_verdict = (agent41_data or {}).get("integrity_verdict", "PASS")
    integrity_issues  = (agent41_data or {}).get("integrity_issues", [])

    steps = ["Deploy previous change set version via Copado"]

    if destructive:
        steps.append("Restore deleted metadata from version control")
        steps.append("Verify data integrity post-restore")

    if dep_depth >= 3:
        steps.append("Review cross-object dependency chain before rollback")

    # REQ-31 Gap 1: integrity issues from Agent 41 increase rollback risk
    if integrity_verdict == "FAIL":
        steps.append("Resolve change set integrity issues before attempting rollback")

    # REQ-31 Gap 2: external dependencies are hard to roll back
    if has_external_deps:
        steps.append(
            "Verify external service configuration (Named Credentials/Connected Apps) "
            "in target org before rollback — metadata revert does not undo external state"
        )

    # REQ-31 Gap 3: MAJOR release (schema change) increases rollback complexity
    if release_type == "MAJOR":
        steps.append("Schema change present — data migration may be required before rollback")

    # Risk assessment
    if destructive and dep_depth >= 2:
        risk = "HIGH"
        feasible = False
        verdict = "NOT_FEASIBLE"
    elif destructive or dep_depth >= 3 or integrity_verdict == "FAIL" or has_external_deps or release_type == "MAJOR":
        risk = "MEDIUM"
        feasible = True
        verdict = "RISKY"
    else:
        risk = "LOW"
        feasible = True
        verdict = "FEASIBLE"

    # NOT_FEASIBLE is informational — future Gate G12 could enforce this
    return feasible, risk, steps, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent13_data: dict | None,
    agent41_data: dict | None,
    agent8_data: dict | None = None,
    agent40_data: dict | None = None,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=55)

    if agent13_data:
        scorer.add("metadata_available", True, +10)
    else:
        scorer.add("no_metadata", 0, -10)

    if agent41_data:
        scorer.add("change_set_data_available", True, +6)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a rollback readiness narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    feasible: bool,
    risk: str,
    steps: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Rollback feasible: {feasible}\n"
        f"Risk level: {risk}\n"
        f"Verdict: {verdict}\n"
        f"Steps: {steps}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
