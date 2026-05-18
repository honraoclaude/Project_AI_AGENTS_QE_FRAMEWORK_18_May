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
    agent13_data = _get_agent_data(state, "13")
    agent41_data = _get_agent_data(state, "41")

    feasible, risk, steps, verdict = _assess_rollback(agent13_data, agent41_data)

    trace_msg = _build_trace_message(story_id, feasible, risk, steps, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent13_data, agent41_data)
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
    agent13_data: dict | None,
    agent41_data: dict | None,
) -> tuple[bool, str, list[str], str]:
    """Returns (feasible, risk_level, steps, verdict)."""
    destructive   = (agent13_data or {}).get("has_destructive_changes", False)
    dep_depth     = (agent13_data or {}).get("dependency_depth", 0)
    int_verdict   = (agent41_data or {}).get("integrity_verdict", "PASS")

    steps = ["Deploy previous change set version via Copado"]

    if destructive:
        steps.append("Restore deleted metadata from version control")
        steps.append("Verify data integrity post-restore")

    if dep_depth >= 3:
        steps.append("Review cross-object dependency chain before rollback")

    # Risk assessment
    if destructive and dep_depth >= 2:
        risk = "HIGH"
        feasible = False
        verdict = "NOT_FEASIBLE"
    elif destructive or dep_depth >= 3:
        risk = "MEDIUM"
        feasible = True
        verdict = "RISKY"
    else:
        risk = "LOW"
        feasible = True
        verdict = "FEASIBLE"

    return feasible, risk, steps, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent13_data: dict | None,
    agent41_data: dict | None,
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
