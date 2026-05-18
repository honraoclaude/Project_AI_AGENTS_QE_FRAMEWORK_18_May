"""
Agent 49 — Post-Release Monitor
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=50)

Runs in Release Final Batch (parallel with Agents 48, 50) — after Gate G12.
Has access to Agent 46.

Purpose:
  Monitors post-release health signals and alerts the ops team to emerging
  issues. In production, polls Salesforce Event Log Files and Apex Exception
  events over a configurable window (default: 30 minutes post-deployment).

  This agent is a stub in v1 — real monitoring integration is Phase 2.
  Infers monitor status from production validation verdict.

Output data keys consumed by downstream:
  monitoring_active   → bool
  alerts_triggered    → list   (any critical alerts fired post-release)
  health_status       → str    (NOMINAL / DEGRADED / CRITICAL)
  monitor_verdict     → str    (MONITORING / ALERT / SKIPPED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 49
AGENT_NAME = "Post-Release Monitor"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_monitor_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "monitor_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising post-release monitoring status. "
                "State whether monitoring is active, any alerts triggered, "
                "and the current health status."
            ),
        },
        "monitor_concern": {
            "type": "string",
            "enum": ["none", "apex_exceptions", "governor_breach",
                     "degraded_performance", "deployment_not_done"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for post-release monitoring
of a Salesforce FSC Wealth Management deployment.
You will receive the production health status and any alerts.
Write a clear 2–3 sentence narrative confirming monitoring status,
any concerns observed, and recommended actions for the ops team.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent46_data = _get_agent_data(state, "46")

    active, alerts, health_status, verdict = _check_monitoring(agent46_data)

    trace_msg = _build_trace_message(story_id, active, alerts, health_status, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent46_data, active)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = f"Post-release monitoring for {story_id}: active={active}, health={health_status} — verdict={verdict}"
    why = trace.get("narrative", "Post-Release Monitor assessed deployment health.")

    data = {
        "monitoring_active": active,
        "alerts_triggered": alerts,
        "health_status": health_status,
        "monitor_verdict": verdict,
        "monitor_concern": trace.get("monitor_concern", "none"),
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


# ── Deterministic monitoring check ───────────────────────────────────────────

def _check_monitoring(agent46_data: dict | None) -> tuple[bool, list[str], str, str]:
    """Returns (monitoring_active, alerts, health_status, verdict)."""
    if not agent46_data:
        return False, [], "UNKNOWN", "SKIPPED"

    prod_healthy = agent46_data.get("production_healthy", False)
    prod_verdict = agent46_data.get("prod_verdict", "SKIPPED")

    if prod_verdict == "SKIPPED" or not prod_healthy:
        return False, [], "UNKNOWN", "SKIPPED"

    # Stub: nominal health when production validated healthy
    return True, [], "NOMINAL", "MONITORING"


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent46_data: dict | None,
    active: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=50)

    if agent46_data:
        scorer.add("production_data_available", True, +12)
    else:
        scorer.add("no_production_data", 0, -10)

    if active:
        scorer.add("monitoring_active", True, +8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a post-release monitoring narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    active: bool,
    alerts: list[str],
    health_status: str,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Monitoring active: {active}\n"
        f"Alerts: {alerts or ['none']}\n"
        f"Health status: {health_status}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
