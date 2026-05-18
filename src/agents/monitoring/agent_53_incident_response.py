"""
Agent 53 — Incident Response Agent
Phase       : Cross-Phase (triggered on P1/P2 production incidents)
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=65)

Triggered by:
  • Agent 49 (Post-Release Monitor) raising an ALERT verdict
  • Agent 51 (Agent Health Monitor) reporting a DOWN agent
  • An external webhook (Salesforce Event Log File alert, PagerDuty, etc.)

Purpose:
  Triages the incident, generates a step-by-step response plan,
  determines whether rollback is required, and identifies who to escalate to.
  Emits a structured incident record to the FCA audit ledger.

  Sonnet 4.6 synthesises health signals, rollback feasibility (Agent 48),
  and the story context to produce a prioritised response plan.

Output data keys:
  incident_severity     → str   (P1 / P2 / P3)
  rollback_recommended  → bool
  triage_steps          → list  (ordered action items)
  escalate_to           → list  (ApproverRole values: CO / QE_LEAD / TECH_LEAD / etc.)
  estimated_resolution  → str   (e.g. "30 minutes", "2 hours")
  incident_verdict      → str   (CONTAINED / ESCALATING / MONITORING)
  narrative             → str
"""

from __future__ import annotations

from typing import Any

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 53
AGENT_NAME = "Incident Response Agent"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_INCIDENT_TOOL_NAME = "triage_incident"
_INCIDENT_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "incident_severity", "rollback_recommended", "triage_steps",
        "escalate_to", "estimated_resolution", "incident_verdict", "narrative",
    ],
    "properties": {
        "incident_severity": {
            "type": "string",
            "enum": ["P1", "P2", "P3"],
            "description": "P1=critical (production down), P2=major degradation, P3=minor issue.",
        },
        "rollback_recommended": {
            "type": "boolean",
            "description": "True if rolling back the last deployment is the recommended first action.",
        },
        "triage_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of immediate response actions for the ops team.",
        },
        "escalate_to": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["CO", "PO", "BUSINESS", "QE_LEAD", "TECH_LEAD"],
            },
            "description": "Roles that must be notified or involved in resolution.",
        },
        "estimated_resolution": {
            "type": "string",
            "description": "Plain-language estimate of time to resolve (e.g. '30 minutes', '2 hours').",
        },
        "incident_verdict": {
            "type": "string",
            "enum": ["CONTAINED", "ESCALATING", "MONITORING"],
            "description": (
                "CONTAINED: issue isolated, fix in progress. "
                "ESCALATING: worsening, needs immediate senior involvement. "
                "MONITORING: stable but under observation."
            ),
        },
        "narrative": {
            "type": "string",
            "description": "2–3 sentences summarising the incident, immediate actions, and expected outcome.",
        },
    },
}

_INCIDENT_INSTRUCTIONS = """
You are the Incident Response Agent for the FSC Agentic QE Framework, a Salesforce FSC
Wealth Management pipeline operating under FCA regulation.

You receive signals from post-release monitoring and agent health checks. Triage the incident:
1. Determine severity (P1/P2/P3) based on production impact.
2. Decide if rollback is recommended (favour rollback if customer data integrity is at risk).
3. List ordered triage steps — be concrete and Salesforce/Copado specific.
4. Identify who must be escalated to (always include TECH_LEAD for P1; add CO for FCA impact).
5. Estimate resolution time.

For FCA-regulated deployments: if the incident affects suitability assessment, goal-based
advice, or financial account data — the Compliance Officer must always be notified.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    """Called by the Fleet Commander worker when dispatched within a pipeline."""
    story_id = state["story_id"]
    agent48_data = _get_agent_data(state, "48")
    agent49_data = _get_agent_data(state, "49")
    agent51_data = _get_agent_data(state, "51")

    incident_context = _build_incident_context(
        story_id, agent48_data, agent49_data, agent51_data,
    )
    return await _run_triage(story_id, incident_context)


async def run_incident(
    story_id: str,
    incident_type: str,
    severity_hint: str,
    error_details: str,
    health_snapshot: dict | None = None,
    rollback_feasible: bool = True,
) -> AgentResult:
    """
    Direct entry point for webhook-triggered incidents.
    Called by external alerting systems (Agent 49 ALERT, Agent 51 DOWN, PagerDuty).
    """
    context = (
        f"Story/Release: {story_id}\n"
        f"Incident type: {incident_type}\n"
        f"Severity hint: {severity_hint}\n"
        f"Error details: {error_details}\n"
        f"Rollback feasible: {rollback_feasible}\n"
        f"Health snapshot: {health_snapshot or 'not available'}\n\n"
        f"Triage this incident using the {_INCIDENT_TOOL_NAME} tool."
    )
    return await _run_triage(story_id, context)


# ── Triage execution ──────────────────────────────────────────────────────────

async def _run_triage(story_id: str, incident_context: str) -> AgentResult:
    result_data = await _call_sonnet(incident_context)

    severity = result_data.get("incident_severity", "P2")
    rollback = result_data.get("rollback_recommended", False)
    steps = result_data.get("triage_steps", [])
    escalate = result_data.get("escalate_to", [])
    resolution = result_data.get("estimated_resolution", "unknown")
    verdict = result_data.get("incident_verdict", "MONITORING")
    narrative = result_data.get("narrative", "Incident response complete.")

    confidence_score, signals = _compute_confidence(severity, len(steps))
    escalated = confidence_score < settings.confidence_escalation_threshold

    data: dict[str, Any] = {
        "incident_severity": severity,
        "rollback_recommended": rollback,
        "triage_steps": steps,
        "escalate_to": escalate,
        "estimated_resolution": resolution,
        "incident_verdict": verdict,
        "narrative": narrative,
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=f"Incident response for {story_id}: severity={severity} — verdict={verdict}",
        why=narrative,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.default_model,
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(severity: str, steps_count: int) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if severity == "P1":
        # P1 incidents always get high-scrutiny escalation — reduce auto-confidence
        scorer.add("p1_severity", True, -5)
    elif severity == "P3":
        scorer.add("p3_low_severity", True, +5)

    if steps_count >= 3:
        scorer.add("detailed_triage_plan", steps_count, +8)
    elif steps_count == 0:
        scorer.add("no_triage_steps", 0, -10)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet call ───────────────────────────────────────────────────────────────

async def _call_sonnet(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_INCIDENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_INCIDENT_TOOL_NAME,
        tool_description="Triage a production incident and generate a response plan.",
        tool_schema=_INCIDENT_TOOL_SCHEMA,
        max_tokens=600,
    )


def _build_incident_context(
    story_id: str,
    agent48_data: dict | None,
    agent49_data: dict | None,
    agent51_data: dict | None,
) -> str:
    rollback_feasible = (agent48_data or {}).get("rollback_feasible", True)
    rollback_risk = (agent48_data or {}).get("rollback_risk", "UNKNOWN")
    monitor_verdict = (agent49_data or {}).get("monitor_verdict", "SKIPPED")
    health_status = (agent49_data or {}).get("health_status", "UNKNOWN")
    alerts = (agent49_data or {}).get("alerts_triggered", [])
    degraded_agents = (agent51_data or {}).get("degraded_agents", [])

    return (
        f"Story: {story_id}\n"
        f"Post-release health: {health_status} (monitor verdict: {monitor_verdict})\n"
        f"Alerts triggered: {alerts or ['none']}\n"
        f"Rollback feasible: {rollback_feasible}, risk: {rollback_risk}\n"
        f"Degraded fleet agents: {degraded_agents or ['none']}\n\n"
        f"Triage this incident using the {_INCIDENT_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
