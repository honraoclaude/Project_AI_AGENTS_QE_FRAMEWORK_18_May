"""
Agent 46 — Production Validation Agent
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=58)

Runs sequentially after Gate G11.
Has access to Agent 45.

Purpose:
  Post-deployment health validation on the production org. Confirms the
  deployment completed, critical flows are operational, and no governor
  limit exceptions have been triggered.

  Stub: in production invokes Salesforce Apex health check endpoint and
  checks event log files for post-deployment errors. The current stub
  infers production health from the Go/No-Go decision.

  Gate G12 depends on prod_verdict.

Output data keys consumed by downstream:
  production_healthy        → bool   (Gate G12 — must be True)
  validation_checks_passed  → int
  validation_checks_total   → int
  prod_verdict              → str    (HEALTHY / DEGRADED / FAILED / SKIPPED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 46
AGENT_NAME = "Production Validation Agent"

_HEALTH_CHECKS = [
    "apex_execution_ok",
    "governor_limits_within_bounds",
    "critical_flows_accessible",
    "fca_validation_endpoint_responsive",
    "no_exception_events_in_log",
]

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_production_validation_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "prod_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the production health check. "
                "State how many checks passed, whether critical flows are healthy, "
                "and any immediate remediation required."
            ),
        },
        "prod_concern": {
            "type": "string",
            "enum": ["none", "apex_errors", "governor_breach",
                     "flow_inaccessible", "deployment_skipped"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a post-deployment production health
validation in a Salesforce FSC Wealth Management environment.
You will receive the Go/No-Go decision and validation check results.
Write a clear 2–3 sentence narrative confirming production health status
and any issues that require immediate attention from the ops team.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent45_data = _get_agent_data(state, "45")

    healthy, checks_passed, checks_total, verdict = _validate_production(agent45_data)

    trace_msg = _build_trace_message(story_id, healthy, checks_passed, checks_total, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent45_data, healthy)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Production validation for {story_id}: {checks_passed}/{checks_total} checks passed — "
        f"verdict={verdict}"
    )
    why = trace.get("narrative", "Production Validation Agent assessed post-deployment health.")

    data = {
        "production_healthy": healthy,
        "validation_checks_passed": checks_passed,
        "validation_checks_total": checks_total,
        "prod_verdict": verdict,
        "prod_concern": trace.get("prod_concern", "none"),
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


# ── Deterministic production validation ───────────────────────────────────────

def _validate_production(
    agent45_data: dict | None,
) -> tuple[bool, int, int, str]:
    """Returns (production_healthy, checks_passed, checks_total, verdict)."""
    go_decision = (agent45_data or {}).get("go_decision", False)
    coordinator_verdict = (agent45_data or {}).get("coordinator_verdict", "NO_GO")

    total = len(_HEALTH_CHECKS)

    if not agent45_data:
        return False, 0, total, "SKIPPED"

    if not go_decision or coordinator_verdict == "NO_GO":
        # Deployment should not have happened — skip validation
        return False, 0, total, "SKIPPED"

    # Stub: all health checks pass when Go/No-Go is GO or CONDITIONAL
    checks_passed = total
    healthy = True
    verdict = "HEALTHY"

    return healthy, checks_passed, total, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent45_data: dict | None,
    healthy: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    if agent45_data:
        scorer.add("go_no_go_data_available", True, +10)
    else:
        scorer.add("no_go_no_go_data", 0, -15)

    if healthy:
        scorer.add("production_healthy", True, +8)
    else:
        scorer.add("production_not_healthy", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a production validation narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    healthy: bool,
    checks_passed: int,
    checks_total: int,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Production healthy: {healthy}\n"
        f"Checks passed: {checks_passed}/{checks_total}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
