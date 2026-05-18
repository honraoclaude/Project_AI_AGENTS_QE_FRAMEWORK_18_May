"""
Agent 37 — Performance Test Agent
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=58)

Runs in Testing Batch 3 (parallel with Agents 28, 31).
Has access to Agents 20, 27.

Purpose:
  Assesses performance test requirements and results based on Agent 20's
  risk estimate. If performance risk is HIGH, a performance test run is
  mandatory. In production this triggers JMeter/Salesforce CLI load tests;
  the current stub estimates requirements from the risk signal.

  Haiku writes the narrative; all scoring is deterministic.

Output data keys consumed by downstream:
  perf_test_required     → bool (Gate G5 — mandatory for HIGH-risk stories)
  perf_test_verdict      → str  (PASS / WARN / FAIL / SKIPPED)
  response_time_ok       → bool (informational)
  governor_limits_ok     → bool (informational)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 37
AGENT_NAME = "Performance Test Agent"

_RESPONSE_TIME_THRESHOLD_MS = 3000  # 3 seconds max for FSC pages

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_performance_test_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "performance_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the performance test outcome. "
                "Note whether tests were required, whether they ran, "
                "response times, and governor limit status."
            ),
        },
        "performance_concern": {
            "type": "string",
            "enum": ["none", "response_time_breach", "governor_limit_breach",
                     "test_skipped_for_high_risk", "multiple"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce FSC performance test assessment.
You will receive the performance risk level, whether tests were required/run, and results.
Write a clear 2–3 sentence narrative explaining whether performance tests ran, what was found,
and what the developer must do if performance issues were detected.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent20_data = _get_agent_data(state, "20")
    agent27_data = _get_agent_data(state, "27")

    # ── Deterministic assessment ──────────────────────────────────────────────
    required, resp_ok, gov_ok, verdict = _assess_performance(agent20_data, agent27_data)

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(story_id, agent20_data, required, resp_ok, gov_ok, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent20_data, agent27_data, verdict)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Performance test for {story_id}: required={required}, "
        f"response_ok={resp_ok}, governor_ok={gov_ok} — verdict={verdict}"
    )
    why = trace.get("narrative", "Performance Test Agent evaluated performance requirements.")

    data = {
        "perf_test_required": required,
        "perf_test_verdict": verdict,
        "response_time_ok": resp_ok,
        "governor_limits_ok": gov_ok,
        "performance_concern": trace.get("performance_concern", "none"),
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


# ── Deterministic performance assessment ──────────────────────────────────────

def _assess_performance(
    agent20_data: dict | None,
    agent27_data: dict | None,
) -> tuple[bool, bool, str, str]:
    """Returns (perf_test_required, response_time_ok, governor_limits_ok, verdict)."""
    perf_risk = (agent20_data or {}).get("performance_risk_level", "LOW")
    soql_loop = (agent20_data or {}).get("soql_loop_risk", False)
    gov_exposure = (agent20_data or {}).get("governor_limit_exposure", "LOW")

    required = perf_risk == "HIGH" or soql_loop

    # Stub: in production, run actual performance test and check response times
    # For now, derive from risk signals
    response_time_ok = perf_risk != "HIGH"
    governor_limits_ok = gov_exposure != "HIGH" and not soql_loop

    if not required:
        verdict = "SKIPPED"
    elif required and response_time_ok and governor_limits_ok:
        verdict = "PASS"
    elif required and (not response_time_ok or not governor_limits_ok):
        verdict = "FAIL"
    else:
        verdict = "WARN"

    return required, response_time_ok, governor_limits_ok, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent20_data: dict | None,
    agent27_data: dict | None,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    if agent20_data:
        scorer.add("performance_risk_signal_available", True, +12)
    else:
        scorer.add("no_performance_risk_signal", 0, -10)

    crt_verdict = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED")
    if crt_verdict == "PASS":
        scorer.add("crt_passed", True, +5)

    if verdict == "FAIL":
        scorer.add("performance_fail", True, -8)
    elif verdict == "SKIPPED":
        scorer.add("performance_test_not_required", 0, +3)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a performance test narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent20_data: dict | None,
    required: bool,
    resp_ok: bool,
    gov_ok: bool,
    verdict: str,
) -> str:
    risk = (agent20_data or {}).get("performance_risk_level", "UNKNOWN")
    soql = (agent20_data or {}).get("soql_loop_risk", False)
    gov_exp = (agent20_data or {}).get("governor_limit_exposure", "UNKNOWN")
    return (
        f"Story: {story_id}\n"
        f"Performance risk: {risk}\n"
        f"SOQL loop risk: {soql}\n"
        f"Governor limit exposure: {gov_exp}\n"
        f"Performance test required: {required}\n"
        f"Response time OK: {resp_ok}\n"
        f"Governor limits OK: {gov_ok}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
