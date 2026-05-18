"""
Agent 36 — UAT Coordination Agent
Phase       : Testing
PACT        : Collaborative
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs sequentially after Agent 35, before Gate G6.
Has access to Agents 3, 29, 33, 34, 35.

Purpose:
  Coordinates the UAT sign-off workflow. Determines whether Compliance Officer
  (CO) approval is required based on FCA classification and UAT test results.
  Issues the async sign-off request (stub: in production sends HMAC-signed
  email link per DD-001). Tracks whether sign-off has been received.

  Deterministic gate logic; Haiku writes the coordination narrative.

Output data keys consumed by downstream:
  uat_sign_off_required   → bool   (Gate G6 — CO must sign off HIGH/MEDIUM FCA)
  uat_sign_off_received   → bool   (stub: False in production until email link clicked)
  uat_coordination_verdict → str   (SIGNED_OFF / PENDING / NOT_REQUIRED / BLOCKED)
  sign_off_request_sent   → bool
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 36
AGENT_NAME = "UAT Coordination Agent"

_FCA_SIGN_OFF_CLASSES = {"HIGH", "MEDIUM"}

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_uat_coordination_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "coordination_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising UAT coordination status. "
                "State whether CO sign-off is required, whether a request was sent, "
                "and what must happen before the story can proceed to release."
            ),
        },
        "coordination_concern": {
            "type": "string",
            "enum": ["none", "sign_off_pending", "uat_failures_block_sign_off",
                     "coverage_below_threshold", "no_uat_tests"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a UAT coordination workflow
in a Salesforce FSC Wealth Management platform under FCA regulation.
You will receive UAT test results, FCA classification, defect status, and
sign-off workflow state.
Write a clear 2–3 sentence narrative explaining the UAT sign-off status,
who needs to act, and what must be resolved before release.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent29_data = _get_agent_data(state, "29")
    agent33_data = _get_agent_data(state, "33")
    agent34_data = _get_agent_data(state, "34")
    agent35_data = _get_agent_data(state, "35")

    required, received, sent, verdict = _coordinate_uat(
        agent3_data, agent29_data, agent33_data, agent34_data, agent35_data,
    )

    trace_msg = _build_trace_message(
        story_id, required, received, sent, verdict,
        agent3_data, agent29_data, agent33_data,
    )
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent3_data, agent29_data, agent33_data, agent34_data, verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"UAT coordination for {story_id}: sign_off_required={required}, "
        f"received={received} — verdict={verdict}"
    )
    why = trace.get("narrative", "UAT Coordination Agent assessed sign-off workflow.")

    data = {
        "uat_sign_off_required": required,
        "uat_sign_off_received": received,
        "uat_coordination_verdict": verdict,
        "sign_off_request_sent": sent,
        "coordination_concern": trace.get("coordination_concern", "none"),
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


# ── Deterministic UAT coordination logic ─────────────────────────────────────

def _coordinate_uat(
    agent3_data: dict | None,
    agent29_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    agent35_data: dict | None,
) -> tuple[bool, bool, bool, str]:
    """Returns (sign_off_required, sign_off_received, request_sent, verdict)."""
    fca_class  = (agent3_data or {}).get("fca_classification", "LOW")
    uat_verdict = (agent29_data or {}).get("uat_verdict", "PASS")
    co_required = (agent29_data or {}).get("co_sign_off_required", False)
    coverage_verdict = (agent33_data or {}).get("coverage_verdict", "PASS")
    defect_verdict   = (agent34_data or {}).get("defect_verdict", "PASS")

    # CO sign-off required for HIGH/MEDIUM FCA or when agent 29 explicitly requires it
    sign_off_required = fca_class in _FCA_SIGN_OFF_CLASSES or co_required

    # Stub: sign-off is never pre-received in CI — real sign-off arrives via email link (DD-001)
    sign_off_received = False

    # Cannot send sign-off request if defects are blocking or coverage has failed
    blockers_exist = defect_verdict == "FAIL" or coverage_verdict == "FAIL"

    if not sign_off_required:
        verdict = "NOT_REQUIRED"
        request_sent = False
    elif blockers_exist:
        verdict = "BLOCKED"
        request_sent = False
    elif sign_off_received:
        verdict = "SIGNED_OFF"
        request_sent = True
    else:
        # Stub: in production, sends HMAC-signed email link per DD-001
        verdict = "PENDING"
        request_sent = True

    return sign_off_required, sign_off_received, request_sent, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent29_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    if agent3_data:
        scorer.add("fca_classification_known", True, +8)
    else:
        scorer.add("fca_classification_unknown", 0, -8)

    if agent29_data:
        scorer.add("uat_test_results_available", True, +6)
    else:
        scorer.add("no_uat_results", 0, -5)

    if agent33_data:
        scorer.add("coverage_data_available", True, +4)

    if verdict == "BLOCKED":
        scorer.add("sign_off_blocked_by_defects", True, -8)
    elif verdict == "SIGNED_OFF":
        scorer.add("sign_off_complete", True, +8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a UAT coordination narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    required: bool,
    received: bool,
    sent: bool,
    verdict: str,
    agent3_data: dict | None,
    agent29_data: dict | None,
    agent33_data: dict | None,
) -> str:
    fca_class    = (agent3_data or {}).get("fca_classification", "UNKNOWN")
    uat_count    = (agent29_data or {}).get("uat_test_count", 0)
    uat_verdict  = (agent29_data or {}).get("uat_verdict", "PASS")
    coverage_pct = (agent33_data or {}).get("overall_coverage_pct", 0.0)

    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n"
        f"UAT Tests: {uat_count} tests, verdict={uat_verdict}\n"
        f"Coverage: {coverage_pct:.1f}%\n"
        f"CO Sign-Off Required: {required}\n"
        f"Sign-Off Received: {received}\n"
        f"Sign-Off Request Sent: {sent}\n"
        f"Coordination Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
