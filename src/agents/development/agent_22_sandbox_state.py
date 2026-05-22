"""
Agent 22 — Sandbox State Agent
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=58)

Runs after Batch 4 (sequential).
Has access to Agents 11, 13, 17.

Purpose:
  Assesses whether the target sandbox is healthy enough for deployment.
  Checks SFDX format compliance (Agent 17), branch traceability (Agent 11),
  and metadata scope readiness (Agent 13). In production, this would call
  a Copado sandbox health API; in the current stub it reasons from upstream
  agent signals.

  Haiku generates the sandbox state narrative; all scoring is deterministic.

Output data keys consumed by downstream:
  sandbox_ready          → bool (Gate G2 — deploy gate)
  sandbox_health_score   → int  (0–100)
  sandbox_blockers       → list (issues preventing deployment)
  sandbox_verdict        → str  (READY / DEGRADED / BLOCKED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 22
AGENT_NAME = "Sandbox State Agent"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_sandbox_state_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "deployment_risk"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the sandbox state assessment. "
                "Note any blockers, SFDX format issues, and whether the sandbox "
                "is ready for deployment. Be specific and actionable."
            ),
        },
        "deployment_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low: Sandbox is healthy and ready for deployment. "
                "medium: Minor issues present but not blocking. "
                "high: Critical blockers prevent deployment."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce sandbox state assessment.
You will receive signals from branch tracing, SFDX format validation, and metadata
dependency mapping. Write a clear 2–3 sentence narrative describing the sandbox
readiness for deployment, any blockers, and what the developer must resolve before
proceeding. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent11_data = _get_agent_data(state, "11")
    agent13_data = _get_agent_data(state, "13")
    agent17_data = _get_agent_data(state, "17")

    # ── Deterministic analysis ────────────────────────────────────────────────
    health_score, blockers, ready, verdict = _assess_sandbox_state(
        agent11_data, agent13_data, agent17_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, agent11_data, agent13_data, agent17_data,
        health_score, blockers, verdict,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        agent11_data, agent13_data, agent17_data, ready,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Sandbox state for {story_id}: health={health_score}/100, "
        f"{len(blockers)} blocker(s) — verdict={verdict}"
    )
    why = trace.get(
        "narrative",
        "Sandbox State Agent assessed deployment readiness from upstream agent signals.",
    )

    data = {
        "sandbox_ready": ready,
        "sandbox_health_score": health_score,
        "sandbox_blockers": blockers,
        "sandbox_verdict": verdict,
        "deployment_risk": trace.get("deployment_risk", "medium"),
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


# ── Deterministic sandbox state assessment ───────────────────────────────────

def _assess_sandbox_state(
    agent11_data: dict | None,
    agent13_data: dict | None,
    agent17_data: dict | None,
) -> tuple[int, list[str], bool, str]:
    """
    Returns (health_score, blockers, sandbox_ready, verdict).
    Health score 0–100; ≥70 = READY, 40–69 = DEGRADED, <40 = BLOCKED.
    """
    blockers: list[str] = []
    health_score = 100

    # Branch signal from Agent 11
    branch_found = (agent11_data or {}).get("branch_found", False)
    naming_valid = (agent11_data or {}).get("naming_convention_valid", False)
    branch_stale = (agent11_data or {}).get("branch_stale", False)

    if not branch_found:
        blockers.append("No story branch found — cannot deploy untraced changes")
        health_score -= 35
    elif not naming_valid:
        blockers.append("Branch naming convention violation — traceability risk")
        health_score -= 15
    if branch_stale:
        blockers.append("Branch is stale (>14 days) — merge conflicts likely")
        health_score -= 10

    # SFDX format signal from Agent 17
    sfdx_valid = (agent17_data or {}).get("sfdx_format_valid", True)
    sfdx_verdict = (agent17_data or {}).get("sfdx_verdict", "PASS")
    invalid_files = (agent17_data or {}).get("invalid_files", [])

    if not sfdx_valid:
        if sfdx_verdict == "FAIL":
            blockers.append(f"SFDX format FAIL: {len(invalid_files)} legacy-format file(s)")
            health_score -= 25
        else:
            blockers.append(f"SFDX format WARN: {len(invalid_files)} file(s) need migration")
            health_score -= 10

    # Metadata scope signal from Agent 13
    scope_delta = (agent13_data or {}).get("scope_delta_objects", [])
    depth = (agent13_data or {}).get("dependency_depth", 0)

    if scope_delta:
        blockers.append(
            f"Metadata scope expanded at code-time: {len(scope_delta)} unexpected object(s)"
        )
        health_score -= 10

    if depth >= 4:
        health_score -= 5

    health_score = max(0, min(100, health_score))

    if health_score >= 70:
        verdict = "READY"
        ready = True
    elif health_score >= 40:
        verdict = "DEGRADED"
        ready = False
    else:
        verdict = "BLOCKED"
        ready = False

    return health_score, blockers, ready, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent11_data: dict | None,
    agent13_data: dict | None,
    agent17_data: dict | None,
    ready: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    if agent11_data:
        scorer.add("branch_context_available", True, +10)
    else:
        scorer.add("no_branch_context", 0, -8)

    if agent13_data:
        scorer.add("metadata_scope_available", True, +8)

    if agent17_data:
        scorer.add("sfdx_format_signal_available", True, +7)

    if not ready:
        scorer.add("sandbox_not_ready", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a sandbox state narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent11_data: dict | None,
    agent13_data: dict | None,
    agent17_data: dict | None,
    health_score: int,
    blockers: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Branch found: {(agent11_data or {}).get('branch_found', 'unknown')}\n"
        f"Branch stale: {(agent11_data or {}).get('branch_stale', 'unknown')}\n"
        f"SFDX verdict: {(agent17_data or {}).get('sfdx_verdict', 'unknown')}\n"
        f"Metadata scope delta: {(agent13_data or {}).get('scope_delta_objects', [])}\n"
        f"Sandbox health score: {health_score}/100\n"
        f"Blockers: {blockers or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
