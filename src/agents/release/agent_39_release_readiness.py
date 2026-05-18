"""
Agent 39 — Release Readiness Assessor
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=63)

Runs in Release Batch 1 (parallel with Agent 47).
Has access to Agents 23, 33, 34, 35, 36.

Purpose:
  Aggregates phase-level verdicts from Development and Testing phases to
  determine whether the story is ready for release. Checks that all upstream
  gates passed and no unresolved blockers remain.

  This is the primary pre-release health check — Gate G7 depends on it.

Output data keys consumed by downstream:
  release_ready       → bool   (Gate G7 — must be True to proceed)
  readiness_blockers  → list   (reasons story is not ready)
  readiness_verdict   → str    (READY / BLOCKED / PARTIAL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 39
AGENT_NAME = "Release Readiness Assessor"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_readiness_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "readiness_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising release readiness status. "
                "State which phases passed, any remaining blockers, "
                "and what must happen before the story can be released."
            ),
        },
        "readiness_concern": {
            "type": "string",
            "enum": ["none", "testing_incomplete", "unresolved_defects",
                     "uat_pending", "coverage_below_threshold"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a release readiness assessment
in a Salesforce FSC Wealth Management CI/CD pipeline under FCA regulation.
You will receive verdicts from the Development and Testing phases.
Write a clear 2–3 sentence narrative explaining whether the story is ready
for release, what passed, and what must be resolved before deployment can proceed.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent23_data = _get_agent_data(state, "23")
    agent33_data = _get_agent_data(state, "33")
    agent34_data = _get_agent_data(state, "34")
    agent35_data = _get_agent_data(state, "35")
    agent36_data = _get_agent_data(state, "36")

    ready, blockers, verdict = _assess_readiness(
        agent23_data, agent33_data, agent34_data, agent35_data, agent36_data,
    )

    trace_msg = _build_trace_message(story_id, ready, blockers, verdict, agent33_data, agent36_data)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent23_data, agent33_data, agent34_data, agent36_data, verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Release readiness for {story_id}: ready={ready}, "
        f"{len(blockers)} blocker(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Release Readiness Assessor evaluated all phase outputs.")

    data = {
        "release_ready": ready,
        "readiness_blockers": blockers,
        "readiness_verdict": verdict,
        "readiness_concern": trace.get("readiness_concern", "none"),
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


# ── Deterministic readiness assessment ───────────────────────────────────────

def _assess_readiness(
    agent23_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    agent35_data: dict | None,
    agent36_data: dict | None,
) -> tuple[bool, list[str], str]:
    """Returns (release_ready, blockers, verdict)."""
    blockers: list[str] = []

    dev_verdict = (agent23_data or {}).get("development_verdict", "PASS")
    if dev_verdict in ("FAIL", "PARTIAL"):
        blockers.append(f"Development phase verdict: {dev_verdict}")

    cov_verdict = (agent33_data or {}).get("coverage_verdict", "PASS")
    if cov_verdict == "FAIL":
        pct = (agent33_data or {}).get("overall_coverage_pct", 0.0)
        blockers.append(f"Test coverage {pct:.0f}% below FCA threshold")

    def_verdict = (agent34_data or {}).get("defect_verdict", "PASS")
    if def_verdict == "FAIL":
        critical = (agent34_data or {}).get("critical_defects", [])
        blockers.append(f"Unresolved critical defects: {critical}")

    rca_verdict = (agent35_data or {}).get("rca_verdict", "NO_ACTION_REQUIRED")
    if rca_verdict == "INCOMPLETE":
        blockers.append("Root cause analysis incomplete — unresolved defects lack fix plan")

    uat_verdict = (agent36_data or {}).get("uat_coordination_verdict", "NOT_REQUIRED")
    if uat_verdict == "BLOCKED":
        blockers.append("UAT sign-off blocked — defects must be resolved first")

    ready = len(blockers) == 0

    if not ready:
        verdict = "BLOCKED"
    elif uat_verdict == "PENDING":
        verdict = "PARTIAL"  # technically ready but awaiting async CO approval
    else:
        verdict = "READY"

    return ready, blockers, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent23_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    agent36_data: dict | None,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=63)

    sources = sum(1 for d in [agent23_data, agent33_data, agent34_data, agent36_data] if d)
    if sources >= 3:
        scorer.add("comprehensive_phase_data", sources, +10)
    elif sources >= 1:
        scorer.add("partial_phase_data", sources, +4)
    else:
        scorer.add("no_phase_data", 0, -10)

    if verdict == "READY":
        scorer.add("all_phases_clear", True, +8)
    elif verdict == "BLOCKED":
        scorer.add("release_blocked", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a release readiness narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    ready: bool,
    blockers: list[str],
    verdict: str,
    agent33_data: dict | None,
    agent36_data: dict | None,
) -> str:
    cov_pct  = (agent33_data or {}).get("overall_coverage_pct", 0.0)
    uat_coord = (agent36_data or {}).get("uat_coordination_verdict", "NOT_REQUIRED")
    return (
        f"Story: {story_id}\n"
        f"Release ready: {ready}\n"
        f"Verdict: {verdict}\n"
        f"Coverage: {cov_pct:.1f}%\n"
        f"UAT coordination: {uat_coord}\n"
        f"Blockers: {blockers or ['none']}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
