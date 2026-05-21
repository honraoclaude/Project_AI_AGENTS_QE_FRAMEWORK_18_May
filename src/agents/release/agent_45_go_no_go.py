"""
Agent 45 — Go/No-Go Coordinator
Phase       : Release
PACT        : Targeted
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs sequentially after Gate G10.
Has access to Agents 39, 41, 43, 44, 36.

Purpose:
  Final pre-production gate decision. Aggregates verdicts from all release
  preparation stages to issue a GO or NO_GO decision. Any hard failure in
  the release pipeline → NO_GO. All clear → GO.

  The Go/No-Go decision is the last deterministic check before production
  deployment. Gate G11 enforces it.

Output data keys consumed by downstream:
  go_decision      → bool   (Gate G11 — must be True for prod deployment)
  no_go_reasons    → list   (reasons for NO_GO if any)
  coordinator_verdict → str (GO / NO_GO / CONDITIONAL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 45
AGENT_NAME = "Go/No-Go Coordinator"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_go_no_go_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "coordinator_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the Go/No-Go decision. "
                "State what passed, any remaining blockers, "
                "and whether production deployment is approved."
            ),
        },
        "coordinator_concern": {
            "type": "string",
            "enum": ["none", "integrity_issue", "smoke_failure",
                     "fca_evidence_incomplete", "uat_pending"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for the final Go/No-Go release decision
in a Salesforce FSC Wealth Management CI/CD pipeline under FCA regulation.
You will receive verdicts from readiness assessment, change set integrity,
smoke tests, and FCA evidence compilation.
Write a clear 2–3 sentence narrative confirming the release decision, stating
what passed and what (if anything) is blocking production deployment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent36_data = _get_agent_data(state, "36")
    agent39_data = _get_agent_data(state, "39")
    agent41_data = _get_agent_data(state, "41")
    agent43_data = _get_agent_data(state, "43")
    agent44_data = _get_agent_data(state, "44")

    go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(
        agent36_data, agent39_data, agent41_data, agent43_data, agent44_data,
    )

    trace_msg = _build_trace_message(story_id, go, reasons, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent39_data, agent41_data, agent43_data, agent44_data, go,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Go/No-Go for {story_id}: go={go}, {len(reasons)} blocker(s) — "
        f"verdict={verdict}"
    )
    why = trace.get("narrative", "Go/No-Go Coordinator assessed all release gates.")

    data = {
        "go_decision": go,
        "no_go_reasons": reasons,
        "coordinator_verdict": verdict,
        "coordinator_concern": trace.get("coordinator_concern", "none"),
        "narrative": trace.get("narrative", ""),
        "minimax_loss_analysis": minimax_loss,
        "coalition_verdict": coalition_verdict,
        "coalition_dissent": coalition_dissent,
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


# ── Deterministic go/no-go logic ──────────────────────────────────────────────

_GATE_LOSS_MAP = {
    "readiness_blocked":    {"loss_type": "FCA_AUDIT_FAILURE",   "severity": "CRITICAL"},
    "integrity_failed":     {"loss_type": "METADATA_CORRUPTION", "severity": "HIGH"},
    "smoke_failed":         {"loss_type": "PRODUCTION_INCIDENT", "severity": "CRITICAL"},
    "fca_evidence_missing": {"loss_type": "REGULATORY_BREACH",   "severity": "CRITICAL"},
    "uat_pending":          {"loss_type": "STAKEHOLDER_SIGN_OFF","severity": "MEDIUM"},
}


def _make_decision(
    agent36_data: dict | None,
    agent39_data: dict | None,
    agent41_data: dict | None,
    agent43_data: dict | None,
    agent44_data: dict | None,
) -> tuple[bool, list[str], str, list[dict], str, list[str]]:
    """Returns (go_decision, no_go_reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent)."""
    reasons: list[str] = []
    loss_keys: list[str] = []

    readiness_verdict = (agent39_data or {}).get("readiness_verdict", "READY")
    if readiness_verdict == "BLOCKED":
        blockers = (agent39_data or {}).get("readiness_blockers", [])
        reasons.append(f"Release readiness BLOCKED: {blockers}")
        loss_keys.append("readiness_blocked")

    integrity_verdict = (agent41_data or {}).get("integrity_verdict", "PASS")
    if integrity_verdict == "FAIL":
        issues = (agent41_data or {}).get("integrity_issues", [])
        reasons.append(f"Change set integrity FAILED: {issues}")
        loss_keys.append("integrity_failed")

    smoke_verdict = (agent43_data or {}).get("smoke_verdict", "PASS")
    if smoke_verdict == "FAIL":
        failed = (agent43_data or {}).get("smoke_failed", 0)
        reasons.append(f"Smoke tests FAILED: {failed} failure(s)")
        loss_keys.append("smoke_failed")

    evidence_verdict = (agent44_data or {}).get("evidence_verdict", "COMPLETE")
    if evidence_verdict == "MISSING":
        gaps = (agent44_data or {}).get("evidence_gaps", [])
        reasons.append(f"FCA evidence MISSING: {gaps}")
        loss_keys.append("fca_evidence_missing")

    uat_coord = (agent36_data or {}).get("uat_coordination_verdict", "NOT_REQUIRED")
    if uat_coord == "BLOCKED":
        reasons.append("UAT coordination BLOCKED — CO sign-off cannot proceed")
        loss_keys.append("uat_pending")

    go = len(reasons) == 0

    # CONDITIONAL: technically clear but CO sign-off still awaited (PENDING)
    if go and uat_coord == "PENDING":
        verdict = "CONDITIONAL"
    elif go:
        verdict = "GO"
    else:
        verdict = "NO_GO"

    # Minimax loss analysis: map each blocking reason to its consequence
    minimax_loss = [
        {"gate": key, **_GATE_LOSS_MAP.get(key, {"loss_type": "UNKNOWN", "severity": "HIGH"})}
        for key in loss_keys
    ]

    # Coalition: all 5 sources must agree for unanimous GO
    coalition_inputs = {
        "readiness":  readiness_verdict,
        "integrity":  integrity_verdict,
        "smoke":      smoke_verdict,
        "evidence":   evidence_verdict,
        "uat":        uat_coord,
    }
    coalition_passes = {
        k: v in ("READY", "PASS", "COMPLETE", "GO", "NOT_REQUIRED", "SIGNED_OFF", "PENDING")
        for k, v in coalition_inputs.items()
    }
    coalition_unanimous = all(coalition_passes.values())
    coalition_verdict = "UNANIMOUS_GO" if coalition_unanimous else "DISSENT_NO_GO"
    coalition_dissent = [k for k, v in coalition_passes.items() if not v]

    return go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent39_data: dict | None,
    agent41_data: dict | None,
    agent43_data: dict | None,
    agent44_data: dict | None,
    go: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    sources = sum(1 for d in [agent39_data, agent41_data, agent43_data, agent44_data] if d)
    if sources >= 3:
        scorer.add("comprehensive_gate_data", sources, +10)
    elif sources >= 1:
        scorer.add("partial_gate_data", sources, +4)
    else:
        scorer.add("no_gate_data", 0, -12)

    if go:
        scorer.add("go_decision", True, +5)
    else:
        scorer.add("no_go_decision", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a Go/No-Go decision narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    go: bool,
    reasons: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Go decision: {go}\n"
        f"Verdict: {verdict}\n"
        f"No-go reasons: {reasons or ['none']}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
