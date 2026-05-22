"""
Agent 32 — Regression Risk Assessor
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=63)

Runs in Testing Batch 1 (parallel with Agents 24, 25).
Has access to Agents 3, 8, 13, 18, 23.

Purpose:
  Assesses regression risk from the story's changed component set.
  Identifies components that are shared with other active stories,
  FSC core objects that have downstream consumers, and the depth
  of the dependency chain that could surface regressions.

  Haiku writes the narrative; scoring is deterministic Python.

Output data keys consumed by downstream:
  regression_risk_level  → str  (LOW / MEDIUM / HIGH)
  regression_risk_factors→ list (informational)
  shared_components      → list (components touched by other stories)
  recommended_regression_suite → str (SMOKE / REGRESSION / FULL)
  regression_verdict     → str  (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 32
AGENT_NAME = "Regression Risk Assessor"

# FSC objects with many downstream consumers — high regression blast radius
_HIGH_BLAST_OBJECTS = frozenset({
    "financialaccount", "financialholding", "financialgoal",
    "revenueschedule", "lead", "opportunity", "account", "contact",
})

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_regression_risk_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "regression_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the regression risk. "
                "Mention specific high-blast-radius objects, dependency depth, "
                "and which regression suite is recommended."
            ),
        },
        "regression_concern": {
            "type": "string",
            "enum": ["none", "shared_components", "high_blast_radius",
                     "deep_dependency_chain", "multiple"],
            "description": "Primary regression concern.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce regression risk assessment.
You will receive changed components, FSC objects in scope, dependency depth, and risk signals.
Write a clear 2–3 sentence narrative identifying the regression risk level, which objects
could cause regressions in other parts of the system, and which test suite to run.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent13_data = _get_agent_data(state, "13")
    agent18_data = _get_agent_data(state, "18")
    agent23_data = _get_agent_data(state, "23")

    # ── Deterministic analysis ────────────────────────────────────────────────
    risk_level, factors, shared, suite, verdict = _assess_regression_risk(
        agent3_data, agent13_data, agent18_data, agent23_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(
        story_id, agent13_data, risk_level, factors, shared, suite, verdict,
    )
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent3_data, agent13_data, agent18_data, risk_level,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Regression risk for {story_id}: risk={risk_level}, "
        f"suite={suite}, {len(shared)} shared component(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Regression Risk Assessor evaluated blast radius.")

    data = {
        "regression_risk_level": risk_level,
        "regression_risk_factors": factors,
        "shared_components": shared,
        "shared_components_stub": True,  # REQ-23: real cross-story data requires sprint query
        "recommended_regression_suite": suite,
        "regression_verdict": verdict,
        "regression_concern": trace.get("regression_concern", "none"),
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


# ── Deterministic regression risk assessment ──────────────────────────────────

def _assess_regression_risk(
    agent3_data: dict | None,
    agent13_data: dict | None,
    agent18_data: dict | None,
    agent23_data: dict | None,
) -> tuple[str, list[str], list[str], str, str]:
    """Returns (risk_level, factors, shared_components, suite, verdict)."""
    factors: list[str] = []

    objects = [o.lower() for o in (agent13_data or {}).get("detected_objects", [])]
    depth = (agent13_data or {}).get("dependency_depth", 0)
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    regulated = (agent18_data or {}).get("regulated_components", [])
    merge_risk = (agent18_data or {}).get("merge_risk_components", [])

    # Shared components — stub: use merge_risk as proxy (multiple editors = shared code)
    shared = list(merge_risk)

    # High blast-radius objects
    blast = [o for o in objects if o in _HIGH_BLAST_OBJECTS]
    if blast:
        factors.append(f"High blast-radius FSC objects: {blast}")

    # Dependency depth
    if depth >= 3:
        factors.append(f"Deep dependency chain (depth={depth}) — wide regression surface")
    elif depth >= 2:
        factors.append(f"Moderate dependency chain (depth={depth})")

    # FCA regulated components
    if regulated:
        factors.append(f"FCA-regulated components changed: {regulated}")

    # Shared components (merge risk proxy)
    if shared:
        factors.append(f"Components edited by multiple developers: {shared}")

    # REQ-23: Development phase verdict elevates regression risk
    dev_verdict = (agent23_data or {}).get("development_verdict", "PASS")
    if dev_verdict in ("PARTIAL", "FAIL"):
        factors.append(f"Development verdict was {dev_verdict} — elevated regression risk")

    # Risk scoring
    risk_score = 0
    if len(blast) >= 2:
        risk_score += 3
    elif blast:
        risk_score += 1
    if depth >= 3:
        risk_score += 2
    elif depth >= 2:
        risk_score += 1
    if regulated:
        risk_score += 2
    if shared:
        risk_score += 1
    if fca_class == "HIGH":
        risk_score += 1
    if dev_verdict in ("PARTIAL", "FAIL"):
        risk_score += 2

    if risk_score >= 5:
        risk_level = "HIGH"
        suite = "FULL"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
        suite = "REGRESSION"
    else:
        risk_level = "LOW"
        suite = "SMOKE"

    if not factors:
        factors.append("No significant regression risk indicators detected")

    verdict = "FAIL" if risk_level == "HIGH" else ("WARN" if risk_level == "MEDIUM" else "PASS")
    return risk_level, factors, shared, suite, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent13_data: dict | None,
    agent18_data: dict | None,
    risk_level: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=63)

    if agent13_data:
        scorer.add("metadata_scope_available", True, +10)
    else:
        scorer.add("no_metadata_scope", 0, -8)

    if agent18_data:
        scorer.add("component_attribution_available", True, +7)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if risk_level == "HIGH":
        scorer.add("high_risk_detected", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a regression risk narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent13_data: dict | None,
    risk_level: str,
    factors: list[str],
    shared: list[str],
    suite: str,
    verdict: str,
) -> str:
    objects = (agent13_data or {}).get("detected_objects", [])
    depth = (agent13_data or {}).get("dependency_depth", 0)
    return (
        f"Story: {story_id}\n"
        f"Objects in scope: {objects or ['unknown']}\n"
        f"Dependency depth: {depth}\n"
        f"Shared components: {shared or ['none']}\n"
        f"Regression risk: {risk_level}\n"
        f"Risk factors:\n" + "\n".join(f"  - {f}" for f in factors) + "\n"
        f"Recommended suite: {suite}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
