"""
Agent 20 — Performance Risk Estimator
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs in Batch 4 (parallel with Agent 21).
Has access to Agents 3, 8, 13, 16.

Purpose:
  Estimates performance risk from the metadata scope and dependency depth.
  Flags SOQL-in-loops risk, governor limit exposure, and large-data-volume
  objects in the changed set. Uses PMD-detected patterns (from Agent 14)
  where available to augment the estimate.

  Haiku generates the narrative; all scoring is deterministic Python.

Output data keys consumed by downstream:
  performance_risk_level   → str  (LOW / MEDIUM / HIGH)
  performance_risk_factors → list (informational)
  soql_loop_risk           → bool (Gate G3 — code quality gate)
  governor_limit_exposure  → str  (LOW / MEDIUM / HIGH)
  performance_verdict      → str  (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 20
AGENT_NAME = "Performance Risk Estimator"

# Objects with known large data volumes in FSC wealth management
_HIGH_VOLUME_OBJECTS = frozenset({
    "financialaccount", "financialholding", "revenueschedule",
    "financialtransaction", "financialgoal", "lead", "opportunity",
    "case", "task", "event",
})

# Objects that typically trigger cross-object SOQL chains
_CROSS_OBJECT_OBJECTS = frozenset({
    "suitability__c", "riskprofile__c", "appropriateness__c",
    "financialaccount", "financialholding",
})

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_performance_risk_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "performance_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the performance risk estimate. "
                "Mention specific objects causing concern, governor limit exposure, "
                "and what the developer must do to mitigate."
            ),
        },
        "performance_concern": {
            "type": "string",
            "enum": ["none", "governor_limits", "soql_patterns", "large_data_volume", "multiple"],
            "description": "Primary performance concern identified.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce performance risk estimate.
You will receive the changed objects, dependency depth, bulk risk level, and PMD findings.
Write a clear 2–3 sentence narrative identifying performance risks (governor limits,
SOQL patterns, large data volumes) and what the developer must do to address them.
Be specific and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data = _get_agent_data(state, "3")
    agent8_data = _get_agent_data(state, "8")
    agent13_data = _get_agent_data(state, "13")
    agent14_data = _get_agent_data(state, "14")
    agent16_data = _get_agent_data(state, "16")

    # ── Deterministic analysis ────────────────────────────────────────────────
    risk_level, factors, soql_loop_risk, gov_exposure, verdict = _estimate_performance_risk(
        agent3_data, agent8_data, agent13_data, agent14_data, agent16_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, agent13_data, agent16_data, agent14_data, risk_level, factors, verdict,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        agent3_data, agent13_data, agent16_data, risk_level,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Performance risk for {story_id}: risk={risk_level}, "
        f"soql_loop_risk={soql_loop_risk}, governor_exposure={gov_exposure} — verdict={verdict}"
    )
    why = trace.get(
        "narrative",
        "Performance Risk Estimator assessed the metadata scope for governor limit exposure.",
    )

    data = {
        "performance_risk_level": risk_level,
        "performance_risk_factors": factors,
        "soql_loop_risk": soql_loop_risk,
        "governor_limit_exposure": gov_exposure,
        "performance_verdict": verdict,
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


# ── Deterministic performance risk analysis ───────────────────────────────────

def _estimate_performance_risk(
    agent3_data: dict | None,
    agent8_data: dict | None,
    agent13_data: dict | None,
    agent14_data: dict | None,
    agent16_data: dict | None,
) -> tuple[str, list[str], bool, str, str]:
    """
    Returns (risk_level, factors, soql_loop_risk, governor_exposure, verdict).
    """
    factors: list[str] = []

    # Gather objects from Agent 13 (code-time) or Agent 8 (refinement-time)
    objects: list[str] = []
    depth = 0
    if agent13_data:
        objects = [o.lower() for o in agent13_data.get("detected_objects", [])]
        depth = agent13_data.get("dependency_depth", 0)
    elif agent8_data:
        objects = [o.lower() for o in agent8_data.get("detected_objects", [])]
        depth = agent8_data.get("dependency_depth", 0)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    bulk_risk = (agent16_data or {}).get("bulk_risk_level", "LOW")

    # PMD SOQL-in-loop detection
    soql_loop_risk = False
    if agent14_data:
        violations = agent14_data.get("critical_violations", [])
        soql_rules = {"ApexSOQLInjection", "OperationWithLimitsInLoop"}
        soql_loop_risk = any(
            v.get("rule_name") in soql_rules for v in violations
        )
        if soql_loop_risk:
            factors.append("PMD detected SOQL/DML patterns in loop context")

    # High-volume object exposure
    high_vol = [o for o in objects if o in _HIGH_VOLUME_OBJECTS]
    if high_vol:
        factors.append(f"High-volume objects in scope: {high_vol}")

    # Cross-object SOQL chain depth
    if depth >= 3:
        factors.append(f"Deep dependency chain (depth={depth}) increases SOQL query count")
    elif depth >= 2:
        factors.append(f"Moderate dependency chain (depth={depth})")

    # Bulk risk from Agent 16
    if bulk_risk == "HIGH":
        factors.append("Agent 16 assessed HIGH bulk/async risk")

    # Governor limit exposure scoring
    gov_score = 0
    if soql_loop_risk:
        gov_score += 3
    if len(high_vol) >= 2:
        gov_score += 2
    elif high_vol:
        gov_score += 1
    if depth >= 3:
        gov_score += 2
    elif depth >= 2:
        gov_score += 1
    if bulk_risk == "HIGH":
        gov_score += 2
    elif bulk_risk == "MEDIUM":
        gov_score += 1

    if gov_score >= 5:
        governor_exposure = "HIGH"
    elif gov_score >= 2:
        governor_exposure = "MEDIUM"
    else:
        governor_exposure = "LOW"

    # Overall risk level
    if soql_loop_risk or governor_exposure == "HIGH" or (fca_class == "HIGH" and depth >= 3):
        risk_level = "HIGH"
    elif governor_exposure == "MEDIUM" or depth >= 2 or bulk_risk == "HIGH":
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if not factors:
        factors.append("No significant performance risk indicators detected")

    # Verdict
    if risk_level == "HIGH" or soql_loop_risk:
        verdict = "FAIL"
    elif risk_level == "MEDIUM":
        verdict = "WARN"
    else:
        verdict = "PASS"

    return risk_level, factors, soql_loop_risk, governor_exposure, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent13_data: dict | None,
    agent16_data: dict | None,
    risk_level: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    if agent13_data:
        scorer.add("code_time_metadata_available", True, +10)
    else:
        scorer.add("no_code_time_metadata", 0, -8)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if agent16_data:
        scorer.add("bulk_risk_context_available", True, +5)

    if risk_level == "HIGH":
        scorer.add("high_risk_detected", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a performance risk narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent13_data: dict | None,
    agent16_data: dict | None,
    agent14_data: dict | None,
    risk_level: str,
    factors: list[str],
    verdict: str,
) -> str:
    objects = (agent13_data or {}).get("detected_objects", [])
    depth = (agent13_data or {}).get("dependency_depth", 0)
    bulk_risk = (agent16_data or {}).get("bulk_risk_level", "UNKNOWN")
    critical_violations = (agent14_data or {}).get("critical_violations", [])
    return (
        f"Story: {story_id}\n"
        f"Objects in scope: {objects or ['unknown']}\n"
        f"Dependency depth: {depth}\n"
        f"Bulk/async risk: {bulk_risk}\n"
        f"PMD critical violations: {len(critical_violations)}\n"
        f"Performance risk: {risk_level}\n"
        f"Risk factors:\n" + "\n".join(f"  - {f}" for f in factors) + "\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
