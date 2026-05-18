"""
Agent 16 — Bulk/Async Quality
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Development Batch 2 (parallel with Agents 12, 14, 15).
Has access to Agents 3 (FCA class), 8 (refinement depth), 13 (metadata scope).

Purpose:
  Deterministically assesses the risk of Salesforce Apex governor limit violations
  — bulkification gaps, SOQL-in-loop risk, CPU limit exposure — based on the
  FSC object dependency depth and the scope of changed metadata.

  Risk heuristics (order = priority):
    HIGH-FCA + dependency_depth ≥ 2  → HIGH bulk risk
    dependency_depth ≥ 3             → HIGH bulk risk (deep chains = large data volumes)
    ≥ 2 detected FSC objects         → MEDIUM
    1 detected object + depth ≥ 1   → MEDIUM
    Otherwise                        → LOW

  Haiku generates the narrative — risk classification is pure Python.

Output data keys consumed by downstream:
  bulk_risk_level      → str  (Gate G3 — HIGH risk triggers mandatory bulkification review)
  bulk_risk_factors    → list (Agent 19 BDD Gherkin — generates bulk test scenarios if HIGH)
  async_recommended    → bool (Agent 23 audit trail — async pattern documented)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 16
AGENT_NAME = "Bulk/Async Quality"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_bulk_risk_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "bulk_risk_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining the governor limit and bulkification risk "
                "for this code change. Note the dependency depth, FSC objects involved, "
                "and whether async patterns or governor limit guards are recommended."
            ),
        },
        "bulk_risk_concern": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": (
                "none: 0 FSC objects, low FCA, no dependencies. "
                "low: 1 object, depth ≤ 1, LOW FCA. "
                "medium: 2 objects or depth 2 or MEDIUM FCA. "
                "high: Depth ≥ 3, or HIGH FCA with depth ≥ 2."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Bulk/Async Quality assessment.
You will receive the FSC dependency depth, detected FSC objects, and FCA
classification for a code change. Write a clear 2–3 sentence narrative explaining
the governor limit risk and whether the developer needs to implement bulkification
patterns, async processing (Queueable/Batch), or CPU-limit guards.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent3_data = _get_agent_data(state, "3")
    agent8_data = _get_agent_data(state, "8")
    agent13_data = _get_agent_data(state, "13")

    # ── Deterministic analysis ────────────────────────────────────────────────
    risk_level, factors, async_recommended = _analyse_bulk_risk(
        agent3_data, agent8_data, agent13_data,
    )

    # ── Haiku trace generation ────────────────────────────────────────────────
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    detected = (agent13_data or {}).get("detected_objects", [])
    depth = (agent13_data or {}).get("dependency_depth",
             (agent8_data or {}).get("dependency_depth", 0))

    trace_message = _build_trace_message(
        story_id, fca_class, detected, depth, risk_level, factors,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(agent3_data, agent13_data, depth)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Bulk risk for {story_id}: risk_level={risk_level}, "
        f"depth={depth}, async_recommended={async_recommended}"
    )
    why = trace.get(
        "narrative",
        "Bulk/Async Quality assessed governor limit risk based on FSC dependency depth.",
    )

    data = {
        "bulk_risk_level": risk_level,
        "bulk_risk_factors": factors,
        "async_recommended": async_recommended,
        "dependency_depth_assessed": depth,
        "detected_objects_count": len(detected),
        "fca_classification": fca_class,
        "bulk_risk_concern": trace.get("bulk_risk_concern", "none"),
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


# ── Deterministic bulk risk analysis ─────────────────────────────────────────

def _analyse_bulk_risk(
    agent3_data: dict | None,
    agent8_data: dict | None,
    agent13_data: dict | None,
) -> tuple[str, list[str], bool]:
    """
    Classify bulk/governor-limit risk from dependency depth and FSC object count.
    Returns (risk_level, factors, async_recommended).
    Pure Python — no LLM involved.
    """
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    detected = (agent13_data or {}).get("detected_objects", [])
    # Prefer development-time depth (Agent 13); fall back to refinement prediction (Agent 8)
    depth = (agent13_data or {}).get("dependency_depth",
             (agent8_data or {}).get("dependency_depth", 0))

    factors: list[str] = []

    if depth >= 3:
        factors.append(
            f"Dependency depth {depth} — DML operations may cascade across "
            "FinancialAccount, Individual, and Household; bulkify all trigger logic"
        )
    if fca_class == "HIGH" and depth >= 2:
        factors.append(
            "HIGH-FCA story with depth ≥ 2 — Suitability/RiskProfile inserts at scale "
            "trigger CPU and DML governor limits; consider Queueable Apex"
        )
    if len(detected) >= 3:
        factors.append(
            f"{len(detected)} FSC objects in scope — multi-object DML requires "
            "collection-based processing (no per-record triggers inside loops)"
        )
    if depth >= 2 and not factors:
        factors.append(
            "Dependency depth indicates cross-object operations — verify no SOQL in trigger loops"
        )

    # Risk level
    if (fca_class == "HIGH" and depth >= 2) or depth >= 3:
        risk_level = "HIGH"
    elif len(detected) >= 2 or depth >= 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    async_recommended = risk_level == "HIGH" or depth >= 3

    if not factors:
        factors.append("No significant governor limit risk identified for this change scope.")

    return risk_level, factors, async_recommended


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent13_data: dict | None,
    depth: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    # Signal 1: metadata scope available (Agent 13)
    if agent13_data:
        scorer.add("metadata_scope_available", True, +8)
    else:
        scorer.add("metadata_scope_unavailable", True, -10)

    # Signal 2: FCA class known
    if agent3_data:
        scorer.add("fca_class_available", True, +5)

    # Signal 3: dependency depth known
    if depth > 0:
        scorer.add("dependency_depth_known", depth, +5)
    elif agent13_data:
        scorer.add("zero_depth_confirmed", 0, +3)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a bulk/governor-limit risk narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    fca_class: str,
    detected: list[str],
    depth: int,
    risk_level: str,
    factors: list[str],
) -> str:
    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n"
        f"Detected FSC objects: {detected or ['none']}\n"
        f"Dependency depth: {depth}\n"
        f"Bulk risk level: {risk_level}\n"
        f"Risk factors: {factors}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
