"""
Agent 15 — Apex Security Scanner
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Development Batch 2 (parallel with Agents 12, 14, 16).
Has access to Agents 3 (FCA class), 8 (refinement deps), 13 (metadata scope).

Purpose:
  Deterministically assesses the security risk posture of the code change
  based on FSC object types, FCA classification, and metadata types changed.

  Does NOT run live code scanning (that is Agent 14 via PMD).
  Instead, applies rule-based heuristics:
    • Objects in _HIGH_RISK_OBJECTS → CRUD/FLS review required
    • Apex Trigger metadata type detected → sharing model review required
    • HIGH/MEDIUM FCA + high-risk objects → security_verdict = REVIEW_REQUIRED

  Haiku generates the narrative — analysis is pure Python.

Output data keys consumed by downstream:
  security_verdict          → str  (Gate G3 — BLOCK on unreviewed HIGH-FCA)
  crud_fls_review_required  → bool (Agent 23 audit trail — must appear in FCA evidence)
  sharing_model_review_required → bool (Agent 23 audit trail)
  security_flags            → list (CO email — specific concerns)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 15
AGENT_NAME = "Apex Security Scanner"

# FSC objects where CRUD/FLS enforcement is non-negotiable under FCA rules.
# FinancialAccount and FinancialHolding are standard FSC AUM objects regulated under COBS 9A/COBS 4.
# Standard FSC objects have no __c suffix — Agent 13 returns them as bare lowercase names.
_HIGH_RISK_OBJECTS = frozenset({
    "suitability__c",
    "riskprofile__c",
    "appropriateness__c",
    "vulnerablecustomerindicator__c",
    "financialaccount",
    "financialholding",
})

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_security_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "security_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining the security risk posture of this code change. "
                "Note which FSC objects require CRUD/FLS enforcement, whether the sharing "
                "model needs review, and the recommended action for the developer."
            ),
        },
        "security_concern": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": (
                "none: No FSC-regulated objects, LOW FCA classification. "
                "low: Non-high-risk FSC objects or LOW FCA. "
                "medium: High-risk FSC objects present but no triggers, or MEDIUM FCA. "
                "high: HIGH-FCA + high-risk objects + Apex triggers."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for an Apex Security risk assessment.
You will receive a list of FSC objects present in the code change, the FCA
classification, and identified security flags. Write a clear 2–3 sentence
narrative explaining the security risk and what the developer must review before
the code can be promoted. Reference CRUD/FLS and sharing model requirements
specifically when relevant to FSC regulated objects.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent3_data = _get_agent_data(state, "3")
    agent13_data = _get_agent_data(state, "13")

    # ── Deterministic analysis ────────────────────────────────────────────────
    risk_level, flags, crud_required, sharing_required, verdict = (
        _analyse_security_risk(agent3_data, agent13_data)
    )

    # ── Haiku trace generation ────────────────────────────────────────────────
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    detected = (agent13_data or {}).get("detected_objects", [])
    trace_message = _build_trace_message(
        story_id, fca_class, detected, flags, risk_level, crud_required, sharing_required,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(agent3_data, agent13_data, detected)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Security assessment for {story_id}: risk_level={risk_level}, "
        f"verdict={verdict}, crud_fls_required={crud_required}"
    )
    why = trace.get(
        "narrative",
        "Apex Security Scanner assessed CRUD/FLS and sharing model risk based on FSC object types.",
    )

    data = {
        "security_risk_level": risk_level,
        "security_verdict": verdict,
        "security_flags": flags,
        "crud_fls_review_required": crud_required,
        "sharing_model_review_required": sharing_required,
        "fca_classification": fca_class,
        "detected_objects_assessed": detected,
        "security_concern": trace.get("security_concern", "none"),
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


# ── Deterministic security analysis ──────────────────────────────────────────

def _analyse_security_risk(
    agent3_data: dict | None,
    agent13_data: dict | None,
) -> tuple[str, list[str], bool, bool, str]:
    """
    Derive security risk level from FSC object types and FCA classification.
    Returns (risk_level, flags, crud_fls_required, sharing_required, verdict).
    Pure Python — no LLM involved.
    """
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    detected = set((agent13_data or {}).get("detected_objects", []))
    depth = (agent13_data or {}).get("dependency_depth", 0)

    high_risk_present = bool(detected & _HIGH_RISK_OBJECTS)
    crud_required = high_risk_present
    # Sharing model review only needed when BOTH FCA tier is elevated AND high-risk FSC objects
    # are in scope — pure config/permission changes don't warrant it.
    sharing_required = fca_class in ("HIGH", "MEDIUM") and high_risk_present

    flags: list[str] = []

    if high_risk_present:
        objects_named = sorted(detected & _HIGH_RISK_OBJECTS)
        flags.append(
            f"CRUD/FLS enforcement required on: {', '.join(objects_named)}"
        )

    if sharing_required:
        flags.append(
            f"{fca_class}-FCA story with regulated FSC objects — sharing model must be reviewed for all changed Apex classes"
        )
    elif fca_class in ("HIGH", "MEDIUM") and not high_risk_present:
        flags.append(
            f"{fca_class}-FCA story — no regulated FSC objects detected; sharing model review not required for this change"
        )

    if depth >= 3:
        flags.append(
            "Deep dependency chain (depth ≥ 3) — cross-object DML operations require bulkification and sharing review"
        )

    # Risk level: HIGH/MEDIUM FCA + no FSC objects → LOW (no regulated object risk)
    if fca_class == "HIGH" and high_risk_present:
        risk_level = "HIGH"
    elif fca_class == "MEDIUM" and high_risk_present:
        risk_level = "MEDIUM"
    elif high_risk_present:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    verdict = "PASS" if risk_level == "LOW" else "REVIEW_REQUIRED"

    return risk_level, flags, crud_required, sharing_required, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent13_data: dict | None,
    detected: list[str],
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    # Signal 1: FCA classification known → risk assessment is grounded
    if agent3_data:
        scorer.add("fca_class_available", True, +8)
    else:
        scorer.add("fca_class_unavailable", True, -8)

    # Signal 2: metadata scope available → know what objects are in play
    if agent13_data:
        scorer.add("metadata_scope_available", True, +8)
    else:
        scorer.add("metadata_scope_unavailable", True, -10)

    # Signal 3: FSC objects detected → can make specific security assessment
    if detected:
        scorer.add("fsc_objects_in_scope", len(detected), +5)
    else:
        scorer.add("no_fsc_objects_detected", 0, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an Apex security risk narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    fca_class: str,
    detected: list[str],
    flags: list[str],
    risk_level: str,
    crud_required: bool,
    sharing_required: bool,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n"
        f"Detected FSC objects: {detected or ['none']}\n"
        f"Security risk level: {risk_level}\n"
        f"CRUD/FLS review required: {crud_required}\n"
        f"Sharing model review required: {sharing_required}\n"
        f"Security flags: {flags or ['none']}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
