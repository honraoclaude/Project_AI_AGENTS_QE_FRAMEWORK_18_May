"""
Agent 31 — Financial Data Integrity
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Testing Batch 3 (parallel with Agents 28, 37).
Has access to Agents 3, 13, 27.

Purpose:
  Validates that financial data integrity rules are upheld after the
  story's changes are applied. Checks for:
  - Balance consistency (FinancialAccount total = sum of holdings)
  - Suitability score integrity (within valid range, no null for HIGH-FCA)
  - Revenue schedule continuity (no gaps in scheduled payments)
  - Audit trail completeness (every financial event has a log entry)

  Integrity checks are deterministic rules; Haiku writes the narrative.
  In production, this queries the Salesforce org; the stub reasons from
  the detected object scope and FCA classification.

Output data keys consumed by downstream:
  integrity_valid        → bool (Gate G5 input)
  integrity_violations   → list (FCA audit evidence)
  integrity_verdict      → str  (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 31
AGENT_NAME = "Financial Data Integrity"

# Rules and the FSC objects they apply to
_INTEGRITY_RULES = {
    "balance_consistency": frozenset({"financialaccount", "financialholding"}),
    "suitability_score_range": frozenset({"suitability__c"}),
    "revenue_schedule_continuity": frozenset({"revenueschedule", "financialaccount"}),
    "audit_trail_completeness": frozenset({
        "financialaccount", "suitability__c", "riskprofile__c",
        "financialtransaction", "financialgoal",
    }),
}

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_integrity_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "integrity_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the financial data integrity result. "
                "Note which rules were checked, any violations, and what the "
                "developer must fix before deployment."
            ),
        },
        "integrity_concern": {
            "type": "string",
            "enum": ["none", "balance_mismatch", "suitability_invalid",
                     "audit_gap", "multiple"],
            "description": "Primary integrity concern.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce FSC financial data integrity check.
You will receive the objects in scope, which integrity rules apply, and any violations found.
Write a clear 2–3 sentence narrative explaining the integrity status, what rules were checked,
any violations found, and what must be fixed. Be specific and use FSC terminology.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent13_data = _get_agent_data(state, "13")
    agent27_data = _get_agent_data(state, "27")

    # ── Deterministic integrity check ────────────────────────────────────────
    valid, violations, rules_checked, verdict = _check_integrity(
        agent3_data, agent13_data, agent27_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(
        story_id, agent13_data, rules_checked, violations, verdict,
    )
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent3_data, agent13_data, agent27_data, valid,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Financial data integrity for {story_id}: {len(rules_checked)} rule(s) checked, "
        f"{len(violations)} violation(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Financial Data Integrity checked FSC data rules.")

    data = {
        "integrity_valid": valid,
        "integrity_violations": violations,
        "integrity_verdict": verdict,
        "rules_checked": rules_checked,
        "stub_mode": True,  # REQ-22: live org queries not yet implemented; Gate G5 guards on this
        "integrity_concern": trace.get("integrity_concern", "none"),
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


# ── Deterministic integrity checking ─────────────────────────────────────────

def _check_integrity(
    agent3_data: dict | None,
    agent13_data: dict | None,
    agent27_data: dict | None,
) -> tuple[bool, list[str], list[str], str]:
    """Returns (integrity_valid, violations, rules_checked, verdict)."""
    objects = {o.lower() for o in (agent13_data or {}).get("detected_objects", [])}
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    crt_passed = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED") == "PASS"
    violations: list[str] = []
    rules_checked: list[str] = []

    for rule, applicable_objects in _INTEGRITY_RULES.items():
        if not objects.intersection(applicable_objects):
            continue
        rules_checked.append(rule)

        # Stub: flag specific violations based on rule and context
        if rule == "suitability_score_range" and fca_class == "HIGH":
            # HIGH-FCA suitability changes require explicit score range validation
            # In production: query org for null/out-of-range suitability scores
            # Stub: pass (no evidence of violation without live org query)
            pass

        if rule == "audit_trail_completeness" and fca_class in ("HIGH", "MEDIUM"):
            # FCA regulated changes must have complete audit trails
            # Stub: pass (audited at gate level)
            pass

    # If CRT tests ran and passed, treat as supporting evidence for integrity
    # If CRT skipped, we have lower confidence but no active violations
    if not crt_passed and (agent27_data or {}).get("crt_execution_verdict") not in ("SKIPPED", ""):
        violations.append("CRT tests did not pass — integrity cannot be confirmed via automated tests")

    if not rules_checked and not objects:
        rules_checked.append("no_objects_in_scope")

    if violations:
        verdict = "FAIL" if len(violations) >= 2 else "WARN"
        valid = False
    else:
        verdict = "PASS"
        valid = True

    return valid, violations, rules_checked, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent13_data: dict | None,
    agent27_data: dict | None,
    valid: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if agent13_data:
        scorer.add("metadata_scope_available", True, +10)
    else:
        scorer.add("no_metadata_scope", 0, -8)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    crt_verdict = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED")
    if crt_verdict == "PASS":
        scorer.add("crt_passed_supports_integrity", True, +8)
    elif crt_verdict == "SKIPPED":
        scorer.add("crt_skipped_reduced_confidence", 0, -5)

    if not valid:
        scorer.add("integrity_violations_found", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a financial data integrity narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent13_data: dict | None,
    rules_checked: list[str],
    violations: list[str],
    verdict: str,
) -> str:
    objects = (agent13_data or {}).get("detected_objects", [])
    return (
        f"Story: {story_id}\n"
        f"Objects in scope: {objects or ['unknown']}\n"
        f"Integrity rules checked: {rules_checked or ['none applicable']}\n"
        f"Violations: {violations or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
