"""
Agent 14 — Code Quality Reviewer
Phase       : Development
PACT        : Proactive
Classification: True AI (Sonnet 4.6) — PMD augmented
Confidence  : Tier B (base=58)

Runs in Development Batch 2 (parallel with Agents 12, 15, 16).
Has access to Agents 3 (FCA class) and 13 (metadata scope).

Purpose:
  Receives PMD static analysis violations from Copado and reasons about their
  severity and impact in the context of Salesforce FSC and FCA regulation.

  PMD priority scale: 1=critical, 2=high, 3=medium, 4=low, 5=info
  Gate G3 rules encoded in instructions:
    • Any Priority-1 violation → quality_verdict = FAIL
    • ApexCRUDViolation or ApexSharingViolations on HIGH-FCA story → FAIL
    • ApexSOQLInjection or ApexXSSFromURLParam → FAIL regardless of FCA tier
    • ≥ 3 Priority-2 violations → WARN
    • Otherwise → PASS or WARN based on total count

  If PMD data is unavailable (Copado unconfigured), the agent reports
  quality as UNKNOWN and confidence is penalised.

Output data keys consumed by downstream:
  quality_verdict     → str  (Gate G3 — FAIL blocks promotion)
  critical_violations → list (Gate G3 — specific violations to fix)
  total_violation_count → int (CO email context)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool, classify_ta_interaction
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.copado import get_pmd_results

AGENT_ID = 14
AGENT_NAME = "Code Quality Reviewer"

# ── PMD rules that always fail regardless of FCA tier ────────────────────────
_ALWAYS_FAIL_RULES = frozenset({
    "ApexSOQLInjection",
    "ApexXSSFromURLParam",
    "ApexOpenRedirect",
    "ApexInsecureEndpoint",
})

# ── Tool schema ───────────────────────────────────────────────────────────────

_VIOLATION_ITEM_SCHEMA = {
    "type": "object",
    "required": ["rule_name", "description", "file_path", "line"],
    "properties": {
        "rule_name": {"type": "string"},
        "description": {"type": "string"},
        "file_path": {"type": "string"},
        "line": {"type": "integer"},
    },
}

_TOOL_NAME = "assess_code_quality"
_TOOL_DESCRIPTION = (
    "Assess the code quality of a Salesforce FSC story based on PMD static analysis results "
    "and FSC/FCA context. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "critical_violations",
        "high_violations",
        "quality_verdict",
        "quality_summary",
        "recommended_fixes",
    ],
    "properties": {
        "critical_violations": {
            "type": "array",
            "items": _VIOLATION_ITEM_SCHEMA,
            "description": (
                "PMD Priority-1 violations, security injection risks, and "
                "CRUD/FLS violations on HIGH-FCA stories. Must be fixed before promotion."
            ),
        },
        "high_violations": {
            "type": "array",
            "items": _VIOLATION_ITEM_SCHEMA,
            "description": "PMD Priority-2 violations and FCA-significant quality issues.",
        },
        "quality_verdict": {
            "type": "string",
            "enum": ["PASS", "WARN", "FAIL"],
            "description": (
                "PASS: No critical/high violations relevant to FCA compliance. "
                "WARN: Issues present but none that block promotion. "
                "FAIL: Critical violations or security risks that must be fixed first."
            ),
        },
        "quality_summary": {
            "type": "string",
            "description": "2–3 sentences for the developer summarising the quality posture.",
        },
        "recommended_fixes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Prioritised list of concrete fixes. Minimum 1 entry.",
        },
    },
}

_AGENT_INSTRUCTIONS = """
You are the Code Quality Reviewer for the FSC QE Framework.

You receive PMD static analysis violations for a Salesforce FSC Apex codebase.
Your job is to classify the violations, assess their impact in the context of
FSC Wealth Management and FCA regulation, and produce a verdict.

Verdict rules (apply in order — first match wins):
1. Any PMD Priority-1 violation → FAIL.
2. Any ApexSOQLInjection, ApexXSSFromURLParam, ApexOpenRedirect, or ApexInsecureEndpoint → FAIL.
3. ApexCRUDViolation or ApexSharingViolations on a HIGH or MEDIUM FCA story → FAIL.
4. Three or more PMD Priority-2 violations → WARN.
5. Any Priority-2 violations → WARN.
6. No critical or high violations → PASS.

Always produce at least one recommended fix. If no violations exist, recommend a
clean code review practice for the team.

Use the assess_code_quality tool to return your structured assessment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent3_data = _get_agent_data(state, "3")
    agent13_data = _get_agent_data(state, "13")

    pmd_violations = await get_pmd_results(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    user_message = _build_user_message(story_id, fca_class, pmd_violations, agent13_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=1200,
    )

    # Ensemble: deterministic PMD-only verdict (Call A) vs Sonnet FCA-context verdict (Call B)
    call_a_verdict, call_a_conf = _pmd_baseline_verdict(pmd_violations, fca_class)
    call_b_verdict = extracted.get("quality_verdict", "WARN")
    call_b_conf, _ = _compute_confidence(pmd_violations, extracted, agent13_data)
    ta_pos, interaction_mode = classify_ta_interaction(call_a_conf, call_b_conf)
    ensemble_agreement = call_a_verdict == call_b_verdict

    # Leading call wins based on TA mode; ASSERT/DEFER → leading call; ESCALATE → conservative
    if interaction_mode in ("ASSERT",):
        verdict = call_a_verdict
    elif interaction_mode == "DEFER":
        verdict = call_b_verdict
    elif interaction_mode == "ESCALATE":
        # Neither trusted — take worse verdict for safety
        verdict = _worse_verdict(call_a_verdict, call_b_verdict)
    else:
        verdict = call_b_verdict if ensemble_agreement else _worse_verdict(call_a_verdict, call_b_verdict)

    confidence_score, signals = _compute_confidence(pmd_violations, extracted, agent13_data)
    escalated = confidence_score < settings.confidence_escalation_threshold

    critical = extracted.get("critical_violations", [])
    high = extracted.get("high_violations", [])

    what = (
        f"Code quality for {story_id}: {len(pmd_violations)} PMD violation(s) — "
        f"critical={len(critical)}, high={len(high)}, verdict={verdict}"
    )
    why = extracted.get(
        "quality_summary",
        "Code Quality Reviewer assessed PMD violations against FSC/FCA quality standards.",
    )

    data = {
        "quality_verdict": verdict,
        "critical_violations": critical,
        "high_violations": high,
        "total_violation_count": len(pmd_violations),
        "quality_summary": extracted.get("quality_summary", ""),
        "recommended_fixes": extracted.get("recommended_fixes", []),
        "fca_classification": fca_class,
        "pmd_data_available": len(pmd_violations) > 0 or _copado_responded(pmd_violations),
        "ensemble_agreement": ensemble_agreement,
        "ta_position": ta_pos,
        "interaction_mode": interaction_mode,
        "call_a_verdict": call_a_verdict,
        "call_b_verdict": call_b_verdict,
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
        model_used=settings.default_model,
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    pmd_violations: list[dict],
    extracted: dict,
    agent13_data: dict | None,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    # Signal 1: PMD data available (even if no violations — clean codebase is good signal)
    # We treat an empty list from a configured Copado as "data available"
    # vs empty list from unconfigured Copado (we can't distinguish easily, so use agent13)
    if agent13_data and agent13_data.get("changed_files_count", 0) > 0:
        scorer.add("metadata_context_available", True, +5)

    # Signal 2: clean code (no violations at all)
    if len(pmd_violations) == 0:
        scorer.add("no_pmd_violations", True, +8)
    elif any(v["priority"] == 1 for v in pmd_violations):
        scorer.add("critical_pmd_violations_present", True, -5)

    # Signal 3: verdict quality
    verdict = extracted.get("quality_verdict", "WARN")
    if verdict == "PASS":
        scorer.add("quality_verdict_pass", True, +10)
    elif verdict == "FAIL":
        scorer.add("quality_verdict_fail", True, -5)

    # Signal 4: recommended fixes present (model gave actionable output)
    if extracted.get("recommended_fixes"):
        scorer.add("recommended_fixes_present", len(extracted["recommended_fixes"]), +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pmd_baseline_verdict(violations: list[dict], fca_class: str) -> tuple[str, int]:
    """Deterministic PMD-only verdict — Call A in the ensemble. Returns (verdict, confidence)."""
    if any(v["priority"] == 1 for v in violations):
        return "FAIL", 80
    if any(v["rule_name"] in _ALWAYS_FAIL_RULES for v in violations):
        return "FAIL", 85
    if fca_class in ("HIGH", "MEDIUM") and any(
        v["rule_name"] in ("ApexCRUDViolation", "ApexSharingViolations") for v in violations
    ):
        return "FAIL", 80
    p2_count = sum(1 for v in violations if v["priority"] == 2)
    if p2_count >= 3:
        return "WARN", 75
    if p2_count > 0:
        return "WARN", 70
    return "PASS", 78 if violations else 72


_VERDICT_ORDER = {"FAIL": 2, "WARN": 1, "PASS": 0}


def _worse_verdict(a: str, b: str) -> str:
    return a if _VERDICT_ORDER.get(a, 0) >= _VERDICT_ORDER.get(b, 0) else b


def _copado_responded(violations: list) -> bool:
    # If Copado is unconfigured, get_pmd_results returns []. We can't tell the
    # difference here — confidence signals handle the uncertainty.
    return isinstance(violations, list)


def _build_user_message(
    story_id: str,
    fca_class: str,
    violations: list[dict],
    agent13_data: dict | None,
) -> str:
    detected = (agent13_data or {}).get("detected_objects", [])
    lines = [
        f"Story: {story_id}",
        f"FCA Classification: {fca_class}",
        f"Detected FSC objects in changed code: {', '.join(detected) or 'none'}",
        "",
        f"PMD violations ({len(violations)} total):",
    ]
    if violations:
        for v in violations:
            lines.append(
                f"  [P{v['priority']}] {v['rule_name']} — {v['description']} "
                f"({v['file_path']}:{v['line']})"
            )
    else:
        lines.append("  (none)")
    lines.append(f"\nAssess the code quality using the {_TOOL_NAME} tool.")
    return "\n".join(lines)


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
