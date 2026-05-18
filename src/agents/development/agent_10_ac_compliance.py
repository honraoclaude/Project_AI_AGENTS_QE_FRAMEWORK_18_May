"""
Agent 10 — AC Compliance Verifier
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=68 — deterministic + strong upstream data)

Runs in Development Batch 1 (parallel with Agent 11, Agent 13).
Has access to Refinement agent results (Agents 3, 5, 6).

Purpose:
  At the start of Development, verifies that the Acceptance Criteria defined
  during Refinement (Agent 5) are still present in the Jira story and that
  coverage completeness has not degraded.

  Deterministic checks:
    1. Current AC count in Jira vs Agent 5's refinement-time count
    2. Coverage completeness (happy_path, error_paths, edge_cases, regulatory)
    3. For HIGH/MEDIUM-FCA stories: regulatory scenario presence is mandatory

  Haiku generates the narrative trace — the analysis itself is pure Python.

Output data keys consumed by downstream:
  compliance_verdict     → str   (Gate G2 — blocks on FAIL for HIGH-FCA)
  missing_coverage_types → list  (Agent 19 BDD Gherkin — fills gaps in scenarios)
  ac_count_current       → int   (Agent 23 Story-to-Code Tracer — audit trail)
  ac_delta               → int   (Gate G2 — negative delta = ACs removed)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 10
AGENT_NAME = "AC Compliance Verifier"

# ── Haiku tool for narrative ──────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_ac_compliance_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "compliance_risk"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the AC compliance state at Development start. "
                "Note whether ACs match the refinement baseline, any coverage gaps, "
                "and the recommended action for the developer."
            ),
        },
        "compliance_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low: ACs match refinement, all coverage types present. "
                "medium: Minor AC drift or 1–2 non-critical coverage types missing. "
                "high: ACs removed since refinement, or regulatory coverage missing on HIGH/MEDIUM-FCA story."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for an automated AC Compliance check at
the start of the Development phase. You will receive the current AC state of a Jira
story compared against the ACs generated during Refinement. Write a clear 2–3 sentence
narrative explaining whether the story is development-ready and what the developer
should address. Be factual and actionable. Reference the FCA classification if relevant.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    story = await get_story(story_id)
    current_acs = await get_acceptance_criteria(story_id)

    agent3_data = _get_agent_data(state, "3")
    agent5_data = _get_agent_data(state, "5")

    # ── Deterministic analysis ────────────────────────────────────────────────
    current_count, refinement_count, ac_delta, missing_coverage, verdict = (
        _analyse_ac_compliance(current_acs, agent5_data, agent3_data)
    )

    # ── Haiku trace generation ────────────────────────────────────────────────
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    trace_message = _build_trace_message(
        story, current_count, refinement_count, ac_delta,
        missing_coverage, verdict, fca_class,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        agent5_data, agent3_data, current_acs, ac_delta, missing_coverage,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"AC compliance for {story_id}: {current_count} ACs current "
        f"(refinement={refinement_count}, delta={ac_delta:+d}) — verdict={verdict}"
    )
    why = trace.get(
        "narrative",
        "AC Compliance Verifier compared current Jira ACs against the Refinement baseline.",
    )

    data = {
        "ac_count_current": current_count,
        "ac_count_at_refinement": refinement_count,
        "ac_delta": ac_delta,
        "compliance_verdict": verdict,
        "missing_coverage_types": missing_coverage,
        "compliance_risk": trace.get("compliance_risk", "low"),
        "narrative": trace.get("narrative", ""),
        "fca_classification": fca_class,
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


# ── Deterministic AC compliance analysis ─────────────────────────────────────

def _analyse_ac_compliance(
    current_acs: list[dict],
    agent5_data: dict | None,
    agent3_data: dict | None,
) -> tuple[int, int, int, list[str], str]:
    """
    Compare current Jira ACs against the Refinement baseline (Agent 5).
    Returns (current_count, refinement_count, ac_delta, missing_coverage, verdict).
    Pure Python — no LLM involved.
    """
    current_count = len(current_acs)
    refinement_count = (agent5_data or {}).get("ac_clause_count", 0)
    ac_delta = current_count - refinement_count

    coverage = (agent5_data or {}).get("coverage_assessment", {})
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")

    missing_coverage: list[str] = []
    for coverage_type, present in coverage.items():
        if not present:
            missing_coverage.append(coverage_type)

    # HIGH/MEDIUM FCA always requires a regulatory scenario
    if fca_class in ("HIGH", "MEDIUM") and not coverage.get("regulatory", True):
        if "regulatory" not in missing_coverage:
            missing_coverage.append("regulatory")

    # Verdict
    if ac_delta < 0 or ("regulatory" in missing_coverage and fca_class in ("HIGH", "MEDIUM")):
        verdict = "FAIL"
    elif missing_coverage:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"

    return current_count, refinement_count, ac_delta, missing_coverage, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent5_data: dict | None,
    agent3_data: dict | None,
    current_acs: list[dict],
    ac_delta: int,
    missing_coverage: list[str],
) -> tuple[int, dict]:
    scorer = TierBScorer(base=68)

    # Signal 1: Agent 5 available → have refinement baseline to compare against
    if agent5_data:
        scorer.add("refinement_baseline_available", True, +8)
    else:
        scorer.add("refinement_baseline_missing", True, -15)

    # Signal 2: ACs present in Jira
    if len(current_acs) >= 1:
        scorer.add("acs_present_in_jira", len(current_acs), +5)
    else:
        scorer.add("no_acs_in_jira", 0, -10)

    # Signal 3: AC count matches refinement baseline
    if agent5_data and ac_delta == 0:
        scorer.add("ac_count_matches_refinement", True, +5)
    elif ac_delta < 0:
        scorer.add("acs_removed_since_refinement", abs(ac_delta), -8)

    # Signal 4: Coverage types — each missing type costs confidence
    if not missing_coverage:
        scorer.add("all_coverage_types_present", True, +5)
    else:
        penalty = min(len(missing_coverage) * 3, 9)
        scorer.add("missing_coverage_types", len(missing_coverage), -penalty)

    # Signal 5: HIGH/MEDIUM-FCA with missing regulatory coverage — serious gap
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    if fca_class in ("HIGH", "MEDIUM") and "regulatory" in missing_coverage:
        scorer.add("regulated_story_missing_regulatory_coverage", fca_class, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an explainability narrative for an AC compliance check.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story: dict,
    current_count: int,
    refinement_count: int,
    ac_delta: int,
    missing_coverage: list[str],
    verdict: str,
    fca_class: str,
) -> str:
    return (
        f"Story: {story['story_id']} — {story['summary']}\n"
        f"FCA Classification: {fca_class}\n\n"
        f"AC count at Development start: {current_count}\n"
        f"AC count at Refinement baseline (Agent 5): {refinement_count}\n"
        f"AC delta: {ac_delta:+d}\n"
        f"Missing coverage types: {missing_coverage or ['none']}\n"
        f"Compliance verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
