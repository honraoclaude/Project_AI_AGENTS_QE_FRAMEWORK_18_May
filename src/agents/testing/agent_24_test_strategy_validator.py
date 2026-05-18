"""
Agent 24 — Test Strategy Validator
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs first in Testing phase (Batch 1, parallel with Agents 25, 32).
Has access to Agents 3, 6, 9, 19, 21, 23.

Purpose:
  Validates that the test strategy is appropriate for the story's FCA
  classification and risk profile. Checks that:
  - Gherkin scenario count meets minimum coverage expectations
  - FCA HIGH/MEDIUM stories have negative and boundary test scenarios
  - Test data strategy is in place (Agent 21 verdict)
  - BDD coverage matches the risk anticipation from Agent 9

  All checks are deterministic; Haiku writes the explanation.

Output data keys consumed by downstream:
  strategy_valid         → bool (Gate G5 prerequisite)
  strategy_verdict       → str  (PASS / WARN / FAIL)
  strategy_gaps          → list (informational)
  fca_scenario_coverage  → bool (Gate G5 — FCA regulated stories)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 24
AGENT_NAME = "Test Strategy Validator"

_MIN_SCENARIOS_HIGH  = 4
_MIN_SCENARIOS_MEDIUM = 3
_MIN_SCENARIOS_LOW   = 2

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_strategy_validation_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "strategy_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the test strategy validation result. "
                "Note FCA classification, scenario count vs requirement, gaps, "
                "and what the QE engineer must do to address any shortfalls."
            ),
        },
        "strategy_concern": {
            "type": "string",
            "enum": ["none", "insufficient_scenarios", "missing_fca_coverage",
                     "no_test_data", "multiple"],
            "description": "Primary strategy concern, or 'none' if strategy is sound.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a test strategy validation.
You will receive FCA classification, scenario counts, coverage flags, and gap details.
Write a clear 2–3 sentence narrative explaining whether the test strategy is adequate
for the story's risk level, what gaps exist, and what the QE engineer must do.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent6_data  = _get_agent_data(state, "6")
    agent19_data = _get_agent_data(state, "19")
    agent21_data = _get_agent_data(state, "21")
    agent23_data = _get_agent_data(state, "23")

    # ── Deterministic validation ──────────────────────────────────────────────
    valid, verdict, gaps, fca_covered = _validate_strategy(
        agent3_data, agent6_data, agent19_data, agent21_data, agent23_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(story_id, agent3_data, agent19_data, gaps, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent3_data, agent19_data, agent21_data, valid,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Test strategy validation for {story_id}: valid={valid}, "
        f"fca_covered={fca_covered}, {len(gaps)} gap(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Test Strategy Validator assessed coverage adequacy.")

    data = {
        "strategy_valid": valid,
        "strategy_verdict": verdict,
        "strategy_gaps": gaps,
        "fca_scenario_coverage": fca_covered,
        "strategy_concern": trace.get("strategy_concern", "none"),
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


# ── Deterministic validation ──────────────────────────────────────────────────

def _validate_strategy(
    agent3_data: dict | None,
    agent6_data: dict | None,
    agent19_data: dict | None,
    agent21_data: dict | None,
    agent23_data: dict | None,
) -> tuple[bool, str, list[str], bool]:
    """Returns (valid, verdict, gaps, fca_scenario_coverage)."""
    gaps: list[str] = []
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")

    # Minimum scenario requirement
    scenario_count = (agent19_data or {}).get("scenario_count", 0)
    min_required = {
        "HIGH": _MIN_SCENARIOS_HIGH,
        "MEDIUM": _MIN_SCENARIOS_MEDIUM,
    }.get(fca_class, _MIN_SCENARIOS_LOW)

    if scenario_count < min_required:
        gaps.append(
            f"Insufficient Gherkin scenarios: {scenario_count} present, "
            f"{min_required} required for {fca_class}-FCA story"
        )

    # FCA coverage for regulated stories
    fca_covered = (agent19_data or {}).get("fca_coverage_present", False)
    if fca_class in ("HIGH", "MEDIUM") and not fca_covered:
        gaps.append(
            f"FCA {fca_class} story missing negative/boundary Gherkin scenarios"
        )

    # Test data strategy
    data_verdict = (agent21_data or {}).get("data_verdict", "")
    if data_verdict == "INCOMPLETE":
        gaps.append("Test data strategy is INCOMPLETE — no seed records designed")
    elif data_verdict == "WARN":
        gaps.append("Test data strategy has gaps — review required before execution")

    # Vulnerable Customer profiles for HIGH/MEDIUM
    vulnerable_profiles = (agent21_data or {}).get("vulnerable_profiles", [])
    if fca_class in ("HIGH", "MEDIUM") and not vulnerable_profiles:
        gaps.append(
            f"FCA {fca_class} story missing Vulnerable Customer test profiles"
        )

    # Development phase must have passed
    dev_verdict = (agent23_data or {}).get("development_verdict", "")
    if dev_verdict == "FAIL":
        gaps.append("Development phase FAILED — testing cannot begin until resolved")

    # Determine verdict
    critical_gaps = [g for g in gaps if "FAILED" in g or "INCOMPLETE" in g
                     or "missing negative" in g]
    if critical_gaps:
        verdict = "FAIL"
        valid = False
    elif gaps:
        verdict = "WARN"
        valid = True  # warn but don't block
    else:
        verdict = "PASS"
        valid = True

    return valid, verdict, gaps, fca_covered


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent19_data: dict | None,
    agent21_data: dict | None,
    valid: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if agent19_data:
        count = agent19_data.get("scenario_count", 0)
        if count > 0:
            scorer.add("gherkin_scenarios_available", count, +8)
        else:
            scorer.add("no_gherkin_scenarios", 0, -10)
    else:
        scorer.add("no_gherkin_agent_data", 0, -8)

    if agent21_data:
        scorer.add("test_data_strategy_available", True, +5)

    if not valid:
        scorer.add("strategy_invalid", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a test strategy validation narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent3_data: dict | None,
    agent19_data: dict | None,
    gaps: list[str],
    verdict: str,
) -> str:
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    scenario_count = (agent19_data or {}).get("scenario_count", 0)
    fca_covered = (agent19_data or {}).get("fca_coverage_present", False)
    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n"
        f"Gherkin scenarios: {scenario_count}\n"
        f"FCA scenario coverage: {fca_covered}\n"
        f"Strategy gaps: {gaps or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
