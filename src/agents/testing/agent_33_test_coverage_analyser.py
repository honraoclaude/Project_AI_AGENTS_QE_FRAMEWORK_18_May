"""
Agent 33 — Test Coverage Analyser
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Testing Batch 4 (parallel with Agents 34, 38).
Has access to Agents 19, 24, 26, 27, 29, 30.

Purpose:
  Aggregates test coverage across all test types (Gherkin, CRT, UAT, FCA)
  and computes an overall coverage score. Identifies which ACs and
  regulatory rules have test coverage and which are uncovered.

  Pure aggregation — no LLM required for the scoring itself. Haiku writes
  the narrative.

Output data keys consumed by downstream:
  overall_coverage_pct   → float (Gate G5 — minimum threshold check)
  coverage_by_type       → dict  (Gherkin/CRT/UAT/FCA breakdown)
  uncovered_acs          → list  (informational)
  coverage_verdict       → str   (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 33
AGENT_NAME = "Test Coverage Analyser"

_MIN_COVERAGE_HIGH   = 90.0
_MIN_COVERAGE_MEDIUM = 85.0
_MIN_COVERAGE_LOW    = 75.0

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_coverage_analysis_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "coverage_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising test coverage across all types. "
                "State the overall percentage, breakdown by type, any uncovered ACs, "
                "and what must be done to meet the FCA classification threshold."
            ),
        },
        "coverage_concern": {
            "type": "string",
            "enum": ["none", "below_threshold", "fca_scenarios_uncovered",
                     "uat_uncovered", "multiple"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a test coverage analysis.
You will receive coverage percentages across Gherkin, CRT, UAT, and FCA scenario types.
Write a clear 2–3 sentence narrative explaining the overall coverage picture,
what is below threshold, and what the QE engineer must add to reach compliance.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent4_data  = _get_agent_data(state, "4")
    agent5_data  = _get_agent_data(state, "5")
    agent19_data = _get_agent_data(state, "19")
    agent26_data = _get_agent_data(state, "26")
    agent27_data = _get_agent_data(state, "27")
    agent29_data = _get_agent_data(state, "29")
    agent30_data = _get_agent_data(state, "30")

    # ── Deterministic aggregation ─────────────────────────────────────────────
    overall_pct, by_type, uncovered, verdict = _analyse_coverage(
        agent3_data, agent4_data, agent5_data, agent19_data, agent26_data,
        agent27_data, agent29_data, agent30_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(story_id, overall_pct, by_type, uncovered, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent19_data, agent26_data, agent29_data, agent30_data, overall_pct,
        scenarios_truncated=(agent26_data or {}).get("scenarios_truncated", False),
        truncated_count=(agent26_data or {}).get("truncated_scenario_count", 0),
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Test coverage for {story_id}: {overall_pct:.0f}% overall, "
        f"{len(uncovered)} uncovered AC(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Test Coverage Analyser aggregated all test type coverage.")

    data = {
        "overall_coverage_pct": overall_pct,
        "coverage_by_type": by_type,
        "uncovered_acs": uncovered,
        "coverage_verdict": verdict,
        "coverage_concern": trace.get("coverage_concern", "none"),
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


# ── Deterministic coverage aggregation ───────────────────────────────────────

def _analyse_coverage(
    agent3_data: dict | None,
    agent4_data: dict | None,
    agent5_data: dict | None,
    agent19_data: dict | None,
    agent26_data: dict | None,
    agent27_data: dict | None,
    agent29_data: dict | None,
    agent30_data: dict | None,
) -> tuple[float, dict, list[str], str]:
    """Returns (overall_pct, coverage_by_type, uncovered_acs, verdict)."""
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    vulnerable_customer_impact = (agent4_data or {}).get("vulnerable_customer_impact", False)
    expected_acs = (agent5_data or {}).get("ac_count", 0)
    ac_clauses = (agent5_data or {}).get("ac_clauses", [])

    gherkin_count = (agent19_data or {}).get("scenario_count", 0)
    vc_coverage_present = (agent19_data or {}).get("vulnerable_customer_coverage_present", False)
    crt_coverage = (agent26_data or {}).get("automation_coverage", 0.0)
    crt_executed = (agent27_data or {}).get("tests_executed", 0)
    uat_count = (agent29_data or {}).get("uat_test_count", 0)
    fca_count = (agent30_data or {}).get("fca_scenario_count", 0)

    by_type = {
        "gherkin": gherkin_count,
        "crt_automation_pct": crt_coverage,
        "crt_executed": crt_executed,
        "uat": uat_count,
        "fca_regulatory": fca_count,
        "vulnerable_customer_covered": vc_coverage_present,
    }

    # REQ-24 Gap 1: Gherkin normalised against expected AC count
    components: list[float] = []
    if gherkin_count > 0:
        if expected_acs > 0:
            components.append(min(100.0, gherkin_count / expected_acs * 100))
        else:
            components.append(min(100.0, gherkin_count * 20.0))  # fallback: no AC count
    if crt_coverage > 0:
        components.append(crt_coverage)
    if uat_count > 0 and expected_acs > 0:
        components.append(min(100.0, uat_count / expected_acs * 100))
    elif uat_count > 0:
        components.append(min(100.0, uat_count * 25.0))

    overall_pct = sum(components) / len(components) if components else 0.0
    overall_pct = min(100.0, overall_pct)

    # REQ-24 Gap 2: Uncovered ACs — use real AC descriptions from Agent 05 when available
    uncovered: list[str] = []
    if expected_acs > 0 and uat_count < expected_acs:
        if ac_clauses:
            uncovered = [
                c.get("description", f"AC{i+1}")
                for i, c in enumerate(ac_clauses[uat_count:])
            ]
        else:
            for i in range(uat_count, expected_acs):
                uncovered.append(f"AC{i+1}")

    # FCA check thresholds
    min_threshold = {
        "HIGH": _MIN_COVERAGE_HIGH,
        "MEDIUM": _MIN_COVERAGE_MEDIUM,
    }.get(fca_class, _MIN_COVERAGE_LOW)

    # REQ-24 Gap 3+4: verdict escalation logic
    if overall_pct < min_threshold:
        verdict = "FAIL"
    elif fca_class == "HIGH" and fca_count == 0:
        verdict = "FAIL"   # HIGH-FCA with no regulatory scenarios is a compliance gap
    elif (
        fca_class in ("HIGH", "MEDIUM")
        and vulnerable_customer_impact
        and not vc_coverage_present
    ):
        verdict = "FAIL"   # VC impact required but no VC scenario
    elif uncovered or (fca_class == "MEDIUM" and fca_count == 0):
        verdict = "WARN"
    else:
        verdict = "PASS"

    return overall_pct, by_type, uncovered, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent19_data: dict | None,
    agent26_data: dict | None,
    agent29_data: dict | None,
    agent30_data: dict | None,
    overall_pct: float,
    scenarios_truncated: bool = False,
    truncated_count: int = 0,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    agents_available = sum(1 for d in [agent19_data, agent26_data, agent29_data, agent30_data] if d)
    if agents_available >= 3:
        scorer.add("comprehensive_test_data", agents_available, +10)
    elif agents_available >= 1:
        scorer.add("partial_test_data", agents_available, +4)
    else:
        scorer.add("no_test_data", 0, -10)

    if overall_pct >= 80:
        scorer.add("high_coverage", overall_pct, +5)
    elif overall_pct < 50:
        scorer.add("low_coverage", overall_pct, -5)

    # REQ-24 Gap 5: CRT scenario truncation means automation_coverage is an overestimate
    if scenarios_truncated:
        scorer.add("crt_scenario_truncated", truncated_count, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a test coverage analysis narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    overall_pct: float,
    by_type: dict,
    uncovered: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Overall coverage: {overall_pct:.1f}%\n"
        f"Gherkin scenarios: {by_type.get('gherkin', 0)}\n"
        f"CRT automation: {by_type.get('crt_automation_pct', 0):.0f}%\n"
        f"UAT tests: {by_type.get('uat', 0)}\n"
        f"FCA regulatory scenarios: {by_type.get('fca_regulatory', 0)}\n"
        f"Uncovered ACs: {uncovered or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
