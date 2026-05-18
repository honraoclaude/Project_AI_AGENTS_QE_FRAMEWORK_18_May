"""
Agent 12 — Apex Coverage Analyser
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=68)

Runs in Development Batch 2 (parallel with Agents 14, 15, 16).
Has access to Agents 3 (FCA class), 6 (test design target), 13 (metadata scope).

Purpose:
  Reads the latest Apex test run results from Copado and determines whether
  the story's code meets its required coverage threshold:
    HIGH/MEDIUM FCA: 85% (mandatory — maps to Gate G3)
    LOW FCA:         75%

  If no test results are available (Copado unconfigured or no run yet),
  the verdict is UNKNOWN and confidence is heavily penalised.

  Haiku generates the narrative — the pass/fail logic is pure Python.

Output data keys consumed by downstream:
  coverage_pct      → int  (Gate G3 — primary gate signal)
  coverage_threshold → int  (Gate G3 — threshold used for comparison)
  coverage_passed   → bool (Gate G3 — direct gate input)
  coverage_verdict  → str  (CO email context for HIGH-FCA stories)
  tests_failed      → int  (Gate G3 — any failed tests block regardless of coverage)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.copado import get_apex_test_results

AGENT_ID = 12
AGENT_NAME = "Apex Coverage Analyser"

_DEFAULT_THRESHOLD_HIGH = 85
_DEFAULT_THRESHOLD_LOW = 75

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_coverage_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "coverage_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining the current Apex test coverage state, "
                "whether it meets the required threshold, and what the developer "
                "must do if coverage is insufficient."
            ),
        },
        "coverage_concern": {
            "type": "string",
            "enum": ["none", "low", "critical"],
            "description": (
                "none: Coverage meets threshold, all tests passing. "
                "low: Coverage within 5% of threshold or minor test failures. "
                "critical: Coverage below threshold or tests failing."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for an Apex test coverage analysis.
You will receive the actual coverage percentage, the required threshold, and test
run statistics. Write a clear 2–3 sentence narrative explaining the coverage status
and what the developer must do if coverage is insufficient. Be factual and actionable.
Reference the FCA classification if coverage is below threshold for a regulated story.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent3_data = _get_agent_data(state, "3")
    agent6_data = _get_agent_data(state, "6")

    test_results = await get_apex_test_results(story_id)

    # ── Deterministic analysis ────────────────────────────────────────────────
    coverage_pct, threshold, tests_run, tests_failed, verdict, gap = (
        _analyse_coverage(test_results, agent3_data, agent6_data)
    )

    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")

    # ── Haiku trace generation ────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, coverage_pct, threshold, tests_run, tests_failed, verdict, fca_class,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        test_results, coverage_pct, threshold, verdict, tests_run,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Coverage analysis for {story_id}: {coverage_pct}% "
        f"(threshold={threshold}%, gap={gap}%) — {verdict}"
    )
    why = trace.get(
        "narrative",
        "Apex Coverage Analyser compared actual test coverage against the FCA-derived threshold.",
    )

    data = {
        "coverage_pct": coverage_pct,
        "coverage_threshold": threshold,
        "coverage_gap_pct": gap,
        "coverage_passed": verdict == "PASS",
        "coverage_verdict": verdict,
        "tests_run": tests_run,
        "tests_passed": test_results.get("tests_passed", 0),
        "tests_failed": tests_failed,
        "fca_classification": fca_class,
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


# ── Deterministic coverage analysis ──────────────────────────────────────────

def _analyse_coverage(
    test_results: dict,
    agent3_data: dict | None,
    agent6_data: dict | None,
) -> tuple[int, int, int, int, str, int]:
    """
    Compare coverage_pct from Copado against the required threshold.
    Returns (coverage_pct, threshold, tests_run, tests_failed, verdict, gap).
    Pure Python — no LLM involved.
    """
    coverage_pct = test_results.get("coverage_pct", 0)
    tests_run = test_results.get("tests_run", 0)
    tests_failed = test_results.get("tests_failed", 0)

    # Threshold: explicit from Agent 6 > derived from FCA class > system default
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    agent6_target = (agent6_data or {}).get("coverage_target_pct", 0)

    if agent6_target:
        threshold = agent6_target
    elif fca_class in ("HIGH", "MEDIUM"):
        threshold = _DEFAULT_THRESHOLD_HIGH
    else:
        threshold = _DEFAULT_THRESHOLD_LOW

    # Verdict
    if tests_run == 0:
        verdict = "UNKNOWN"
        gap = threshold  # gap = full threshold if no tests run
    elif tests_failed > 0:
        verdict = "FAIL"  # failing tests override coverage number
        gap = max(threshold - coverage_pct, 0)
    elif coverage_pct >= threshold:
        verdict = "PASS"
        gap = 0
    else:
        verdict = "FAIL"
        gap = threshold - coverage_pct

    return coverage_pct, threshold, tests_run, tests_failed, verdict, gap


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    test_results: dict,
    coverage_pct: int,
    threshold: int,
    verdict: str,
    tests_run: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=68)

    # Signal 1: test results available (Copado configured and tests ran)
    data_available = bool(test_results.get("test_run_id") or tests_run > 0)
    if data_available:
        scorer.add("coverage_data_available", True, +8)
    else:
        scorer.add("no_coverage_data", True, -15)

    # Signal 2: tests ran
    if tests_run >= 5:
        scorer.add("adequate_test_count", tests_run, +5)
    elif tests_run == 0 and data_available:
        scorer.add("no_tests_run", 0, -10)

    # Signal 3: coverage verdict
    if verdict == "PASS":
        scorer.add("coverage_passed", coverage_pct, +10)
    elif verdict == "FAIL":
        scorer.add("coverage_failed", coverage_pct, -8)
    else:  # UNKNOWN
        scorer.add("coverage_unknown", True, -10)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an Apex coverage narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    coverage_pct: int,
    threshold: int,
    tests_run: int,
    tests_failed: int,
    verdict: str,
    fca_class: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n\n"
        f"Actual coverage: {coverage_pct}%\n"
        f"Required threshold: {threshold}%\n"
        f"Tests run: {tests_run}\n"
        f"Tests failed: {tests_failed}\n"
        f"Coverage verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
