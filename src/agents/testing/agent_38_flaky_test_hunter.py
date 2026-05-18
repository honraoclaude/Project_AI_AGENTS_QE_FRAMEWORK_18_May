"""
Agent 38 — Flaky Test Hunter
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=62)

Runs in Testing Batch 4 (parallel with Agents 33, 34).
Has access to Agent 27.

Purpose:
  Detects flaky tests in CRT results by analysing self-heal patterns,
  intermittent failures, and tests that passed after retry. Quarantines
  tests that exhibit flaky behaviour to prevent false-positive gate passage.

  Deterministic detection logic; Haiku writes the narrative and recommendation.

Output data keys consumed by downstream:
  flaky_tests             → list   (test IDs flagged as flaky)
  flaky_count             → int
  quarantine_recommended  → list   (test IDs to quarantine)
  flaky_verdict           → str    (PASS / WARN / QUARANTINE_REQUIRED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 38
AGENT_NAME = "Flaky Test Hunter"

_FLAKY_THRESHOLD = 2          # ≥ this many flaky tests → WARN
_QUARANTINE_THRESHOLD = 3     # ≥ this many flaky tests → QUARANTINE_REQUIRED

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_flaky_test_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "flaky_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising flaky test findings. "
                "State which tests are flaky, the likely root cause (locator drift, "
                "timing, data dependency), and what the QE engineer must do."
            ),
        },
        "flaky_concern": {
            "type": "string",
            "enum": ["none", "locator_drift", "timing_issue",
                     "data_dependency", "excessive_flakiness"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a flaky test detection analysis
in a Salesforce FSC Wealth Management CRT suite.
You will receive a list of tests flagged as flaky, including their self-heal
status and retry patterns.
Write a clear 2–3 sentence narrative identifying the likely root cause of flakiness
(locator drift from self-heal, timing issues, data dependency) and what the QE
engineer must do to stabilise the suite.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent27_data = _get_agent_data(state, "27")

    flaky_tests, quarantine, verdict = _detect_flaky_tests(agent27_data)
    flaky_count = len(flaky_tests)

    trace_msg = _build_trace_message(story_id, flaky_tests, quarantine, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent27_data, flaky_count)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Flaky test detection for {story_id}: {flaky_count} flaky test(s) — "
        f"verdict={verdict}"
    )
    why = trace.get("narrative", "Flaky Test Hunter analysed CRT execution patterns.")

    data = {
        "flaky_tests": flaky_tests,
        "flaky_count": flaky_count,
        "quarantine_recommended": quarantine,
        "flaky_verdict": verdict,
        "flaky_concern": trace.get("flaky_concern", "none"),
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


# ── Deterministic flaky detection ─────────────────────────────────────────────

def _detect_flaky_tests(
    agent27_data: dict | None,
) -> tuple[list[str], list[str], str]:
    """Returns (flaky_test_ids, quarantine_ids, verdict)."""
    if not agent27_data:
        return [], [], "PASS"

    crt_results: list[dict] = agent27_data.get("crt_results", [])
    if not crt_results:
        return [], [], "PASS"

    flaky: list[str] = []
    for test in crt_results:
        test_id = test.get("test_id", "")
        self_healed = test.get("self_healed", False)
        retry_passed = test.get("retry_passed", False)
        status = test.get("status", "PASSED")

        # A test is flaky if it self-healed OR passed only after retry
        if self_healed or retry_passed:
            flaky.append(test_id)
        # Also flag tests marked INTERMITTENT
        elif status == "INTERMITTENT":
            flaky.append(test_id)

    flaky_count = len(flaky)
    quarantine = flaky if flaky_count >= _QUARANTINE_THRESHOLD else []

    if flaky_count == 0:
        verdict = "PASS"
    elif flaky_count >= _QUARANTINE_THRESHOLD:
        verdict = "QUARANTINE_REQUIRED"
    else:
        verdict = "WARN"

    return flaky, quarantine, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent27_data: dict | None,
    flaky_count: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    if agent27_data:
        tests_executed = agent27_data.get("tests_executed", 0)
        if tests_executed > 0:
            scorer.add("crt_results_available", tests_executed, +12)
        else:
            scorer.add("no_tests_executed", 0, -5)
    else:
        scorer.add("no_crt_data", 0, -15)

    if flaky_count == 0:
        scorer.add("no_flaky_tests", 0, +5)
    elif flaky_count >= _QUARANTINE_THRESHOLD:
        scorer.add("high_flakiness", flaky_count, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a flaky test detection narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    flaky_tests: list[str],
    quarantine: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Flaky tests detected: {flaky_tests or ['none']}\n"
        f"Quarantine recommended: {quarantine or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
