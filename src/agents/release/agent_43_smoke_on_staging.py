"""
Agent 43 — Smoke-on-Staging
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs sequentially after Agent 42 (Dry-Run).
Has access to Agents 32, 42.

Purpose:
  Runs smoke tests on the staging environment after a successful dry-run deployment.
  Test suite depth is determined by regression risk level from Agent 32:
    - HIGH regression risk → FULL suite (all scenarios)
    - MEDIUM regression risk → REGRESSION suite
    - LOW regression risk → SMOKE suite only

  Stub: in production executes CRT smoke scenarios against staging org.
  Gate G9 depends on this verdict.

Output data keys consumed by downstream:
  smoke_tests_passed  → bool   (Gate G9 — must be True)
  smoke_test_count    → int
  smoke_failed        → int
  smoke_suite         → str    (FULL / REGRESSION / SMOKE)
  smoke_verdict       → str    (PASS / FAIL / SKIPPED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 43
AGENT_NAME = "Smoke-on-Staging"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_smoke_test_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "smoke_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the smoke test result on staging. "
                "State which suite ran, how many tests passed/failed, "
                "and whether the story is cleared for production deployment."
            ),
        },
        "smoke_concern": {
            "type": "string",
            "enum": ["none", "smoke_failures", "full_suite_failures",
                     "dry_run_not_done", "regression_risk_unresolved"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a smoke test run on a Salesforce FSC
staging environment in the release pipeline.
You will receive the dry-run outcome, regression risk level, and smoke test results.
Write a clear 2–3 sentence narrative explaining which test suite ran, what the results
were, and whether the story is ready for production deployment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent32_data = _get_agent_data(state, "32")
    agent42_data = _get_agent_data(state, "42")

    passed, test_count, failed, suite, verdict = _run_smoke_tests(agent32_data, agent42_data)

    trace_msg = _build_trace_message(story_id, passed, test_count, failed, suite, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent32_data, agent42_data, passed)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Smoke tests for {story_id}: {test_count} run, {failed} failed, "
        f"suite={suite} — verdict={verdict}"
    )
    why = trace.get("narrative", "Smoke-on-Staging ran post-deployment tests.")

    data = {
        "smoke_tests_passed": passed,
        "smoke_test_count": test_count,
        "smoke_failed": failed,
        "smoke_suite": suite,
        "smoke_verdict": verdict,
        "smoke_concern": trace.get("smoke_concern", "none"),
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


# ── Deterministic smoke test simulation ──────────────────────────────────────

def _run_smoke_tests(
    agent32_data: dict | None,
    agent42_data: dict | None,
) -> tuple[bool, int, int, str, str]:
    """Returns (passed, test_count, failed_count, suite, verdict)."""
    dry_run_success = (agent42_data or {}).get("dry_run_success", False)
    dry_run_verdict = (agent42_data or {}).get("dry_run_verdict", "SKIPPED")
    regression_risk = (agent32_data or {}).get("regression_risk", "LOW")
    suite_type      = (agent32_data or {}).get("recommended_suite", "SMOKE")

    if not dry_run_success or dry_run_verdict in ("FAIL", "SKIPPED"):
        return False, 0, 0, "SMOKE", "SKIPPED"

    # Suite depth drives test count (stub values)
    if regression_risk == "HIGH" or suite_type == "FULL":
        suite = "FULL"
        test_count = 20
    elif regression_risk == "MEDIUM" or suite_type == "REGRESSION":
        suite = "REGRESSION"
        test_count = 10
    else:
        suite = "SMOKE"
        test_count = 5

    # Stub: all smoke tests pass on a valid dry-run
    failed = 0
    passed = True
    verdict = "PASS"

    return passed, test_count, failed, suite, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent32_data: dict | None,
    agent42_data: dict | None,
    smoke_passed: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    if agent42_data:
        scorer.add("dry_run_data_available", True, +8)
    else:
        scorer.add("no_dry_run_data", 0, -10)

    if agent32_data:
        scorer.add("regression_risk_data", True, +5)

    if smoke_passed:
        scorer.add("smoke_tests_passed", True, +5)
    else:
        scorer.add("smoke_tests_failed_or_skipped", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a smoke test narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    passed: bool,
    test_count: int,
    failed: int,
    suite: str,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Smoke suite: {suite}\n"
        f"Tests run: {test_count}\n"
        f"Tests failed: {failed}\n"
        f"Smoke passed: {passed}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
