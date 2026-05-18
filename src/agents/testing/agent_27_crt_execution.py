"""
Agent 27 — CRT Execution Agent
Phase       : Testing
PACT        : Autonomous
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=62)

Runs sequentially after Batch 2 (needs Agent 26's CRT test cases).
Has access to Agents 25, 26.

Purpose:
  Triggers Copado Robotic Testing (CRT) execution for the designed test
  cases and collects results. In production this calls the Copado REST API
  to initiate a test run and polls for completion. The current stub
  simulates execution from the Agent 25 environment state and Agent 26
  test case definitions.

  Haiku generates the execution narrative; pass/fail tallying is deterministic.

Output data keys consumed by downstream:
  crt_results            → list (Agent 28 Self-Heal Reviewer input)
  crt_pass_count         → int  (coverage metric)
  crt_fail_count         → int  (Gate G5 input)
  crt_execution_verdict  → str  (PASS / PARTIAL / FAIL / SKIPPED)
  tests_executed         → int
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 27
AGENT_NAME = "CRT Execution Agent"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_crt_execution_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "execution_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the CRT execution result. "
                "State how many tests ran, how many passed/failed, "
                "and what the QE engineer must investigate."
            ),
        },
        "execution_concern": {
            "type": "string",
            "enum": ["none", "test_failures", "env_instability",
                     "execution_skipped", "partial_run"],
            "description": "Primary CRT execution concern.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Copado Robotic Testing (CRT) execution.
You will receive the execution summary: test count, pass count, fail count, and any errors.
Write a clear 2–3 sentence narrative explaining the execution result, what failed and why
(if known), and what the QE engineer must do next. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent25_data = _get_agent_data(state, "25")
    agent26_data = _get_agent_data(state, "26")

    # ── Deterministic execution simulation ───────────────────────────────────
    results, executed, passed, failed, verdict = _simulate_execution(
        agent25_data, agent26_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(story_id, executed, passed, failed, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(
        agent25_data, agent26_data, executed, verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"CRT execution for {story_id}: {executed} test(s) run, "
        f"{passed} passed, {failed} failed — verdict={verdict}"
    )
    why = trace.get("narrative", "CRT Execution Agent ran automated test suite.")

    data = {
        "crt_results": results,
        "crt_pass_count": passed,
        "crt_fail_count": failed,
        "crt_execution_verdict": verdict,
        "tests_executed": executed,
        "execution_concern": trace.get("execution_concern", "none"),
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


# ── Deterministic execution simulation ───────────────────────────────────────

def _simulate_execution(
    agent25_data: dict | None,
    agent26_data: dict | None,
) -> tuple[list[dict], int, int, int, str]:
    """
    Simulate CRT execution from environment state and test case definitions.
    In production: call Copado CRT REST API.
    Returns (results, executed, passed, failed, verdict).
    """
    env_ready = (agent25_data or {}).get("env_ready", False)
    crt_connected = (agent25_data or {}).get("crt_connected", False)
    test_cases = (agent26_data or {}).get("crt_test_cases", [])
    design_verdict = (agent26_data or {}).get("crt_design_verdict", "INCOMPLETE")

    # Cannot execute if env not ready or CRT not connected
    if not env_ready or not crt_connected:
        return [], 0, 0, 0, "SKIPPED"

    # Cannot execute if no test cases designed
    if not test_cases or design_verdict == "INCOMPLETE":
        return [], 0, 0, 0, "SKIPPED"

    # Stub: simulate all tests pass (production would call Copado API)
    results = [
        {
            "test_id": tc.get("test_id", f"CRT-{i+1:03d}"),
            "title": tc.get("title", ""),
            "status": "PASSED",
            "duration_ms": 1200,
            "error_message": None,
            "self_healed": False,
        }
        for i, tc in enumerate(test_cases)
    ]

    executed = len(results)
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = executed - passed

    if executed == 0:
        verdict = "SKIPPED"
    elif failed == 0:
        verdict = "PASS"
    elif failed <= executed // 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return results, executed, passed, failed, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent25_data: dict | None,
    agent26_data: dict | None,
    executed: int,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    env_ready = (agent25_data or {}).get("env_ready", False)
    if env_ready:
        scorer.add("env_ready_for_execution", True, +10)
    else:
        scorer.add("env_not_ready", 0, -15)

    test_count = (agent26_data or {}).get("crt_test_count", 0)
    if test_count > 0:
        scorer.add("crt_test_cases_available", test_count, +8)
    else:
        scorer.add("no_crt_test_cases", 0, -10)

    if executed > 0:
        scorer.add("tests_executed", executed, +5)
    elif verdict == "SKIPPED":
        scorer.add("execution_skipped", 0, -10)

    if verdict == "FAIL":
        scorer.add("tests_failed", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a CRT execution narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    executed: int,
    passed: int,
    failed: int,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Tests executed: {executed}\n"
        f"Tests passed: {passed}\n"
        f"Tests failed: {failed}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
