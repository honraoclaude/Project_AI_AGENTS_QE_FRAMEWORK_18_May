"""
Agent 42 — Dry-Run Agent
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=58)

Runs sequentially after Gate G8.
Has access to Agents 25, 41.

Purpose:
  Simulates a Salesforce deployment dry-run by checking environment readiness
  and change set integrity. In production this invokes Copado's deploy API
  in validate-only mode; the current stub infers success from upstream signals.

Output data keys consumed by downstream:
  dry_run_success  → bool   (Gate G9 input — must be True for smoke tests to run)
  dry_run_errors   → list   (deployment errors found in dry-run)
  dry_run_verdict  → str    (PASS / FAIL / SKIPPED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 42
AGENT_NAME = "Dry-Run Agent"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_dry_run_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "dry_run_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the dry-run result. "
                "State whether the deployment simulation passed, any errors found, "
                "and what must be resolved before smoke testing can proceed."
            ),
        },
        "dry_run_concern": {
            "type": "string",
            "enum": ["none", "deployment_error", "env_not_ready",
                     "change_set_invalid", "validation_timeout"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce deployment dry-run
(validate-only mode) in an FSC Wealth Management CI/CD pipeline.
You will receive environment readiness and change set integrity signals.
Write a clear 2–3 sentence narrative explaining whether the dry-run succeeded,
what errors were found if any, and what must be fixed before smoke tests can run.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent25_data = _get_agent_data(state, "25")
    agent41_data = _get_agent_data(state, "41")

    success, errors, verdict = _simulate_dry_run(agent25_data, agent41_data)

    trace_msg = _build_trace_message(story_id, success, errors, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent25_data, agent41_data, success)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = f"Dry-run for {story_id}: success={success}, {len(errors)} error(s) — verdict={verdict}"
    why = trace.get("narrative", "Dry-Run Agent simulated deployment.")

    data = {
        "dry_run_success": success,
        "dry_run_errors": errors,
        "dry_run_verdict": verdict,
        "dry_run_concern": trace.get("dry_run_concern", "none"),
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


# ── Deterministic dry-run simulation ─────────────────────────────────────────

def _simulate_dry_run(
    agent25_data: dict | None,
    agent41_data: dict | None,
) -> tuple[bool, list[str], str]:
    """Returns (success, errors, verdict)."""
    errors: list[str] = []

    env_verdict     = (agent25_data or {}).get("env_verdict", "READY")
    env_ready       = (agent25_data or {}).get("env_ready", True)
    integrity_valid = (agent41_data or {}).get("integrity_valid", True)
    integrity_v     = (agent41_data or {}).get("integrity_verdict", "PASS")

    if not env_ready or env_verdict == "BLOCKED":
        errors.append(f"Staging environment not ready: {env_verdict}")

    if not integrity_valid or integrity_v == "FAIL":
        errors.append(f"Change set integrity failed: {integrity_v}")

    success = len(errors) == 0

    if not agent25_data and not agent41_data:
        verdict = "SKIPPED"
        success = False
    elif success:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return success, errors, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent25_data: dict | None,
    agent41_data: dict | None,
    success: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    if agent25_data:
        scorer.add("env_data_available", True, +8)
    else:
        scorer.add("no_env_data", 0, -8)

    if agent41_data:
        scorer.add("integrity_data_available", True, +8)
    else:
        scorer.add("no_integrity_data", 0, -5)

    if success:
        scorer.add("dry_run_succeeded", True, +5)
    else:
        scorer.add("dry_run_failed", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a dry-run deployment narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    success: bool,
    errors: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Dry-run success: {success}\n"
        f"Errors: {errors or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
