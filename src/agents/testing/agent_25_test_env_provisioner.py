"""
Agent 25 — Test Environment Provisioner
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs in Testing Batch 1 (parallel with Agents 24, 32).
Has access to Agents 21, 22.

Purpose:
  Checks that the test environment (sandbox) is provisioned and ready
  for test execution. Verifies that Agent 22's sandbox health signals
  are still valid, test data has been seeded (from Agent 21 strategy),
  and the CRT environment connection is available.

  In production, this calls a Copado sandbox health API; the current
  implementation reasons from upstream agent signals as a stub.

Output data keys consumed by downstream:
  env_ready              → bool (Gate G5 prerequisite)
  env_verdict            → str  (READY / DEGRADED / BLOCKED)
  env_blockers           → list (issues blocking test execution)
  crt_connected          → bool (Agent 27 CRT Execution prerequisite)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 25
AGENT_NAME = "Test Environment Provisioner"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_env_provisioning_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "provisioning_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the test environment readiness. "
                "Mention sandbox health, data seeding status, CRT connectivity, "
                "and what must be resolved before test execution can begin."
            ),
        },
        "provisioning_concern": {
            "type": "string",
            "enum": ["none", "sandbox_degraded", "data_not_seeded",
                     "crt_unavailable", "multiple"],
            "description": "Primary provisioning concern, or 'none' if environment is ready.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce test environment provisioning check.
You will receive sandbox health signals, test data seeding status, and CRT connectivity info.
Write a clear 2–3 sentence narrative explaining whether the environment is ready for automated
test execution, what issues exist, and what must be resolved. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent21_data = _get_agent_data(state, "21")
    agent22_data = _get_agent_data(state, "22")

    # ── Deterministic check ───────────────────────────────────────────────────
    env_ready, verdict, blockers, crt_connected = _check_environment(
        agent21_data, agent22_data,
    )

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(
        story_id, agent21_data, agent22_data, blockers, verdict,
    )
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent21_data, agent22_data, env_ready)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Test environment for {story_id}: ready={env_ready}, "
        f"crt_connected={crt_connected}, {len(blockers)} blocker(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Test Environment Provisioner assessed sandbox readiness.")

    data = {
        "env_ready": env_ready,
        "env_verdict": verdict,
        "env_blockers": blockers,
        "crt_connected": crt_connected,
        "provisioning_concern": trace.get("provisioning_concern", "none"),
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


# ── Deterministic environment check ──────────────────────────────────────────

def _check_environment(
    agent21_data: dict | None,
    agent22_data: dict | None,
) -> tuple[bool, str, list[str], bool]:
    """Returns (env_ready, verdict, blockers, crt_connected)."""
    blockers: list[str] = []

    # Sandbox health from Agent 22
    sandbox_ready = (agent22_data or {}).get("sandbox_ready", True)
    sandbox_verdict = (agent22_data or {}).get("sandbox_verdict", "UNKNOWN")
    sandbox_blockers = (agent22_data or {}).get("sandbox_blockers", [])

    if not sandbox_ready:
        blockers.append(
            f"Sandbox not ready (Agent 22: {sandbox_verdict}): {sandbox_blockers}"
        )
    elif sandbox_verdict == "DEGRADED":
        blockers.append("Sandbox is DEGRADED — intermittent failures possible")

    # Test data seeding from Agent 21
    data_verdict = (agent21_data or {}).get("data_verdict", "")

    if data_verdict == "INCOMPLETE":
        blockers.append("Test data strategy INCOMPLETE — no records to seed")
    elif data_verdict == "WARN":
        blockers.append("Test data strategy has gaps — some scenarios may lack data")

    # CRT connectivity — stub: assume connected if sandbox is ready
    # In production this calls the Copado CRT health endpoint
    crt_connected = sandbox_ready and sandbox_verdict != "BLOCKED"

    if not crt_connected:
        blockers.append("CRT not connected — automated test execution unavailable")

    # Verdict
    critical = [b for b in blockers if "not ready" in b or "INCOMPLETE" in b
                or "not connected" in b]
    if critical:
        verdict = "BLOCKED"
        env_ready = False
    elif blockers:
        verdict = "DEGRADED"
        env_ready = True  # degraded but not blocked
    else:
        verdict = "READY"
        env_ready = True

    return env_ready, verdict, blockers, crt_connected


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent21_data: dict | None,
    agent22_data: dict | None,
    env_ready: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    if agent22_data:
        scorer.add("sandbox_health_signal_available", True, +10)
    else:
        scorer.add("no_sandbox_health_signal", 0, -8)

    if agent21_data:
        scorer.add("test_data_strategy_available", True, +7)
    else:
        scorer.add("no_test_data_strategy", 0, -5)

    if not env_ready:
        scorer.add("environment_not_ready", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a test environment provisioning narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    agent21_data: dict | None,
    agent22_data: dict | None,
    blockers: list[str],
    verdict: str,
) -> str:
    sandbox_verdict = (agent22_data or {}).get("sandbox_verdict", "UNKNOWN")
    health_score = (agent22_data or {}).get("sandbox_health_score", 0)
    data_verdict = (agent21_data or {}).get("data_verdict", "UNKNOWN")
    seed_count = (agent21_data or {}).get("seed_record_count", 0)
    return (
        f"Story: {story_id}\n"
        f"Sandbox verdict: {sandbox_verdict} (health={health_score}/100)\n"
        f"Test data verdict: {data_verdict} ({seed_count} seed record type(s))\n"
        f"Environment blockers: {blockers or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
