"""
Agent 28 — CRT Self-Heal Reviewer
Phase       : Testing
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=60)

Runs in Testing Batch 3 (parallel with Agents 31, 37).
Has access to Agent 27 (CRT execution results).

Purpose:
  Reviews CRT self-healed test steps for correctness. Copado Robotic
  Testing can auto-heal broken locators — this agent flags self-healed
  steps that may have healed incorrectly (e.g. healed to a wrong element)
  and flags them for manual review.

  Self-healed tests that pass may be masking regressions. This agent
  ensures the QE engineer reviews each self-heal before accepting results.

Output data keys consumed by downstream:
  self_healed_count      → int  (informational)
  suspect_self_heals     → list (tests needing manual review)
  self_heal_verdict      → str  (PASS / WARN / REVIEW_REQUIRED)
  auto_heal_risk         → str  (LOW / MEDIUM / HIGH)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 28
AGENT_NAME = "CRT Self-Heal Reviewer"

_SUSPECT_HEAL_THRESHOLD = 2  # more than this many self-heals → HIGH risk

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_self_heal_review_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "heal_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the self-heal review. "
                "Note how many tests self-healed, whether any are suspect, "
                "and what the QE engineer must manually verify."
            ),
        },
        "heal_concern": {
            "type": "string",
            "enum": ["none", "suspect_heals", "excessive_healing", "fca_test_healed"],
            "description": "Primary self-heal concern.",
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are reviewing Copado Robotic Testing self-heal events for a Salesforce FSC platform.
Self-healed tests may have healed to wrong elements, masking regressions.
Write a clear 2–3 sentence narrative explaining which tests self-healed, whether they are
suspect (FCA-tagged tests, complex assertions), and what the QE engineer must check manually.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent27_data = _get_agent_data(state, "27")

    # ── Deterministic review ─────────────────────────────────────────────────
    healed_count, suspect, risk, verdict = _review_self_heals(agent27_data)

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_msg = _build_trace_message(story_id, healed_count, suspect, risk, verdict)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent27_data, healed_count, verdict)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Self-heal review for {story_id}: {healed_count} self-healed test(s), "
        f"{len(suspect)} suspect — verdict={verdict}"
    )
    why = trace.get("narrative", "CRT Self-Heal Reviewer assessed auto-healed tests.")

    data = {
        "self_healed_count": healed_count,
        "suspect_self_heals": suspect,
        "self_heal_verdict": verdict,
        "auto_heal_risk": risk,
        "heal_concern": trace.get("heal_concern", "none"),
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


# ── Deterministic self-heal review ────────────────────────────────────────────

def _review_self_heals(
    agent27_data: dict | None,
) -> tuple[int, list[str], str, str]:
    """Returns (healed_count, suspect_tests, auto_heal_risk, verdict)."""
    crt_results = (agent27_data or {}).get("crt_results", [])

    healed = [r for r in crt_results if r.get("self_healed")]
    healed_count = len(healed)

    # Flag FCA-tagged or complex tests that self-healed as suspect
    suspect = []
    for r in healed:
        test_id = r.get("test_id", "")
        title = r.get("title", "").lower()
        # Suspect if title contains FCA/regulatory keywords
        if any(kw in title for kw in ("fca", "suitability", "regulatory", "vulnerable",
                                       "consumer duty", "cobs", "compliance")):
            suspect.append(test_id)
        # Or if test was previously flagged as FCA (tagged @fca)
        if "fca" in " ".join(r.get("tags", [])).lower():
            if test_id not in suspect:
                suspect.append(test_id)

    # Risk level
    if healed_count == 0:
        risk = "LOW"
    elif len(suspect) > 0 or healed_count > _SUSPECT_HEAL_THRESHOLD:
        risk = "HIGH"
    else:
        risk = "MEDIUM"

    # Verdict
    if len(suspect) > 0:
        verdict = "REVIEW_REQUIRED"
    elif healed_count > _SUSPECT_HEAL_THRESHOLD:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return healed_count, suspect, risk, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent27_data: dict | None,
    healed_count: int,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    executed = (agent27_data or {}).get("tests_executed", 0)
    if executed > 0:
        scorer.add("crt_results_available", executed, +12)
    else:
        scorer.add("no_crt_results", 0, -10)

    if healed_count == 0:
        scorer.add("no_self_heals", True, +5)
    elif healed_count > _SUSPECT_HEAL_THRESHOLD:
        scorer.add("excessive_self_heals", healed_count, -8)

    if verdict == "REVIEW_REQUIRED":
        scorer.add("suspect_heals_found", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a CRT self-heal review narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    healed_count: int,
    suspect: list[str],
    risk: str,
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Self-healed tests: {healed_count}\n"
        f"Suspect self-heals: {suspect or ['none']}\n"
        f"Auto-heal risk: {risk}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
