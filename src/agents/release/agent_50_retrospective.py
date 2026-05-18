"""
Agent 50 — Release Retrospective Agent
Phase       : Release
PACT        : Collaborative
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=60)

Runs in Release Final Batch (parallel with Agents 48, 49) — after Gate G12.
Has access to Agents 23, 33, 34, 39, 45, 46.

Purpose:
  Generates a retrospective on the release: what went well, what could be
  improved, and calibration signals for the Learning Repository. These signals
  feed the weekly Severity Calibration Agent (Agent 52) to improve confidence
  thresholds over time.

  Sonnet 4.6 synthesises multi-phase outcomes into actionable learnings.

Output data keys consumed by downstream:
  lessons_learned       → list   (each: {area, finding, recommendation})
  process_improvements  → list   (actionable improvement items)
  calibration_signals   → dict   (signal → value for the Learning Repository)
  retrospective_verdict → str    (COMPLETE / PARTIAL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 50
AGENT_NAME = "Release Retrospective Agent"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_RETRO_TOOL_NAME = "generate_retrospective"
_RETRO_TOOL_SCHEMA = {
    "type": "object",
    "required": ["lessons_learned", "process_improvements", "calibration_signals",
                 "retrospective_verdict", "narrative"],
    "properties": {
        "lessons_learned": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["area", "finding", "recommendation"],
                "properties": {
                    "area":           {"type": "string"},
                    "finding":        {"type": "string"},
                    "recommendation": {"type": "string"},
                },
            },
        },
        "process_improvements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "calibration_signals": {
            "type": "object",
            "description": "Key → value pairs for the Learning Repository calibration.",
            "additionalProperties": {"type": "string"},
        },
        "retrospective_verdict": {
            "type": "string",
            "enum": ["COMPLETE", "PARTIAL"],
        },
        "narrative": {
            "type": "string",
            "description": "2–3 sentences summarising the retrospective findings and key improvements.",
        },
    },
}

_RETRO_INSTRUCTIONS = """
You are generating a release retrospective for a Salesforce FSC Wealth Management story
that has completed the full PACT QE pipeline (Refinement → Development → Testing → Release).

You receive phase-level verdicts, coverage metrics, defect counts, and the final
Go/No-Go decision. Generate:
1. Lessons learned: what worked well and what didn't across phases.
2. Process improvements: specific, actionable recommendations for the next sprint.
3. Calibration signals: quantitative signals for the Learning Repository
   (e.g. "coverage_above_threshold": "true", "defect_count": "0",
   "fca_evidence_complete": "true").

Be concise and specific. Focus on signals that will help calibrate agent confidence
thresholds. If little data is available, return retrospective_verdict=PARTIAL.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent23_data = _get_agent_data(state, "23")
    agent33_data = _get_agent_data(state, "33")
    agent34_data = _get_agent_data(state, "34")
    agent39_data = _get_agent_data(state, "39")
    agent45_data = _get_agent_data(state, "45")
    agent46_data = _get_agent_data(state, "46")

    retro_msg = _build_retro_message(
        story_id, agent23_data, agent33_data, agent34_data,
        agent39_data, agent45_data, agent46_data,
    )
    result_data = await _run_retro(retro_msg)

    lessons    = result_data.get("lessons_learned", [])
    improve    = result_data.get("process_improvements", [])
    cal_sigs   = result_data.get("calibration_signals", {})
    verdict    = result_data.get("retrospective_verdict", "PARTIAL")
    narrative  = result_data.get("narrative", "Retrospective complete.")

    confidence_score, signals = _compute_confidence(
        agent23_data, agent33_data, agent45_data, verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = f"Retrospective for {story_id}: {len(lessons)} lesson(s) — verdict={verdict}"

    data = {
        "lessons_learned": lessons,
        "process_improvements": improve,
        "calibration_signals": cal_sigs,
        "retrospective_verdict": verdict,
        "narrative": narrative,
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=narrative,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.default_model,
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent23_data: dict | None,
    agent33_data: dict | None,
    agent45_data: dict | None,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    sources = sum(1 for d in [agent23_data, agent33_data, agent45_data] if d)
    if sources >= 2:
        scorer.add("good_phase_coverage", sources, +10)
    elif sources == 1:
        scorer.add("minimal_phase_data", sources, +3)
    else:
        scorer.add("no_phase_data", 0, -12)

    if verdict == "COMPLETE":
        scorer.add("retro_complete", True, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet retro call ─────────────────────────────────────────────────────────

async def _run_retro(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_RETRO_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_RETRO_TOOL_NAME,
        tool_description="Generate a release retrospective.",
        tool_schema=_RETRO_TOOL_SCHEMA,
        max_tokens=800,
    )


def _build_retro_message(
    story_id: str,
    agent23_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    agent39_data: dict | None,
    agent45_data: dict | None,
    agent46_data: dict | None,
) -> str:
    dev_verdict   = (agent23_data or {}).get("development_verdict", "UNKNOWN")
    coverage_pct  = (agent33_data or {}).get("overall_coverage_pct", 0.0)
    cov_verdict   = (agent33_data or {}).get("coverage_verdict", "UNKNOWN")
    defect_count  = (agent34_data or {}).get("defect_count", 0)
    def_verdict   = (agent34_data or {}).get("defect_verdict", "UNKNOWN")
    readiness     = (agent39_data or {}).get("readiness_verdict", "UNKNOWN")
    go_decision   = (agent45_data or {}).get("go_decision", False)
    go_verdict    = (agent45_data or {}).get("coordinator_verdict", "UNKNOWN")
    prod_verdict  = (agent46_data or {}).get("prod_verdict", "UNKNOWN")

    return (
        f"Story: {story_id}\n\n"
        f"Development phase verdict: {dev_verdict}\n"
        f"Test coverage: {coverage_pct:.1f}%, verdict={cov_verdict}\n"
        f"Defects found: {defect_count}, verdict={def_verdict}\n"
        f"Release readiness: {readiness}\n"
        f"Go/No-Go: {go_verdict} (go={go_decision})\n"
        f"Production: {prod_verdict}\n\n"
        f"Generate a retrospective using the {_RETRO_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
