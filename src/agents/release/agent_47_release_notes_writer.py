"""
Agent 47 — Release Notes Writer
Phase       : Release
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=62)

Runs in Release Batch 1 (parallel with Agent 39).
Has access to Agents 5, 19, 23, 33.

Purpose:
  Generates human-readable release notes from the story's acceptance criteria,
  Gherkin scenarios, development summary, and coverage report. Produces both
  technical notes for the QE team and a regulatory summary for Compliance Officers.

  Sonnet 4.6 synthesises structured inputs into polished release documentation.

Output data keys consumed by downstream:
  release_title      → str   (concise title for the release)
  release_notes      → str   (full human-readable notes)
  regulatory_notes   → str   (FCA/compliance-specific section)
  notes_verdict      → str   (COMPLETE / PARTIAL / FAILED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 47
AGENT_NAME = "Release Notes Writer"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_NOTES_TOOL_NAME = "write_release_notes"
_NOTES_TOOL_SCHEMA = {
    "type": "object",
    "required": ["release_title", "release_notes", "regulatory_notes",
                 "notes_verdict", "narrative"],
    "properties": {
        "release_title": {
            "type": "string",
            "description": "Concise title (max 80 chars) for this release.",
        },
        "release_notes": {
            "type": "string",
            "description": (
                "Full release notes for QE and development teams. "
                "Include: what changed, acceptance criteria covered, "
                "test coverage summary, and any known limitations."
            ),
        },
        "regulatory_notes": {
            "type": "string",
            "description": (
                "FCA and compliance-specific section for the Compliance Officer. "
                "Reference regulatory rules covered, Consumer Duty implications, "
                "and any FCA-relevant test scenarios run."
            ),
        },
        "notes_verdict": {
            "type": "string",
            "enum": ["COMPLETE", "PARTIAL", "FAILED"],
            "description": (
                "COMPLETE when all sections are fully populated, "
                "PARTIAL when some source data is missing, "
                "FAILED if critical story information is absent."
            ),
        },
        "narrative": {
            "type": "string",
            "description": "1–2 sentences confirming the release notes were generated and what they cover.",
        },
    },
}

_NOTES_INSTRUCTIONS = """
You are generating release notes for a Salesforce FSC Wealth Management feature release
under FCA regulatory oversight.

You receive the story's acceptance criteria, BDD Gherkin scenarios, development phase
summary, and test coverage figures.

Write release notes that are:
- Clear and concise for QE engineers and developers
- Accurate: reflect what was actually tested and covered
- Compliant: include a regulatory section calling out FCA-relevant changes
- Honest about any coverage gaps or limitations

If source data is sparse, still produce a best-effort document and mark verdict=PARTIAL.
If no meaningful data is available, mark verdict=FAILED.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent5_data  = _get_agent_data(state, "5")
    agent19_data = _get_agent_data(state, "19")
    agent23_data = _get_agent_data(state, "23")
    agent33_data = _get_agent_data(state, "33")

    notes_msg = _build_notes_message(story_id, agent5_data, agent19_data, agent23_data, agent33_data)
    result_data = await _run_notes(notes_msg)

    title      = result_data.get("release_title", f"{story_id} Release")
    notes      = result_data.get("release_notes", "")
    reg_notes  = result_data.get("regulatory_notes", "")
    verdict    = result_data.get("notes_verdict", "PARTIAL")
    narrative  = result_data.get("narrative", "Release notes generated.")

    confidence_score, signals = _compute_confidence(agent5_data, agent19_data, agent23_data, verdict)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = f"Release notes for {story_id}: verdict={verdict}"

    data = {
        "release_title": title,
        "release_notes": notes,
        "regulatory_notes": reg_notes,
        "notes_verdict": verdict,
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
    agent5_data: dict | None,
    agent19_data: dict | None,
    agent23_data: dict | None,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    sources = sum(1 for d in [agent5_data, agent19_data, agent23_data] if d)
    if sources >= 2:
        scorer.add("good_source_coverage", sources, +10)
    elif sources == 1:
        scorer.add("minimal_source_data", sources, +2)
    else:
        scorer.add("no_source_data", 0, -12)

    if verdict == "COMPLETE":
        scorer.add("notes_complete", True, +5)
    elif verdict == "FAILED":
        scorer.add("notes_failed", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet notes call ─────────────────────────────────────────────────────────

async def _run_notes(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_NOTES_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_NOTES_TOOL_NAME,
        tool_description="Write release notes for this story.",
        tool_schema=_NOTES_TOOL_SCHEMA,
        max_tokens=800,
    )


def _build_notes_message(
    story_id: str,
    agent5_data: dict | None,
    agent19_data: dict | None,
    agent23_data: dict | None,
    agent33_data: dict | None,
) -> str:
    acs            = (agent5_data or {}).get("acceptance_criteria", [])
    ac_count       = (agent5_data or {}).get("ac_count", len(acs))
    scenarios      = (agent19_data or {}).get("scenarios", "")
    scenario_count = (agent19_data or {}).get("scenario_count", 0)
    dev_verdict    = (agent23_data or {}).get("development_verdict", "UNKNOWN")
    dev_summary    = (agent23_data or {}).get("narrative", "")
    coverage_pct   = (agent33_data or {}).get("overall_coverage_pct", 0.0)
    cov_verdict    = (agent33_data or {}).get("coverage_verdict", "UNKNOWN")

    return (
        f"Story: {story_id}\n\n"
        f"Acceptance Criteria ({ac_count} ACs):\n{acs or '(not available)'}\n\n"
        f"BDD Gherkin Scenarios ({scenario_count} scenarios):\n{scenarios or '(not available)'}\n\n"
        f"Development Phase: verdict={dev_verdict}\n{dev_summary}\n\n"
        f"Test Coverage: {coverage_pct:.1f}%, verdict={cov_verdict}\n\n"
        f"Generate release notes using the {_NOTES_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
