"""
Agent 19 — BDD Gherkin Writer
Phase       : Development
PACT        : Proactive
Classification: True AI Agent (Claude Sonnet 4.6)
Confidence  : Tier B (base=70)

Runs sequentially after Batch 3.
Has access to Agents 3, 5, 10, 13.

Purpose:
  Generates BDD Gherkin scenarios from the story's acceptance criteria,
  FCA classification, and changed metadata scope. Each AC becomes one or
  more Given/When/Then scenarios. FCA-regulated stories require negative
  test scenarios and boundary conditions.

  This is a True AI agent — Sonnet 4.6 handles the full generation.
  No deterministic pre-processing; structured output via tool use.

Output data keys consumed by downstream:
  gherkin_scenarios     → list (Agent 27 CRT Execution)
  scenario_count        → int  (Agent 23 story-to-code tracer)
  gherkin_verdict       → str  (PASS / PARTIAL / INCOMPLETE)
  fca_coverage_present  → bool (Gate G2 — regulated stories need negative tests)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 19
AGENT_NAME = "BDD Gherkin Writer"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_GHERKIN_TOOL_NAME = "generate_gherkin_scenarios"
_GHERKIN_TOOL_SCHEMA = {
    "type": "object",
    "required": ["scenarios", "scenario_count", "gherkin_verdict", "fca_coverage_present", "coverage_gaps"],
    "properties": {
        "scenarios": {
            "type": "array",
            "description": "List of BDD Gherkin scenarios, each as a complete Feature/Scenario block.",
            "items": {
                "type": "object",
                "required": ["title", "tags", "steps"],
                "properties": {
                    "title": {"type": "string", "description": "Scenario title"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags e.g. @fca, @negative, @boundary, @smoke",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps in Given/When/Then format",
                    },
                },
            },
        },
        "scenario_count": {
            "type": "integer",
            "description": "Total number of scenarios generated.",
        },
        "gherkin_verdict": {
            "type": "string",
            "enum": ["PASS", "PARTIAL", "INCOMPLETE"],
            "description": (
                "PASS: All ACs covered with appropriate depth. "
                "PARTIAL: Coverage partial or boundary tests missing. "
                "INCOMPLETE: Unable to generate scenarios from available ACs."
            ),
        },
        "fca_coverage_present": {
            "type": "boolean",
            "description": "True if negative/boundary FCA test scenarios are included.",
        },
        "coverage_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ACs or behaviours not covered by generated scenarios.",
        },
    },
}

_GHERKIN_INSTRUCTIONS = """
You are a BDD Gherkin scenario writer for a Salesforce FSC Wealth Management platform
operating under FCA regulation. Your task is to convert acceptance criteria into
well-structured Gherkin Feature/Scenario blocks.

Rules:
1. Each acceptance criterion becomes at least one Scenario.
2. For FCA-classified HIGH or MEDIUM stories: add @negative and @boundary scenarios
   covering rejection cases, limit breaches, and regulatory boundary conditions.
3. Use FSC domain terminology: Suitability, Risk Profile, Financial Account, Goal,
   Financial Holding, Vulnerable Customer, Consumer Duty.
4. Tag scenarios: @smoke (happy path), @regression, @fca (regulatory), @negative,
   @boundary as appropriate.
5. Steps must be concrete and testable — no vague language like "the system works correctly".
6. Use data tables or Examples where multiple input variations apply.
7. If ACs are ambiguous or missing, note the gap in coverage_gaps and set verdict PARTIAL.
8. If no ACs are available at all, set verdict INCOMPLETE.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data = _get_agent_data(state, "3")
    agent5_data = _get_agent_data(state, "5")
    agent10_data = _get_agent_data(state, "10")
    agent13_data = _get_agent_data(state, "13")

    story = await get_story(story_id)
    acs = await get_acceptance_criteria(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    ac_count = len(acs)
    refined_ac_count = (agent5_data or {}).get("ac_clause_count", 0)
    compliance_verdict = (agent10_data or {}).get("coverage_verdict", "")
    objects_in_scope = (agent13_data or {}).get("detected_objects", [])

    user_message = _build_prompt(
        story_id=story_id,
        story=story,
        acs=acs,
        fca_class=fca_class,
        refined_ac_count=refined_ac_count,
        compliance_verdict=compliance_verdict,
        objects_in_scope=objects_in_scope,
    )

    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_GHERKIN_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_GHERKIN_TOOL_NAME,
        tool_description="Generate BDD Gherkin scenarios for the story's acceptance criteria.",
        tool_schema=_GHERKIN_TOOL_SCHEMA,
        max_tokens=2000,
    )

    scenarios = result.get("scenarios", [])
    scenario_count = result.get("scenario_count", len(scenarios))
    verdict = result.get("gherkin_verdict", "INCOMPLETE")
    fca_coverage = result.get("fca_coverage_present", False)
    gaps = result.get("coverage_gaps", [])

    confidence_score, signals = _compute_confidence(
        acs, agent3_data, agent5_data, scenario_count, fca_class, fca_coverage,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"BDD Gherkin for {story_id}: {scenario_count} scenario(s) generated from "
        f"{ac_count} AC(s) — verdict={verdict}"
    )
    why = (
        f"Generated {scenario_count} Gherkin scenario(s) for a {fca_class}-FCA story. "
        f"FCA coverage: {'present' if fca_coverage else 'absent'}. "
        f"Gaps: {len(gaps)}."
    )

    data = {
        "gherkin_scenarios": scenarios,
        "scenario_count": scenario_count,
        "gherkin_verdict": verdict,
        "fca_coverage_present": fca_coverage,
        "coverage_gaps": gaps,
        "ac_count": ac_count,
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
        model_used=settings.default_model,
    )


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    acs: list,
    agent3_data: dict | None,
    agent5_data: dict | None,
    scenario_count: int,
    fca_class: str,
    fca_coverage: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=70)

    if acs:
        scorer.add("acs_available", len(acs), +8)
    else:
        scorer.add("no_acs_available", 0, -20)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if agent5_data:
        scorer.add("refined_ac_baseline_available", True, +5)

    if scenario_count > 0:
        scorer.add("scenarios_generated", scenario_count, +5)
    else:
        scorer.add("no_scenarios_generated", 0, -15)

    if fca_class in ("HIGH", "MEDIUM") and not fca_coverage:
        scorer.add("regulated_story_missing_fca_scenarios", True, -10)

    if fca_class in ("HIGH", "MEDIUM") and fca_coverage:
        scorer.add("regulated_story_has_fca_scenarios", True, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    story_id: str,
    story: dict,
    acs: list,
    fca_class: str,
    refined_ac_count: int,
    compliance_verdict: str,
    objects_in_scope: list,
) -> str:
    ac_text = "\n".join(
        f"  AC{i+1}: {ac.get('description', str(ac))}" for i, ac in enumerate(acs)
    ) or "  (no acceptance criteria available)"

    return (
        f"Story ID: {story_id}\n"
        f"Title: {story.get('summary', 'N/A')}\n"
        f"FCA Classification: {fca_class}\n"
        f"Acceptance Criteria ({len(acs)} present, {refined_ac_count} expected from refinement):\n"
        f"{ac_text}\n"
        f"AC Compliance Verdict: {compliance_verdict or 'N/A'}\n"
        f"Metadata in scope: {objects_in_scope or ['not yet determined']}\n\n"
        f"Generate BDD Gherkin scenarios using the {_GHERKIN_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
