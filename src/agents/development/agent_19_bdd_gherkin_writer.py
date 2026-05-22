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

from src.agents.base import ShapleyAttributor, TierBScorer, _ta_mult, build_system, call_with_tool, get_agent_result
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 19
AGENT_NAME = "BDD Gherkin Writer"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_GHERKIN_TOOL_NAME = "generate_gherkin_scenarios"
_GHERKIN_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "scenarios", "scenario_count", "gherkin_verdict",
        "fca_coverage_present", "vulnerable_customer_coverage_present", "coverage_gaps",
    ],
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
                        "description": "Tags e.g. @fca, @negative, @boundary, @smoke, @vulnerable_customer, @bulk",
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
        "vulnerable_customer_coverage_present": {
            "type": "boolean",
            "description": (
                "True if at least one scenario tagged @vulnerable_customer is present. "
                "Must be True when vulnerable_customer_impact=True from Agent 04. "
                "A story with vulnerable_customer_impact=True but no @vulnerable_customer "
                "scenario should receive gherkin_verdict=PARTIAL, not PASS."
            ),
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
   @boundary, @vulnerable_customer, @bulk as appropriate.
5. Steps must be concrete and testable — no vague language like "the system works correctly".
6. Use data tables or Examples where multiple input variations apply.
7. If ACs are ambiguous or missing, note the gap in coverage_gaps and set verdict PARTIAL.
8. If no ACs are available at all, set verdict INCOMPLETE.
9. Vulnerable Customer: when vulnerable_customer_impact=True, generate at least one scenario
   tagged @vulnerable_customer covering the FG21/1 pathway. Do not rely on AC text alone —
   the vulnerable customer obligation is mandatory regardless of how ACs are worded.
   A story with vulnerable_customer_impact=True but no @vulnerable_customer scenario must
   have gherkin_verdict=PARTIAL and set vulnerable_customer_coverage_present=False.
10. Bulk risk: when bulk_risk_level=HIGH, generate at least one scenario tagged @bulk
    covering the specified bulk risk factors (e.g. DML governor limits, large data volumes,
    Queueable Apex async patterns). Include volume numbers in the scenario steps.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data, agent3_conf = get_agent_result(state, "3")
    agent4_data = _get_agent_data(state, "4")
    agent5_data, agent5_conf = get_agent_result(state, "5")
    agent10_data = _get_agent_data(state, "10")
    agent13_data, agent13_conf = get_agent_result(state, "13")
    agent16_data = _get_agent_data(state, "16")

    story = await get_story(story_id)
    acs = await get_acceptance_criteria(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    ac_count = len(acs)
    refined_ac_count = (agent5_data or {}).get("ac_clause_count", 0)
    compliance_verdict = (agent10_data or {}).get("coverage_verdict", "")
    objects_in_scope = (agent13_data or {}).get("detected_objects", [])
    vulnerable_customer_impact = (agent4_data or {}).get("vulnerable_customer_impact", False)
    bulk_risk_level = (agent16_data or {}).get("bulk_risk_level", "LOW")
    bulk_risk_factors = (agent16_data or {}).get("bulk_risk_factors", [])

    user_message = _build_prompt(
        story_id=story_id,
        story=story,
        acs=acs,
        fca_class=fca_class,
        refined_ac_count=refined_ac_count,
        compliance_verdict=compliance_verdict,
        objects_in_scope=objects_in_scope,
        vulnerable_customer_impact=vulnerable_customer_impact,
        bulk_risk_level=bulk_risk_level,
        bulk_risk_factors=bulk_risk_factors,
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
    vc_coverage = result.get("vulnerable_customer_coverage_present", False)
    gaps = result.get("coverage_gaps", [])
    bulk_test_scenarios_generated = any("@bulk" in tag for s in scenarios for tag in s.get("tags", []))

    # Coalition Shapley: 3 sources contribute to scenario generation
    agent5_trust = (agent5_data or {}).get("generation_mode_trust", 0.8)
    agent5_shapley_mult = 1.0 if agent5_trust >= 0.8 else 0.5
    attributor = ShapleyAttributor()
    attributor.add_agent("5_ac_clauses", agent5_conf,  ac_count > 0,                  agent5_shapley_mult)
    attributor.add_agent("3_fca_class",  agent3_conf,  agent3_data is not None,       _ta_mult(agent3_conf))
    attributor.add_agent("13_metadata",  agent13_conf, bool((agent13_data or {}).get("detected_objects")), _ta_mult(agent13_conf))
    shapley = attributor.compute()

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
        "vulnerable_customer_coverage_present": vc_coverage,
        "bulk_test_scenarios_generated": bulk_test_scenarios_generated,
        "coverage_gaps": gaps,
        "ac_count": ac_count,
        "shapley_attribution": shapley,
        "ac_source_trust": agent5_trust,
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
    vulnerable_customer_impact: bool = False,
    bulk_risk_level: str = "LOW",
    bulk_risk_factors: list | None = None,
) -> str:
    ac_text = "\n".join(
        f"  AC{i+1}: {ac.get('description', str(ac))}" for i, ac in enumerate(acs)
    ) or "  (no acceptance criteria available)"

    vc_line = (
        "Vulnerable Customer Impact: TRUE — generate ≥1 scenario tagged @vulnerable_customer (FG21/1 mandatory)"
        if vulnerable_customer_impact
        else "Vulnerable Customer Impact: FALSE"
    )

    bulk_factors_text = ", ".join(bulk_risk_factors or []) or "none"
    bulk_line = (
        f"Bulk Risk Level: {bulk_risk_level} — generate ≥1 scenario tagged @bulk "
        f"covering: {bulk_factors_text}"
        if bulk_risk_level == "HIGH"
        else f"Bulk Risk Level: {bulk_risk_level}"
    )

    return (
        f"Story ID: {story_id}\n"
        f"Title: {story.get('summary', 'N/A')}\n"
        f"FCA Classification: {fca_class}\n"
        f"{vc_line}\n"
        f"{bulk_line}\n"
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
