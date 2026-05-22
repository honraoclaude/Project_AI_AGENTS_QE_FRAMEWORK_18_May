"""
Agent 26 — CRT Scenario Designer
Phase       : Testing
PACT        : Proactive
Classification: True AI Agent (Claude Sonnet 4.6)
Confidence  : Tier B (base=68)

Runs in Testing Batch 2 (parallel with Agents 29, 30).
Has access to Agents 3, 19, 21, 24, 32.

Purpose:
  Converts Gherkin scenarios into Copado Robotic Testing (CRT) automation
  scripts. Each Gherkin scenario becomes a structured CRT test case with
  step actions, locators, and data references. FCA scenarios get additional
  assertion steps for regulatory compliance.

  True AI (Sonnet 4.6) generates the CRT test structures from Gherkin.

Output data keys consumed by downstream:
  crt_test_cases       → list (Agent 27 CRT Execution input)
  crt_test_count       → int  (coverage metric)
  automation_coverage  → float (% of Gherkin scenarios automated)
  crt_design_verdict   → str  (PASS / PARTIAL / INCOMPLETE)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 26
AGENT_NAME = "CRT Scenario Designer"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_CRT_TOOL_NAME = "design_crt_test_cases"
_CRT_TOOL_SCHEMA = {
    "type": "object",
    "required": ["crt_test_cases", "crt_test_count", "automation_coverage",
                 "crt_design_verdict", "design_notes"],
    "properties": {
        "crt_test_cases": {
            "type": "array",
            "description": "CRT test case definitions derived from Gherkin scenarios.",
            "items": {
                "type": "object",
                "required": ["test_id", "title", "tags", "steps", "data_references"],
                "properties": {
                    "test_id": {"type": "string"},
                    "title": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["action", "target", "value"],
                            "properties": {
                                "action": {"type": "string",
                                           "description": "CRT action: click, input, assert, navigate, wait"},
                                "target": {"type": "string",
                                           "description": "Salesforce UI locator or API endpoint"},
                                "value": {"type": "string",
                                          "description": "Input value or expected assertion value"},
                            },
                        },
                    },
                    "data_references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Seed data record names this test depends on",
                    },
                },
            },
        },
        "crt_test_count": {"type": "integer"},
        "automation_coverage": {
            "type": "number",
            "description": "Percentage of Gherkin scenarios covered by CRT tests (0–100).",
        },
        "crt_design_verdict": {
            "type": "string",
            "enum": ["PASS", "PARTIAL", "INCOMPLETE"],
        },
        "design_notes": {
            "type": "string",
            "description": "Notes on scenarios that could not be fully automated.",
        },
    },
}

_CRT_INSTRUCTIONS = """
You are a Copado Robotic Testing (CRT) automation engineer for a Salesforce FSC platform.
Your task is to convert Gherkin BDD scenarios into CRT test case structures.

Rules:
1. Each Gherkin scenario becomes one CRT test case.
2. Given steps → navigation or data setup actions.
3. When steps → user interaction actions (click, input).
4. Then steps → assertion actions verifying expected outcomes.
5. FCA-tagged scenarios (@fca, @negative) must include explicit compliance assertions.
6. Reference seed data records by name from the test data strategy.
7. Use Salesforce Lightning UI locators (e.g., "Record Page > Suitability Score field").
8. If a scenario cannot be fully automated (e.g., requires manual inspection), note it
   in design_notes and mark it PARTIAL coverage.
9. If no Gherkin scenarios are available, return INCOMPLETE.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent6_data  = _get_agent_data(state, "6")
    agent19_data = _get_agent_data(state, "19")
    agent21_data = _get_agent_data(state, "21")
    agent32_data = _get_agent_data(state, "32")

    gherkin_scenarios = (agent19_data or {}).get("gherkin_scenarios", [])
    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    seed_records = (agent21_data or {}).get("test_data_strategy", {}).get("seed_records", [])
    regression_suite = (agent32_data or {}).get("recommended_regression_suite", "SMOKE")
    manual_test_present = "ManualTest" in (agent6_data or {}).get("test_tools", [])

    scenarios_truncated = len(gherkin_scenarios) > 10
    truncated_scenario_count = max(0, len(gherkin_scenarios) - 10)

    user_message = _build_prompt(
        story_id, fca_class, gherkin_scenarios, seed_records, regression_suite,
        manual_test_present=manual_test_present,
        scenarios_truncated=scenarios_truncated,
        truncated_count=truncated_scenario_count,
    )

    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_CRT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_CRT_TOOL_NAME,
        tool_description="Design CRT automated test cases from Gherkin scenarios.",
        tool_schema=_CRT_TOOL_SCHEMA,
        max_tokens=3000,
    )

    test_cases = result.get("crt_test_cases", [])
    test_count = result.get("crt_test_count", len(test_cases))
    coverage = result.get("automation_coverage", 0.0)
    verdict = result.get("crt_design_verdict", "INCOMPLETE")
    notes = result.get("design_notes", "")

    confidence_score, signals = _compute_confidence(
        agent19_data, agent21_data, test_count, coverage, verdict,
        scenarios_truncated=scenarios_truncated,
        manual_test_present=manual_test_present,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"CRT design for {story_id}: {test_count} test case(s) designed, "
        f"{coverage:.0f}% automation coverage — verdict={verdict}"
    )
    why = notes or f"Designed {test_count} CRT test cases from {len(gherkin_scenarios)} Gherkin scenarios."

    data = {
        "crt_test_cases": test_cases,
        "crt_test_count": test_count,
        "automation_coverage": coverage,
        "crt_design_verdict": verdict,
        "design_notes": notes,
        "gherkin_scenario_count": len(gherkin_scenarios),
        "scenarios_truncated": scenarios_truncated,
        "truncated_scenario_count": truncated_scenario_count,
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
    agent19_data: dict | None,
    agent21_data: dict | None,
    test_count: int,
    coverage: float,
    verdict: str,
    scenarios_truncated: bool = False,
    manual_test_present: bool = False,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=68)

    scenario_count = (agent19_data or {}).get("scenario_count", 0)
    if scenario_count > 0:
        scorer.add("gherkin_scenarios_available", scenario_count, +8)
    else:
        scorer.add("no_gherkin_scenarios", 0, -15)

    if agent21_data:
        scorer.add("test_data_available", True, +5)

    if test_count > 0:
        scorer.add("crt_tests_designed", test_count, +5)
    else:
        scorer.add("no_crt_tests", 0, -10)

    if coverage >= 80:
        scorer.add("high_automation_coverage", coverage, +5)
    elif coverage < 50 and scenario_count > 0 and not manual_test_present:
        scorer.add("low_automation_coverage", coverage, -5)

    if verdict == "INCOMPLETE":
        scorer.add("incomplete_design", True, -10)

    if scenarios_truncated:
        scorer.add("scenarios_truncated", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    story_id: str,
    fca_class: str,
    gherkin_scenarios: list,
    seed_records: list,
    regression_suite: str,
    manual_test_present: bool = False,
    scenarios_truncated: bool = False,
    truncated_count: int = 0,
) -> str:
    capped = gherkin_scenarios[:10]
    scenario_text = "\n".join(
        f"  Scenario {i+1}: [{', '.join(s.get('tags', []))}] {s.get('title', '')}\n"
        f"    Steps: {'; '.join(s.get('steps', []))}"
        for i, s in enumerate(capped)
    ) or "  (no Gherkin scenarios available)"

    truncation_note = (
        f"NOTE: {truncated_count} scenario(s) omitted due to prompt length limit — "
        f"acknowledge this gap in design_notes.\n"
        if scenarios_truncated else ""
    )
    manual_note = (
        "NOTE: ManualTest flag present — scenarios tagged @manual are deliberately not automated; "
        "exclude them from automation_coverage calculation and note in design_notes.\n"
        if manual_test_present else ""
    )

    seed_names = [r.get("object_name", "") for r in seed_records]

    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n"
        f"Regression suite required: {regression_suite}\n"
        f"Available seed data objects: {seed_names or ['none']}\n"
        f"{truncation_note}"
        f"{manual_note}"
        f"Gherkin scenarios ({len(gherkin_scenarios)}):\n{scenario_text}\n\n"
        f"Design CRT test cases using the {_CRT_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
