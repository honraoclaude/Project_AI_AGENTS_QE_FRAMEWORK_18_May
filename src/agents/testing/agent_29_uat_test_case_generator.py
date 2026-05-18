"""
Agent 29 — UAT Test Case Generator
Phase       : Testing
PACT        : Collaborative
Classification: True AI Agent (Claude Sonnet 4.6)
Confidence  : Tier B (base=67)

Runs in Testing Batch 2 (parallel with Agents 26, 30).
Has access to Agents 3, 5, 19, 21.

Purpose:
  Generates business-readable UAT test cases for human testers and
  Compliance Officers. Unlike CRT cases (machine-executable), UAT cases
  use plain English steps that a business analyst or Compliance Officer
  can follow. FCA HIGH/MEDIUM stories require explicit regulatory pass/fail
  criteria that can be cited in the FCA evidence pack.

  True AI (Sonnet 4.6) generates the UAT test cases.

Output data keys consumed by downstream:
  uat_test_cases       → list (Agent 36 UAT Coordination input)
  uat_test_count       → int  (coverage metric)
  co_sign_off_required → bool (Gate G6 — Compliance Officer sign-off gate)
  uat_verdict          → str  (PASS / WARN / INCOMPLETE)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 29
AGENT_NAME = "UAT Test Case Generator"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_UAT_TOOL_NAME = "generate_uat_test_cases"
_UAT_TOOL_SCHEMA = {
    "type": "object",
    "required": ["uat_test_cases", "uat_test_count", "co_sign_off_required",
                 "uat_verdict", "regulatory_assertions"],
    "properties": {
        "uat_test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["test_id", "title", "ac_reference", "preconditions",
                             "steps", "expected_result", "regulatory_flag"],
                "properties": {
                    "test_id": {"type": "string"},
                    "title": {"type": "string"},
                    "ac_reference": {"type": "string",
                                     "description": "AC ID this test covers"},
                    "preconditions": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"},
                              "description": "Plain English steps for business tester"},
                    "expected_result": {"type": "string"},
                    "regulatory_flag": {
                        "type": "boolean",
                        "description": "True if this test must be signed off by Compliance Officer",
                    },
                },
            },
        },
        "uat_test_count": {"type": "integer"},
        "co_sign_off_required": {
            "type": "boolean",
            "description": "True if any test requires Compliance Officer sign-off.",
        },
        "uat_verdict": {
            "type": "string",
            "enum": ["PASS", "WARN", "INCOMPLETE"],
        },
        "regulatory_assertions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "FCA-specific assertions that must appear in the evidence pack.",
        },
    },
}

_UAT_INSTRUCTIONS = """
You are a QE lead generating UAT test cases for a Salesforce FSC Wealth Management platform
regulated by the FCA. Your test cases are for business analysts and Compliance Officers,
not automation engineers — use plain, jargon-free English.

Rules:
1. Each acceptance criterion gets at least one UAT test case.
2. FCA HIGH or MEDIUM classified stories must include regulatory test cases that
   a Compliance Officer can sign off on. These are marked regulatory_flag=true.
3. Steps must be written for non-technical testers: navigate to Record, click button,
   observe field value — not code or API calls.
4. Include explicit expected results that can be objectively verified.
5. Regulatory assertions must reference the FCA rule being tested
   (e.g. "Suitability assessment passes FCA COBS 9 requirement").
6. If ACs are unavailable, set INCOMPLETE and explain in regulatory_assertions.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent5_data  = _get_agent_data(state, "5")
    agent19_data = _get_agent_data(state, "19")
    agent21_data = _get_agent_data(state, "21")

    story = await get_story(story_id)
    acs   = await get_acceptance_criteria(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    gherkin = (agent19_data or {}).get("gherkin_scenarios", [])
    vulnerable_profiles = (agent21_data or {}).get("vulnerable_profiles", [])

    user_message = _build_prompt(
        story_id, story, acs, fca_class, gherkin, vulnerable_profiles,
    )

    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_UAT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_UAT_TOOL_NAME,
        tool_description="Generate UAT test cases for business testers and Compliance Officers.",
        tool_schema=_UAT_TOOL_SCHEMA,
        max_tokens=2500,
    )

    test_cases = result.get("uat_test_cases", [])
    test_count = result.get("uat_test_count", len(test_cases))
    co_required = result.get("co_sign_off_required", False)
    verdict = result.get("uat_verdict", "INCOMPLETE")
    reg_assertions = result.get("regulatory_assertions", [])

    confidence_score, signals = _compute_confidence(
        acs, agent3_data, agent19_data, test_count, fca_class, co_required,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"UAT test cases for {story_id}: {test_count} test(s) generated, "
        f"CO sign-off required={co_required} — verdict={verdict}"
    )
    why = (
        f"Generated {test_count} UAT test case(s) for {len(acs)} AC(s) on a "
        f"{fca_class}-FCA story. {len(reg_assertions)} regulatory assertion(s) captured."
    )

    data = {
        "uat_test_cases": test_cases,
        "uat_test_count": test_count,
        "co_sign_off_required": co_required,
        "uat_verdict": verdict,
        "regulatory_assertions": reg_assertions,
        "ac_count": len(acs),
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
    agent19_data: dict | None,
    test_count: int,
    fca_class: str,
    co_required: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=67)

    if acs:
        scorer.add("acs_available", len(acs), +8)
    else:
        scorer.add("no_acs_available", 0, -15)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if agent19_data and agent19_data.get("scenario_count", 0) > 0:
        scorer.add("gherkin_context_available", True, +5)

    if test_count > 0:
        scorer.add("uat_tests_generated", test_count, +5)
    else:
        scorer.add("no_uat_tests", 0, -10)

    if fca_class in ("HIGH", "MEDIUM") and not co_required:
        scorer.add("regulated_story_co_not_flagged", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    story_id: str,
    story: dict,
    acs: list,
    fca_class: str,
    gherkin: list,
    vulnerable_profiles: list,
) -> str:
    ac_text = "\n".join(
        f"  AC{i+1}: {ac.get('description', str(ac))}" for i, ac in enumerate(acs)
    ) or "  (no acceptance criteria available)"
    gherkin_titles = [s.get("title", "") for s in gherkin[:5]]
    return (
        f"Story: {story_id} — {story.get('summary', 'N/A')}\n"
        f"FCA Classification: {fca_class}\n"
        f"Acceptance Criteria:\n{ac_text}\n"
        f"Gherkin scenario titles (for context): {gherkin_titles or ['none']}\n"
        f"Vulnerable Customer profiles available: {vulnerable_profiles or ['none']}\n\n"
        f"Generate UAT test cases using the {_UAT_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
