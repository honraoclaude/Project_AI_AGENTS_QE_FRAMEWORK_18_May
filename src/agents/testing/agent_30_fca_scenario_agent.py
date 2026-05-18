"""
Agent 30 — FCA Scenario Agent
Phase       : Testing
PACT        : Proactive
Classification: True AI Agent (Claude Sonnet 4.6)
Confidence  : Tier B (base=70)

Runs in Testing Batch 2 (parallel with Agents 26, 29).
Has access to Agents 3, 4, 5, 9, 19.

Purpose:
  Generates FCA-specific regulatory test scenarios beyond what the general
  BDD Gherkin Writer produces. Covers Consumer Duty obligations, COBS rules,
  MiFID II suitability requirements, and Vulnerable Customer protections.
  These scenarios feed the FCA Evidence Pack (Agent 44) and Gate G6.

  True AI (Sonnet 4.6) — FCA domain knowledge requires genuine reasoning.

Output data keys consumed by downstream:
  fca_test_scenarios    → list (Agent 44 FCA Evidence Pack)
  consumer_duty_covered → bool (Gate G5 — Consumer Duty obligation check)
  cobs_scenarios_count  → int  (regulatory coverage metric)
  fca_scenario_verdict  → str  (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 30
AGENT_NAME = "FCA Scenario Agent"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_FCA_TOOL_NAME = "generate_fca_regulatory_scenarios"
_FCA_TOOL_SCHEMA = {
    "type": "object",
    "required": ["fca_test_scenarios", "consumer_duty_covered",
                 "cobs_scenarios_count", "fca_scenario_verdict", "regulatory_gaps"],
    "properties": {
        "fca_test_scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scenario_id", "regulation", "title",
                             "description", "pass_criteria", "fail_criteria"],
                "properties": {
                    "scenario_id": {"type": "string"},
                    "regulation": {
                        "type": "string",
                        "description": "FCA rule reference e.g. COBS 9, Consumer Duty, MiFID II Art.25",
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "pass_criteria": {"type": "string",
                                      "description": "Observable evidence of compliance"},
                    "fail_criteria": {"type": "string",
                                      "description": "Observable evidence of non-compliance"},
                },
            },
        },
        "consumer_duty_covered": {
            "type": "boolean",
            "description": "True if Consumer Duty obligations are explicitly tested.",
        },
        "cobs_scenarios_count": {"type": "integer"},
        "fca_scenario_verdict": {
            "type": "string",
            "enum": ["PASS", "WARN", "FAIL"],
            "description": (
                "PASS: All applicable FCA rules have test coverage. "
                "WARN: Some rules have partial coverage. "
                "FAIL: Critical FCA rules have no coverage."
            ),
        },
        "regulatory_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "FCA rules without test coverage.",
        },
    },
}

_FCA_INSTRUCTIONS = """
You are a senior FCA compliance expert generating regulatory test scenarios for a
Salesforce FSC Wealth Management platform. Your scenarios must be directly traceable
to FCA regulatory obligations.

Applicable regulations for FSC Wealth Management:
- COBS 9: Suitability assessment for personal recommendations
- COBS 4: Communicating with clients (clear, fair, not misleading)
- Consumer Duty (PS22/9): Good outcomes for retail customers
- MiFID II Article 25: Appropriateness and suitability
- FCA Consumer Duty — Vulnerable Customer provisions (FG21/1)

Rules:
1. For HIGH-FCA stories: generate scenarios covering COBS 9 suitability, Consumer Duty
   good outcome requirements, and Vulnerable Customer provisions.
2. For MEDIUM-FCA stories: cover Consumer Duty and basic COBS requirements.
3. For LOW-FCA stories: generate a minimal Consumer Duty scenario only.
4. Each scenario must have observable pass/fail criteria — not vague statements.
5. If the story has no FCA implications, return WARN with an explanation.
6. Always check: does this feature make it harder for a Vulnerable Customer to get
   good outcomes? If yes, flag it.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data = _get_agent_data(state, "3")
    agent4_data = _get_agent_data(state, "4")
    agent5_data = _get_agent_data(state, "5")
    agent9_data = _get_agent_data(state, "9")
    agent19_data = _get_agent_data(state, "19")

    story = await get_story(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    consumer_duty_risk = (agent4_data or {}).get("consumer_duty_risk", "LOW")
    risk_level = (agent9_data or {}).get("risk_level", "LOW")
    existing_scenarios = (agent19_data or {}).get("gherkin_scenarios", [])

    user_message = _build_prompt(
        story_id, story, fca_class, consumer_duty_risk, risk_level, existing_scenarios,
    )

    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_FCA_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_FCA_TOOL_NAME,
        tool_description="Generate FCA regulatory test scenarios.",
        tool_schema=_FCA_TOOL_SCHEMA,
        max_tokens=2500,
    )

    fca_scenarios = result.get("fca_test_scenarios", [])
    consumer_duty = result.get("consumer_duty_covered", False)
    cobs_count = result.get("cobs_scenarios_count", 0)
    verdict = result.get("fca_scenario_verdict", "WARN")
    gaps = result.get("regulatory_gaps", [])

    confidence_score, signals = _compute_confidence(
        agent3_data, agent4_data, agent9_data, len(fca_scenarios), fca_class,
        consumer_duty, verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"FCA scenarios for {story_id}: {len(fca_scenarios)} scenario(s), "
        f"Consumer Duty covered={consumer_duty}, COBS={cobs_count} — verdict={verdict}"
    )
    why = (
        f"Generated {len(fca_scenarios)} FCA regulatory test scenario(s) for a "
        f"{fca_class}-FCA story. Consumer Duty: {'covered' if consumer_duty else 'not covered'}. "
        f"{len(gaps)} regulatory gap(s)."
    )

    data = {
        "fca_test_scenarios": fca_scenarios,
        "consumer_duty_covered": consumer_duty,
        "cobs_scenarios_count": cobs_count,
        "fca_scenario_verdict": verdict,
        "regulatory_gaps": gaps,
        "fca_scenario_count": len(fca_scenarios),
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
    agent3_data: dict | None,
    agent4_data: dict | None,
    agent9_data: dict | None,
    scenario_count: int,
    fca_class: str,
    consumer_duty: bool,
    verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=70)

    if agent3_data:
        scorer.add("fca_classification_available", True, +7)
    else:
        scorer.add("no_fca_classification", 0, -10)

    if agent4_data:
        scorer.add("consumer_duty_assessment_available", True, +5)

    if agent9_data:
        scorer.add("risk_anticipation_available", True, +5)

    if scenario_count > 0:
        scorer.add("fca_scenarios_generated", scenario_count, +5)
    else:
        scorer.add("no_fca_scenarios", 0, -10)

    if fca_class in ("HIGH", "MEDIUM") and not consumer_duty:
        scorer.add("regulated_story_consumer_duty_not_covered", True, -10)

    if verdict == "FAIL":
        scorer.add("fca_scenario_fail", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    story_id: str,
    story: dict,
    fca_class: str,
    consumer_duty_risk: str,
    risk_level: str,
    existing_scenarios: list,
) -> str:
    existing_titles = [s.get("title", "") for s in existing_scenarios[:5]]
    return (
        f"Story: {story_id} — {story.get('summary', 'N/A')}\n"
        f"Description: {story.get('description', 'N/A')[:500]}\n"
        f"FCA Classification: {fca_class}\n"
        f"Consumer Duty risk: {consumer_duty_risk}\n"
        f"Risk anticipation level: {risk_level}\n"
        f"Existing Gherkin scenarios (for context, avoid duplication):\n"
        + "\n".join(f"  - {t}" for t in existing_titles) if existing_titles
        else "  none"
        + f"\n\nGenerate FCA regulatory scenarios using the {_FCA_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
