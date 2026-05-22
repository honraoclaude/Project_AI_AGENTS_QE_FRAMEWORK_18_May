"""
Agent 6 — Test Design Strategy
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Runs in Batch 3 (parallel with Agent 5) after Agents 1–4 complete.

Purpose:
  Designs the test pyramid for this story — what levels of testing are needed,
  what Apex classes need unit tests, which FSC object interactions need integration
  tests, how many CRT scenarios to write, and where defect risk is highest.

  Coverage targets are driven by FCA tier (Agent 3):
    HIGH / MEDIUM → 85% Apex line coverage (PACT targeted standard)
    LOW           → 75% Apex line coverage (Salesforce platform minimum)

Output data keys consumed by downstream:
  coverage_target_pct    → int   (Development gate G2 enforces this)
  apex_unit_test_classes → list  (Agent 12 Apex Coverage Checker)
  crt_recommended_count  → int   (Agent 26 CRT Scenario Designer)
  risk_areas             → list  (Agent 9 Risk Anticipation)
  test_strategy_summary  → str   (Jira comment posted by Fleet Commander)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 6
AGENT_NAME = "Test Design Strategy"

# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOL_NAME = "design_test_strategy"
_TOOL_DESCRIPTION = (
    "Design a test pyramid strategy for a Salesforce FSC user story. "
    "Specify coverage targets, which components need each test level, "
    "and where the highest defect risk lies. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "coverage_target_pct",
        "apex_unit_test_classes",
        "integration_test_scope",
        "ui_test_components",
        "crt_recommended_count",
        "test_tools",
        "risk_areas",
        "test_strategy_summary",
    ],
    "properties": {
        "coverage_target_pct": {
            "type": "integer",
            "enum": [75, 85],
            "description": "Apex line coverage target. 85 for HIGH/MEDIUM FCA. 75 for LOW.",
        },
        "apex_unit_test_classes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Apex classes that need dedicated unit test coverage for this story. "
                "Name the class and what logic it contains. "
                "e.g. ['SuitabilityAssessmentService — validation and record creation logic', "
                "'SuitabilityTriggerHandler — before-insert validation']. "
                "Empty list if story has no Apex."
            ),
        },
        "integration_test_scope": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "FSC object interactions that need integration-level testing. "
                "e.g. ['Suitability__c → RiskProfile__c lookup integrity', "
                "'VulnerableCustomerIndicator__c → Flow branch routing']. "
                "Empty list if story has no cross-object interactions."
            ),
        },
        "ui_test_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "LWC components or Screen Flows that need UI-level automated testing. "
                "e.g. ['Screen Flow: Suitability Assessment — step navigation', "
                "'LWC suitability-form — Consumer Duty checkbox visibility']. "
                "Empty list for non-UI stories."
            ),
        },
        "crt_recommended_count": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Recommended number of CRT (Continuous Regression Testing) scenarios. "
                "HIGH-FCA: ≥4. MEDIUM: ≥2. LOW: 0–1."
            ),
        },
        "test_tools": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["ApexUnit", "CRT", "Jest", "LWCTest", "Selenium", "Copado", "Postman", "ManualTest"],
            },
            "description": (
                "Test frameworks/tools required for this story's test pyramid. "
                "Postman: include when story touches external REST/SOAP callouts, Named Credentials, or ConnectedApps. "
                "ManualTest: include when story has exploratory, accessibility, or UAT scenarios that are explicitly not automatable."
            ),
        },
        "risk_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific areas where defects are most likely. "
                "e.g. ['Bulkification risk in Apex trigger on Suitability__c', "
                "'Consumer Duty checkbox not rendered for all vulnerable customer record types']. "
                "Minimum 1 entry."
            ),
        },
        "test_strategy_summary": {
            "type": "string",
            "description": (
                "2–3 sentence plain-English summary of the test approach. "
                "Written to be posted directly to Jira as a comment."
            ),
        },
    },
}

_AGENT_INSTRUCTIONS = """
You are the Test Design Strategy Agent for the FSC QE Framework.

Your job is to design a test pyramid for a Salesforce FSC user story based on its
FCA regulatory tier, FSC object scope, and acceptance criteria structure.

Coverage targets (non-negotiable):
- HIGH-FCA or MEDIUM-FCA story → 85% Apex line coverage target (PACT standard)
- LOW-FCA story                → 75% Apex line coverage target (Salesforce minimum)

Test pyramid guidance for FSC stories:

Apex Unit Tests (ApexUnit):
- Required for every Apex trigger, service class, and helper on objects this story touches.
- Test both happy-path and error-path logic at the class level.
- For HIGH-FCA: test the exact regulatory validation logic in isolation.

Integration Tests:
- Required when the story creates or modifies cross-object relationships.
  e.g. Suitability__c linking to RiskProfile__c and FinancialAccount.
- Test lookup integrity, rollup accuracy, validation rule behaviour across objects.

UI Tests (CRT / Jest / LWCTest):
- Required for Screen Flows and LWC components that the adviser or client interacts with.
- CRT is the primary tool — it runs inside Salesforce and handles governor limits.
- Jest/LWCTest for LWC unit logic (not Salesforce-server-dependent behaviour).

CRT Scenario Count:
- HIGH-FCA: recommend ≥4 scenarios (happy path + vulnerable customer + ≥2 error paths)
- MEDIUM-FCA: recommend ≥2 scenarios
- LOW-FCA: recommend 0–1 scenario (1 if there is UI interaction)

Risk Areas — always include:
- Bulkification risk if the story involves an Apex trigger or Flow on a high-volume object
- FLS (Field-Level Security) risk if new fields are added to FSC objects
- Governor limit risk if the trigger/flow processes many related records
- Consumer Duty risk if VulnerableCustomerIndicator__c is involved

Tool selection guidance:
- Include Postman when the story references external REST/SOAP callouts, Named Credentials,
  External Data Sources, Connected Apps, or integration with external AUM/financial data providers.
- Include ManualTest when the story explicitly contains exploratory testing, accessibility
  validation, or UAT scenarios that are deliberately out of scope for automation.

Use the design_test_strategy tool to return your assessment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent1_data = _get_agent_data(state, "1")
    agent2_data = _get_agent_data(state, "2")
    agent3_data = _get_agent_data(state, "3")

    story = await get_story(story_id)

    user_message = _build_user_message(story, agent1_data, agent2_data, agent3_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=2048,
    )

    confidence_score, signals = _compute_confidence(agent1_data, agent2_data, agent3_data, extracted)
    escalated = confidence_score < settings.confidence_escalation_threshold
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")

    cov_pct = extracted.get("coverage_target_pct", 75)
    what = (
        f"Test strategy for {story_id}: coverage={cov_pct}%, "
        f"apex_classes={len(extracted.get('apex_unit_test_classes', []))}, "
        f"crt_scenarios={extracted.get('crt_recommended_count', 0)}, "
        f"risk_areas={len(extracted.get('risk_areas', []))}"
    )
    why = (
        f"Test Design Strategy Agent applied PACT coverage standards for a {fca_class}-FCA story. "
        f"Coverage target is {cov_pct}% "
        f"({'85% PACT standard' if cov_pct == 85 else '75% platform minimum'})."
    )

    data = {
        "coverage_target_pct": cov_pct,
        "apex_unit_test_classes": extracted.get("apex_unit_test_classes", []),
        "integration_test_scope": extracted.get("integration_test_scope", []),
        "ui_test_components": extracted.get("ui_test_components", []),
        "crt_recommended_count": extracted.get("crt_recommended_count", 0),
        "test_tools": extracted.get("test_tools", []),
        "risk_areas": extracted.get("risk_areas", []),
        "test_strategy_summary": extracted.get("test_strategy_summary", ""),
        "fca_classification_context": fca_class,
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


# ── Confidence scoring (Tier B) ───────────────────────────────────────────────

def _compute_confidence(
    agent1_data: dict | None,
    agent2_data: dict | None,
    agent3_data: dict | None,
    extracted: dict,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")

    # Signal 1: FCA tier → determines coverage target clarity
    if fca_class in ("HIGH", "MEDIUM"):
        scorer.add("fca_known_elevated", fca_class, +10)
    elif fca_class == "LOW":
        scorer.add("fca_low", fca_class, +7)
    else:
        scorer.add("fca_unclassified", fca_class, -8)

    # Signal 2: FSC objects → determines scope of Apex/integration test design
    fsc_objects = (agent1_data or {}).get("fsc_objects", [])
    if len(fsc_objects) >= 2:
        scorer.add("fsc_objects_rich", len(fsc_objects), +8)
    elif len(fsc_objects) == 1:
        scorer.add("fsc_objects_single", len(fsc_objects), +4)
    else:
        scorer.add("fsc_objects_none", 0, -5)

    # Signal 3: INVEST score from Agent 2 — well-defined story → reliable test design
    invest_score = (agent2_data or {}).get("invest_score", 0)
    if invest_score >= 80:
        scorer.add("invest_pass", invest_score, +5)
    elif invest_score > 0:
        scorer.add("invest_fail", invest_score, -3)

    # Signal 4: CRT scenarios recommended — confirms test scope was identified
    crt_count = extracted.get("crt_recommended_count", 0)
    if crt_count > 0:
        scorer.add("crt_scenarios_identified", crt_count, +5)

    # Signal 5: Risk areas identified — confirms thorough analysis
    risk_count = len(extracted.get("risk_areas", []))
    if risk_count >= 2:
        scorer.add("risk_areas_identified", risk_count, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None


def _build_user_message(
    story: dict,
    agent1_data: dict | None,
    agent2_data: dict | None,
    agent3_data: dict | None,
) -> str:
    agent1_section = ""
    if agent1_data:
        fsc_objs = ", ".join(agent1_data.get("fsc_objects", [])) or "None"
        fsc_comps = ", ".join(agent1_data.get("fsc_components", [])) or "None"
        agent1_section = (
            f"\n\nAGENT 1 — STORY INTENT:\n"
            f"Goal: {agent1_data.get('goal', 'UNKNOWN')}\n"
            f"Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"FSC Objects: {fsc_objs}\n"
            f"FSC Components: {fsc_comps}"
        )

    agent2_section = ""
    if agent2_data:
        agent2_section = (
            f"\n\nAGENT 2 — INVEST SCORE:\n"
            f"INVEST Score: {agent2_data.get('invest_score', 'N/A')}/100 "
            f"({agent2_data.get('invest_verdict', 'N/A')})"
        )

    agent3_section = ""
    if agent3_data:
        triggers = ", ".join(agent3_data.get("fca_triggers", [])) or "None"
        agent3_section = (
            f"\n\nAGENT 3 — FCA CLASSIFICATION:\n"
            f"FCA Tier: {agent3_data.get('fca_classification', 'UNCLASSIFIED')}\n"
            f"FCA Triggers: {triggers}\n"
            f"Enhanced Testing Required: {agent3_data.get('enhanced_testing_required', False)}"
        )

    return (
        f"Design a test pyramid strategy for the following Jira user story.\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{agent1_section}"
        f"{agent2_section}"
        f"{agent3_section}\n\n"
        f"Use the design_test_strategy tool to return your test pyramid design."
    )
