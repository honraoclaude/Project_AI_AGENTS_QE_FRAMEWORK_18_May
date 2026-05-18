"""
Agent 7 — Data Need Agent
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Runs in Batch 2 (parallel with Agents 2 and 3), so only Agent 1's output
is available. Agent 7 reads Agent 1's fsc_objects, goal, and description
richness to identify what test data the story requires.

Purpose:
  Identifies all Salesforce records that must exist before tests can execute.
  Maps the insertion order (parent records before children), flags sensitive
  data fields that need masking, and recommends factory patterns.

  This output is consumed by Agent 21 (Test Data Architect) in the
  Development phase to generate actual test data factories.

Output data keys consumed by downstream:
  required_records          → list  (Agent 21 Test Data Architect)
  data_isolation_strategy   → str   (Agent 21, Agent 26 CRT Scenario Designer)
  sensitive_data_fields     → list  (Agent 21 — masking requirements)
  data_dependencies_ordered → list  (insertion order; Agent 21)
  risks                     → list  (Agent 9 Risk Anticipation)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 7
AGENT_NAME = "Data Need Agent"

_TOOL_NAME = "identify_data_needs"
_TOOL_DESCRIPTION = (
    "Identify the Salesforce test data requirements for a Jira user story. "
    "Specify which records must exist, their insertion order, and any sensitive "
    "data handling requirements. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "required_records",
        "data_isolation_strategy",
        "sensitive_data_present",
        "sensitive_data_fields",
        "factory_classes_recommended",
        "data_dependencies_ordered",
        "data_volume",
        "risks",
    ],
    "properties": {
        "required_records": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["object_api_name", "min_record_count", "key_field_values", "setup_method"],
                "properties": {
                    "object_api_name": {"type": "string"},
                    "min_record_count": {"type": "integer", "minimum": 1},
                    "key_field_values": {
                        "type": "object",
                        "description": "Field name → example value pairs required for tests to pass.",
                    },
                    "setup_method": {
                        "type": "string",
                        "enum": ["TestSetup", "TestFactory", "StaticData", "MockData"],
                    },
                },
            },
            "description": (
                "Every Salesforce record type required before tests can run. "
                "Include only records that tests actively depend on."
            ),
        },
        "data_isolation_strategy": {
            "type": "string",
            "enum": ["per_test_setup_teardown", "per_class_setup", "shared_org_data"],
            "description": (
                "per_test_setup_teardown: each test creates and deletes its own records (safest). "
                "per_class_setup: @TestSetup creates data once per test class (faster, recommended for most FSC). "
                "shared_org_data: tests use pre-existing org data (fragile — avoid for regulated objects)."
            ),
        },
        "sensitive_data_present": {
            "type": "boolean",
            "description": "True if the story touches fields containing PII, financial amounts, or vulnerability indicators.",
        },
        "sensitive_data_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Fields that must use synthetic/masked values in tests. "
                "e.g. ['VulnerableCustomerIndicator__c — must use synthetic flag, never real client data', "
                "'AUM — use synthetic financial amounts only']. "
                "Empty list if sensitive_data_present is False."
            ),
        },
        "factory_classes_recommended": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Apex test utility classes to create or reuse. "
                "e.g. ['TestDataFactory.createRiskProfile()', 'SuitabilityTestSetup.createVulnerableClient()']. "
                "Empty list if only simple inline data is needed."
            ),
        },
        "data_dependencies_ordered": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Insertion order for records — parent before child. "
                "e.g. ['1. Individual (client)', '2. FinancialAccount', '3. RiskProfile__c', '4. Suitability__c']. "
                "Empty list if only one record type is needed."
            ),
        },
        "data_volume": {
            "type": "string",
            "enum": ["minimal", "moderate", "complex"],
            "description": (
                "minimal: 1–2 object types, simple field values. "
                "moderate: 3–5 object types or specific field value combinations. "
                "complex: deep dependency chains, bulk data, or multiple test data variants."
            ),
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Data-related test risks. "
                "e.g. ['Suitability__c requires a linked RiskProfile__c — tests will fail if setup order is wrong', "
                "'VulnerableCustomerIndicator__c must be reset between tests to avoid state leakage']. "
                "Minimum 1 entry."
            ),
        },
    },
}

_AGENT_INSTRUCTIONS = """
You are the Data Need Agent for the FSC QE Framework.

Your job is to identify what Salesforce test data must exist before the acceptance
criteria for this story can be executed. You work from the story description and
the FSC object model provided in your domain context.

Key FSC data dependency rules you must apply:
1. Suitability__c requires: Individual (client) → FinancialAccount → RiskProfile__c
2. RiskProfile__c requires: Individual (client) → FinancialAccount
3. FinancialGoal / Goal__c requires: Individual → FinancialAccount
4. FinancialHolding requires: Individual → FinancialAccount
5. VulnerableCustomerIndicator__c: a flag on Individual — always use synthetic values
6. Appropriateness__c requires: Individual → FinancialAccount

Data isolation rules for FSC:
- Prefer per_class_setup (@TestSetup) for most FSC tests — it's faster and
  governor limits are isolated per class.
- Use per_test_setup_teardown only when tests genuinely modify records in ways
  that would affect subsequent tests.
- NEVER recommend shared_org_data for objects in HIGH or MEDIUM FCA tier.
  State pollution across tests is a compliance risk.

Sensitive data:
- VulnerableCustomerIndicator__c must ALWAYS use synthetic boolean flags — never
  real client vulnerability status.
- Financial amounts (AUM, fees, balances) must use synthetic round numbers.
- Any PII fields (name, email, DOB) on Individual must use test-only synthetic values.

Factory pattern:
- Recommend named factory methods that tests can reuse.
- If the story creates a new FSC object or field, recommend a new factory method.

Use the identify_data_needs tool to return your assessment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent1_data = _get_agent_data(state, "1")

    story = await get_story(story_id)

    user_message = _build_user_message(story, agent1_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=2048,
    )

    confidence_score, signals = _compute_confidence(agent1_data, extracted)
    escalated = confidence_score < settings.confidence_escalation_threshold

    records = extracted.get("required_records", [])
    what = (
        f"Data needs for {story_id}: {len(records)} record type(s) required, "
        f"volume={extracted['data_volume']}, "
        f"isolation={extracted['data_isolation_strategy']}, "
        f"sensitive={extracted['sensitive_data_present']}"
    )
    why = (
        "Data Need Agent mapped FSC object dependency rules to identify the minimum "
        "Salesforce records required before acceptance criteria can execute. "
        f"Insertion order: {' → '.join(extracted.get('data_dependencies_ordered', ['N/A'])[:3])}."
    )

    data = {
        "required_records": records,
        "data_isolation_strategy": extracted["data_isolation_strategy"],
        "sensitive_data_present": extracted["sensitive_data_present"],
        "sensitive_data_fields": extracted.get("sensitive_data_fields", []),
        "factory_classes_recommended": extracted.get("factory_classes_recommended", []),
        "data_dependencies_ordered": extracted.get("data_dependencies_ordered", []),
        "data_volume": extracted["data_volume"],
        "risks": extracted.get("risks", []),
        "required_record_count": len(records),
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

def _compute_confidence(agent1_data: dict | None, extracted: dict) -> tuple[int, dict]:
    scorer = TierBScorer(base=55)

    # Signal 1: FSC object count → quality of data need identification
    fsc_objects = (agent1_data or {}).get("fsc_objects", [])
    if len(fsc_objects) >= 3:
        scorer.add("fsc_objects_rich", len(fsc_objects), +12)
    elif len(fsc_objects) >= 1:
        scorer.add("fsc_objects_present", len(fsc_objects), +7)
    else:
        scorer.add("fsc_objects_absent", 0, -5)

    # Signal 2: description richness → reliable data extraction
    word_count = (agent1_data or {}).get("description_word_count", 0)
    if word_count >= 100:
        scorer.add("description_rich", word_count, +8)
    elif word_count >= 50:
        scorer.add("description_moderate", word_count, +4)
    elif word_count < 30:
        scorer.add("description_sparse", word_count, -8)

    # Signal 3: dependency chain found → non-trivial, more complete analysis
    dep_count = len(extracted.get("data_dependencies_ordered", []))
    if dep_count >= 2:
        scorer.add("dependency_chain_found", dep_count, +5)

    # Signal 4: sensitive data flagged → confirms thorough analysis
    if extracted.get("sensitive_data_present"):
        scorer.add("sensitive_data_identified", True, +5)

    # Signal 5: volume complexity alignment
    data_volume = extracted.get("data_volume", "minimal")
    if data_volume == "complex" and len(fsc_objects) >= 3:
        scorer.add("volume_complexity_aligned", True, +5)
    elif data_volume == "complex" and len(fsc_objects) < 2:
        scorer.add("volume_complexity_mismatch", True, -5)  # mismatch is uncertain

    scorer.cap(92).floor(20)
    return scorer.build()


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None


def _build_user_message(story: dict, agent1_data: dict | None) -> str:
    agent1_section = ""
    if agent1_data:
        fsc_objs = ", ".join(agent1_data.get("fsc_objects", [])) or "None"
        fsc_comps = ", ".join(agent1_data.get("fsc_components", [])) or "None"
        agent1_section = (
            f"\n\nAGENT 1 — STORY INTENT:\n"
            f"Goal: {agent1_data.get('goal', 'UNKNOWN')}\n"
            f"Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"FSC Objects: {fsc_objs}\n"
            f"FSC Components: {fsc_comps}\n"
            f"Story Summary: {agent1_data.get('story_summary', '')}"
        )

    return (
        f"Identify the Salesforce test data requirements for the following Jira user story.\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{agent1_section}\n\n"
        f"Use the identify_data_needs tool to return your assessment."
    )
