"""Tests for Agent 6 — Test Design Strategy."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_06_test_design import (
    _build_user_message,
    _compute_confidence,
    _TOOL_NAME,
    _TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio so that the firm meets its regulatory obligation.\n\n"
        "The Suitability__c record must link to RiskProfile__c and FinancialAccount.\n\n"
        "For vulnerable customers (VulnerableCustomerIndicator__c = true) the flow "
        "must present an additional Consumer Duty confirmation step."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "High",
    "labels": [],
    "components": ["Suitability"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

AGENT1_DATA = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount"],
    "fsc_components": ["Screen Flow — Suitability Assessment", "Apex trigger on Suitability__c"],
    "description_word_count": 80,
}

AGENT2_DATA_PASS = {"invest_score": 85, "invest_verdict": "PASS"}
AGENT2_DATA_FAIL = {"invest_score": 45, "invest_verdict": "FAIL"}

AGENT3_DATA_HIGH = {
    "fca_classification": "HIGH",
    "fca_triggers": ["Suitability__c", "RiskProfile__c"],
    "enhanced_testing_required": True,
}

AGENT3_DATA_LOW = {
    "fca_classification": "LOW",
    "fca_triggers": [],
    "enhanced_testing_required": False,
}

MOCK_STRATEGY_HIGH = {
    "coverage_target_pct": 85,
    "apex_unit_test_classes": [
        "SuitabilityAssessmentService — validation and record creation",
        "SuitabilityTriggerHandler — before-insert validation",
    ],
    "integration_test_scope": [
        "Suitability__c → RiskProfile__c lookup integrity",
        "VulnerableCustomerIndicator__c → Flow branch routing",
    ],
    "ui_test_components": [
        "Screen Flow: Suitability Assessment — step navigation and Consumer Duty checkbox",
    ],
    "crt_recommended_count": 4,
    "test_tools": ["ApexUnit", "CRT", "LWCTest"],
    "risk_areas": [
        "Bulkification risk in SuitabilityTriggerHandler on high-volume FinancialAccount inserts",
        "Consumer Duty checkbox not rendered for all VulnerableCustomerIndicator__c record types",
    ],
    "test_strategy_summary": (
        "This HIGH-FCA story requires 85% Apex coverage targeting the Suitability trigger and "
        "service class. CRT will cover 4 scenarios including the vulnerable customer pathway. "
        "Integration tests verify the cross-object RiskProfile → FinancialAccount chain."
    ),
}

AGENT1_DATA_ONE_OBJECT = {**AGENT1_DATA, "fsc_objects": ["Suitability__c"]}

AGENT3_DATA_MEDIUM = {
    "fca_classification": "MEDIUM",
    "fca_triggers": ["FinancialAccount"],
    "enhanced_testing_required": True,
}

STORY_LABEL_CHANGE = {
    "story_id": "FSC-2500",
    "summary": "Update button label",
    "description": "Change the Save button to Submit.",
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": "po@firm.com",
}

STORY_NO_DESCRIPTION = {**STORY_LABEL_CHANGE, "story_id": "FSC-2501", "description": None}

MOCK_STRATEGY_LOW = {
    "coverage_target_pct": 75,
    "apex_unit_test_classes": [],
    "integration_test_scope": [],
    "ui_test_components": ["Lightning Page — Account Detail button label"],
    "crt_recommended_count": 0,
    "test_tools": ["CRT"],
    "risk_areas": ["Label change may be overridden by a custom CSS class in managed package."],
    "test_strategy_summary": "LOW-FCA UI label change. 75% coverage standard. 1 CRT smoke scenario recommended.",
}


class TestConfidenceScoring:
    def test_high_fca_rich_story_scores_well(self):
        score, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert score >= 75

    def test_unclassified_fca_reduces_confidence(self):
        agent3_unclassified = {**AGENT3_DATA_HIGH, "fca_classification": "UNCLASSIFIED"}
        score_classified, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        score_unclassified, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, agent3_unclassified, MOCK_STRATEGY_HIGH)
        assert score_classified > score_unclassified

    def test_invest_fail_reduces_confidence(self):
        score_pass, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        score_fail, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_FAIL, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert score_pass > score_fail

    def test_no_fsc_objects_reduces_confidence(self):
        empty_agent1 = {**AGENT1_DATA, "fsc_objects": []}
        score_full, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        score_empty, _ = _compute_confidence(empty_agent1, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert score_full > score_empty

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, MOCK_STRATEGY_LOW)
        assert score >= 20

    def test_crt_signal_present_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert "crt_scenarios_identified" in signals

    def test_fsc_objects_single_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_ONE_OBJECT, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert "fsc_objects_single" in signals
        assert signals["fsc_objects_single"] == 1

    def test_fca_known_elevated_stores_fca_class_high(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert signals["fca_known_elevated"] == "HIGH"

    def test_fca_low_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_LOW, MOCK_STRATEGY_LOW)
        assert "fca_low" in signals
        assert signals["fca_low"] == "LOW"

    def test_fca_medium_signal_stores_medium(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_MEDIUM, MOCK_STRATEGY_HIGH)
        assert signals["fca_known_elevated"] == "MEDIUM"

    def test_fsc_objects_rich_stores_count(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert signals["fsc_objects_rich"] == 3

    def test_crt_scenarios_identified_stores_count(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert signals["crt_scenarios_identified"] == 4

    def test_risk_areas_identified_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH, MOCK_STRATEGY_HIGH)
        assert "risk_areas_identified" in signals
        assert signals["risk_areas_identified"] == 2


@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA}
        state["agent_results"]["2"] = {"data": AGENT2_DATA_PASS}
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert result.agent_id == 6
        assert result.agent_name == "Test Design Strategy"
        assert result.confidence.tier == "B"

    async def test_coverage_target_85_for_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert result.data["coverage_target_pct"] == 85

    async def test_coverage_target_75_for_low_fca(self):
        state = initial_story_state("FSC-2500")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_LOW}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_LOW
            result = await run(state)

        assert result.data["coverage_target_pct"] == 75

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        for key in ["coverage_target_pct", "apex_unit_test_classes", "crt_recommended_count",
                    "risk_areas", "test_strategy_summary"]:
            assert key in result.data

    async def test_runs_standalone_without_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert result.agent_id == 6

    async def test_zero_context_causes_escalation(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_LOW
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_fca_classification_context_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert result.data["fca_classification_context"] == "HIGH"

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_test_strategy_summary_is_non_empty_string(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert isinstance(result.data["test_strategy_summary"], str)
        assert len(result.data["test_strategy_summary"]) > 0

    async def test_medium_fca_run_coverage_target_85(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_MEDIUM}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert result.data["coverage_target_pct"] == 85
        assert result.data["fca_classification_context"] == "MEDIUM"


# ── REQ-04: Postman + ManualTest enum tests ───────────────────────────────────

MOCK_STRATEGY_POSTMAN = {
    **MOCK_STRATEGY_HIGH,
    "test_tools": ["ApexUnit", "CRT", "Postman"],
}

MOCK_STRATEGY_MANUAL = {
    **MOCK_STRATEGY_HIGH,
    "test_tools": ["ApexUnit", "ManualTest"],
}


@pytest.mark.asyncio
class TestPostmanManualTestREQ04:
    async def test_postman_in_test_tools_passes_through(self):
        """REQ-04: Postman must be a valid test_tools enum value."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": {"fca_classification": "HIGH", "fca_triggers": []}}

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_POSTMAN
            result = await run(state)

        assert "Postman" in result.data["test_tools"]

    async def test_manual_test_in_test_tools_passes_through(self):
        """REQ-04: ManualTest must be a valid test_tools enum value."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_MANUAL
            result = await run(state)

        assert "ManualTest" in result.data["test_tools"]

    async def test_test_tools_list_in_output_data(self):
        """REQ-04: test_tools list always present in output data."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_06_test_design.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_06_test_design.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_STRATEGY_HIGH
            result = await run(state)

        assert "test_tools" in result.data
        assert isinstance(result.data["test_tools"], list)


# ── Tests: prompt content ─────────────────────────────────────────────────────

class TestPromptContent:
    def test_prompt_includes_story_id(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "FSC-2417" in msg

    def test_prompt_includes_summary(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert STORY_SUITABILITY["summary"] in msg

    def test_prompt_includes_components(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "COMPONENTS:" in msg

    def test_prompt_empty_components_renders_as_none(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "COMPONENTS: None" in msg

    def test_prompt_empty_description_shows_empty(self):
        msg = _build_user_message(STORY_NO_DESCRIPTION, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "(empty)" in msg

    def test_prompt_includes_agent1_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "AGENT 1 — STORY INTENT:" in msg
        assert "FSC Components:" in msg

    def test_prompt_agent1_section_absent_when_no_agent1_data(self):
        msg = _build_user_message(STORY_SUITABILITY, None, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "AGENT 1 — STORY INTENT:" not in msg

    def test_prompt_includes_agent2_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "AGENT 2 — INVEST SCORE:" in msg

    def test_prompt_agent2_section_absent_when_no_agent2_data(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, None, AGENT3_DATA_HIGH)
        assert "AGENT 2 — INVEST SCORE:" not in msg

    def test_prompt_includes_agent3_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert "AGENT 3 — FCA CLASSIFICATION:" in msg
        assert "Enhanced Testing Required:" in msg

    def test_prompt_agent3_section_absent_when_no_agent3_data(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, None)
        assert "AGENT 3 — FCA CLASSIFICATION:" not in msg

    def test_prompt_ends_with_tool_instruction(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA, AGENT2_DATA_PASS, AGENT3_DATA_HIGH)
        assert _TOOL_NAME in msg
        assert msg.strip().endswith("design.")


# ── Tests: schema contract ────────────────────────────────────────────────────

class TestSchemaContract:
    def test_tool_schema_has_eight_required_fields(self):
        expected = {
            "coverage_target_pct",
            "apex_unit_test_classes",
            "integration_test_scope",
            "ui_test_components",
            "crt_recommended_count",
            "test_tools",
            "risk_areas",
            "test_strategy_summary",
        }
        assert set(_TOOL_SCHEMA["required"]) == expected

    def test_coverage_target_pct_enum_has_two_values(self):
        assert _TOOL_SCHEMA["properties"]["coverage_target_pct"]["enum"] == [75, 85]

    def test_crt_recommended_count_has_minimum_zero(self):
        assert _TOOL_SCHEMA["properties"]["crt_recommended_count"]["minimum"] == 0

    def test_test_tools_is_array_of_enum(self):
        prop = _TOOL_SCHEMA["properties"]["test_tools"]
        assert prop["type"] == "array"
        assert len(prop["items"]["enum"]) == 8

    def test_apex_unit_test_classes_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["apex_unit_test_classes"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_integration_test_scope_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["integration_test_scope"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_ui_test_components_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["ui_test_components"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_risk_areas_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["risk_areas"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_test_strategy_summary_is_string(self):
        assert _TOOL_SCHEMA["properties"]["test_strategy_summary"]["type"] == "string"
