"""Tests for Agent 7 — Data Need Agent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_07_data_need import (
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
        "for a client's retirement portfolio. The Suitability__c record must link to "
        "RiskProfile__c and FinancialAccount. Vulnerable customers require a Consumer "
        "Duty confirmation step."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "High",
    "labels": [],
    "components": ["Suitability"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

AGENT1_DATA_RICH = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount", "VulnerableCustomerIndicator__c"],
    "fsc_components": ["Screen Flow — Suitability Assessment"],
    "story_summary": "Records COBS 9.2 suitability assessment.",
    "description_word_count": 90,
}

AGENT1_DATA_SPARSE = {
    "goal": "Update a label.",
    "persona": "Operations/Admin",
    "fsc_objects": [],
    "fsc_components": [],
    "story_summary": "UI label change.",
    "description_word_count": 12,
}

MOCK_DATA_NEED_COMPLEX = {
    "required_records": [
        {
            "object_api_name": "Individual",
            "min_record_count": 2,
            "key_field_values": {"FirstName": "TestClient", "LastName": "Synthetic"},
            "setup_method": "TestSetup",
        },
        {
            "object_api_name": "FinancialAccount",
            "min_record_count": 1,
            "key_field_values": {"FinancialAccountNumber": "TEST-001", "Status": "Active"},
            "setup_method": "TestSetup",
        },
        {
            "object_api_name": "RiskProfile__c",
            "min_record_count": 1,
            "key_field_values": {"risk_level__c": "Moderate", "investment_horizon__c": "Long-term"},
            "setup_method": "TestFactory",
        },
        {
            "object_api_name": "VulnerableCustomerIndicator__c",
            "min_record_count": 1,
            "key_field_values": {"IsVulnerable__c": True},
            "setup_method": "TestFactory",
        },
    ],
    "data_isolation_strategy": "per_class_setup",
    "sensitive_data_present": True,
    "sensitive_data_fields": [
        "VulnerableCustomerIndicator__c.IsVulnerable__c — synthetic boolean only",
        "FinancialAccount.AUM__c — synthetic financial amounts only",
    ],
    "factory_classes_recommended": [
        "TestDataFactory.createRiskProfile(financialAccount, 'Moderate')",
        "SuitabilityTestSetup.createVulnerableClient()",
    ],
    "data_dependencies_ordered": [
        "1. Individual (client account)",
        "2. FinancialAccount — linked to Individual",
        "3. RiskProfile__c — linked to FinancialAccount",
        "4. VulnerableCustomerIndicator__c — flag on Individual",
        "5. Suitability__c — linked to RiskProfile__c and FinancialAccount",
    ],
    "data_volume": "complex",
    "risks": [
        "RiskProfile__c must be inserted before Suitability__c — incorrect setup order causes FK violations.",
        "VulnerableCustomerIndicator__c must be reset between test classes to avoid state leakage.",
    ],
}

AGENT1_DATA_TWO_OBJECTS = {**AGENT1_DATA_RICH, "fsc_objects": ["Suitability__c", "RiskProfile__c"]}
AGENT1_DATA_VERY_RICH = {**AGENT1_DATA_RICH, "description_word_count": 120}

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

MOCK_DATA_NEED_MINIMAL = {
    "required_records": [
        {
            "object_api_name": "Account",
            "min_record_count": 1,
            "key_field_values": {"Name": "Test Account"},
            "setup_method": "TestSetup",
        },
    ],
    "data_isolation_strategy": "per_class_setup",
    "sensitive_data_present": False,
    "sensitive_data_fields": [],
    "factory_classes_recommended": [],
    "data_dependencies_ordered": [],
    "data_volume": "minimal",
    "risks": ["No FSC objects involved — minimal data risk."],
}


class TestConfidenceScoring:
    def test_rich_agent1_data_scores_high(self):
        score, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert score >= 75

    def test_sparse_agent1_data_scores_lower(self):
        score_rich, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        score_sparse, _ = _compute_confidence(AGENT1_DATA_SPARSE, MOCK_DATA_NEED_MINIMAL)
        assert score_rich > score_sparse

    def test_no_agent1_data_reduces_confidence(self):
        with_agent1, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        without_agent1, _ = _compute_confidence(None, MOCK_DATA_NEED_COMPLEX)
        assert with_agent1 > without_agent1

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, MOCK_DATA_NEED_MINIMAL)
        assert score >= 20

    def test_dependency_chain_boosts_confidence(self):
        score_complex, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        simple = {**MOCK_DATA_NEED_COMPLEX, "data_dependencies_ordered": []}
        score_simple, _ = _compute_confidence(AGENT1_DATA_RICH, simple)
        assert score_complex > score_simple

    def test_sensitive_data_boosts_confidence(self):
        """Identifying sensitive data fields signals thorough analysis."""
        score_with, _ = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        no_sensitive = {**MOCK_DATA_NEED_COMPLEX, "sensitive_data_present": False}
        score_without, _ = _compute_confidence(AGENT1_DATA_RICH, no_sensitive)
        assert score_with > score_without

    def test_fsc_objects_present_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_TWO_OBJECTS, MOCK_DATA_NEED_COMPLEX)
        assert "fsc_objects_present" in signals
        assert signals["fsc_objects_present"] == 2

    def test_description_rich_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_VERY_RICH, MOCK_DATA_NEED_COMPLEX)
        assert "description_rich" in signals
        assert signals["description_rich"] == 120

    def test_volume_complexity_mismatch_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_SPARSE, MOCK_DATA_NEED_COMPLEX)
        assert "volume_complexity_mismatch" in signals

    def test_fsc_objects_rich_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert "fsc_objects_rich" in signals
        assert signals["fsc_objects_rich"] == 4

    def test_fsc_objects_absent_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_SPARSE, MOCK_DATA_NEED_MINIMAL)
        assert "fsc_objects_absent" in signals
        assert signals["fsc_objects_absent"] == 0

    def test_description_moderate_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert "description_moderate" in signals

    def test_description_sparse_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_SPARSE, MOCK_DATA_NEED_MINIMAL)
        assert "description_sparse" in signals

    def test_dependency_chain_found_stores_count(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert signals["dependency_chain_found"] == 5

    def test_sensitive_data_identified_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert "sensitive_data_identified" in signals
        assert signals["sensitive_data_identified"] is True

    def test_volume_complexity_aligned_key_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, MOCK_DATA_NEED_COMPLEX)
        assert "volume_complexity_aligned" in signals
        assert signals["volume_complexity_aligned"] is True


@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert result.agent_id == 7
        assert result.agent_name == "Data Need Agent"
        assert result.confidence.tier == "B"

    async def test_required_records_count_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert result.data["required_record_count"] == 4
        assert result.data["sensitive_data_present"] is True

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        for key in ["required_records", "data_isolation_strategy", "sensitive_data_fields",
                    "data_dependencies_ordered", "risks"]:
            assert key in result.data

    async def test_runs_standalone(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert result.agent_id == 7

    async def test_no_context_causes_escalation(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_MINIMAL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_data_volume_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert result.data["data_volume"] == "complex"

    async def test_factory_classes_recommended_is_list(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert isinstance(result.data["factory_classes_recommended"], list)

    async def test_signals_key_in_data_is_dict(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_RICH}

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert isinstance(result.data["signals"], dict)


# ── REQ-05: fca_context_available + defensive default tests ──────────────────

MOCK_DATA_NEED_SHARED_ORG = {
    **MOCK_DATA_NEED_COMPLEX,
    "data_isolation_strategy": "shared_org_data",
    "fca_context_available": False,
}


@pytest.mark.asyncio
class TestFcaContextAvailableREQ05:
    async def test_fca_context_available_always_false(self):
        """REQ-05: Agent 07 always emits fca_context_available=False (runs before Agent 03)."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX
            result = await run(state)

        assert "fca_context_available" in result.data
        assert result.data["fca_context_available"] is False

    async def test_shared_org_data_overridden_to_per_class_setup(self):
        """REQ-05: LLM returning shared_org_data must be overridden to per_class_setup."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_SHARED_ORG
            result = await run(state)

        assert result.data["data_isolation_strategy"] != "shared_org_data", (
            "shared_org_data must be overridden when fca_context_available=False"
        )
        assert result.data["data_isolation_strategy"] == "per_class_setup"

    async def test_per_class_setup_not_overridden(self):
        """REQ-05: per_class_setup must not be changed by the defensive override."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_07_data_need.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_07_data_need.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_llm.return_value = MOCK_DATA_NEED_COMPLEX  # isolation=per_class_setup
            result = await run(state)

        assert result.data["data_isolation_strategy"] == "per_class_setup"


# ── Tests: prompt content ─────────────────────────────────────────────────────

class TestPromptContent:
    def test_prompt_includes_story_id(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA_RICH)
        assert "FSC-2417" in msg

    def test_prompt_includes_summary(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA_RICH)
        assert STORY_SUITABILITY["summary"] in msg

    def test_prompt_includes_components(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA_RICH)
        assert "COMPONENTS:" in msg

    def test_prompt_empty_components_renders_as_none(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, AGENT1_DATA_RICH)
        assert "COMPONENTS: None" in msg

    def test_prompt_empty_description_shows_empty(self):
        msg = _build_user_message(STORY_NO_DESCRIPTION, AGENT1_DATA_RICH)
        assert "(empty)" in msg

    def test_prompt_includes_agent1_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA_RICH)
        assert "AGENT 1 — STORY INTENT:" in msg
        assert "Story Summary:" in msg

    def test_prompt_agent1_section_absent_when_no_agent1_data(self):
        msg = _build_user_message(STORY_SUITABILITY, None)
        assert "AGENT 1 — STORY INTENT:" not in msg

    def test_prompt_ends_with_tool_instruction(self):
        msg = _build_user_message(STORY_SUITABILITY, AGENT1_DATA_RICH)
        assert _TOOL_NAME in msg
        assert msg.strip().endswith("assessment.")


# ── Tests: schema contract ────────────────────────────────────────────────────

class TestSchemaContract:
    def test_tool_schema_has_nine_required_fields(self):
        expected = {
            "required_records",
            "data_isolation_strategy",
            "sensitive_data_present",
            "sensitive_data_fields",
            "factory_classes_recommended",
            "data_dependencies_ordered",
            "data_volume",
            "risks",
            "fca_context_available",
        }
        assert set(_TOOL_SCHEMA["required"]) == expected

    def test_required_records_is_array(self):
        assert _TOOL_SCHEMA["properties"]["required_records"]["type"] == "array"

    def test_record_schema_has_four_required_fields(self):
        record_schema = _TOOL_SCHEMA["properties"]["required_records"]["items"]
        expected = {"object_api_name", "min_record_count", "key_field_values", "setup_method"}
        assert set(record_schema["required"]) == expected

    def test_setup_method_enum_has_four_values(self):
        record_schema = _TOOL_SCHEMA["properties"]["required_records"]["items"]
        assert record_schema["properties"]["setup_method"]["enum"] == [
            "TestSetup", "TestFactory", "StaticData", "MockData"
        ]

    def test_min_record_count_has_minimum_one(self):
        record_schema = _TOOL_SCHEMA["properties"]["required_records"]["items"]
        assert record_schema["properties"]["min_record_count"]["minimum"] == 1

    def test_data_isolation_strategy_enum_has_three_values(self):
        assert _TOOL_SCHEMA["properties"]["data_isolation_strategy"]["enum"] == [
            "per_test_setup_teardown", "per_class_setup", "shared_org_data"
        ]

    def test_data_volume_enum_has_three_values(self):
        assert _TOOL_SCHEMA["properties"]["data_volume"]["enum"] == ["minimal", "moderate", "complex"]

    def test_sensitive_data_present_is_boolean(self):
        assert _TOOL_SCHEMA["properties"]["sensitive_data_present"]["type"] == "boolean"

    def test_fca_context_available_is_boolean(self):
        assert _TOOL_SCHEMA["properties"]["fca_context_available"]["type"] == "boolean"

    def test_sensitive_data_fields_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["sensitive_data_fields"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_risks_is_array_of_strings(self):
        prop = _TOOL_SCHEMA["properties"]["risks"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"
