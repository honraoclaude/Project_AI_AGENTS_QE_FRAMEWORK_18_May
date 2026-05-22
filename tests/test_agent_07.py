"""Tests for Agent 7 — Data Need Agent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_07_data_need import _compute_confidence, run
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
