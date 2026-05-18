"""Tests for Agent 6 — Test Design Strategy."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_06_test_design import _compute_confidence, run
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
