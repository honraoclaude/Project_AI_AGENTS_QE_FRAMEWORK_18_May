"""Tests for Agent 21 — Test Data Architect (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_21_test_data_architect import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH", "ensemble_agreement": True}
AGENT3_LOW  = {"fca_classification": "LOW",  "ensemble_agreement": True}

AGENT5_DATA = {"ac_count": 3, "acs_generated": True}

AGENT7_DATA = {
    "data_requirements": [
        "FinancialAccount with balance > £100k",
        "Suitability record with HIGH risk profile",
    ],
    "data_complexity": "HIGH",
}

AGENT13_DATA = {
    "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
    "dependency_depth": 2,
}

AGENT19_DATA = {
    "gherkin_scenarios": [
        {
            "title": "HIGH-risk client fails suitability check",
            "tags": ["@fca", "@negative"],
            "steps": ["Given a HIGH-risk client", "When suitability runs", "Then it fails"],
        },
        {
            "title": "Valid portfolio rebalancing",
            "tags": ["@smoke"],
            "steps": ["Given a valid portfolio", "When rebalancing", "Then FSC rules apply"],
        },
    ],
    "scenario_count": 2,
}

MOCK_STORY = {
    "key": "FSC-2417",
    "summary": "Suitability Assessment Enhancement",
}

MOCK_DATA_PASS = {
    "seed_records": [
        {
            "object_name": "FinancialAccount",
            "record_count": 5,
            "key_fields": ["Balance__c", "RiskProfile__c"],
            "purpose": "Base account records for suitability tests",
        },
        {
            "object_name": "Suitability__c",
            "record_count": 3,
            "key_fields": ["Score__c", "Classification__c"],
            "purpose": "Suitability assessment records",
        },
    ],
    "requires_anonymisation": True,
    "anonymisation_fields": ["Name", "NationalInsurance__c", "DateOfBirth__c"],
    "vulnerable_profiles": [
        "VCI_01: Client with cognitive impairment marker",
        "VCI_02: Client in financial distress",
    ],
    "data_verdict": "PASS",
    "data_setup_notes": "Use TestDataFactory.createSuitabilityBundle() for setup.",
    "coverage_gaps": [],
}

MOCK_DATA_WARN = {
    "seed_records": [
        {
            "object_name": "FinancialAccount",
            "record_count": 2,
            "key_fields": ["Balance__c"],
            "purpose": "Basic accounts",
        },
    ],
    "requires_anonymisation": False,
    "anonymisation_fields": [],
    "vulnerable_profiles": [],
    "data_verdict": "WARN",
    "data_setup_notes": "Vulnerable Customer profiles not available — manual setup required.",
    "coverage_gaps": ["Vulnerable Customer scenario lacks data strategy"],
}

MOCK_DATA_INCOMPLETE = {
    "seed_records": [],
    "requires_anonymisation": False,
    "anonymisation_fields": [],
    "vulnerable_profiles": [],
    "data_verdict": "INCOMPLETE",
    "data_setup_notes": "Insufficient context to design data strategy.",
    "coverage_gaps": ["All scenarios lack data strategy"],
}


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score >= 75

    def test_no_data_needs_baseline_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, None, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score_with > score_without

    def test_no_gherkin_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, None,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score_with > score_without

    def test_high_fca_missing_vulnerable_profiles_penalised(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert score_with > score_without

    def test_incomplete_verdict_penalised(self):
        score_pass, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_incomplete, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=0, verdict="INCOMPLETE", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert score_pass > score_incomplete

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=5, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01", "VCI_02"],
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(
            None, None, None, None,
            seed_record_count=0, verdict="INCOMPLETE", fca_class="LOW",
            vulnerable_profiles=[],
        )
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["7"] = {"data": AGENT7_DATA}
        state["agent_results"]["13"] = {"data": AGENT13_DATA}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.agent_id == 21
        assert result.agent_name == "Test Data Architect"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        for key in ["test_data_strategy", "requires_anonymisation",
                    "vulnerable_profiles", "data_verdict"]:
            assert key in result.data

    async def test_pass_verdict_with_full_context(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["data_verdict"] == "PASS"
        assert result.data["requires_anonymisation"] is True
        assert len(result.data["vulnerable_profiles"]) >= 1

    async def test_warn_when_vulnerable_profiles_missing(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_WARN
            result = await run(state)

        assert result.data["data_verdict"] == "WARN"

    async def test_incomplete_without_context(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_INCOMPLETE
            result = await run(state)

        assert result.data["data_verdict"] == "INCOMPLETE"
        assert result.data["seed_record_count"] == 0

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"
