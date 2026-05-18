"""Tests for Agent 29 — UAT Test Case Generator (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_29_uat_test_case_generator import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT19_DATA = {
    "scenario_count": 3,
    "gherkin_scenarios": [
        {"title": "HIGH-risk client suitability fails", "tags": ["@fca"]},
        {"title": "Valid portfolio", "tags": ["@smoke"]},
    ],
}

AGENT21_DATA = {
    "vulnerable_profiles": ["VCI_01: Cognitive impairment"],
    "data_verdict": "PASS",
}

MOCK_ACS = [
    {"id": "AC1", "description": "Given HIGH-risk client, suitability must fail if score < 50"},
    {"id": "AC2", "description": "Given VCI flag, additional review is triggered"},
]

MOCK_STORY = {"key": "FSC-2417", "summary": "Suitability Enhancement"}

MOCK_UAT_PASS = {
    "uat_test_cases": [
        {
            "test_id": "UAT-001",
            "title": "HIGH-risk client fails suitability assessment",
            "ac_reference": "AC1",
            "preconditions": ["Logged in as Adviser", "Test account with HIGH risk profile exists"],
            "steps": [
                "Navigate to the Suitability Assessment record for FSC-TEST-001",
                "Click 'Run Assessment'",
                "Observe the Assessment Result field",
            ],
            "expected_result": "Assessment Result shows FAILED and regulatory alert is visible",
            "regulatory_flag": True,
        },
        {
            "test_id": "UAT-002",
            "title": "Vulnerable Customer review triggered",
            "ac_reference": "AC2",
            "preconditions": ["VCI flag set on client record"],
            "steps": [
                "Navigate to the Client record",
                "Trigger any financial assessment",
                "Observe whether additional review task is created",
            ],
            "expected_result": "Additional review task created and assigned to compliance team",
            "regulatory_flag": True,
        },
    ],
    "uat_test_count": 2,
    "co_sign_off_required": True,
    "uat_verdict": "PASS",
    "regulatory_assertions": [
        "Suitability assessment complies with COBS 9 — unsuitable products blocked for HIGH-risk clients",
        "Vulnerable Customer provisions per FG21/1 triggered correctly",
    ],
}

MOCK_UAT_INCOMPLETE = {
    "uat_test_cases": [],
    "uat_test_count": 0,
    "co_sign_off_required": False,
    "uat_verdict": "INCOMPLETE",
    "regulatory_assertions": ["No ACs available to generate UAT tests from"],
}

MOCK_UAT_WARN = {
    "uat_test_cases": [
        {
            "test_id": "UAT-001",
            "title": "Basic suitability pass",
            "ac_reference": "AC1",
            "preconditions": [],
            "steps": ["Navigate to suitability", "Check result"],
            "expected_result": "Result is visible",
            "regulatory_flag": False,
        }
    ],
    "uat_test_count": 1,
    "co_sign_off_required": False,
    "uat_verdict": "WARN",
    "regulatory_assertions": [],
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_high_fca_scores_well(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT19_DATA, 2, "HIGH", True)
        assert score >= 70

    def test_no_acs_heavily_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT19_DATA, 2, "HIGH", True)
        score_without, _ = _compute_confidence([], AGENT3_HIGH, AGENT19_DATA, 0, "HIGH", False)
        assert score_with > score_without

    def test_high_fca_without_co_flag_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT19_DATA, 2, "HIGH", True)
        score_without, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT19_DATA, 2, "HIGH", False)
        assert score_with > score_without

    def test_low_fca_no_co_not_penalised(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_LOW, AGENT19_DATA, 2, "LOW", False)
        assert score >= 60

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT19_DATA, 3, "HIGH", True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence([], None, None, 0, "LOW", False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}
        state["agent_results"]["21"] = {"data": AGENT21_DATA}

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_UAT_PASS
            result = await run(state)

        assert result.agent_id == 29
        assert result.agent_name == "UAT Test Case Generator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_UAT_PASS
            result = await run(state)

        for key in ["uat_test_cases", "uat_test_count",
                    "co_sign_off_required", "uat_verdict", "regulatory_assertions"]:
            assert key in result.data

    async def test_co_required_for_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_UAT_PASS
            result = await run(state)

        assert result.data["co_sign_off_required"] is True

    async def test_incomplete_without_acs(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_UAT_INCOMPLETE
            result = await run(state)

        assert result.data["uat_verdict"] == "INCOMPLETE"

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_UAT_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"
