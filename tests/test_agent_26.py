"""Tests for Agent 26 — CRT Scenario Designer (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_26_crt_scenario_designer import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT19_FULL = {
    "scenario_count": 4,
    "fca_coverage_present": True,
    "gherkin_scenarios": [
        {"title": "HIGH-risk client fails suitability", "tags": ["@fca", "@negative"],
         "steps": ["Given HIGH-risk client", "When suitability runs", "Then FAIL"]},
        {"title": "Valid portfolio rebalancing", "tags": ["@smoke"],
         "steps": ["Given valid portfolio", "When rebalancing", "Then FSC rules"]},
        {"title": "Vulnerable customer detected", "tags": ["@fca"],
         "steps": ["Given VCI client", "When check runs", "Then alert raised"]},
        {"title": "Boundary score at threshold", "tags": ["@boundary"],
         "steps": ["Given score=50", "When assessed", "Then PASSED"]},
    ],
}

AGENT19_EMPTY = {"scenario_count": 0, "gherkin_scenarios": [], "fca_coverage_present": False}

AGENT21_DATA = {
    "test_data_strategy": {
        "seed_records": [
            {"object_name": "FinancialAccount", "record_count": 3, "key_fields": [], "purpose": ""},
            {"object_name": "Suitability__c", "record_count": 2, "key_fields": [], "purpose": ""},
        ]
    }
}

MOCK_CRT_PASS = {
    "crt_test_cases": [
        {
            "test_id": "CRT-001",
            "title": "HIGH-risk client fails suitability check",
            "tags": ["@fca", "@negative", "@smoke"],
            "steps": [
                {"action": "navigate", "target": "Suitability Record Page", "value": ""},
                {"action": "assert", "target": "Suitability Score field", "value": "< 50"},
                {"action": "assert", "target": "Assessment Result", "value": "FAILED"},
            ],
            "data_references": ["FinancialAccount", "Suitability__c"],
        },
    ],
    "crt_test_count": 1,
    "automation_coverage": 25.0,
    "crt_design_verdict": "PARTIAL",
    "design_notes": "3 of 4 scenarios require manual inspection for FCA compliance assertion.",
}

MOCK_CRT_FULL = {
    "crt_test_cases": [
        {"test_id": f"CRT-{i:03d}", "title": f"Test {i}", "tags": ["@smoke"],
         "steps": [{"action": "assert", "target": "Field", "value": "Value"}],
         "data_references": []}
        for i in range(1, 5)
    ],
    "crt_test_count": 4,
    "automation_coverage": 100.0,
    "crt_design_verdict": "PASS",
    "design_notes": "",
}

MOCK_CRT_INCOMPLETE = {
    "crt_test_cases": [],
    "crt_test_count": 0,
    "automation_coverage": 0.0,
    "crt_design_verdict": "INCOMPLETE",
    "design_notes": "No Gherkin scenarios available to design from.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_scenarios_high_coverage_scores_well(self):
        score, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        assert score >= 75

    def test_no_gherkin_scenarios_heavily_penalised(self):
        score_with, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        score_without, _ = _compute_confidence(AGENT19_EMPTY, None, 0, 0.0, "INCOMPLETE")
        assert score_with > score_without

    def test_low_automation_coverage_penalised(self):
        score_high, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        score_low, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 1, 20.0, "PARTIAL")
        assert score_high > score_low

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 5, 100.0, "PASS")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, 0, 0.0, "INCOMPLETE")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_FULL}
        state["agent_results"]["21"] = {"data": AGENT21_DATA}

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert result.agent_id == 26
        assert result.agent_name == "CRT Scenario Designer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        for key in ["crt_test_cases", "crt_test_count",
                    "automation_coverage", "crt_design_verdict"]:
            assert key in result.data

    async def test_pass_verdict_with_full_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["19"] = {"data": AGENT19_FULL}

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert result.data["crt_design_verdict"] == "PASS"
        assert result.data["crt_test_count"] == 4

    async def test_incomplete_when_no_scenarios(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["19"] = {"data": AGENT19_EMPTY}

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_INCOMPLETE
            result = await run(state)

        assert result.data["crt_design_verdict"] == "INCOMPLETE"
        assert result.data["crt_test_count"] == 0

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"


# ── REQ-17: Scenario truncation detection ─────────────────────────────────────

class TestScenarioTruncationREQ17:
    def test_truncation_flag_in_confidence(self):
        score_without, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS", scenarios_truncated=False)
        score_with, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS", scenarios_truncated=True)
        assert score_without > score_with

    def test_no_truncation_penalty_when_false(self):
        score, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS", scenarios_truncated=False)
        assert "crt_scenario_truncated" not in signals

    def test_manual_test_suppresses_low_coverage_penalty(self):
        score_with_manual, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 2, 20.0, "PARTIAL", manual_test_present=True)
        score_without_manual, _ = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 2, 20.0, "PARTIAL", manual_test_present=False)
        assert score_with_manual >= score_without_manual


@pytest.mark.asyncio
class TestScenarioTruncationRunREQ17:
    async def test_12_scenarios_gives_scenarios_truncated_true(self):
        twelve_scenarios = [
            {"title": f"Scenario {i}", "tags": ["@smoke"],
             "steps": ["Given step", "When action", "Then result"]}
            for i in range(12)
        ]
        state = initial_story_state("FSC-2417")
        state["agent_results"]["19"] = {"data": {
            "scenario_count": 12,
            "fca_coverage_present": True,
            "gherkin_scenarios": twelve_scenarios,
        }}

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert result.data["scenarios_truncated"] is True
        assert result.data["truncated_scenario_count"] == 2

    async def test_4_scenarios_gives_scenarios_truncated_false(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["19"] = {"data": AGENT19_FULL}

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert result.data["scenarios_truncated"] is False
        assert result.data["truncated_scenario_count"] == 0
