"""Tests for Agent 26 — CRT Scenario Designer (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_26_crt_scenario_designer import (
    _build_prompt,
    _compute_confidence,
    _CRT_TOOL_NAME,
    _CRT_TOOL_SCHEMA,
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

    def test_gherkin_scenarios_available_key_and_value(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        assert signals["gherkin_scenarios_available"] == 4

    def test_no_gherkin_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(AGENT19_EMPTY, None, 0, 0.0, "INCOMPLETE")
        assert "no_gherkin_scenarios" in signals

    def test_test_data_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        assert "test_data_available" in signals

    def test_crt_tests_designed_key_and_value(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        assert signals["crt_tests_designed"] == 4

    def test_no_crt_tests_key_in_signals(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 0, 0.0, "INCOMPLETE")
        assert "no_crt_tests" in signals

    def test_high_automation_coverage_key_and_value(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS")
        assert signals["high_automation_coverage"] == 100.0

    def test_low_automation_coverage_key_in_signals(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 1, 20.0, "PARTIAL")
        assert "low_automation_coverage" in signals

    def test_incomplete_design_key_in_signals(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 0, 0.0, "INCOMPLETE")
        assert "incomplete_design" in signals

    def test_scenarios_truncated_key_in_signals_when_true(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS",
                                         scenarios_truncated=True)
        assert "scenarios_truncated" in signals

    def test_scenarios_truncated_not_in_signals_when_false(self):
        _, signals = _compute_confidence(AGENT19_FULL, AGENT21_DATA, 4, 100.0, "PASS",
                                         scenarios_truncated=False)
        assert "scenarios_truncated" not in signals


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

    async def test_escalated_when_no_upstream_data(self):
        # base=68, no gherkin→-15, no crt tests→-10, INCOMPLETE→-10 = 33 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_INCOMPLETE
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_26_crt_scenario_designer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRT_FULL
            result = await run(state)

        assert isinstance(result.data["signals"], dict)


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


# ── Prompt builder unit tests ─────────────────────────────────────────────────

_SIMPLE_SCENARIO = [{"title": "Test scenario", "tags": ["@smoke"],
                     "steps": ["Given step", "When action", "Then result"]}]

_SEED_RECORDS = [{"object_name": "FinancialAccount", "record_count": 2,
                  "key_fields": [], "purpose": ""}]


class TestBuildPrompt:
    def test_includes_story_id(self):
        prompt = _build_prompt("FSC-2417", "HIGH", [], [], "SMOKE")
        assert "FSC-2417" in prompt

    def test_includes_fca_class(self):
        prompt = _build_prompt("FSC-2417", "HIGH", [], [], "SMOKE")
        assert "HIGH" in prompt

    def test_includes_regression_suite(self):
        prompt = _build_prompt("FSC-2417", "LOW", [], [], "FULL")
        assert "FULL" in prompt

    def test_no_scenarios_shows_placeholder(self):
        prompt = _build_prompt("FSC-2417", "LOW", [], [], "SMOKE")
        assert "(no Gherkin scenarios available)" in prompt

    def test_scenario_titles_shown_when_present(self):
        prompt = _build_prompt("FSC-2417", "HIGH", _SIMPLE_SCENARIO, [], "SMOKE")
        assert "Test scenario" in prompt

    def test_truncation_note_when_truncated(self):
        prompt = _build_prompt("FSC-2417", "HIGH", _SIMPLE_SCENARIO, [], "SMOKE",
                               scenarios_truncated=True, truncated_count=3)
        assert "omitted" in prompt

    def test_no_truncation_note_when_not_truncated(self):
        prompt = _build_prompt("FSC-2417", "HIGH", _SIMPLE_SCENARIO, [], "SMOKE",
                               scenarios_truncated=False)
        assert "omitted" not in prompt

    def test_manual_note_when_present(self):
        prompt = _build_prompt("FSC-2417", "LOW", [], [], "SMOKE",
                               manual_test_present=True)
        assert "ManualTest" in prompt

    def test_no_manual_note_when_absent(self):
        prompt = _build_prompt("FSC-2417", "LOW", [], [], "SMOKE",
                               manual_test_present=False)
        assert "ManualTest" not in prompt

    def test_seed_names_in_prompt(self):
        prompt = _build_prompt("FSC-2417", "HIGH", [], _SEED_RECORDS, "SMOKE")
        assert "FinancialAccount" in prompt

    def test_no_seed_records_shows_none(self):
        prompt = _build_prompt("FSC-2417", "HIGH", [], [], "SMOKE")
        assert "['none']" in prompt

    def test_ends_with_tool_name(self):
        prompt = _build_prompt("FSC-2417", "LOW", [], [], "SMOKE")
        assert _CRT_TOOL_NAME in prompt
        assert prompt.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_five_required_fields(self):
        assert set(_CRT_TOOL_SCHEMA["required"]) == {
            "crt_test_cases", "crt_test_count", "automation_coverage",
            "crt_design_verdict", "design_notes",
        }

    def test_crt_design_verdict_enum_has_three_values(self):
        assert _CRT_TOOL_SCHEMA["properties"]["crt_design_verdict"]["enum"] == [
            "PASS", "PARTIAL", "INCOMPLETE"
        ]

    def test_test_case_items_have_five_required_fields(self):
        item_schema = _CRT_TOOL_SCHEMA["properties"]["crt_test_cases"]["items"]
        assert set(item_schema["required"]) == {
            "test_id", "title", "tags", "steps", "data_references"
        }

    def test_step_items_have_three_required_fields(self):
        step_schema = (
            _CRT_TOOL_SCHEMA["properties"]["crt_test_cases"]["items"]
            ["properties"]["steps"]["items"]
        )
        assert set(step_schema["required"]) == {"action", "target", "value"}
