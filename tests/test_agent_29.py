"""Tests for Agent 29 — UAT Test Case Generator (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_29_uat_test_case_generator import (
    _build_prompt,
    _compute_confidence,
    _UAT_TOOL_NAME,
    _UAT_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

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
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        assert score >= 70

    def test_no_acs_heavily_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        score_without, _ = _compute_confidence([], AGENT3_HIGH, None, AGENT19_DATA, 0, "HIGH", False)
        assert score_with > score_without

    def test_high_fca_without_co_flag_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        score_without, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", False)
        assert score_with > score_without

    def test_low_fca_no_co_not_penalised(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_LOW, None, AGENT19_DATA, 2, "LOW", False)
        assert score >= 60

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 3, "HIGH", True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence([], None, None, None, 0, "LOW", False)
        assert score >= 20

    def test_acs_available_key_and_value(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        assert signals["acs_available"] == 2

    def test_no_acs_available_key_in_signals(self):
        _, signals = _compute_confidence([], AGENT3_HIGH, None, None, 0, "HIGH", True)
        assert "no_acs_available" in signals

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        assert "fca_classification_available" in signals

    def test_refined_ac_clauses_available_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_WITH_CLAUSES,
                                         AGENT19_DATA, 2, "HIGH", True)
        assert "refined_ac_clauses_available" in signals

    def test_gherkin_context_available_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        assert "gherkin_context_available" in signals

    def test_gherkin_context_not_added_when_scenario_count_zero(self):
        agent19_empty = {"scenario_count": 0, "gherkin_scenarios": []}
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, agent19_empty, 2, "HIGH", True)
        assert "gherkin_context_available" not in signals

    def test_uat_tests_generated_key_and_value(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, AGENT19_DATA, 2, "HIGH", True)
        assert signals["uat_tests_generated"] == 2

    def test_no_uat_tests_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, None, None, 0, "HIGH", True)
        assert "no_uat_tests" in signals

    def test_regulated_story_co_not_flagged_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_MEDIUM, None, None, 2, "MEDIUM", False)
        assert "regulated_story_co_not_flagged" in signals


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

    async def test_escalated_when_no_upstream_data(self):
        # base=67, no_acs→-15, no_uat_tests→-10 = 42 < 60
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

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
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

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
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

        assert isinstance(result.data["signals"], dict)

    async def test_medium_fca_co_not_overridden(self):
        """MEDIUM-FCA: code falls through both branches, accepting LLM co_required as-is."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_MEDIUM}

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
            mock_sonnet.return_value = MOCK_UAT_CO_FALSE
            result = await run(state)

        assert result.data["co_sign_off_required"] is False


# ── REQ-20: CO override, VC instruction, Agent 05 consumption ─────────────────

MOCK_UAT_CO_FALSE = {
    "uat_test_cases": [{"test_id": "UAT-001", "title": "Test", "ac_reference": "AC1",
                        "preconditions": [], "steps": [], "expected_result": "Pass",
                        "regulatory_flag": False}],
    "uat_test_count": 1,
    "co_sign_off_required": False,   # LLM incorrectly returns False for HIGH-FCA
    "uat_verdict": "PASS",
    "regulatory_assertions": [],
}

MOCK_UAT_WITH_VC = {
    "uat_test_cases": [
        {"test_id": "UAT-001", "title": "Vulnerable customer receives appropriate outcome",
         "ac_reference": "AC1", "preconditions": ["Customer flagged as vulnerable"],
         "steps": ["Step 1"], "expected_result": "FG21/1 outcome met", "regulatory_flag": True},
    ],
    "uat_test_count": 1,
    "co_sign_off_required": True,
    "uat_verdict": "PASS",
    "regulatory_assertions": ["FG21/1 vulnerable customer outcome verified"],
}

AGENT5_WITH_CLAUSES = {
    "ac_count": 2,
    "ac_clauses": [
        {"description": "AC1: System blocks unsuitable products for HIGH-risk clients", "scenario_type": "regulatory"},
        {"description": "AC2: Vulnerable customer alert raised", "scenario_type": "vulnerable_customer"},
    ],
}

AGENT21_WITH_VC_PROFILES = {
    "vulnerable_profiles": ["Customer with cognitive impairment", "Customer in financial difficulty"],
    "data_verdict": "PASS",
}


MOCK_STORY_SIMPLE = {"summary": "Test story", "description": "Test description", "acceptance_criteria": []}


@pytest.mark.asyncio
class TestCOOverrideREQ20:
    async def test_high_fca_co_false_overridden_to_true(self):
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
            mock_story.return_value = MOCK_STORY_SIMPLE
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_UAT_CO_FALSE
            result = await run(state)

        assert result.data["co_sign_off_required"] is True

    async def test_low_fca_co_false_not_overridden(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_LOW}
        state["agent_results"]["21"] = {"data": AGENT21_DATA}

        with (
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_29_uat_test_case_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.testing.agent_29_uat_test_case_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY_SIMPLE
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_UAT_CO_FALSE
            result = await run(state)

        assert result.data["co_sign_off_required"] is False

    async def test_agent5_data_improves_confidence(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["5"]  = {"data": AGENT5_WITH_CLAUSES}
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
            mock_story.return_value = MOCK_STORY_SIMPLE
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_UAT_PASS
            result = await run(state)

        assert result.confidence.raw_score > 0


# ── Prompt builder unit tests ─────────────────────────────────────────────────

_AC_CLAUSES = [
    {"description": "Block unsuitable products for HIGH-risk", "scenario_type": "regulatory"},
    {"description": "Raise alert for vulnerable customers", "scenario_type": "vulnerable_customer"},
]

_GHERKIN = [{"title": "HIGH-risk suitability fails", "tags": ["@fca"]}]
_VP = ["Customer with cognitive impairment"]


class TestBuildPrompt:
    def test_includes_story_id(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [])
        assert "FSC-2417" in prompt

    def test_includes_fca_class(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [])
        assert "HIGH" in prompt

    def test_no_acs_and_no_clauses_shows_placeholder(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, [], "HIGH", [], [])
        assert "(no acceptance criteria available)" in prompt

    def test_raw_acs_shown_when_no_clauses(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [])
        assert "AC1" in prompt

    def test_ac_clauses_used_when_present(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [],
                               ac_clauses=_AC_CLAUSES)
        assert "Block unsuitable products" in prompt
        assert "[regulatory]" in prompt

    def test_gherkin_titles_shown_when_present(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", _GHERKIN, [])
        assert "HIGH-risk suitability fails" in prompt

    def test_no_gherkin_shows_none(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [])
        assert "['none']" in prompt

    def test_vulnerable_profiles_shown(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], _VP)
        assert "cognitive impairment" in prompt

    def test_no_vulnerable_profiles_shows_none(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "HIGH", [], [])
        # Both gherkin and VP show ['none'] when absent; VP is last
        assert "['none']" in prompt

    def test_ends_with_tool_name(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, MOCK_ACS, "LOW", [], [])
        assert _UAT_TOOL_NAME in prompt
        assert prompt.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_five_required_fields(self):
        assert set(_UAT_TOOL_SCHEMA["required"]) == {
            "uat_test_cases", "uat_test_count", "co_sign_off_required",
            "uat_verdict", "regulatory_assertions",
        }

    def test_uat_verdict_enum_has_three_values(self):
        assert _UAT_TOOL_SCHEMA["properties"]["uat_verdict"]["enum"] == [
            "PASS", "WARN", "INCOMPLETE"
        ]

    def test_test_case_items_have_seven_required_fields(self):
        item_schema = _UAT_TOOL_SCHEMA["properties"]["uat_test_cases"]["items"]
        assert set(item_schema["required"]) == {
            "test_id", "title", "ac_reference", "preconditions",
            "steps", "expected_result", "regulatory_flag",
        }

    def test_regulatory_assertions_is_array(self):
        assert _UAT_TOOL_SCHEMA["properties"]["regulatory_assertions"]["type"] == "array"
