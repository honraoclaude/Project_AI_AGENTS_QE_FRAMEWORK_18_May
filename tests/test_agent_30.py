"""Tests for Agent 30 — FCA Scenario Agent (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_30_fca_scenario_agent import (
    _build_prompt,
    _compute_confidence,
    _FCA_TOOL_NAME,
    _FCA_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT4_HIGH = {"consumer_duty_risk": "HIGH", "consumer_duty_verdict": "REVIEW_REQUIRED"}
AGENT4_LOW  = {"consumer_duty_risk": "LOW",  "consumer_duty_verdict": "PASS"}

AGENT9_HIGH = {"risk_level": "HIGH", "risk_verdict": "CRITICAL"}
AGENT9_LOW  = {"risk_level": "LOW",  "risk_verdict": "LOW"}

AGENT19_DATA = {
    "scenario_count": 3,
    "gherkin_scenarios": [
        {"title": "HIGH-risk suitability fail", "tags": ["@fca"]},
    ],
}

MOCK_STORY = {"key": "FSC-2417", "summary": "Suitability Enhancement",
              "description": "Enhance suitability scoring for HIGH-risk clients."}

MOCK_FCA_PASS = {
    "fca_test_scenarios": [
        {
            "scenario_id": "FCA-001",
            "regulation": "COBS 9.2",
            "title": "Suitability assessment blocks unsuitable investment for HIGH-risk client",
            "description": "Verify COBS 9 compliance: system must block recommendations unsuitable for client risk profile",
            "pass_criteria": "System displays UNSUITABLE warning and blocks progression to order placement",
            "fail_criteria": "System allows investment to proceed without suitability warning",
        },
        {
            "scenario_id": "FCA-002",
            "regulation": "Consumer Duty PS22/9",
            "title": "Good outcome delivered for Vulnerable Customer",
            "description": "Verify Consumer Duty compliance for clients with VCI flag",
            "pass_criteria": "VCI-flagged client receives enhanced review task before any recommendation",
            "fail_criteria": "VCI flag is ignored and client proceeds without enhanced review",
        },
    ],
    "consumer_duty_covered": True,
    "cobs_scenarios_count": 1,
    "fca_scenario_verdict": "PASS",
    "regulatory_gaps": [],
}

MOCK_FCA_WARN = {
    "fca_test_scenarios": [
        {
            "scenario_id": "FCA-001",
            "regulation": "Consumer Duty PS22/9",
            "title": "Basic Consumer Duty check",
            "description": "Minimal Consumer Duty scenario for LOW-FCA story",
            "pass_criteria": "Client receives appropriate information",
            "fail_criteria": "Client does not receive required information",
        }
    ],
    "consumer_duty_covered": True,
    "cobs_scenarios_count": 0,
    "fca_scenario_verdict": "WARN",
    "regulatory_gaps": ["MiFID II Article 25 appropriateness not tested"],
}

MOCK_FCA_FAIL = {
    "fca_test_scenarios": [],
    "consumer_duty_covered": False,
    "cobs_scenarios_count": 0,
    "fca_scenario_verdict": "FAIL",
    "regulatory_gaps": [
        "COBS 9 suitability not tested",
        "Consumer Duty not tested",
        "Vulnerable Customer provisions not tested",
    ],
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_high_fca_context_scores_well(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS"
        )
        assert score >= 75

    def test_no_fca_classification_heavily_penalised(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS"
        )
        score_without, _ = _compute_confidence(
            None, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS"
        )
        assert score_with > score_without

    def test_high_fca_consumer_duty_not_covered_penalised(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS"
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", False, "PASS"
        )
        assert score_with > score_without

    def test_fail_verdict_reduces_confidence(self):
        score_pass, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS"
        )
        score_fail, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 0, "HIGH", False, "FAIL"
        )
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 3, "HIGH", True, "PASS"
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, 0, "LOW", False, "FAIL")
        assert score >= 20

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS")
        assert "fca_classification_available" in signals

    def test_no_fca_classification_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS")
        assert "no_fca_classification" in signals

    def test_consumer_duty_assessment_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS")
        assert "consumer_duty_assessment_available" in signals

    def test_risk_anticipation_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS")
        assert "risk_anticipation_available" in signals

    def test_fca_scenarios_generated_stores_count(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", True, "PASS")
        assert signals["fca_scenarios_generated"] == 2

    def test_no_fca_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 0, "HIGH", True, "PASS")
        assert "no_fca_scenarios" in signals

    def test_regulated_story_consumer_duty_not_covered_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 2, "HIGH", False, "PASS")
        assert "regulated_story_consumer_duty_not_covered" in signals

    def test_medium_fca_consumer_duty_not_covered_penalised(self):
        score_covered, _ = _compute_confidence(AGENT3_MEDIUM, AGENT4_HIGH, AGENT9_HIGH, 2, "MEDIUM", True, "PASS")
        score_not_covered, signals = _compute_confidence(AGENT3_MEDIUM, AGENT4_HIGH, AGENT9_HIGH, 2, "MEDIUM", False, "PASS")
        assert score_covered > score_not_covered
        assert "regulated_story_consumer_duty_not_covered" in signals

    def test_fca_scenario_fail_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT4_HIGH, AGENT9_HIGH, 0, "HIGH", False, "FAIL")
        assert "fca_scenario_fail" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["4"]  = {"data": AGENT4_HIGH}
        state["agent_results"]["9"]  = {"data": AGENT9_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert result.agent_id == 30
        assert result.agent_name == "FCA Scenario Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        for key in ["fca_test_scenarios", "consumer_duty_covered",
                    "cobs_scenarios_count", "fca_scenario_verdict", "regulatory_gaps"]:
            assert key in result.data

    async def test_consumer_duty_covered_for_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["4"] = {"data": AGENT4_HIGH}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert result.data["consumer_duty_covered"] is True
        assert result.data["fca_scenario_verdict"] == "PASS"

    async def test_fail_verdict_when_no_fca_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_FAIL
            result = await run(state)

        assert result.data["fca_scenario_verdict"] == "FAIL"
        assert len(result.data["regulatory_gaps"]) >= 1

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_escalated_when_no_upstream_data(self):
        # base=70, no_fca_classification→-10, no_fca_scenarios→-10, fca_scenario_fail→-8 = 42 < 60
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_medium_fca_consumer_duty_penalty_in_signals(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_MEDIUM}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = {**MOCK_FCA_PASS, "consumer_duty_covered": False}
            result = await run(state)

        assert "regulated_story_consumer_duty_not_covered" in result.data["signals"]


# ── Ensemble and TA integration tests ────────────────────────────────────────

@pytest.mark.asyncio
class TestEnsembleAndTA:
    async def test_ensemble_agreement_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert "ensemble_agreement" in result.data
        assert isinstance(result.data["ensemble_agreement"], bool)

    async def test_ta_position_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert "ta_position" in result.data
        assert "interaction_mode" in result.data
        assert result.data["ta_position"] in ("OK_OK", "OK_NOT_OK", "NOT_OK_OK", "NOT_OK_NOT_OK")

    async def test_call_scenario_counts_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert "call_a_scenario_count" in result.data
        assert "call_b_scenario_count" in result.data
        assert isinstance(result.data["call_a_scenario_count"], int)
        assert isinstance(result.data["call_b_scenario_count"], int)

    async def test_both_calls_same_mock_agreement_true(self):
        """When both LLM calls return the same mock, ensemble_agreement should be True."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            result = await run(state)

        assert result.data["ensemble_agreement"] is True

    async def test_ensemble_agreement_uses_call_a_result(self):
        """REQ-21 Bug 2: when ensemble_agreement=True, call_a (permissive) result is used."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        # call_a returns WARN (1 scenario), call_b returns PASS (2 scenarios)
        # When they agree → call_a should be used (both return PASS → agreement → call_a)
        # Use side_effect to return different values: first call=call_a, second=call_b
        # Both return same verdict (PASS) so they agree — result should be call_a (1 scenario)
        call_a_result = {**MOCK_FCA_PASS, "fca_test_scenarios": [MOCK_FCA_PASS["fca_test_scenarios"][0]]}
        call_b_result = MOCK_FCA_PASS  # 2 scenarios

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.side_effect = [call_a_result, call_b_result]
            result = await run(state)

        assert result.data["ensemble_agreement"] is True
        # call_a scenario count (1) should be used, not call_b (2)
        assert result.data["fca_scenario_count"] == result.data["call_a_scenario_count"]

    async def test_prompt_ends_with_tool_instruction_when_no_existing_scenarios(self):
        """REQ-21 Bug 1: prompt must end with tool instruction even when no existing scenarios."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        # No agent_results["19"] → no existing Gherkin scenarios → existing_titles is empty

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            await run(state)

        # Both calls should have the tool instruction
        for call in mock_sonnet.call_args_list:
            user_msg = call.kwargs.get("user_message", "")
            assert "generate" in user_msg.lower() or "tool" in user_msg.lower(), (
                "Prompt must include tool instruction regardless of existing scenarios"
            )


# ── REQ-21 Bug 3: vulnerable_customer_impact wired into prompt ────────────────

AGENT4_WITH_VC = {
    "consumer_duty_risk": "HIGH",
    "consumer_duty_verdict": "REVIEW_REQUIRED",
    "vulnerable_customer_impact": True,
}

AGENT4_WITHOUT_VC = {
    "consumer_duty_risk": "HIGH",
    "consumer_duty_verdict": "REVIEW_REQUIRED",
    "vulnerable_customer_impact": False,
}


@pytest.mark.asyncio
class TestVulnerableCustomerImpactREQ21:
    async def test_vc_impact_true_included_in_prompt(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["4"] = {"data": AGENT4_WITH_VC}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            await run(state)

        first_call_msg = mock_sonnet.call_args_list[0].kwargs.get("user_message", "")
        assert "TRUE" in first_call_msg or "FG21/1" in first_call_msg

    async def test_vc_impact_false_not_mandatory(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["4"] = {"data": AGENT4_WITHOUT_VC}

        with (
            patch("src.agents.testing.agent_30_fca_scenario_agent.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.testing.agent_30_fca_scenario_agent.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_FCA_PASS
            await run(state)

        first_call_msg = mock_sonnet.call_args_list[0].kwargs.get("user_message", "")
        assert "FALSE" in first_call_msg or "Vulnerable Customer Impact: FALSE" in first_call_msg


# ── Prompt builder unit tests ─────────────────────────────────────────────────

_EXISTING_SCENARIO = {"title": "Suitability fails for HIGH-risk client"}


class TestBuildPrompt:
    def test_includes_story_id(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [])
        assert "FSC-2417" in prompt

    def test_includes_fca_class(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [])
        assert "FCA Classification: HIGH" in prompt

    def test_includes_consumer_duty_risk(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "LOW", "HIGH", "LOW", [])
        assert "Consumer Duty risk: HIGH" in prompt

    def test_includes_risk_level(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "CRITICAL", [])
        assert "Risk anticipation level: CRITICAL" in prompt

    def test_vc_impact_true_shows_mandatory(self):
        prompt = _build_prompt(
            "FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [],
            vulnerable_customer_impact=True,
        )
        assert "TRUE" in prompt
        assert "FG21/1 scenario is mandatory" in prompt

    def test_vc_impact_false_shows_false(self):
        prompt = _build_prompt(
            "FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [],
            vulnerable_customer_impact=False,
        )
        assert "Vulnerable Customer Impact: FALSE" in prompt

    def test_existing_titles_shown(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [_EXISTING_SCENARIO])
        assert "Suitability fails for HIGH-risk client" in prompt

    def test_no_existing_scenarios_shows_none(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [])
        assert "  none" in prompt

    def test_ends_with_tool_name(self):
        prompt = _build_prompt("FSC-2417", MOCK_STORY, "HIGH", "HIGH", "HIGH", [])
        assert _FCA_TOOL_NAME in prompt
        assert prompt.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_five_required_fields(self):
        assert set(_FCA_TOOL_SCHEMA["required"]) == {
            "fca_test_scenarios", "consumer_duty_covered",
            "cobs_scenarios_count", "fca_scenario_verdict", "regulatory_gaps",
        }

    def test_fca_scenario_verdict_enum_has_three_values(self):
        assert _FCA_TOOL_SCHEMA["properties"]["fca_scenario_verdict"]["enum"] == [
            "PASS", "WARN", "FAIL",
        ]

    def test_scenario_item_has_six_required_fields(self):
        item_required = set(
            _FCA_TOOL_SCHEMA["properties"]["fca_test_scenarios"]["items"]["required"]
        )
        assert item_required == {
            "scenario_id", "regulation", "title",
            "description", "pass_criteria", "fail_criteria",
        }

    def test_regulatory_gaps_is_array(self):
        assert _FCA_TOOL_SCHEMA["properties"]["regulatory_gaps"]["type"] == "array"
