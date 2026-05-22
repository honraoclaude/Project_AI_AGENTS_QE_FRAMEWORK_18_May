"""Tests for Agent 30 — FCA Scenario Agent (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_30_fca_scenario_agent import (
    _compute_confidence,
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
