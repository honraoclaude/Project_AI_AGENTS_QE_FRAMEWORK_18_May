"""Tests for Agent 19 — BDD Gherkin Writer (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_19_bdd_gherkin_writer import (
    _build_prompt,
    _compute_confidence,
    _GHERKIN_TOOL_NAME,
    _GHERKIN_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH",   "ensemble_agreement": True}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM", "ensemble_agreement": True}
AGENT3_LOW    = {"fca_classification": "LOW",    "ensemble_agreement": True}
AGENT5_DATA   = {"ac_count": 3, "acs_generated": True}
AGENT10_PASS  = {"coverage_verdict": "PASS", "current_ac_count": 3}
AGENT13_DATA  = {"detected_objects": ["suitability__c", "riskprofile__c"]}

MOCK_ACS = [
    {"id": "AC1", "description": "Given a HIGH-risk client, When suitability check runs, Then it must fail if score < 50"},
    {"id": "AC2", "description": "Given a valid portfolio, When rebalancing is triggered, Then FSC rules apply"},
    {"id": "AC3", "description": "Given a Vulnerable Customer flag, Then additional review is required"},
]

MOCK_STORY = {
    "key": "FSC-2417",
    "summary": "Suitability Assessment Enhancement",
    "description": "Enhance suitability scoring for HIGH-risk FSC clients.",
}

MOCK_GHERKIN_PASS = {
    "scenarios": [
        {
            "title": "HIGH-risk client fails suitability with score below threshold",
            "tags": ["@fca", "@negative", "@smoke"],
            "steps": [
                "Given a HIGH-risk client with suitability score 40",
                "When the suitability assessment runs",
                "Then the assessment result is FAILED",
                "And a regulatory alert is raised",
            ],
        },
        {
            "title": "Valid portfolio triggers FSC rebalancing rules",
            "tags": ["@smoke", "@regression"],
            "steps": [
                "Given a valid FSC portfolio",
                "When rebalancing is triggered",
                "Then FSC rebalancing rules are applied",
            ],
        },
    ],
    "scenario_count": 2,
    "gherkin_verdict": "PASS",
    "fca_coverage_present": True,
    "coverage_gaps": [],
}

MOCK_GHERKIN_WARN = {
    "scenarios": [
        {
            "title": "Basic suitability pass",
            "tags": ["@smoke"],
            "steps": ["Given a client", "When check runs", "Then result is PASSED"],
        }
    ],
    "scenario_count": 1,
    "gherkin_verdict": "WARN",
    "fca_coverage_present": False,
    "coverage_gaps": ["AC3: Vulnerable Customer scenario not covered"],
}

MOCK_GHERKIN_INCOMPLETE = {
    "scenarios": [],
    "scenario_count": 0,
    "gherkin_verdict": "INCOMPLETE",
    "fca_coverage_present": False,
    "coverage_gaps": ["No ACs available to generate scenarios"],
}


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_high_fca_with_coverage_scores_well(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        assert score >= 75

    def test_no_acs_penalises_heavily(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        score_without, _ = _compute_confidence([], AGENT3_HIGH, AGENT5_DATA, 0, "HIGH", False)
        assert score_with > score_without

    def test_high_fca_missing_fca_scenarios_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        score_without, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", False)
        assert score_with > score_without

    def test_no_scenarios_generated_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        score_without, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 0, "HIGH", False)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 5, "HIGH", True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence([], None, None, 0, "LOW", False)
        assert score >= 20

    def test_low_fca_not_penalised_for_missing_fca_scenarios(self):
        score_high, _ = _compute_confidence(MOCK_ACS, AGENT3_LOW, AGENT5_DATA, 2, "LOW", False)
        score_low_penalised, _ = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 2, "HIGH", False)
        assert score_high >= score_low_penalised

    def test_acs_available_stores_count(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        assert signals["acs_available"] == len(MOCK_ACS)

    def test_no_acs_available_key_in_signals(self):
        _, signals = _compute_confidence([], AGENT3_HIGH, AGENT5_DATA, 0, "HIGH", False)
        assert "no_acs_available" in signals

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        assert "fca_classification_available" in signals

    def test_refined_ac_baseline_available_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        assert "refined_ac_baseline_available" in signals

    def test_scenarios_generated_stores_count(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 3, "HIGH", True)
        assert signals["scenarios_generated"] == 3

    def test_no_scenarios_generated_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 0, "HIGH", False)
        assert "no_scenarios_generated" in signals

    def test_regulated_story_missing_fca_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 2, "HIGH", False)
        assert "regulated_story_missing_fca_scenarios" in signals

    def test_regulated_story_has_fca_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(MOCK_ACS, AGENT3_HIGH, AGENT5_DATA, 2, "HIGH", True)
        assert "regulated_story_has_fca_scenarios" in signals

    def test_medium_fca_missing_fca_scenarios_penalised(self):
        score_with, _ = _compute_confidence(MOCK_ACS, AGENT3_MEDIUM, AGENT5_DATA, 2, "MEDIUM", True)
        score_without, _ = _compute_confidence(MOCK_ACS, AGENT3_MEDIUM, AGENT5_DATA, 2, "MEDIUM", False)
        assert score_with > score_without


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_DATA}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert result.agent_id == 19
        assert result.agent_name == "BDD Gherkin Writer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        for key in ["gherkin_scenarios", "scenario_count", "gherkin_verdict", "fca_coverage_present"]:
            assert key in result.data

    async def test_pass_verdict_when_all_acs_covered(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert result.data["gherkin_verdict"] == "PASS"
        assert result.data["fca_coverage_present"] is True

    async def test_incomplete_verdict_when_no_acs(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_GHERKIN_INCOMPLETE
            result = await run(state)

        assert result.data["gherkin_verdict"] == "INCOMPLETE"
        assert result.data["scenario_count"] == 0

    async def test_warn_verdict_for_partial_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_WARN
            result = await run(state)

        assert result.data["gherkin_verdict"] == "WARN"
        assert len(result.data["coverage_gaps"]) >= 1

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_escalated_when_no_acs_and_no_scenarios(self):
        # base=70, no_acs_available=-20, no_scenarios_generated=-15 → 35 < 60
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = []
            mock_sonnet.return_value = MOCK_GHERKIN_INCOMPLETE
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_ac_count_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert "ac_count" in result.data
        assert result.data["ac_count"] == len(MOCK_ACS)


# ── Shapley attribution tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestShapleyAttribution:
    async def test_shapley_attribution_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_DATA}
        state["agent_results"]["13"] = {"data": AGENT13_DATA}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert "shapley_attribution" in result.data
        shapley = result.data["shapley_attribution"]
        assert isinstance(shapley, dict)
        assert len(shapley) == 3

    async def test_shapley_attribution_sums_to_100(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_DATA}
        state["agent_results"]["13"] = {"data": AGENT13_DATA}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        shapley = result.data["shapley_attribution"]
        total = sum(shapley.values())
        assert abs(total - 100.0) < 0.01

    async def test_ac_source_trust_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": {**AGENT5_DATA, "generation_mode_trust": 0.6}}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS
            result = await run(state)

        assert "ac_source_trust" in result.data
        assert result.data["ac_source_trust"] == 0.6


# ── REQ-12: vulnerable_customer + bulk risk tests ─────────────────────────────

MOCK_GHERKIN_WITH_VC = {
    "scenarios": [
        {
            "title": "Vulnerable Customer receives Consumer Duty confirmation step",
            "tags": ["@fca", "@vulnerable_customer"],
            "steps": [
                "Given a client with VulnerableCustomerIndicator__c = true",
                "When the Suitability Assessment flow runs",
                "Then the Consumer Duty confirmation step is presented",
                "And the step must be acknowledged before proceeding",
            ],
        },
    ],
    "scenario_count": 1,
    "gherkin_verdict": "PASS",
    "fca_coverage_present": True,
    "vulnerable_customer_coverage_present": True,
    "coverage_gaps": [],
}

MOCK_GHERKIN_WITH_BULK = {
    "scenarios": [
        {
            "title": "Bulk insert of 200 FinancialHolding records does not breach governor limits",
            "tags": ["@bulk", "@regression"],
            "steps": [
                "Given 200 FinancialHolding records are queued for insert",
                "When the trigger fires on all 200 records in a single batch",
                "Then no DML governor limit exception is thrown",
                "And all 200 records are correctly created",
            ],
        },
    ],
    "scenario_count": 1,
    "gherkin_verdict": "PASS",
    "fca_coverage_present": False,
    "vulnerable_customer_coverage_present": False,
    "coverage_gaps": [],
}


@pytest.mark.asyncio
class TestVulnerableCustomerBulkREQ12:
    async def test_vc_impact_true_in_prompt(self):
        """REQ-12: vulnerable_customer_impact=True from Agent 04 appears in prompt."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["4"] = {"data": {"vulnerable_customer_impact": True}}

        captured_message = None

        async def capture(**kwargs):
            nonlocal captured_message
            captured_message = kwargs.get("user_message", "")
            return MOCK_GHERKIN_PASS

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  side_effect=capture),
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            await run(state)

        assert "Vulnerable Customer Impact: TRUE" in captured_message

    async def test_bulk_risk_high_in_prompt(self):
        """REQ-12: bulk_risk_level=HIGH from Agent 16 appears in prompt with factors."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["16"] = {"data": {
            "bulk_risk_level": "HIGH",
            "bulk_risk_factors": ["DML governor limits", "large data volume"],
        }}

        captured_message = None

        async def capture(**kwargs):
            nonlocal captured_message
            captured_message = kwargs.get("user_message", "")
            return MOCK_GHERKIN_PASS

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  side_effect=capture),
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            await run(state)

        assert "Bulk Risk Level: HIGH" in captured_message
        assert "DML governor limits" in captured_message

    async def test_vc_coverage_present_in_output_data(self):
        """REQ-12: vulnerable_customer_coverage_present present in result.data."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["4"] = {"data": {"vulnerable_customer_impact": True}}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_WITH_VC
            result = await run(state)

        assert "vulnerable_customer_coverage_present" in result.data
        assert result.data["vulnerable_customer_coverage_present"] is True

    async def test_bulk_test_scenarios_generated_true_when_bulk_tagged(self):
        """REQ-12: bulk_test_scenarios_generated=True when a @bulk scenario is in output."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["16"] = {"data": {
            "bulk_risk_level": "HIGH",
            "bulk_risk_factors": ["DML governor limits"],
        }}

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_WITH_BULK
            result = await run(state)

        assert "bulk_test_scenarios_generated" in result.data
        assert result.data["bulk_test_scenarios_generated"] is True

    async def test_bulk_test_scenarios_generated_false_when_no_bulk_tag(self):
        """REQ-12: bulk_test_scenarios_generated=False when no @bulk tag in any scenario."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_19_bdd_gherkin_writer.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_acs.return_value = MOCK_ACS
            mock_sonnet.return_value = MOCK_GHERKIN_PASS  # no @bulk tags
            result = await run(state)

        assert result.data["bulk_test_scenarios_generated"] is False


# ── _build_prompt unit tests ──────────────────────────────────────────────────

_SIMPLE_STORY = {"summary": "Test Story", "description": "desc"}

class TestBuildPrompt:
    def test_includes_story_id(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "HIGH", 3, "PASS", [])
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "HIGH", 3, "PASS", [])
        assert "HIGH" in msg

    def test_no_acs_shows_placeholder(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, [], "LOW", 0, "", [])
        assert "(no acceptance criteria available)" in msg

    def test_acs_formatted_with_prefix(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "HIGH", 3, "PASS", [])
        assert "AC1:" in msg

    def test_vc_impact_true_shows_mandatory_note(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "HIGH", 3, "PASS", [],
                            vulnerable_customer_impact=True)
        assert "Vulnerable Customer Impact: TRUE" in msg
        assert "mandatory" in msg

    def test_vc_impact_false_shows_false(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "LOW", 0, "", [],
                            vulnerable_customer_impact=False)
        assert "Vulnerable Customer Impact: FALSE" in msg

    def test_bulk_high_includes_factors(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "LOW", 0, "", [],
                            bulk_risk_level="HIGH", bulk_risk_factors=["DML governor limits"])
        assert "Bulk Risk Level: HIGH" in msg
        assert "DML governor limits" in msg

    def test_bulk_low_shows_level_only(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "LOW", 0, "", [],
                            bulk_risk_level="LOW")
        assert "Bulk Risk Level: LOW" in msg

    def test_ends_with_tool_name(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, MOCK_ACS, "HIGH", 3, "PASS", [])
        assert _GHERKIN_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_six_required_fields(self):
        assert set(_GHERKIN_TOOL_SCHEMA["required"]) == {
            "scenarios", "scenario_count", "gherkin_verdict",
            "fca_coverage_present", "vulnerable_customer_coverage_present", "coverage_gaps",
        }

    def test_gherkin_verdict_enum_has_three_values(self):
        assert _GHERKIN_TOOL_SCHEMA["properties"]["gherkin_verdict"]["enum"] == [
            "PASS", "PARTIAL", "INCOMPLETE"
        ]

    def test_scenarios_is_array_type(self):
        assert _GHERKIN_TOOL_SCHEMA["properties"]["scenarios"]["type"] == "array"

    def test_scenario_item_has_three_required_fields(self):
        item_required = set(_GHERKIN_TOOL_SCHEMA["properties"]["scenarios"]["items"]["required"])
        assert item_required == {"title", "tags", "steps"}
