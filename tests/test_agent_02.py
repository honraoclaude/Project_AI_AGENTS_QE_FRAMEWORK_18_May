"""
Tests for Agent 2 — INVEST Quality Agent.

Uses mock Jira and mock LLM so tests run without live infrastructure.
Tests: INVEST score normalisation, confidence scoring, AgentResult shape,
G1 gate readiness signal (invest_score >= 80 required).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_02_invest_quality import (
    _compute_confidence,
    _get_invest_score,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio so that the firm meets its regulatory "
        "obligation before recommending any changes to the FinancialAccount.\n\n"
        "The assessment must capture risk tolerance, investment horizon, and capacity "
        "for loss. The Suitability__c record must link to the client's RiskProfile__c "
        "and the relevant FinancialAccount.\n\n"
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

STORY_LABEL_CHANGE = {
    "story_id": "FSC-2500",
    "summary": "Update button label on Account page",
    "description": "Change the 'Save' button label to 'Submit' on the Account detail page.",
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": "po@firm.com",
}

AC_CLAUSES_FULL = [
    {
        "source": "description",
        "scenario": "Scenario: Adviser records suitability for standard client",
        "given": ["Given the client has a RiskProfile__c with risk_level = Moderate"],
        "when": ["When the adviser completes the Suitability Assessment screen flow"],
        "then": [
            "Then a Suitability__c record is created linked to the FinancialAccount",
            "Then the status field is set to 'Assessment Complete'",
        ],
    },
    {
        "source": "description",
        "scenario": "Scenario: Vulnerable customer — additional confirmation step",
        "given": ["Given VulnerableCustomerIndicator__c is true for the client"],
        "when": ["When the adviser reaches the final step of the Suitability flow"],
        "then": [
            "Then a Consumer Duty confirmation checkbox is displayed",
            "Then the assessment cannot be submitted without checking the box",
        ],
    },
]

MOCK_EXTRACTION_PASS = {
    "independent_score": 18,
    "independent_rationale": "Story stands alone with no explicit dependencies on other open stories.",
    "negotiable_score": 16,
    "negotiable_rationale": "Written as a user goal with clear business outcomes.",
    "valuable_score": 20,
    "valuable_rationale": "Directly addresses COBS 9.2 regulatory obligation — high regulatory value.",
    "estimable_score": 16,
    "estimable_rationale": "FSC objects and flow type are clearly named and bounded.",
    "small_score": 14,
    "small_rationale": "Two scenarios bounded to one screen flow — fits one sprint.",
    "testable_score": 18,
    "testable_rationale": "Two complete GWT scenarios covering happy path and vulnerable customer edge case.",
    "invest_verdict": "PASS",
    "improvement_suggestions": ["Add an error scenario for when RiskProfile__c is missing."],
    "blocking_issues": [],
}

MOCK_EXTRACTION_FAIL = {
    "independent_score": 8,
    "independent_rationale": "Story description implies dependency on a parent epic not yet complete.",
    "negotiable_score": 10,
    "negotiable_rationale": "Partially written as a requirement spec rather than a goal.",
    "valuable_score": 4,
    "valuable_rationale": "No business value stated — purely cosmetic label change.",
    "estimable_score": 12,
    "estimable_rationale": "Simple UI change, easily estimated despite missing context.",
    "small_score": 18,
    "small_rationale": "Very small scope — single field label change.",
    "testable_score": 2,
    "testable_rationale": "No acceptance criteria provided at all.",
    "invest_verdict": "FAIL",
    "improvement_suggestions": [
        "Add acceptance criteria in Given/When/Then format.",
        "State the business or regulatory value of this label change.",
    ],
    "blocking_issues": [
        "No acceptance criteria — Testable score is 2/20.",
        "No business value stated — Valuable score is 4/20.",
    ],
}

AGENT1_DATA_SUITABILITY = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments linked to FinancialAccounts.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount"],
    "fsc_components": ["Screen Flow — Suitability Assessment"],
    "ac_present": True,
    "ac_complete": True,
    "missing_elements": ["none"],
    "flags": ["high_fca_object_detected"],
    "story_summary": "Records COBS 9.2 suitability assessment for retirement portfolios.",
    "story_summary_jira": "Record Suitability Assessment for Retirement Portfolio",
    "description_word_count": 85,
    "ac_clause_count": 2,
    "signals": {},
}

AGENT1_DATA_LABEL = {
    "goal": "Update a button label on the Account detail page.",
    "persona": "Operations/Admin",
    "fsc_objects": [],
    "fsc_components": ["Lightning Page — Account Detail"],
    "ac_present": False,
    "ac_complete": False,
    "missing_elements": ["acceptance_criteria", "error_scenarios"],
    "flags": ["no_acceptance_criteria", "vague_goal", "no_fsc_objects"],
    "story_summary": "Minor UI change to rename a button label.",
    "story_summary_jira": "Update button label on Account page",
    "description_word_count": 15,
    "ac_clause_count": 0,
    "signals": {},
}


# ── INVEST score normalisation unit tests (no LLM, no Jira) ──────────────────

class TestInvestScoreNormalisation:
    def test_pass_story_scores_above_80(self):
        score = _get_invest_score(MOCK_EXTRACTION_PASS)
        assert score >= 80, f"Well-specified HIGH-FCA story should score ≥ 80, got {score}"

    def test_fail_story_scores_below_65(self):
        score = _get_invest_score(MOCK_EXTRACTION_FAIL)
        assert score < 65, f"Vague cosmetic story should score < 65, got {score}"

    def test_score_ceiling_is_100(self):
        perfect = {k: 20 for k in [
            "independent_score", "negotiable_score", "valuable_score",
            "estimable_score", "small_score", "testable_score",
        ]}
        assert _get_invest_score(perfect) == 100

    def test_score_floor_is_0(self):
        zero = {k: 0 for k in [
            "independent_score", "negotiable_score", "valuable_score",
            "estimable_score", "small_score", "testable_score",
        ]}
        assert _get_invest_score(zero) == 0

    def test_normalisation_formula(self):
        # (18+16+20+16+14+18) = 102; 102*100/120 = 85
        assert _get_invest_score(MOCK_EXTRACTION_PASS) == 85


# ── Confidence scoring unit tests (no LLM, no Jira) ──────────────────────────

class TestConfidenceScoring:
    def test_pass_story_confidence_above_70(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_PASS)
        score, _ = _compute_confidence(AGENT1_DATA_SUITABILITY, MOCK_EXTRACTION_PASS, invest_score)
        assert score >= 70, f"Well-specified story with rich Agent 1 data should score ≥ 70, got {score}"

    def test_fail_story_confidence_above_50(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_FAIL)
        score, _ = _compute_confidence(AGENT1_DATA_LABEL, MOCK_EXTRACTION_FAIL, invest_score)
        assert score >= 50, f"Clear-fail story confidence should be ≥ 50 (confident in low score), got {score}"

    def test_score_never_exceeds_92(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_PASS)
        score, _ = _compute_confidence(AGENT1_DATA_SUITABILITY, MOCK_EXTRACTION_PASS, invest_score)
        assert score <= 92, "Tier B cap is 92"

    def test_score_never_below_20(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_FAIL)
        score, _ = _compute_confidence(None, MOCK_EXTRACTION_FAIL, invest_score)
        assert score >= 20, "Tier B floor is 20"

    def test_signals_dict_populated(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_PASS)
        _, signals = _compute_confidence(AGENT1_DATA_SUITABILITY, MOCK_EXTRACTION_PASS, invest_score)
        assert "invest_margin" in signals
        assert any(k.startswith("testable") for k in signals)

    def test_no_agent1_data_reduces_confidence(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_PASS)
        with_agent1, _ = _compute_confidence(AGENT1_DATA_SUITABILITY, MOCK_EXTRACTION_PASS, invest_score)
        without_agent1, _ = _compute_confidence(None, MOCK_EXTRACTION_PASS, invest_score)
        assert with_agent1 > without_agent1, "Agent 1 data should raise confidence"

    def test_borderline_score_is_penalised(self):
        borderline = dict(MOCK_EXTRACTION_PASS)
        # Adjust so invest_score = 80 exactly → margin = 0
        # total needed = 80 * 120 / 100 = 96
        # use 16×6 = 96
        for k in ["independent_score", "negotiable_score", "valuable_score",
                   "estimable_score", "small_score", "testable_score"]:
            borderline[k] = 16
        borderline["testable_score"] = 16
        invest_score = _get_invest_score(borderline)
        score, signals = _compute_confidence(AGENT1_DATA_SUITABILITY, borderline, invest_score)
        # invest_score = 96*100/120 = 80, margin = 0 → penalty -5
        assert signals.get("invest_margin") == 0


# ── Integration tests — full agent run with mocked LLM and Jira ───────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_passing_story(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        assert result.agent_id == 2
        assert result.agent_name == "INVEST Quality Agent"
        assert result.model_used == "claude-sonnet-4-6"
        assert "FSC-2417" in result.what
        assert result.confidence.tier == "B"

    async def test_invest_score_present_and_above_80_for_pass(self):
        invest_score = _get_invest_score(MOCK_EXTRACTION_PASS)
        assert invest_score >= 80

    async def test_returns_invest_score_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        assert "invest_score" in result.data
        assert "invest_verdict" in result.data
        assert "dimension_scores" in result.data
        assert "blocking_issues" in result.data
        assert "improvement_suggestions" in result.data

    async def test_invest_score_correct_value(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        assert result.data["invest_score"] == 85  # (18+16+20+16+14+18)*100//120

    async def test_fail_story_invest_score_below_80(self):
        state = initial_story_state("FSC-2500")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_LABEL}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_EXTRACTION_FAIL

            result = await run(state)

        assert result.data["invest_score"] < 80, "Fail story must score < 80 to trigger G1 block"
        assert len(result.data["blocking_issues"]) > 0

    async def test_dimension_scores_sum_to_total_raw(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        ds = result.data["dimension_scores"]
        expected_total = ds["independent"] + ds["negotiable"] + ds["valuable"] + \
                         ds["estimable"] + ds["small"] + ds["testable"]
        assert ds["total_raw"] == expected_total

    async def test_agent1_available_flag_set_when_state_has_agent1(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        assert result.data["agent1_available"] is True

    async def test_runs_without_agent1_data(self):
        """Agent 2 must work standalone even if Agent 1 result is absent."""
        state = initial_story_state("FSC-2417")  # no agent_results["1"]

        with (
            patch("src.agents.refinement.agent_02_invest_quality.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_02_invest_quality.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_02_invest_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_PASS

            result = await run(state)

        assert result.agent_id == 2
        assert result.data["agent1_available"] is False
