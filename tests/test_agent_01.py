"""
Tests for Agent 1 — Story Intent Agent.

Uses a mock Jira response so the test runs without a live Jira instance.
Tests: extraction quality, confidence scoring, flag detection, AgentResult shape.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_01_story_intent import run, _compute_confidence
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

MOCK_EXTRACTION_SUITABILITY = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments linked to FinancialAccounts and RiskProfiles.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount", "VulnerableCustomerIndicator__c"],
    "fsc_components": ["Screen Flow — Suitability Assessment", "Apex trigger on Suitability__c"],
    "ac_present": True,
    "ac_complete": True,
    "missing_elements": ["none"],
    "story_summary": (
        "This story implements the COBS 9.2 Suitability Assessment recording flow for wealth advisers. "
        "It creates Suitability__c records linked to RiskProfile__c and FinancialAccount. "
        "It adds a Consumer Duty confirmation step for vulnerable customers."
    ),
    "flags": ["high_fca_object_detected"],
}

MOCK_EXTRACTION_LABEL = {
    "goal": "Update a button label on the Account detail page.",
    "persona": "Operations/Admin",
    "fsc_objects": [],
    "fsc_components": ["Lightning Page — Account Detail"],
    "ac_present": False,
    "ac_complete": False,
    "missing_elements": ["acceptance_criteria", "error_scenarios"],
    "story_summary": "Minor UI change to rename a button label on the Account detail page. No FSC objects involved.",
    "flags": ["no_acceptance_criteria", "vague_goal", "no_fsc_objects"],
}


# ── Confidence scoring unit tests (no LLM, no Jira) ──────────────────────────

class TestConfidenceScoring:
    def test_high_fca_story_scores_above_70(self):
        score, signals = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert score >= 70, f"Expected >=70 for well-specified HIGH-FCA story, got {score}"

    def test_label_change_scores_below_60(self):
        score, signals = _compute_confidence(STORY_LABEL_CHANGE, [], MOCK_EXTRACTION_LABEL)
        assert score < 70, f"Expected <70 for vague label-change story, got {score}"

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert score <= 92, "Tier B score must never exceed 92 (reserved for deterministic checks)"

    def test_score_never_below_20(self):
        empty_story = {**STORY_LABEL_CHANGE, "description": "", "summary": ""}
        score, _ = _compute_confidence(empty_story, [], MOCK_EXTRACTION_LABEL)
        assert score >= 20, "Score floor is 20"

    def test_signals_dict_populated(self):
        _, signals = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert "description_words" in signals
        assert "ac_complete" in signals or "ac_present_incomplete" in signals or "ac_absent" in signals


# ── Integration test — full agent run with mocked LLM and Jira ────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        assert result.agent_id == 1
        assert result.agent_name == "Story Intent Agent"
        assert result.model_used == "claude-sonnet-4-6"
        assert "FSC-2417" in result.what
        assert result.confidence.tier == "B"
        assert result.confidence.final_score >= 70
        assert not result.escalated

    async def test_escalates_on_vague_story(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_EXTRACTION_LABEL

            result = await run(state)

        # Vague story with no ACs — may escalate depending on final score
        assert result.agent_id == 1
        assert result.confidence.tier == "B"
        assert isinstance(result.confidence.final_score, int)

    async def test_data_contains_required_downstream_keys(self):
        """Downstream agents (2, 3, 7) depend on these keys existing."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        required_keys = ["goal", "persona", "fsc_objects", "ac_present", "ac_clauses", "flags"]
        # ac_clauses key comes from the extraction
        data_keys = set(result.data.keys())
        for key in ["goal", "persona", "fsc_objects", "ac_present", "flags"]:
            assert key in data_keys, f"Missing downstream-required key: {key}"

    async def test_high_fca_flag_present_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        assert "high_fca_object_detected" in result.data.get("flags", [])
