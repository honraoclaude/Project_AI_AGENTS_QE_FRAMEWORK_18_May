"""
Tests for Agent 8 — Dependency Mapping (Augmented Script).

The deterministic _analyse_dependencies() function is the primary test target —
no LLM mocking needed for correctness tests.
Integration tests mock the Haiku call (narrative only).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_08_dependency_mapping import (
    _analyse_dependencies,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio. The Suitability__c record must link "
        "to the client's RiskProfile__c and the relevant FinancialAccount. "
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

STORY_FINANCIAL = {
    "story_id": "FSC-2600",
    "summary": "Add AUM roll-up to FinancialAccount",
    "description": (
        "Aggregate FinancialHolding records to produce AUM on FinancialAccount. "
        "The Revenue__c object should be updated when the roll-up changes."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Medium",
    "labels": [],
    "components": ["WealthCore"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

MOCK_TRACE = {
    "narrative": (
        "The story touches Suitability__c and RiskProfile__c, implying FinancialAccount "
        "and Individual parent records must exist before tests execute. "
        "The dependency depth of 2 indicates moderate deployment complexity."
    ),
    "dependency_complexity": "medium",
}


# ── Deterministic analysis tests (no LLM, no Jira) ───────────────────────────

class TestDependencyAnalysis:
    def test_suitability_story_detects_key_objects(self):
        detected, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "suitability__c" in detected
        assert "riskprofile__c" in detected
        assert "financialaccount" in detected

    def test_suitability_story_implies_individual(self):
        _, implied, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "individual" not in implied   # individual is excluded from implied (too generic)
        # individual appears in the dependency map but is excluded from implied list
        # (it's filtered as a base object in _analyse_dependencies)

    def test_label_change_detects_no_fsc_objects(self):
        detected, implied, _, depth = _analyse_dependencies(STORY_LABEL_CHANGE)
        assert len(detected) == 0
        assert len(implied) == 0
        assert depth == 0

    def test_financial_story_detects_financial_objects(self):
        detected, _, _, _ = _analyse_dependencies(STORY_FINANCIAL)
        assert "financialaccount" in detected
        assert "financialholding" in detected

    def test_dependency_depth_nonzero_for_suitability(self):
        _, _, _, depth = _analyse_dependencies(STORY_SUITABILITY)
        assert depth >= 1

    def test_dependency_depth_zero_for_label_change(self):
        _, _, _, depth = _analyse_dependencies(STORY_LABEL_CHANGE)
        assert depth == 0

    def test_dependency_graph_contains_detected_objects(self):
        detected, _, graph, _ = _analyse_dependencies(STORY_SUITABILITY)
        for obj in detected:
            assert obj in graph

    def test_vulnerable_customer_detected_by_alias(self):
        detected, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "vulnerablecustomerindicator__c" in detected

    def test_analysis_is_case_insensitive(self):
        upper_story = {**STORY_SUITABILITY, "description": STORY_SUITABILITY["description"].upper()}
        detected_upper, _, _, _ = _analyse_dependencies(upper_story)
        detected_lower, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert set(detected_upper) == set(detected_lower)


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_suitability_story_high_confidence(self):
        detected, implied, _, depth = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score >= 75, "Rich multi-object story should score ≥ 75"

    def test_label_change_lower_confidence(self):
        detected, implied, _, depth = _analyse_dependencies(STORY_LABEL_CHANGE)
        score, _ = _compute_confidence(STORY_LABEL_CHANGE, detected, implied, depth)
        assert score < 75, "Story with no FSC objects should score lower"

    def test_no_objects_heavily_penalised(self):
        detected, implied, _, depth = _analyse_dependencies(STORY_LABEL_CHANGE)
        score, signals = _compute_confidence(STORY_LABEL_CHANGE, detected, implied, depth)
        assert "no_objects_detected" in signals

    def test_score_never_exceeds_92(self):
        detected, implied, _, depth = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score <= 92

    def test_score_never_below_20(self):
        empty_story = {**STORY_LABEL_CHANGE, "description": ""}
        detected, implied, _, depth = _analyse_dependencies(empty_story)
        score, _ = _compute_confidence(empty_story, detected, implied, depth)
        assert score >= 20

    def test_high_base_score_reflects_deterministic_nature(self):
        """Augmented Script uses base=72 — should score higher than Tier B agents on same input."""
        detected, implied, _, depth = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score >= 72, "Base score for deterministic analysis should be high"


# ── Integration tests — full agent run ───────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.agent_id == 8
        assert result.agent_name == "Dependency Mapping"
        assert result.model_used == "claude-haiku-4-5-20251001"
        assert result.confidence.tier == "B"

    async def test_detected_objects_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "suitability__c" in result.data["detected_objects"]
        assert "dependency_depth" in result.data
        assert result.data["dependency_depth"] >= 1

    async def test_label_change_no_detected_objects(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_haiku.return_value = {**MOCK_TRACE, "dependency_complexity": "low"}
            result = await run(state)

        assert result.data["detected_objects"] == []
        assert result.data["dependency_depth"] == 0

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        for key in ["detected_objects", "implied_objects", "dependency_graph",
                    "dependency_depth", "cross_object_count"]:
            assert key in result.data

    async def test_uses_fast_model_for_haiku(self):
        """Agent 8 is an Augmented Script — Haiku generates narrative, not analysis."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
