"""
Tests for Agent 10 — AC Compliance Verifier (Augmented Script).

The deterministic _analyse_ac_compliance() function is the primary test target.
Integration tests mock Jira calls and the Haiku narrative call.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_10_ac_compliance import (
    _analyse_ac_compliance,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": "COBS 9.2 suitability assessment story.",
    "status": "In Development",
    "issue_type": "Story",
    "priority": "High",
    "labels": [],
    "components": ["Suitability"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

AGENT3_HIGH = {
    "fca_classification": "HIGH",
    "ensemble_agreement": True,
    "fca_triggers": ["Suitability__c", "RiskProfile__c"],
}

AGENT3_LOW = {
    "fca_classification": "LOW",
    "ensemble_agreement": True,
    "fca_triggers": [],
}

AGENT5_FULL_COVERAGE = {
    "ac_clause_count": 4,
    "generation_mode": "validated_existing",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": True,
    },
    "remaining_gaps": [],
}

AGENT5_MISSING_REGULATORY = {
    "ac_clause_count": 3,
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": False,
        "regulatory": False,
    },
    "remaining_gaps": ["No regulatory test scenario for COBS 9.2"],
}

AGENT5_MISSING_ERROR_PATHS = {
    "ac_clause_count": 3,
    "generation_mode": "supplemented_existing",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": False,
        "edge_cases": True,
        "regulatory": True,
    },
    "remaining_gaps": [],
}

# 4 ACs matching the refinement baseline
CURRENT_ACS_MATCHING = [
    {"source": "jira", "scenario": "Scenario 1", "given": [], "when": [], "then": []},
    {"source": "jira", "scenario": "Scenario 2", "given": [], "when": [], "then": []},
    {"source": "jira", "scenario": "Scenario 3", "given": [], "when": [], "then": []},
    {"source": "jira", "scenario": "Scenario 4", "given": [], "when": [], "then": []},
]

# Fewer ACs than refinement baseline (some removed)
CURRENT_ACS_REDUCED = [
    {"source": "jira", "scenario": "Scenario 1", "given": [], "when": [], "then": []},
    {"source": "jira", "scenario": "Scenario 2", "given": [], "when": [], "then": []},
]

CURRENT_ACS_EMPTY = []

MOCK_TRACE_PASS = {
    "narrative": (
        "The story has 4 ACs matching the refinement baseline with full coverage across "
        "happy path, error paths, edge cases, and regulatory scenarios. "
        "The story is development-ready with no AC gaps to address."
    ),
    "compliance_risk": "low",
}

MOCK_TRACE_FAIL = {
    "narrative": (
        "This HIGH-FCA story is missing a regulatory AC scenario, which is mandatory for "
        "COBS 9.2 compliance. Additionally, 2 ACs were removed since refinement. "
        "The developer must restore the missing ACs before proceeding."
    ),
    "compliance_risk": "high",
}


# ── Deterministic analysis tests ──────────────────────────────────────────────

class TestAcComplianceAnalysis:
    def test_pass_when_counts_match_and_all_coverage_present(self):
        count, ref, delta, missing, verdict = _analyse_ac_compliance(
            CURRENT_ACS_MATCHING, AGENT5_FULL_COVERAGE, AGENT3_HIGH
        )
        assert verdict == "PASS"
        assert delta == 0
        assert missing == []

    def test_fail_when_ac_count_decreases(self):
        _, _, delta, _, verdict = _analyse_ac_compliance(
            CURRENT_ACS_REDUCED, AGENT5_FULL_COVERAGE, AGENT3_HIGH
        )
        assert delta < 0
        assert verdict == "FAIL"

    def test_fail_when_high_fca_missing_regulatory_coverage(self):
        _, _, _, missing, verdict = _analyse_ac_compliance(
            CURRENT_ACS_MATCHING, AGENT5_MISSING_REGULATORY, AGENT3_HIGH
        )
        assert "regulatory" in missing
        assert verdict == "FAIL"

    def test_partial_when_non_critical_coverage_missing(self):
        # Missing error_paths on a LOW-FCA story → PARTIAL (not FAIL)
        _, _, _, missing, verdict = _analyse_ac_compliance(
            CURRENT_ACS_REDUCED[:3], AGENT5_MISSING_ERROR_PATHS, AGENT3_LOW
        )
        assert "error_paths" in missing
        assert verdict in ("PARTIAL", "FAIL")

    def test_low_fca_missing_regulatory_does_not_fail(self):
        # LOW-FCA: missing regulatory coverage is PARTIAL, not FAIL
        _, _, _, missing, verdict = _analyse_ac_compliance(
            CURRENT_ACS_MATCHING, AGENT5_MISSING_REGULATORY, AGENT3_LOW
        )
        assert verdict == "PARTIAL"

    def test_no_acs_in_jira_with_baseline_gives_fail(self):
        _, _, delta, _, verdict = _analyse_ac_compliance(
            CURRENT_ACS_EMPTY, AGENT5_FULL_COVERAGE, AGENT3_HIGH
        )
        assert delta < 0
        assert verdict == "FAIL"

    def test_no_agent5_data_gives_zero_refinement_count(self):
        count, ref, delta, _, _ = _analyse_ac_compliance(
            CURRENT_ACS_MATCHING, None, AGENT3_HIGH
        )
        assert ref == 0
        assert delta == len(CURRENT_ACS_MATCHING)

    def test_current_count_is_accurate(self):
        count, _, _, _, _ = _analyse_ac_compliance(
            CURRENT_ACS_MATCHING, AGENT5_FULL_COVERAGE, AGENT3_HIGH
        )
        assert count == 4


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_high_score_with_baseline_and_full_coverage(self):
        score, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        assert score >= 75

    def test_low_score_without_refinement_baseline(self):
        score_with, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        score_without, _ = _compute_confidence(
            None, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        assert score_with > score_without

    def test_acs_removed_penalised(self):
        score_delta_zero, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        score_delta_negative, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_REDUCED, -2, []
        )
        assert score_delta_zero > score_delta_negative

    def test_missing_coverage_reduces_confidence(self):
        score_full, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        score_missing, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, ["error_paths"]
        )
        assert score_full > score_missing

    def test_high_fca_missing_regulatory_doubly_penalised(self):
        score_low_fca, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_LOW, CURRENT_ACS_MATCHING, 0, ["regulatory"]
        )
        score_high_fca, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, ["regulatory"]
        )
        assert score_low_fca > score_high_fca

    def test_no_acs_in_jira_penalised(self):
        score_with_acs, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        score_no_acs, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_EMPTY, -4, []
        )
        assert score_with_acs > score_no_acs

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT5_FULL_COVERAGE, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, []
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, CURRENT_ACS_EMPTY, -5, ["regulatory", "error_paths"])
        assert score >= 20

    def test_refinement_baseline_missing_signal_recorded(self):
        _, signals = _compute_confidence(None, AGENT3_HIGH, CURRENT_ACS_MATCHING, 0, [])
        assert "refinement_baseline_missing" in signals


# ── Integration tests — full agent run ───────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_FULL_COVERAGE}

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_MATCHING
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 10
        assert result.agent_name == "AC Compliance Verifier"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"] = {"data": AGENT5_FULL_COVERAGE}

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_MATCHING
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["compliance_verdict", "missing_coverage_types",
                    "ac_count_current", "ac_delta", "ac_count_at_refinement"]:
            assert key in result.data

    async def test_verdict_pass_for_matching_acs(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_FULL_COVERAGE}

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_MATCHING
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["compliance_verdict"] == "PASS"
        assert result.data["ac_delta"] == 0

    async def test_verdict_fail_for_removed_acs(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["5"] = {"data": AGENT5_FULL_COVERAGE}

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_REDUCED  # 2 ACs, baseline was 4
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["compliance_verdict"] == "FAIL"
        assert result.data["ac_delta"] < 0

    async def test_runs_standalone_without_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_MATCHING
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 10
        assert result.data["ac_count_at_refinement"] == 0  # no Agent 5 data

    async def test_uses_fast_model_for_haiku(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_10_ac_compliance.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_10_ac_compliance.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_acs,
            patch("src.agents.development.agent_10_ac_compliance.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_acs.return_value = CURRENT_ACS_MATCHING
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
