"""
Tests for Agent 5 — AC Generator.

Uses mock Jira and mock LLM so tests run without live infrastructure.
Tests: confidence scoring per generation mode, coverage assessment signals,
regulatory scenario requirement for HIGH-FCA, required downstream keys,
standalone mode (no upstream agent data).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_05_ac_generator import (
    _AC_CLAUSE_SCHEMA,
    _build_user_message,
    _TOOL_NAME,
    _TOOL_SCHEMA,
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

AGENT1_DATA = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount"],
    "story_summary": "Records COBS 9.2 suitability assessment for retirement portfolios.",
    "ac_present": False,
    "ac_complete": False,
    "missing_elements": ["acceptance_criteria"],
    "description_word_count": 85,
}

AGENT3_DATA_HIGH = {
    "fca_classification": "HIGH",
    "fca_triggers": ["Suitability__c", "RiskProfile__c", "VulnerableCustomerIndicator__c"],
    "regulatory_obligations": ["COBS 9.2 Suitability", "Consumer Duty PS22/9"],
    "ensemble_agreement": True,
}

AGENT3_DATA_LOW = {
    "fca_classification": "LOW",
    "fca_triggers": [],
    "regulatory_obligations": [],
    "ensemble_agreement": True,
}

AGENT4_DATA_HIGH = {
    "cd_verdict": "COMPLIANT",
    "vulnerable_customer_impact": True,
    "cd_obligations": [
        "PS22/9 Outcome 1 — Suitability Assessment must be fit for purpose.",
        "FG21/1 §4.3 — Vulnerable customers must not be disadvantaged.",
    ],
}

AGENT4_DATA_LOW = {
    "cd_verdict": "NOT_APPLICABLE",
    "vulnerable_customer_impact": False,
    "cd_obligations": [],
}

# Full generation from scratch — 4 scenarios, all coverage types
MOCK_AC_GENERATED = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Adviser records suitability for standard client",
            "scenario_type": "happy_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has a RiskProfile__c with risk_level = Moderate"],
            "when": ["When the adviser completes the Suitability Assessment screen flow"],
            "then": [
                "Then a Suitability__c record is created linked to the FinancialAccount",
                "Then the status field is set to 'Assessment Complete'",
            ],
        },
        {
            "scenario": "Scenario: Vulnerable customer — additional Consumer Duty confirmation",
            "scenario_type": "regulatory",
            "test_category": "UI",
            "fca_relevant": True,
            "given": ["Given VulnerableCustomerIndicator__c is true for the client"],
            "when": ["When the adviser reaches the final step of the Suitability flow"],
            "then": [
                "Then a Consumer Duty confirmation checkbox is displayed",
                "Then the assessment cannot be submitted without checking the box",
            ],
        },
        {
            "scenario": "Scenario: RiskProfile missing — assessment blocked",
            "scenario_type": "error_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has no RiskProfile__c record"],
            "when": ["When the adviser attempts to open the Suitability Assessment flow"],
            "then": [
                "Then an error message 'Risk profile required' is displayed",
                "Then no Suitability__c record is created",
            ],
        },
        {
            "scenario": "Scenario: Assessment already exists — duplicate prevented",
            "scenario_type": "edge_case",
            "test_category": "REGRESSION",
            "fca_relevant": False,
            "given": ["Given a Suitability__c record already exists for this FinancialAccount"],
            "when": ["When the adviser attempts to create a new assessment"],
            "then": [
                "Then the existing assessment is displayed",
                "Then a new Suitability__c record is not created",
            ],
        },
    ],
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": True,
    },
    "gaps_filled": [
        "Happy path scenario",
        "Regulatory: vulnerable customer Consumer Duty confirmation",
        "Error path: missing RiskProfile__c",
        "Edge case: duplicate prevention",
    ],
    "remaining_gaps": ["Performance under concurrent adviser load not specified."],
}

# Validated from existing complete ACs
MOCK_AC_VALIDATED = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Adviser records suitability",
            "scenario_type": "happy_path",
            "test_category": "AUTOMATION_CANDIDATE",
            "fca_relevant": True,
            "given": ["Given the client has a RiskProfile__c"],
            "when": ["When the adviser submits the assessment"],
            "then": ["Then a Suitability__c record is created"],
        },
        {
            "scenario": "Scenario: Consumer Duty confirmation for vulnerable client",
            "scenario_type": "regulatory",
            "test_category": "UI",
            "fca_relevant": True,
            "given": ["Given VulnerableCustomerIndicator__c is true"],
            "when": ["When the adviser completes the flow"],
            "then": ["Then a Consumer Duty checkbox is shown"],
        },
        {
            "scenario": "Scenario: Missing RiskProfile blocks submission",
            "scenario_type": "error_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given no RiskProfile__c exists"],
            "when": ["When the adviser attempts submission"],
            "then": ["Then an error is shown"],
        },
        {
            "scenario": "Scenario: Duplicate assessment blocked",
            "scenario_type": "edge_case",
            "test_category": "REGRESSION",
            "fca_relevant": False,
            "given": ["Given an assessment already exists"],
            "when": ["When the adviser tries to create another"],
            "then": ["Then the duplicate is blocked"],
        },
    ],
    "generation_mode": "validated_existing",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": True,
    },
    "gaps_filled": [],
    "remaining_gaps": [],
}

# Simple LOW-FCA story — generated, minimal coverage
MOCK_AC_LOW_FCA = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Button label shows Submit on Account page",
            "scenario_type": "happy_path",
            "test_category": "UI",
            "fca_relevant": False,
            "given": ["Given the user is on the Account detail page"],
            "when": ["When the page loads"],
            "then": ["Then the button shows 'Submit' instead of 'Save'"],
        },
        {
            "scenario": "Scenario: Label correct on mobile viewport",
            "scenario_type": "edge_case",
            "test_category": "AUTOMATION_CANDIDATE",
            "fca_relevant": False,
            "given": ["Given the user views the Account page on a mobile device"],
            "when": ["When the page renders"],
            "then": ["Then the button label shows 'Submit'"],
        },
    ],
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": False,
        "edge_cases": True,
        "regulatory": False,
    },
    "gaps_filled": ["Happy path", "Mobile viewport edge case"],
    "remaining_gaps": [],
}

# HIGH-FCA but missing regulatory scenario — should penalise confidence
MOCK_AC_NO_REGULATORY = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Adviser records suitability",
            "scenario_type": "happy_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has a RiskProfile__c"],
            "when": ["When the adviser submits"],
            "then": ["Then a record is created"],
        },
        {
            "scenario": "Scenario: Error when RiskProfile missing",
            "scenario_type": "error_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": False,
            "given": ["Given no RiskProfile__c"],
            "when": ["When the adviser submits"],
            "then": ["Then an error shows"],
        },
        {
            "scenario": "Scenario: Duplicate blocked",
            "scenario_type": "edge_case",
            "test_category": "REGRESSION",
            "fca_relevant": False,
            "given": ["Given assessment exists"],
            "when": ["When adviser creates another"],
            "then": ["Then blocked"],
        },
    ],
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": False,  # missing for HIGH-FCA story
    },
    "gaps_filled": ["Happy path", "Error path", "Edge case"],
    "remaining_gaps": ["Regulatory: vulnerable customer Consumer Duty scenario not generated."],
}


# ── Confidence scoring unit tests (no LLM, no Jira) ──────────────────────────

class TestConfidenceScoring:
    def test_validated_existing_full_coverage_scores_highest(self):
        score, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_VALIDATED)
        assert score >= 78, f"Validated full-coverage should score ≥ 78, got {score}"

    def test_generated_from_scratch_full_coverage_scores_well(self):
        score, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_GENERATED)
        assert score >= 65, f"Generated from scratch with full coverage should score ≥ 65, got {score}"

    def test_missing_regulatory_scenario_for_high_fca_reduces_confidence(self):
        with_regulatory, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_GENERATED)
        without_regulatory, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_NO_REGULATORY)
        assert with_regulatory > without_regulatory, "Missing regulatory scenario should lower confidence for HIGH-FCA"

    def test_low_fca_partial_coverage_still_passes_threshold(self):
        score, _ = _compute_confidence(None, AGENT3_DATA_LOW, AGENT4_DATA_LOW, MOCK_AC_LOW_FCA)
        # LOW-FCA: no regulatory penalty; partial coverage acceptable
        assert score >= 20, "Must be at least floor"

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_VALIDATED)
        assert score <= 92, "Tier B cap is 92"

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, MOCK_AC_NO_REGULATORY)
        assert score >= 20, "Tier B floor is 20"

    def test_validated_beats_generated(self):
        validated_score, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_VALIDATED)
        generated_score, _ = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_GENERATED)
        assert validated_score > generated_score, "Validated existing ACs give higher confidence than generated"

    def test_signals_dict_has_generation_mode_signal(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_GENERATED)
        assert "generated_from_scratch" in signals or "supplemented_existing" in signals or "validated_existing" in signals

    def test_full_coverage_signal_present(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_VALIDATED)
        assert "full_coverage" in signals

    def test_supplemented_existing_signal_in_signals(self):
        _, signals = _compute_confidence(AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_SUPPLEMENTED)
        assert "supplemented_existing" in signals
        assert signals["supplemented_existing"] is True

    def test_clause_count_below_2_adds_penalty_signal(self):
        _, signals = _compute_confidence(None, None, None, MOCK_AC_ZERO_CLAUSES)
        assert "clause_count" in signals
        assert signals["clause_count"] == 0

    def test_rich_description_boosts_confidence(self):
        _, signals = _compute_confidence(AGENT1_DATA_RICH, None, None, MOCK_AC_GENERATED)
        assert "rich_description" in signals
        assert signals["rich_description"] == 200

    def test_sparse_description_reduces_confidence(self):
        _, signals = _compute_confidence(AGENT1_DATA_SPARSE, None, None, MOCK_AC_GENERATED)
        assert "sparse_description" in signals
        assert signals["sparse_description"] == 20

    def test_good_coverage_signal_in_signals(self):
        _, signals = _compute_confidence(None, AGENT3_DATA_HIGH, None, MOCK_AC_NO_REGULATORY)
        assert "good_coverage" in signals

    def test_partial_coverage_signal_in_signals(self):
        _, signals = _compute_confidence(None, AGENT3_DATA_LOW, None, MOCK_AC_LOW_FCA)
        assert "partial_coverage" in signals

    def test_cd_obligations_covered_signal_in_signals(self):
        _, signals = _compute_confidence(None, None, AGENT4_DATA_HIGH, MOCK_AC_GENERATED)
        assert "cd_obligations_covered" in signals

    def test_cd_obligations_uncovered_signal_in_signals(self):
        _, signals = _compute_confidence(None, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH, MOCK_AC_NO_REGULATORY)
        assert "cd_obligations_uncovered" in signals


# ── Integration tests — full agent run with mocked LLM and Jira ───────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA}
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}
        state["agent_results"]["4"] = {"data": AGENT4_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        assert result.agent_id == 5
        assert result.agent_name == "AC Generator"
        assert result.model_used == "claude-sonnet-4-6"
        assert result.confidence.tier == "B"

    async def test_ac_clauses_count_correct(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        assert result.data["ac_clause_count"] == 4
        assert result.data["generation_mode"] == "generated_from_scratch"

    async def test_fca_relevant_clause_count(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        # 3 of 4 clauses in MOCK_AC_GENERATED have fca_relevant=True
        assert result.data["fca_relevant_clause_count"] == 3

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        required = [
            "ac_clauses", "ac_clause_count", "generation_mode",
            "coverage_assessment", "gaps_filled", "remaining_gaps",
        ]
        for key in required:
            assert key in result.data, f"Missing downstream-required key: {key}"

    async def test_each_clause_has_gherkin_structure(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        valid_test_categories = {"UNIT", "UI", "FUNCTIONAL", "REGRESSION", "AUTOMATION_CANDIDATE"}
        for clause in result.data["ac_clauses"]:
            assert "scenario" in clause
            assert "given" in clause and isinstance(clause["given"], list)
            assert "when" in clause and isinstance(clause["when"], list)
            assert "then" in clause and isinstance(clause["then"], list)
            assert "scenario_type" in clause
            assert "test_category" in clause
            assert clause["test_category"] in valid_test_categories, (
                f"test_category '{clause['test_category']}' not in allowed values"
            )
            assert "fca_relevant" in clause

    async def test_validated_mode_when_existing_acs_passed(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = [{"scenario": "existing", "given": [], "when": [], "then": []}]
            mock_llm.return_value = MOCK_AC_VALIDATED

            result = await run(state)

        assert result.data["generation_mode"] == "validated_existing"
        assert result.data["existing_ac_count"] == 1

    async def test_runs_without_any_upstream_agent_data(self):
        state = initial_story_state("FSC-2417")  # no agent_results

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        assert result.agent_id == 5
        assert result.data["ac_clause_count"] == 4

    async def test_zero_clauses_causes_escalation(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_ZERO_CLAUSES

            result = await run(state)

        assert result.confidence.escalated is True

    async def test_fca_classification_context_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        assert result.data["fca_classification_context"] == "HIGH"

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED

            result = await run(state)

        assert "FSC-2417" in result.what


# ── Mechanism design signal tests ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestMechanismDesign:
    async def test_generation_mode_trust_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED
            result = await run(state)

        assert "generation_mode_trust" in result.data
        assert result.data["generation_mode_trust"] in (0.6, 0.8, 1.0)

    async def test_generated_from_scratch_trust_is_0_6(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_GENERATED  # generation_mode = "generated_from_scratch"
            result = await run(state)

        assert result.data["generation_mode"] == "generated_from_scratch"
        assert result.data["generation_mode_trust"] == 0.6

    async def test_validated_existing_trust_is_1_0(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = [{"scenario": "existing", "given": [], "when": [], "then": []}]
            mock_llm.return_value = MOCK_AC_VALIDATED  # generation_mode = "validated_existing"
            result = await run(state)

        assert result.data["generation_mode"] == "validated_existing"
        assert result.data["generation_mode_trust"] == 1.0

    async def test_trust_is_downstream_signal_not_self_penalty(self):
        """generation_mode_trust must be in data regardless of FCA class."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_LOW}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_LOW_FCA  # generation_mode = "generated_from_scratch"
            result = await run(state)

        assert "generation_mode_trust" in result.data

    async def test_supplemented_existing_trust_is_0_8(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = [{"scenario": "existing", "given": [], "when": [], "then": []}]
            mock_llm.return_value = MOCK_AC_SUPPLEMENTED

            result = await run(state)

        assert result.data["generation_mode"] == "supplemented_existing"
        assert result.data["generation_mode_trust"] == 0.8


# ── REQ-03: vulnerable_customer scenario_type tests ──────────────────────────

MOCK_AC_WITH_VC = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Adviser records suitability",
            "scenario_type": "happy_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has a RiskProfile__c"],
            "when": ["When the adviser submits the assessment"],
            "then": ["Then a Suitability__c is created"],
        },
        {
            "scenario": "Scenario: RiskProfile missing — blocked",
            "scenario_type": "error_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has no RiskProfile__c"],
            "when": ["When the adviser opens the flow"],
            "then": ["Then an error is shown"],
        },
        {
            "scenario": "Scenario: Duplicate assessment prevented",
            "scenario_type": "edge_case",
            "test_category": "REGRESSION",
            "fca_relevant": False,
            "given": ["Given a Suitability__c already exists"],
            "when": ["When the adviser creates another"],
            "then": ["Then the duplicate is blocked"],
        },
        {
            "scenario": "Scenario: Vulnerable customer pathway triggers Consumer Duty step",
            "scenario_type": "vulnerable_customer",
            "test_category": "UI",
            "fca_relevant": True,
            "given": ["Given VulnerableCustomerIndicator__c is true for the client"],
            "when": ["When the adviser reaches the final suitability step"],
            "then": [
                "Then the Consumer Duty confirmation checkbox is displayed",
                "Then the assessment cannot be submitted without checking the box",
            ],
        },
    ],
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": False,
        "vulnerable_customer": True,
    },
    "gaps_filled": ["happy_path", "error_path", "edge_case", "vulnerable_customer"],
    "remaining_gaps": [],
}

MOCK_AC_VC_MISSING = {
    **MOCK_AC_GENERATED,
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": True,
        "vulnerable_customer": False,
    },
}

MOCK_AC_SUPPLEMENTED = {
    "ac_clauses": [
        {
            "scenario": "Scenario: Adviser records suitability",
            "scenario_type": "happy_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given the client has a RiskProfile__c"],
            "when": ["When the adviser submits"],
            "then": ["Then a Suitability__c is created"],
        },
        {
            "scenario": "Scenario: Missing RiskProfile blocks assessment",
            "scenario_type": "error_path",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given no RiskProfile__c"],
            "when": ["When the flow is opened"],
            "then": ["Then an error is shown"],
        },
        {
            "scenario": "Scenario: Duplicate assessment prevented",
            "scenario_type": "edge_case",
            "test_category": "REGRESSION",
            "fca_relevant": False,
            "given": ["Given a Suitability__c already exists"],
            "when": ["When the adviser creates another"],
            "then": ["Then the duplicate is blocked"],
        },
        {
            "scenario": "Scenario: COBS 9.2 regulatory check",
            "scenario_type": "regulatory",
            "test_category": "FUNCTIONAL",
            "fca_relevant": True,
            "given": ["Given COBS 9.2 applies to this client"],
            "when": ["When the suitability assessment is submitted"],
            "then": ["Then a COBS 9.2 compliance record is created"],
        },
    ],
    "generation_mode": "supplemented_existing",
    "coverage_assessment": {
        "happy_path": True,
        "error_paths": True,
        "edge_cases": True,
        "regulatory": True,
        "vulnerable_customer": False,
    },
    "gaps_filled": ["regulatory: COBS 9.2 scenario added"],
    "remaining_gaps": [],
}

MOCK_AC_ZERO_CLAUSES = {
    "ac_clauses": [],
    "generation_mode": "generated_from_scratch",
    "coverage_assessment": {
        "happy_path": False,
        "error_paths": False,
        "edge_cases": False,
        "regulatory": False,
        "vulnerable_customer": False,
    },
    "gaps_filled": [],
    "remaining_gaps": ["Description too sparse to generate ACs."],
}

AGENT1_DATA_RICH = {**AGENT1_DATA, "description_word_count": 200}
AGENT1_DATA_SPARSE = {**AGENT1_DATA, "description_word_count": 20}
STORY_NO_DESCRIPTION = {**STORY_LABEL_CHANGE, "story_id": "FSC-2501", "description": None}


class TestVulnerableCustomerREQ03:
    def test_vc_impact_true_with_vc_scenario_boosts_confidence(self):
        score_with, _ = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": True, "cd_obligations": []},
            MOCK_AC_WITH_VC,
        )
        score_without, _ = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": False, "cd_obligations": []},
            MOCK_AC_WITH_VC,
        )
        assert score_with > score_without

    def test_vc_impact_true_missing_vc_scenario_penalises_confidence(self):
        score_present, _ = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": True, "cd_obligations": []},
            MOCK_AC_WITH_VC,
        )
        score_missing, _ = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": True, "cd_obligations": []},
            MOCK_AC_VC_MISSING,
        )
        assert score_present > score_missing

    def test_vc_impact_false_no_penalty_when_vc_absent(self):
        _, signals = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": False, "cd_obligations": []},
            MOCK_AC_VC_MISSING,
        )
        assert "vulnerable_customer_scenario_missing" not in signals

    def test_score_never_below_20_with_vc_penalty(self):
        score, _ = _compute_confidence(
            None, None,
            {"vulnerable_customer_impact": True, "cd_obligations": []},
            MOCK_AC_VC_MISSING,
        )
        assert score >= 20


@pytest.mark.asyncio
class TestVulnerableCustomerIntegrationREQ03:
    async def test_vc_scenario_type_present_in_ac_clauses(self):
        """REQ-03: vulnerable_customer_impact=True → ac_clauses contains vc scenario type."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}
        state["agent_results"]["4"] = {"data": AGENT4_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_WITH_VC
            result = await run(state)

        vc_scenarios = [
            c for c in result.data["ac_clauses"]
            if c.get("scenario_type") == "vulnerable_customer"
        ]
        assert len(vc_scenarios) >= 1, "Must have at least 1 vulnerable_customer scenario"

    async def test_coverage_assessment_has_vulnerable_customer_key(self):
        """REQ-03: coverage_assessment dict must include vulnerable_customer bool."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["4"] = {"data": AGENT4_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_WITH_VC
            result = await run(state)

        assert "vulnerable_customer" in result.data["coverage_assessment"]
        assert result.data["coverage_assessment"]["vulnerable_customer"] is True

    async def test_vc_coverage_false_when_vc_scenario_absent(self):
        """REQ-03: coverage_assessment.vulnerable_customer=False when no VC scenario present."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_05_ac_generator.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_05_ac_generator.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_05_ac_generator.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_AC_VC_MISSING
            result = await run(state)

        assert result.data["coverage_assessment"]["vulnerable_customer"] is False


# ── Prompt content unit tests (no LLM, no Jira) ───────────────────────────────

class TestPromptContent:
    def test_prompt_includes_story_id(self):
        msg = _build_user_message(STORY_SUITABILITY, [], AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH)
        assert "FSC-2417" in msg

    def test_prompt_includes_summary(self):
        msg = _build_user_message(STORY_SUITABILITY, [], AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH)
        assert STORY_SUITABILITY["summary"] in msg

    def test_prompt_includes_components(self):
        msg = _build_user_message(STORY_SUITABILITY, [], AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH)
        assert "COMPONENTS:" in msg

    def test_prompt_empty_components_renders_as_none(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, [], None, None, None)
        assert "COMPONENTS: None" in msg

    def test_prompt_empty_description_shows_empty(self):
        msg = _build_user_message(STORY_NO_DESCRIPTION, [], None, None, None)
        assert "(empty)" in msg

    def test_prompt_existing_acs_present_shows_section(self):
        existing = [{"scenario": "Scenario: existing", "given": [], "when": [], "then": []}]
        msg = _build_user_message(STORY_SUITABILITY, existing, None, None, None)
        assert "EXISTING ACCEPTANCE CRITERIA (from Jira):" in msg

    def test_prompt_no_existing_acs_shows_none_in_jira(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, None, None)
        assert "None in Jira." in msg

    def test_prompt_includes_agent1_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, [], AGENT1_DATA, None, None)
        assert "AGENT 1 — STORY INTENT:" in msg
        assert "Goal:" in msg
        assert "Missing Elements:" in msg

    def test_prompt_agent1_section_absent_when_no_agent1_data(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, None, None)
        assert "AGENT 1 — STORY INTENT:" not in msg

    def test_prompt_includes_agent3_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, AGENT3_DATA_HIGH, None)
        assert "AGENT 3 — FCA CLASSIFICATION:" in msg
        assert "FCA Triggers:" in msg

    def test_prompt_agent3_section_absent_when_no_agent3_data(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, None, None)
        assert "AGENT 3 — FCA CLASSIFICATION:" not in msg

    def test_prompt_includes_agent4_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, None, AGENT4_DATA_HIGH)
        assert "AGENT 4 — CONSUMER DUTY:" in msg
        assert "CD Verdict:" in msg
        assert "CD Obligations:" in msg

    def test_prompt_agent4_section_absent_when_no_agent4_data(self):
        msg = _build_user_message(STORY_SUITABILITY, [], None, None, None)
        assert "AGENT 4 — CONSUMER DUTY:" not in msg

    def test_prompt_ends_with_tool_instruction(self):
        msg = _build_user_message(STORY_SUITABILITY, [], AGENT1_DATA, AGENT3_DATA_HIGH, AGENT4_DATA_HIGH)
        assert _TOOL_NAME in msg
        assert msg.strip().endswith("AC set.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_tool_schema_has_five_required_fields(self):
        expected = {
            "generation_mode",
            "coverage_assessment",
            "ac_clauses",
            "gaps_filled",
            "remaining_gaps",
        }
        assert set(_TOOL_SCHEMA["required"]) == expected

    def test_generation_mode_is_enum_with_three_values(self):
        schema = _TOOL_SCHEMA["properties"]["generation_mode"]
        assert schema["enum"] == ["generated_from_scratch", "supplemented_existing", "validated_existing"]

    def test_coverage_assessment_has_five_required_subfields(self):
        expected = {"happy_path", "error_paths", "edge_cases", "regulatory", "vulnerable_customer"}
        subschema = _TOOL_SCHEMA["properties"]["coverage_assessment"]
        assert set(subschema["required"]) == expected

    def test_ac_clauses_is_array(self):
        assert _TOOL_SCHEMA["properties"]["ac_clauses"]["type"] == "array"

    def test_gaps_filled_is_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["gaps_filled"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_remaining_gaps_is_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["remaining_gaps"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_ac_clause_schema_has_seven_required_fields(self):
        expected = {"scenario", "scenario_type", "test_category", "fca_relevant", "given", "when", "then"}
        assert set(_AC_CLAUSE_SCHEMA["required"]) == expected

    def test_scenario_type_enum_has_five_values(self):
        schema = _AC_CLAUSE_SCHEMA["properties"]["scenario_type"]
        assert schema["enum"] == ["happy_path", "error_path", "edge_case", "regulatory", "vulnerable_customer"]

    def test_test_category_enum_has_five_values(self):
        schema = _AC_CLAUSE_SCHEMA["properties"]["test_category"]
        assert set(schema["enum"]) == {"UNIT", "UI", "FUNCTIONAL", "REGRESSION", "AUTOMATION_CANDIDATE"}

    def test_fca_relevant_is_boolean_in_clause_schema(self):
        assert _AC_CLAUSE_SCHEMA["properties"]["fca_relevant"]["type"] == "boolean"
