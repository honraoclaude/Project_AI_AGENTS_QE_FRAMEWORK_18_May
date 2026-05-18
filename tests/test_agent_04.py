"""
Tests for Agent 4 — Consumer Duty Mapper.

Uses mock Jira and mock LLM so tests run without live infrastructure.
Tests: confidence scoring, verdict logic, required downstream keys,
       vulnerable customer detection, NOT_APPLICABLE for LOW-FCA stories.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_04_consumer_duty import _compute_confidence, run
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
        "then": ["Then a Suitability__c record is created linked to the FinancialAccount"],
    },
    {
        "source": "description",
        "scenario": "Scenario: Vulnerable customer — additional confirmation step",
        "given": ["Given VulnerableCustomerIndicator__c is true for the client"],
        "when": ["When the adviser reaches the final step of the Suitability flow"],
        "then": ["Then a Consumer Duty confirmation checkbox is displayed"],
    },
]

AGENT3_DATA_HIGH = {
    "fca_classification": "HIGH",
    "classification_rationale": "Story directly modifies Suitability__c and RiskProfile__c.",
    "fca_triggers": ["Suitability__c", "RiskProfile__c", "VulnerableCustomerIndicator__c", "COBS 9.2"],
    "regulatory_obligations": ["COBS 9.2 Suitability", "Consumer Duty PS22/9", "FG21/1 Vulnerable Customers"],
    "co_signoff_required": True,
    "enhanced_testing_required": True,
    "ensemble_agreement": True,
    "call_a_classification": "HIGH",
    "call_b_classification": "HIGH",
    "tier_gap": 0,
    "signals": {},
}

AGENT3_DATA_LOW = {
    "fca_classification": "LOW",
    "classification_rationale": "Cosmetic UI change — no FSC objects.",
    "fca_triggers": [],
    "regulatory_obligations": [],
    "co_signoff_required": False,
    "enhanced_testing_required": False,
    "ensemble_agreement": True,
    "call_a_classification": "LOW",
    "call_b_classification": "LOW",
    "tier_gap": 0,
    "signals": {},
}

AGENT3_DATA_DISAGREED = {
    "fca_classification": "HIGH",
    "classification_rationale": "Cautious call won.",
    "fca_triggers": ["Suitability__c"],
    "regulatory_obligations": ["COBS 9.2 Suitability"],
    "co_signoff_required": True,
    "enhanced_testing_required": True,
    "ensemble_agreement": False,
    "call_a_classification": "HIGH",
    "call_b_classification": "MEDIUM",
    "tier_gap": 1,
    "signals": {},
}

AGENT1_DATA = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount"],
    "story_summary": "Records COBS 9.2 suitability assessment for retirement portfolios.",
}

MOCK_CD_HIGH_COMPLIANT = {
    "cd_outcomes_affected": ["products_and_services", "consumer_support"],
    "vulnerable_customer_impact": True,
    "vulnerable_customer_rationale": (
        "Story adds Consumer Duty confirmation step for clients where "
        "VulnerableCustomerIndicator__c is true."
    ),
    "cd_obligations": [
        "PS22/9 Outcome 1 — Suitability Assessment must be fit for purpose for retirement clients.",
        "FG21/1 §4.3 — Vulnerable customers must not be systematically disadvantaged by the flow.",
    ],
    "cd_risks": [],
    "cd_evidence_required": [
        "Screenshot of Consumer Duty confirmation step in the Suitability flow.",
        "Test results showing the confirmation step is enforced for VulnerableCustomerIndicator__c = true.",
        "Impact assessment confirming retirement portfolio clients are the target market.",
    ],
    "cd_verdict": "COMPLIANT",
    "cd_rationale": (
        "Story explicitly adds the Consumer Duty confirmation step for vulnerable customers "
        "and links Suitability__c to the correct client objects. No CD gaps identified."
    ),
}

MOCK_CD_NOT_APPLICABLE = {
    "cd_outcomes_affected": ["none"],
    "vulnerable_customer_impact": False,
    "vulnerable_customer_rationale": (
        "Story is a cosmetic label change on a non-financial page. "
        "No vulnerable customer pathway is involved."
    ),
    "cd_obligations": [],
    "cd_risks": [],
    "cd_evidence_required": [],
    "cd_verdict": "NOT_APPLICABLE",
    "cd_rationale": (
        "Story is LOW-FCA — a button label rename with no customer-facing regulatory impact. "
        "Consumer Duty obligations do not apply."
    ),
}

MOCK_CD_AT_RISK = {
    "cd_outcomes_affected": ["products_and_services", "consumer_understanding"],
    "vulnerable_customer_impact": True,
    "vulnerable_customer_rationale": "VulnerableCustomerIndicator__c triggers a special flow.",
    "cd_obligations": [
        "PS22/9 Outcome 3 — information presented during the suitability flow must be clear.",
    ],
    "cd_risks": [
        "No AC covers the error scenario where VulnerableCustomerIndicator__c is true but the "
        "confirmation step is unavailable (e.g. field not populated).",
    ],
    "cd_evidence_required": [
        "Test results covering the vulnerable customer confirmation bypass scenario.",
    ],
    "cd_verdict": "AT_RISK",
    "cd_rationale": (
        "Story touches Consumer Duty outcomes but the acceptance criteria are missing an error "
        "scenario for the vulnerable customer confirmation step."
    ),
}


# ── Confidence scoring unit tests (no LLM, no Jira) ──────────────────────────

class TestConfidenceScoring:
    def test_high_fca_with_agreement_scores_high(self):
        score, _ = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        assert score >= 75, f"HIGH-FCA with Agent 3 agreement should score ≥ 75, got {score}"

    def test_low_fca_not_applicable_scores_high(self):
        score, _ = _compute_confidence(AGENT3_DATA_LOW, MOCK_CD_NOT_APPLICABLE)
        assert score >= 70, f"LOW-FCA NOT_APPLICABLE is a confident easy answer, got {score}"

    def test_agent3_disagreement_reduces_confidence(self):
        agreed_score, _ = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        disagreed_score, _ = _compute_confidence(AGENT3_DATA_DISAGREED, MOCK_CD_HIGH_COMPLIANT)
        assert agreed_score > disagreed_score, "Ensemble disagreement on Agent 3 should lower confidence"

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        assert score <= 92, "Tier B cap is 92"

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, MOCK_CD_NOT_APPLICABLE)
        assert score >= 20, "Tier B floor is 20"

    def test_no_agent3_data_reduces_confidence(self):
        with_agent3, _ = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        without_agent3, _ = _compute_confidence(None, MOCK_CD_HIGH_COMPLIANT)
        assert with_agent3 > without_agent3

    def test_vulnerable_customer_in_triggers_boosts_confidence(self):
        """VulnerableCustomerIndicator__c in triggers → confident detection."""
        score_with, signals_with = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        assert "vulnerable_customer_in_triggers" in signals_with

    def test_signals_dict_populated(self):
        _, signals = _compute_confidence(AGENT3_DATA_HIGH, MOCK_CD_HIGH_COMPLIANT)
        assert len(signals) > 0
        assert any(k.startswith("fca_") for k in signals)

    def test_not_applicable_adds_confidence_signal(self):
        _, signals = _compute_confidence(AGENT3_DATA_LOW, MOCK_CD_NOT_APPLICABLE)
        assert "not_applicable_verdict" in signals


# ── Integration tests — full agent run with mocked LLM and Jira ───────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_high_fca_story(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA}
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        assert result.agent_id == 4
        assert result.agent_name == "Consumer Duty Mapper"
        assert result.model_used == "claude-sonnet-4-6"
        assert result.confidence.tier == "B"

    async def test_compliant_verdict_for_well_specified_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA}
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        assert result.data["cd_verdict"] == "COMPLIANT"
        assert result.data["vulnerable_customer_impact"] is True
        assert "products_and_services" in result.data["cd_outcomes_affected"]
        assert len(result.data["cd_evidence_required"]) > 0

    async def test_not_applicable_for_low_fca_story(self):
        state = initial_story_state("FSC-2500")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_LOW}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_CD_NOT_APPLICABLE

            result = await run(state)

        assert result.data["cd_verdict"] == "NOT_APPLICABLE"
        assert result.data["vulnerable_customer_impact"] is False
        assert result.data["cd_obligations"] == []
        assert result.data["cd_evidence_required"] == []

    async def test_at_risk_verdict_escalates(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = []   # no ACs → AT_RISK
            mock_llm.return_value = MOCK_CD_AT_RISK

            result = await run(state)

        assert result.data["cd_verdict"] == "AT_RISK"
        assert len(result.data["cd_risks"]) > 0

    async def test_data_has_required_downstream_keys(self):
        """Agents 9 and 44 depend on these keys."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        required = [
            "cd_outcomes_affected", "vulnerable_customer_impact",
            "cd_obligations", "cd_risks", "cd_evidence_required",
            "cd_verdict", "cd_rationale",
        ]
        for key in required:
            assert key in result.data, f"Missing downstream-required key: {key}"

    async def test_fca_classification_recorded_in_data(self):
        state = initial_story_state("FSC-2417")
        state["fca_classification"] = "HIGH"
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        assert result.data["fca_classification_from_agent3"] == "HIGH"
        assert result.data["agent3_available"] is True

    async def test_runs_without_agent3_or_agent1_data(self):
        """Agent 4 must not crash if upstream agents haven't run."""
        state = initial_story_state("FSC-2417")  # no agent_results

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        assert result.agent_id == 4
        assert result.data["agent3_available"] is False

    async def test_what_field_contains_story_id_and_verdict(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_DATA_HIGH}

        with (
            patch("src.agents.refinement.agent_04_consumer_duty.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_04_consumer_duty.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_04_consumer_duty.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_CD_HIGH_COMPLIANT

            result = await run(state)

        assert "FSC-2417" in result.what
        assert "COMPLIANT" in result.what
