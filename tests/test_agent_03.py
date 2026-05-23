"""
Tests for Agent 3 — FCA Risk Classifier (Tier D Ensemble).

Uses mock Jira and mock LLM so tests run without live infrastructure.
Tests: ensemble resolution logic (agreement/disagreement), "safer call wins",
tier gap graduated confidence, confidence caps and floors, AgentResult shape.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_03_fca_classifier import (
    _build_user_message,
    _pick_primary_call,
    _resolve_ensemble,
    _TOOL_NAME,
    _TOOL_SCHEMA,
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

STORY_FINANCIAL_ACCOUNT = {
    "story_id": "FSC-2600",
    "summary": "Add AUM roll-up to FinancialAccount",
    "description": (
        "As an Operations user, I want the FinancialAccount to show a real-time AUM "
        "roll-up from all linked FinancialHolding records so that advisers can see "
        "total portfolio value without navigating to each holding."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Medium",
    "labels": [],
    "components": ["WealthCore"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

AC_CLAUSES = [
    {
        "source": "description",
        "scenario": "Scenario: Standard adviser records suitability",
        "given": ["Given the client has a RiskProfile__c"],
        "when": ["When the adviser completes the Suitability Assessment flow"],
        "then": ["Then a Suitability__c record is created"],
    },
]

MOCK_HIGH = {
    "fca_classification": "HIGH",
    "classification_rationale": (
        "Story directly modifies Suitability__c and RiskProfile__c — core COBS 9.2 objects. "
        "VulnerableCustomerIndicator__c triggers additional Consumer Duty obligations."
    ),
    "fca_triggers": ["Suitability__c", "RiskProfile__c", "VulnerableCustomerIndicator__c", "COBS 9.2"],
    "regulatory_obligations": ["COBS 9.2 Suitability", "Consumer Duty PS22/9", "FG21/1 Vulnerable Customers"],
    "co_signoff_required": True,
    "enhanced_testing_required": True,
}

MOCK_MEDIUM = {
    "fca_classification": "MEDIUM",
    "classification_rationale": (
        "Story touches FinancialAccount and FinancialHolding for AUM roll-up. "
        "No Suitability or RiskProfile objects are modified."
    ),
    "fca_triggers": ["FinancialAccount", "FinancialHolding", "AUM"],
    "regulatory_obligations": [],
    "co_signoff_required": False,
    "enhanced_testing_required": True,
}

MOCK_LOW = {
    "fca_classification": "LOW",
    "classification_rationale": (
        "Story is a cosmetic UI change — renaming a button label. "
        "No FSC financial or regulatory objects are touched."
    ),
    "fca_triggers": [],
    "regulatory_obligations": [],
    "co_signoff_required": False,
    "enhanced_testing_required": False,
}

MOCK_UNCLASSIFIED = {
    **MOCK_LOW,
    "fca_classification": "UNCLASSIFIED",
    "classification_rationale": "Insufficient context — description too sparse to classify.",
    "fca_triggers": [],
    "regulatory_obligations": [],
    "co_signoff_required": False,
    "enhanced_testing_required": False,
}

STORY_SPARSE = {
    "story_id": "FSC-9999",
    "summary": "Do the thing",
    "description": "Update it.",  # < 15 meaningful words → UNCLASSIFIED
    "status": "In Progress",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": None,
}

AGENT1_DATA_SUITABILITY = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount", "VulnerableCustomerIndicator__c"],
    "flags": ["high_fca_object_detected"],
    "story_summary": "Records COBS 9.2 suitability assessment for retirement portfolios.",
    "description_word_count": 85,
}


# ── Ensemble resolution unit tests (no LLM, no Jira) ─────────────────────────

class TestEnsembleResolution:
    def test_both_high_returns_high_confidence_85(self):
        classification, confidence, signals = _resolve_ensemble(MOCK_HIGH, MOCK_HIGH)
        assert classification == "HIGH"
        assert confidence == 85
        assert signals["ensemble_agreement"] is True
        assert signals["tier_gap"] == 0

    def test_both_medium_returns_medium_confidence_80(self):
        classification, confidence, signals = _resolve_ensemble(MOCK_MEDIUM, MOCK_MEDIUM)
        assert classification == "MEDIUM"
        assert confidence == 80
        assert signals["ensemble_agreement"] is True

    def test_both_low_returns_low_confidence_78(self):
        classification, confidence, signals = _resolve_ensemble(MOCK_LOW, MOCK_LOW)
        assert classification == "LOW"
        assert confidence == 78
        assert signals["ensemble_agreement"] is True

    def test_high_vs_medium_conservative_wins(self):
        """Adjacent tier gap — HIGH wins (safer), confidence 48."""
        classification, confidence, signals = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert classification == "HIGH"
        assert confidence == 48
        assert signals["ensemble_agreement"] is False
        assert signals["tier_gap"] == 1

    def test_medium_vs_low_conservative_wins(self):
        """Adjacent tier gap — MEDIUM wins (safer), confidence 48."""
        classification, confidence, signals = _resolve_ensemble(MOCK_MEDIUM, MOCK_LOW)
        assert classification == "MEDIUM"
        assert confidence == 48
        assert signals["ensemble_agreement"] is False
        assert signals["tier_gap"] == 1

    def test_high_vs_low_large_gap_lower_confidence(self):
        """Two-tier gap — HIGH wins, confidence 38 (less certain)."""
        classification, confidence, signals = _resolve_ensemble(MOCK_HIGH, MOCK_LOW)
        assert classification == "HIGH"
        assert confidence == 38
        assert signals["tier_gap"] == 2

    def test_disagreement_triggers_escalation(self):
        """All disagreements produce confidence < 60 — auto-escalate threshold."""
        _, confidence, _ = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert confidence < 60, "Disagreement must produce confidence below escalation threshold"

    def test_unclassified_loses_to_real_classification(self):
        """UNCLASSIFIED defers to whichever call has a real tier."""
        unclassified = {**MOCK_LOW, "fca_classification": "UNCLASSIFIED"}
        classification, confidence, signals = _resolve_ensemble(unclassified, MOCK_MEDIUM)
        assert classification == "MEDIUM"
        assert signals["ensemble_agreement"] is False

    def test_order_independence_high_vs_medium(self):
        """Safer-call-wins must be symmetric — swapping call order changes nothing."""
        class_a, conf_a, _ = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        class_b, conf_b, _ = _resolve_ensemble(MOCK_MEDIUM, MOCK_HIGH)
        assert class_a == class_b
        assert conf_a == conf_b

    def test_signals_dict_has_required_keys(self):
        _, _, signals = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert "ensemble_agreement" in signals
        assert "call_a" in signals
        assert "call_b" in signals
        assert "tier_gap" in signals

    # C3 — UNCLASSIFIED agreement → confidence=55 (never hit by existing fixtures)
    def test_both_unclassified_returns_confidence_55(self):
        classification, confidence, signals = _resolve_ensemble(MOCK_UNCLASSIFIED, MOCK_UNCLASSIFIED)
        assert classification == "UNCLASSIFIED"
        assert confidence == 55
        assert signals["ensemble_agreement"] is True

    # C4 — tier_gap=3 → confidence=30 (HIGH vs UNCLASSIFIED; only gap 1 and 2 were tested)
    def test_high_vs_unclassified_three_tier_gap_confidence_30(self):
        classification, confidence, signals = _resolve_ensemble(MOCK_HIGH, MOCK_UNCLASSIFIED)
        assert confidence == 30
        assert signals["tier_gap"] == 3
        assert classification == "HIGH"

    # C5 — conservative_winner in signals never verified
    def test_conservative_winner_in_signals_on_disagreement(self):
        _, _, signals = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert signals["conservative_winner"] == "HIGH"

    # H1 — _pick_primary_call never tested directly
    def test_pick_primary_call_returns_call_a_when_a_matches(self):
        result = _pick_primary_call(MOCK_HIGH, MOCK_MEDIUM, "HIGH")
        assert result is MOCK_HIGH

    def test_pick_primary_call_returns_call_b_when_b_matches(self):
        result = _pick_primary_call(MOCK_MEDIUM, MOCK_HIGH, "HIGH")
        assert result is MOCK_HIGH

    # M1 — MEDIUM vs LOW disagreement: both confs ≥ 60 → TA position OK_OK despite disagreement
    def test_medium_vs_low_disagreement_ta_position_ok_ok(self):
        # MEDIUM conf=80, LOW conf=78 — both ≥ threshold(60) → OK_OK / COLLABORATE
        _, _, signals = _resolve_ensemble(MOCK_MEDIUM, MOCK_LOW)
        assert signals["ta_position"] == "OK_OK"
        assert signals["interaction_mode"] == "COLLABORATE"
        assert signals["ensemble_agreement"] is False  # still a disagreement


# ── Integration tests — full agent run with mocked LLM and Jira ───────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            # Both calls return HIGH → agreement
            mock_llm.return_value = MOCK_HIGH

            result = await run(state)

        assert result.agent_id == 3
        assert result.agent_name == "FCA Risk Classifier"
        assert result.model_used == "claude-sonnet-4-6"
        assert result.confidence.tier == "D"

    async def test_high_agreement_sets_classification_and_co_flag(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            mock_llm.return_value = MOCK_HIGH

            result = await run(state)

        assert result.data["fca_classification"] == "HIGH"
        assert result.data["co_signoff_required"] is True
        assert result.data["enhanced_testing_required"] is True
        assert result.data["ensemble_agreement"] is True
        assert result.confidence.final_score == 85
        assert result.confidence.escalated is False

    async def test_low_agreement_no_escalation(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_LOW

            result = await run(state)

        assert result.data["fca_classification"] == "LOW"
        assert result.data["co_signoff_required"] is False
        assert result.data["enhanced_testing_required"] is False
        assert result.confidence.final_score == 78
        assert result.confidence.escalated is False

    async def test_disagreement_escalates(self):
        """Simulate call-A returning HIGH, call-B returning MEDIUM → escalate."""
        state = initial_story_state("FSC-2417")

        call_count = 0

        async def alternating_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MOCK_HIGH if call_count % 2 == 1 else MOCK_MEDIUM

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  side_effect=alternating_llm),
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES

            result = await run(state)

        assert result.data["fca_classification"] == "HIGH"  # safer wins
        assert result.data["ensemble_agreement"] is False
        assert result.confidence.final_score == 48
        assert result.confidence.escalated is True  # 48 < 60

    async def test_data_has_required_downstream_keys(self):
        """Fleet Commander and downstream agents depend on these keys."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            mock_llm.return_value = MOCK_HIGH

            result = await run(state)

        required = [
            "fca_classification", "co_signoff_required", "enhanced_testing_required",
            "fca_triggers", "regulatory_obligations",
            "ensemble_agreement", "call_a_classification", "call_b_classification",
        ]
        for key in required:
            assert key in result.data, f"Missing downstream-required key: {key}"

    async def test_medium_classification_no_co_but_enhanced_testing(self):
        state = initial_story_state("FSC-2600")

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_FINANCIAL_ACCOUNT
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_MEDIUM

            result = await run(state)

        assert result.data["fca_classification"] == "MEDIUM"
        assert result.data["co_signoff_required"] is False
        assert result.data["enhanced_testing_required"] is True
        assert result.confidence.final_score == 80
        assert result.confidence.escalated is False

    async def test_runs_without_agent1_data(self):
        """Agent 3 must work standalone if Agent 1 hasn't run yet."""
        state = initial_story_state("FSC-2417")  # no agent_results["1"]

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            mock_llm.return_value = MOCK_HIGH

            result = await run(state)

        assert result.agent_id == 3
        assert result.data["fca_classification"] == "HIGH"

    async def test_fca_triggers_populated_for_high(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            mock_llm.return_value = MOCK_HIGH

            result = await run(state)

        assert len(result.data["fca_triggers"]) > 0
        assert "Suitability__c" in result.data["fca_triggers"]

    # H — data dict keys/values never tested

    async def test_classification_rationale_in_data(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            ml.return_value = MOCK_HIGH
            result = await run(state)
        assert isinstance(result.data["classification_rationale"], str)
        assert len(result.data["classification_rationale"]) > 0

    async def test_regulatory_obligations_in_data(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            ml.return_value = MOCK_HIGH
            result = await run(state)
        assert isinstance(result.data["regulatory_obligations"], list)
        assert len(result.data["regulatory_obligations"]) >= 1  # HIGH always has obligations

    async def test_tier_gap_zero_on_agreement(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            ml.return_value = MOCK_HIGH
            result = await run(state)
        assert result.data["tier_gap"] == 0

    async def test_tier_gap_one_on_high_medium_disagreement(self):
        state = initial_story_state("FSC-2417")
        call_count = 0

        async def alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MOCK_HIGH if call_count % 2 == 1 else MOCK_MEDIUM

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", side_effect=alternating),
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            result = await run(state)
        assert result.data["tier_gap"] == 1

    async def test_call_a_and_call_b_values_in_data(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            ml.return_value = MOCK_HIGH
            result = await run(state)
        assert result.data["call_a_classification"] == "HIGH"
        assert result.data["call_b_classification"] == "HIGH"

    async def test_unclassified_agreement_co_false_enhanced_false_escalated(self):
        """Both calls returning UNCLASSIFIED → confidence=55 < 60 → escalated=True."""
        state = initial_story_state("FSC-9999")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SPARSE
            ma.return_value = []
            ml.return_value = MOCK_UNCLASSIFIED
            result = await run(state)
        assert result.data["fca_classification"] == "UNCLASSIFIED"
        assert result.data["co_signoff_required"] is False
        assert result.data["enhanced_testing_required"] is False
        assert result.confidence.final_score == 55
        assert result.confidence.escalated is True

    # M2 — what string content never tested
    async def test_what_string_contains_story_id_and_classification(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story", new_callable=AsyncMock) as ms,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria", new_callable=AsyncMock) as ma,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool", new_callable=AsyncMock) as ml,
        ):
            ms.return_value = STORY_SUITABILITY
            ma.return_value = AC_CLAUSES
            ml.return_value = MOCK_HIGH
            result = await run(state)
        assert "FSC-2417" in result.what
        assert "HIGH" in result.what


# ── Transactional Analysis integration tests ──────────────────────────────────

@pytest.mark.asyncio
class TestTAPosition:
    def test_agreement_produces_ok_ok_ta_position(self):
        _, _, signals = _resolve_ensemble(MOCK_HIGH, MOCK_HIGH)
        assert signals["ta_position"] == "OK_OK"
        assert signals["interaction_mode"] == "COLLABORATE"

    def test_disagreement_produces_non_ok_ok_ta_position(self):
        """HIGH (conf 85) vs UNCLASSIFIED (conf 55 < 60) → OK_NOT_OK."""
        unclassified = {**MOCK_LOW, "fca_classification": "UNCLASSIFIED"}
        _, _, signals = _resolve_ensemble(MOCK_HIGH, unclassified)
        assert signals["ta_position"] != "OK_OK"

    def test_signals_dict_includes_ta_keys(self):
        _, _, signals = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert "ta_position" in signals
        assert "interaction_mode" in signals

    def test_low_vs_low_agreement_ok_ok(self):
        _, _, signals = _resolve_ensemble(MOCK_LOW, MOCK_LOW)
        assert signals["ta_position"] == "OK_OK"

    def test_disagreement_has_ta_rationale(self):
        _, _, signals = _resolve_ensemble(MOCK_HIGH, MOCK_MEDIUM)
        assert "ta_rationale" in signals

    async def test_ta_position_in_result_data_on_agreement(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["1"] = {"data": AGENT1_DATA_SUITABILITY}

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            mock_llm.return_value = MOCK_HIGH
            result = await run(state)

        assert "ta_position" in result.data
        assert "interaction_mode" in result.data
        assert result.data["ta_position"] == "OK_OK"
        assert result.data["interaction_mode"] == "COLLABORATE"

    async def test_ta_position_in_result_data_on_disagreement(self):
        """HIGH (conf 85) vs UNCLASSIFIED (conf 55 < 60) → OK_NOT_OK → ta_position != OK_OK."""
        state = initial_story_state("FSC-2417")
        unclassified = {**MOCK_LOW, "fca_classification": "UNCLASSIFIED"}

        call_count = 0

        async def alternating_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MOCK_HIGH if call_count % 2 == 1 else unclassified

        with (
            patch("src.agents.refinement.agent_03_fca_classifier.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_03_fca_classifier.get_acceptance_criteria",
                  new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_03_fca_classifier.call_with_tool",
                  side_effect=alternating_llm),
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES
            result = await run(state)

        assert result.data["ta_position"] != "OK_OK"
        assert result.data["ensemble_agreement"] is False


# ── Prompt content tests ──────────────────────────────────────────────────────

class TestPromptContent:
    """
    Tests that _build_user_message() produces the prompt both Sonnet calls receive.
    Agent 3 prompt: STORY ID, SUMMARY, COMPONENTS, DESCRIPTION only (no STATUS/PRIORITY).
    Agent 1 section: FSC Objects, Flags, Story Summary only (no goal/persona).
    Final instruction ends with 'classification.' not 'tool.'.
    """

    def test_prompt_includes_story_id(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert "FSC-2417" in msg

    def test_prompt_includes_summary(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert STORY_SUITABILITY["summary"] in msg

    def test_prompt_includes_components(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert "COMPONENTS:" in msg
        assert "Suitability" in msg  # from STORY_SUITABILITY["components"]

    def test_prompt_empty_components_renders_as_none(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, [], None)
        assert "COMPONENTS: None" in msg

    def test_prompt_shows_empty_when_description_is_none(self):
        story = {**STORY_LABEL_CHANGE, "description": None}
        msg = _build_user_message(story, [], None)
        assert "(empty)" in msg

    def test_prompt_ac_present_shows_scenario_structure(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert "ACCEPTANCE CRITERIA:" in msg
        assert "Scenario" in msg
        assert "Given" in msg
        assert "Then" in msg

    def test_prompt_ac_absent_shows_none_provided(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, [], None)
        assert "None provided." in msg

    def test_prompt_includes_agent1_section_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert "AGENT 1 PRE-ANALYSIS" in msg
        # Agent 3 section includes FSC Objects and Story Summary
        assert "Suitability__c" in msg  # from fsc_objects
        assert AGENT1_DATA_SUITABILITY["story_summary"] in msg

    def test_prompt_agent1_section_absent_when_no_agent1_data(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, None)
        assert "AGENT 1 PRE-ANALYSIS" not in msg

    def test_prompt_ends_with_tool_instruction(self):
        # If tool instruction is missing, call_with_tool() will raise RuntimeError.
        # Agent 3's instruction ends with "classification." (not "tool.")
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES, AGENT1_DATA_SUITABILITY)
        assert _TOOL_NAME in msg  # "classify_fca_risk" must appear
        assert msg.strip().endswith("classification.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    """
    Tests that _TOOL_SCHEMA enforces the correct structure for both ensemble calls.
    Both _INSTRUCTIONS_CAUTIOUS and _INSTRUCTIONS_EVIDENCE_BASED use the same schema.
    """

    def test_all_six_fields_are_required(self):
        required = set(_TOOL_SCHEMA["required"])
        expected = {
            "fca_classification", "classification_rationale",
            "fca_triggers", "regulatory_obligations",
            "co_signoff_required", "enhanced_testing_required",
        }
        assert required == expected, f"Required fields mismatch: {required ^ expected}"

    def test_fca_classification_is_enum_with_four_values(self):
        enum = set(_TOOL_SCHEMA["properties"]["fca_classification"]["enum"])
        assert enum == {"HIGH", "MEDIUM", "LOW", "UNCLASSIFIED"}

    def test_classification_rationale_is_string(self):
        assert _TOOL_SCHEMA["properties"]["classification_rationale"]["type"] == "string"

    def test_fca_triggers_is_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["fca_triggers"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_regulatory_obligations_is_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["regulatory_obligations"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_co_signoff_required_is_boolean(self):
        assert _TOOL_SCHEMA["properties"]["co_signoff_required"]["type"] == "boolean"

    def test_enhanced_testing_required_is_boolean(self):
        assert _TOOL_SCHEMA["properties"]["enhanced_testing_required"]["type"] == "boolean"
