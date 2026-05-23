"""Tests for Agent 21 — Test Data Architect (True AI Agent, Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_21_test_data_architect import (
    _build_mechanism_signal,
    _build_prompt,
    _compute_completeness,
    _compute_confidence,
    _DATA_TOOL_NAME,
    _DATA_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH",   "ensemble_agreement": True}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM", "ensemble_agreement": True}
AGENT3_LOW    = {"fca_classification": "LOW",    "ensemble_agreement": True}

AGENT5_DATA = {"ac_count": 3, "acs_generated": True}

AGENT7_DATA = {
    "required_records": [
        "FinancialAccount with balance > £100k",
        "Suitability record with HIGH risk profile",
    ],
    "data_complexity": "HIGH",
}

AGENT7_SHARED_ORG = {
    "required_records": ["FinancialAccount"],
    "data_complexity": "LOW",
    "data_isolation_strategy": "shared_org_data",
    "fca_context_available": False,
}

AGENT7_PER_CLASS = {
    "required_records": ["FinancialAccount"],
    "data_complexity": "LOW",
    "data_isolation_strategy": "per_class_setup",
    "fca_context_available": False,
}

AGENT13_DATA = {
    "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
    "dependency_depth": 2,
}

AGENT19_DATA = {
    "gherkin_scenarios": [
        {
            "title": "HIGH-risk client fails suitability check",
            "tags": ["@fca", "@negative"],
            "steps": ["Given a HIGH-risk client", "When suitability runs", "Then it fails"],
        },
        {
            "title": "Valid portfolio rebalancing",
            "tags": ["@smoke"],
            "steps": ["Given a valid portfolio", "When rebalancing", "Then FSC rules apply"],
        },
    ],
    "scenario_count": 2,
}

MOCK_STORY = {
    "key": "FSC-2417",
    "summary": "Suitability Assessment Enhancement",
}

MOCK_DATA_PASS = {
    "seed_records": [
        {
            "object_name": "FinancialAccount",
            "record_count": 5,
            "key_fields": ["Balance__c", "RiskProfile__c"],
            "purpose": "Base account records for suitability tests",
        },
        {
            "object_name": "Suitability__c",
            "record_count": 3,
            "key_fields": ["Score__c", "Classification__c"],
            "purpose": "Suitability assessment records",
        },
    ],
    "requires_anonymisation": True,
    "anonymisation_fields": ["Name", "NationalInsurance__c", "DateOfBirth__c"],
    "vulnerable_profiles": [
        "VCI_01: Client with cognitive impairment marker",
        "VCI_02: Client in financial distress",
    ],
    "data_verdict": "PASS",
    "data_setup_notes": "Use TestDataFactory.createSuitabilityBundle() for setup.",
    "coverage_gaps": [],
}

MOCK_DATA_WARN = {
    "seed_records": [
        {
            "object_name": "FinancialAccount",
            "record_count": 2,
            "key_fields": ["Balance__c"],
            "purpose": "Basic accounts",
        },
    ],
    "requires_anonymisation": False,
    "anonymisation_fields": [],
    "vulnerable_profiles": [],
    "data_verdict": "WARN",
    "data_setup_notes": "Vulnerable Customer profiles not available — manual setup required.",
    "coverage_gaps": ["Vulnerable Customer scenario lacks data strategy"],
}

MOCK_DATA_INCOMPLETE = {
    "seed_records": [],
    "requires_anonymisation": False,
    "anonymisation_fields": [],
    "vulnerable_profiles": [],
    "data_verdict": "INCOMPLETE",
    "data_setup_notes": "Insufficient context to design data strategy.",
    "coverage_gaps": ["All scenarios lack data strategy"],
}


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score >= 75

    def test_no_data_needs_baseline_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, None, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score_with > score_without

    def test_no_gherkin_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, None,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert score_with > score_without

    def test_high_fca_missing_vulnerable_profiles_penalised(self):
        score_with, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_without, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert score_with > score_without

    def test_incomplete_verdict_penalised(self):
        score_pass, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        score_incomplete, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=0, verdict="INCOMPLETE", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert score_pass > score_incomplete

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=5, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01", "VCI_02"],
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(
            None, None, None, None,
            seed_record_count=0, verdict="INCOMPLETE", fca_class="LOW",
            vulnerable_profiles=[],
        )
        assert score >= 20

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "fca_classification_available" in signals

    def test_data_needs_from_refinement_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "data_needs_from_refinement" in signals

    def test_no_data_needs_baseline_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, None, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "no_data_needs_baseline" in signals

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "metadata_scope_available" in signals

    def test_gherkin_scenarios_available_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "gherkin_scenarios_available" in signals

    def test_no_gherkin_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, None,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "no_gherkin_scenarios" in signals

    def test_seed_records_designed_stores_count(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=3, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert signals["seed_records_designed"] == 3

    def test_no_seed_records_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=0, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=["VCI_01"],
        )
        assert "no_seed_records" in signals

    def test_regulated_story_missing_vulnerable_profiles_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=2, verdict="PASS", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert "regulated_story_missing_vulnerable_profiles" in signals

    def test_incomplete_strategy_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT3_HIGH, AGENT7_DATA, AGENT13_DATA, AGENT19_DATA,
            seed_record_count=0, verdict="INCOMPLETE", fca_class="HIGH",
            vulnerable_profiles=[],
        )
        assert "incomplete_strategy" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["7"] = {"data": AGENT7_DATA}
        state["agent_results"]["13"] = {"data": AGENT13_DATA}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.agent_id == 21
        assert result.agent_name == "Test Data Architect"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        for key in ["test_data_strategy", "requires_anonymisation",
                    "vulnerable_profiles", "data_verdict"]:
            assert key in result.data

    async def test_pass_verdict_with_full_context(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["data_verdict"] == "PASS"
        assert result.data["requires_anonymisation"] is True
        assert len(result.data["vulnerable_profiles"]) >= 1

    async def test_warn_when_vulnerable_profiles_missing(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_WARN
            result = await run(state)

        assert result.data["data_verdict"] == "WARN"

    async def test_incomplete_without_context(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_INCOMPLETE
            result = await run(state)

        assert result.data["data_verdict"] == "INCOMPLETE"
        assert result.data["seed_record_count"] == 0

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_escalated_when_no_upstream_data(self):
        # base=68, no_data_needs_baseline=-5, no_gherkin=-5, no_seed_records=-10, incomplete=-10 → 38 < 60
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_INCOMPLETE
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)


# ── Mechanism design signal tests ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestMechanismDesign:
    async def test_data_design_completeness_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert "data_design_completeness" in result.data
        completeness = result.data["data_design_completeness"]
        assert 0 <= completeness <= 100

    async def test_mechanism_signal_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert "mechanism_signal" in result.data
        signal = result.data["mechanism_signal"]
        assert "vulnerable_profile_missing" in signal
        assert "seed_records_missing" in signal
        assert "downstream_penalty_active" in signal

    async def test_downstream_penalty_active_when_incomplete(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_INCOMPLETE
            result = await run(state)

        assert result.data["mechanism_signal"]["downstream_penalty_active"] is True

    async def test_completeness_100_with_full_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_DATA}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        # MOCK_DATA_PASS has seed_records, anonymisation_fields, and vulnerable_profiles
        assert result.data["data_design_completeness"] >= 70

    async def test_prompt_always_ends_with_tool_instruction_no_gherkin(self):
        """REQ-13 Bug 1: prompt must end with tool instruction even when no Gherkin scenarios."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        # No agent_results["19"] → no Gherkin scenarios

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            await run(state)

        call_kwargs = mock_sonnet.call_args.kwargs
        user_msg = call_kwargs.get("user_message", "")
        assert "tool" in user_msg.lower() or "design" in user_msg.lower(), (
            "Prompt must include tool instruction even with no Gherkin scenarios"
        )

    async def test_required_records_key_read_from_agent7(self):
        """REQ-13 Bug 2: Agent 21 reads required_records (not data_requirements) from Agent 07."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["7"] = {"data": AGENT7_DATA}  # has required_records key

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            await run(state)

        call_kwargs = mock_sonnet.call_args.kwargs
        user_msg = call_kwargs.get("user_message", "")
        # required_records values should appear in the prompt
        assert "FinancialAccount" in user_msg or "Suitability" in user_msg, (
            "required_records content from Agent 07 must appear in the prompt"
        )

    async def test_no_seed_records_reduces_completeness(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result_pass = await run(state)

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_INCOMPLETE
            result_incomplete = await run(state)

        assert result_pass.data["data_design_completeness"] > result_incomplete.data["data_design_completeness"]


# ── REQ-05: isolation_override tests ─────────────────────────────────────────

@pytest.mark.asyncio
class TestIsolationOverrideREQ05:
    async def test_high_fca_shared_org_data_triggers_override(self):
        """REQ-05 Part 2: HIGH-FCA + shared_org_data → isolation_override=True."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["7"] = {"data": AGENT7_SHARED_ORG}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["isolation_override"] is True
        assert "isolation_override_reason" in result.data
        assert result.data["isolation_override_reason"] != ""

    async def test_high_fca_per_class_setup_no_override(self):
        """HIGH-FCA + per_class_setup (not shared_org_data) → no override."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["7"] = {"data": AGENT7_PER_CLASS}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["isolation_override"] is False

    async def test_low_fca_shared_org_data_no_override(self):
        """LOW-FCA + shared_org_data → no override (only HIGH/MEDIUM triggers it)."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_LOW}
        state["agent_results"]["7"] = {"data": AGENT7_SHARED_ORG}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["isolation_override"] is False

    async def test_isolation_override_keys_always_present(self):
        """isolation_override and isolation_override_reason always in output data."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert "isolation_override" in result.data
        assert "isolation_override_reason" in result.data

    async def test_medium_fca_shared_org_data_triggers_override(self):
        """MEDIUM FCA + shared_org_data also triggers isolation override (same rule as HIGH)."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_MEDIUM}
        state["agent_results"]["7"] = {"data": AGENT7_SHARED_ORG}

        with (
            patch("src.agents.development.agent_21_test_data_architect.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.development.agent_21_test_data_architect.call_with_tool",
                  new_callable=AsyncMock) as mock_sonnet,
        ):
            mock_story.return_value = MOCK_STORY
            mock_sonnet.return_value = MOCK_DATA_PASS
            result = await run(state)

        assert result.data["isolation_override"] is True


# ── _build_prompt unit tests ──────────────────────────────────────────────────

_SIMPLE_STORY = {"summary": "Test Story", "description": "desc"}

class TestBuildPrompt:
    def test_includes_story_id(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "HIGH", [], [], [], 3)
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "HIGH", [], [], [], 0)
        assert "HIGH" in msg

    def test_no_gherkin_shows_placeholder(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "LOW", [], [], [], 0)
        assert "(no Gherkin scenarios available)" in msg

    def test_gherkin_titles_shown(self):
        scenarios = [{"title": "Suitability fails for HIGH-risk client", "tags": [], "steps": []}]
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "HIGH", [], scenarios, [], 1)
        assert "Suitability fails for HIGH-risk client" in msg

    def test_no_data_needs_shows_not_captured(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "LOW", [], [], [], 0)
        assert "not captured" in msg

    def test_no_objects_shows_not_determined(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "LOW", [], [], [], 0)
        assert "not yet determined" in msg

    def test_ends_with_tool_name(self):
        msg = _build_prompt("FSC-2417", _SIMPLE_STORY, "HIGH", [], [], [], 0)
        assert _DATA_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── _compute_completeness unit tests ─────────────────────────────────────────

_SEED_RECORD = [{"object_name": "FinancialAccount", "record_count": 1,
                 "key_fields": ["Balance__c"], "purpose": "test"}]

class TestComputeCompleteness:
    def test_full_data_gives_100(self):
        score = _compute_completeness(_SEED_RECORD, "LOW", [], "PASS", ["Name"])
        assert score == 100

    def test_no_seed_records_reduces_score(self):
        score = _compute_completeness([], "LOW", [], "PASS", ["Name"])
        assert score == 70

    def test_regulated_missing_vc_profiles_reduces_score(self):
        score_with = _compute_completeness(_SEED_RECORD, "HIGH", ["VCI_01"], "PASS", ["Name"])
        score_without = _compute_completeness(_SEED_RECORD, "HIGH", [], "PASS", ["Name"])
        assert score_with > score_without
        assert score_without == 75

    def test_no_anon_fields_reduces_score(self):
        score = _compute_completeness(_SEED_RECORD, "LOW", [], "PASS", [])
        assert score == 85

    def test_incomplete_verdict_reduces_score(self):
        score = _compute_completeness(_SEED_RECORD, "LOW", [], "INCOMPLETE", ["Name"])
        assert score == 80


# ── _build_mechanism_signal unit tests ───────────────────────────────────────

class TestBuildMechanismSignal:
    def test_all_keys_present(self):
        signal = _build_mechanism_signal(_SEED_RECORD, "LOW", [], "PASS", ["Name"])
        assert "vulnerable_profile_missing" in signal
        assert "seed_records_missing" in signal
        assert "downstream_penalty_active" in signal

    def test_seed_records_missing_true_when_empty(self):
        signal = _build_mechanism_signal([], "LOW", [], "PASS", ["Name"])
        assert signal["seed_records_missing"] is True

    def test_vulnerable_profile_missing_true_for_high_fca_no_profiles(self):
        signal = _build_mechanism_signal(_SEED_RECORD, "HIGH", [], "PASS", ["Name"])
        assert signal["vulnerable_profile_missing"] is True

    def test_downstream_penalty_active_false_when_completeness_ge_70(self):
        # completeness=100 → downstream_penalty_active=False
        signal = _build_mechanism_signal(_SEED_RECORD, "LOW", [], "PASS", ["Name"])
        assert signal["downstream_penalty_active"] is False


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_seven_required_fields(self):
        assert set(_DATA_TOOL_SCHEMA["required"]) == {
            "seed_records", "requires_anonymisation", "anonymisation_fields",
            "vulnerable_profiles", "data_verdict", "data_setup_notes", "coverage_gaps",
        }

    def test_data_verdict_enum_has_three_values(self):
        assert _DATA_TOOL_SCHEMA["properties"]["data_verdict"]["enum"] == [
            "PASS", "WARN", "INCOMPLETE"
        ]

    def test_seed_records_is_array_type(self):
        assert _DATA_TOOL_SCHEMA["properties"]["seed_records"]["type"] == "array"

    def test_seed_record_item_has_four_required_fields(self):
        item_required = set(_DATA_TOOL_SCHEMA["properties"]["seed_records"]["items"]["required"])
        assert item_required == {"object_name", "record_count", "key_fields", "purpose"}
