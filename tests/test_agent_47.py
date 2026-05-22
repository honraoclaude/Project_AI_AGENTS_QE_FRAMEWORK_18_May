"""Tests for Agent 47 — Release Notes Writer (True AI Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_47_release_notes_writer import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT5_FULL = {
    "ac_count": 4,
    "ac_clauses": [
        {"description": "Adviser can view client suitability score", "scenario_type": "happy_path"},
        {"description": "Score refreshes on portfolio change", "scenario_type": "happy_path"},
        {"description": "FCA suitability check runs on submit", "scenario_type": "regulatory"},
        {"description": "Vulnerable customer flag shown", "scenario_type": "regulatory"},
    ],
}

AGENT5_EMPTY = {"ac_count": 0, "ac_clauses": []}

AGENT19_FULL = {
    "scenario_count": 5,
    "gherkin_scenarios": [
        {"title": "Happy path suitability check", "tags": ["@smoke"]},
        {"title": "FCA suitability failure for HIGH-risk client", "tags": ["@fca"]},
    ],
    "gherkin_verdict": "PASS",
}

AGENT23_PASS = {
    "development_verdict": "PASS",
    "narrative": "All Apex checks passed. Coverage 87%. No security violations.",
}

AGENT33_PASS = {
    "overall_coverage_pct": 92.0,
    "coverage_verdict": "PASS",
}

MOCK_NOTES_COMPLETE = {
    "release_title": "FSC-2417: Suitability Assessment Refresh",
    "release_notes": "This release updates the suitability assessment flow to refresh on portfolio changes...",
    "regulatory_notes": "FCA COBS 9 suitability checks validated. Consumer Duty PS22/9 covered by AC4.",
    "notes_verdict": "COMPLETE",
    "narrative": "Release notes generated covering 4 ACs, 5 Gherkin scenarios, and regulatory compliance summary.",
}

MOCK_NOTES_PARTIAL = {
    "release_title": "FSC-2417: Feature Release",
    "release_notes": "Limited source data available. Change introduces updates to FSC workflow.",
    "regulatory_notes": "No FCA-specific data available — Compliance Officer must review manually.",
    "notes_verdict": "PARTIAL",
    "narrative": "Release notes generated with partial data — some sections are incomplete.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_two_or_more_sources_scores_well(self):
        score, _ = _compute_confidence(AGENT5_FULL, AGENT19_FULL, AGENT23_PASS, "COMPLETE")
        assert score >= 70

    def test_no_source_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT5_FULL, AGENT19_FULL, AGENT23_PASS, "COMPLETE")
        score_without, _ = _compute_confidence(None, None, None, "PARTIAL")
        assert score_with > score_without

    def test_complete_verdict_boosts_confidence(self):
        score_complete, _ = _compute_confidence(AGENT5_FULL, AGENT19_FULL, AGENT23_PASS, "COMPLETE")
        score_failed, _ = _compute_confidence(AGENT5_FULL, AGENT19_FULL, AGENT23_PASS, "FAILED")
        assert score_complete > score_failed

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT5_FULL, AGENT19_FULL, AGENT23_PASS, "COMPLETE")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, "FAILED")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"]  = {"data": AGENT5_FULL}
        state["agent_results"]["19"] = {"data": AGENT19_FULL}
        state["agent_results"]["23"] = {"data": AGENT23_PASS}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        assert result.agent_id == 47
        assert result.agent_name == "Release Notes Writer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        for key in ["release_title", "release_notes", "regulatory_notes", "notes_verdict"]:
            assert key in result.data

    async def test_complete_when_all_sources_available(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"]  = {"data": AGENT5_FULL}
        state["agent_results"]["19"] = {"data": AGENT19_FULL}
        state["agent_results"]["23"] = {"data": AGENT23_PASS}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        assert result.data["notes_verdict"] == "COMPLETE"
        assert len(result.data["release_title"]) > 0
        assert len(result.data["release_notes"]) > 0
        assert len(result.data["regulatory_notes"]) > 0

    async def test_partial_when_no_source_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_PARTIAL
            result = await run(state)

        assert result.data["notes_verdict"] == "PARTIAL"

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_ac_clauses_key_content_reaches_prompt(self):
        """REQ-30 Bug 1: Agent 47 reads ac_clauses (not acceptance_criteria) from Agent 05."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"] = {"data": AGENT5_FULL}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            await run(state)

        user_msg = mock_sonnet.call_args.kwargs.get("user_message", "")
        assert "Adviser can view client suitability score" in user_msg, (
            "AC descriptions from ac_clauses must appear in the prompt — not '(not available)'"
        )

    async def test_gherkin_scenarios_key_content_reaches_prompt(self):
        """REQ-30 Bug 2: Agent 47 reads gherkin_scenarios (not scenarios) from Agent 19."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["19"] = {"data": AGENT19_FULL}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            await run(state)

        user_msg = mock_sonnet.call_args.kwargs.get("user_message", "")
        assert "Happy path suitability check" in user_msg or "FCA suitability" in user_msg, (
            "Gherkin scenario titles from gherkin_scenarios must appear in the prompt"
        )

    async def test_old_wrong_keys_produce_not_available(self):
        """REQ-30: Old keys acceptance_criteria/scenarios must not populate the prompt."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"] = {"data": {
            "acceptance_criteria": ["AC1: old key"],  # wrong key
            "ac_count": 1,
        }}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            await run(state)

        user_msg = mock_sonnet.call_args.kwargs.get("user_message", "")
        assert "AC1: old key" not in user_msg, (
            "Old acceptance_criteria key should not be read"
        )

    async def test_release_title_populated(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"] = {"data": AGENT5_FULL}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        assert result.data["release_title"] == MOCK_NOTES_COMPLETE["release_title"]


# ── REQ-30: new tests ─────────────────────────────────────────────────────────

from src.agents.release.agent_47_release_notes_writer import _build_notes_message

AGENT5_WITH_CLAUSES = {
    "ac_count": 2,
    "ac_clauses": [
        {"description": "Adviser can view suitability score", "scenario_type": "happy_path"},
        {"description": "FCA suitability check runs on submit", "scenario_type": "regulatory"},
    ],
}

AGENT19_WITH_SCENARIOS = {
    "scenario_count": 2,
    "gherkin_scenarios": [
        {"title": "Happy path suitability", "tags": ["@smoke"]},
        {"title": "FCA regulatory check", "tags": ["@fca"]},
    ],
}

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}


class TestREQ30AcClausesKeyFix:
    def test_ac_clause_descriptions_appear_in_prompt(self):
        msg = _build_notes_message("FSC-001", AGENT3_HIGH, AGENT5_WITH_CLAUSES, None, None, None)
        assert "Adviser can view suitability score" in msg

    def test_no_ac_clauses_gives_not_available(self):
        msg = _build_notes_message("FSC-001", AGENT3_HIGH, {"ac_clauses": []}, None, None, None)
        assert "(not available)" in msg


class TestREQ30GherkinScenariosKeyFix:
    def test_gherkin_scenario_titles_appear_in_prompt(self):
        msg = _build_notes_message("FSC-001", AGENT3_HIGH, None, AGENT19_WITH_SCENARIOS, None, None)
        assert "Happy path suitability" in msg

    def test_no_gherkin_scenarios_gives_not_available(self):
        msg = _build_notes_message("FSC-001", AGENT3_HIGH, None, {"gherkin_scenarios": []}, None, None)
        assert "(not available)" in msg


class TestREQ30FcaClassificationInPrompt:
    def test_high_fca_classification_in_prompt(self):
        msg = _build_notes_message("FSC-001", AGENT3_HIGH, None, None, None, None)
        assert "HIGH" in msg

    def test_low_fca_classification_in_prompt(self):
        msg = _build_notes_message("FSC-001", AGENT3_LOW, None, None, None, None)
        assert "LOW" in msg

    def test_no_agent3_defaults_gracefully(self):
        msg = _build_notes_message("FSC-001", None, None, None, None, None)
        assert "FSC-001" in msg
