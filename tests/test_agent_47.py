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
    "acceptance_criteria": [
        "AC1: Adviser can view client suitability score",
        "AC2: Score refreshes on portfolio change",
        "AC3: FCA suitability check runs on submit",
        "AC4: Vulnerable customer flag shown",
    ],
}

AGENT5_EMPTY = {"ac_count": 0, "acceptance_criteria": []}

AGENT19_FULL = {
    "scenario_count": 5,
    "scenarios": "Scenario: Happy path suitability check\nGiven ...\nWhen ...\nThen ...",
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

    async def test_release_title_populated(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["5"] = {"data": AGENT5_FULL}

        with patch("src.agents.release.agent_47_release_notes_writer.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NOTES_COMPLETE
            result = await run(state)

        assert result.data["release_title"] == MOCK_NOTES_COMPLETE["release_title"]
