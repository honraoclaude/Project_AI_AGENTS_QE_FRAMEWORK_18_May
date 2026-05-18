"""
Tests for Agent 11 — Story-to-Branch Tracer (Augmented Script).

The deterministic _analyse_branch() function is the primary test target.
Integration tests mock the Copado call and the Haiku narrative call.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_11_branch_tracer import (
    _analyse_branch,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

BRANCH_VALID = {
    "branch_name": "feature/FSC-2417-suitability-assessment",
    "commit_sha": "a1b2c3d4e5f6",
    "created_date": "2026-05-15T09:00:00+00:00",
    "last_commit_date": "2026-05-17T14:00:00+00:00",
    "author_email": "dev@firm.com",
}

BRANCH_BAD_NAME = {
    "branch_name": "dev/suitability-changes",  # no FSC prefix
    "commit_sha": "a1b2c3d4e5f6",
    "created_date": "2026-05-10T09:00:00+00:00",
    "last_commit_date": "2026-05-17T14:00:00+00:00",
    "author_email": "dev@firm.com",
}

BRANCH_NO_STORY_ID = {
    "branch_name": "feature/FSC-9999-other-story",  # wrong story ID
    "commit_sha": "deadbeef",
    "created_date": "2026-05-10T09:00:00+00:00",
    "last_commit_date": "2026-05-17T14:00:00+00:00",
    "author_email": "dev@firm.com",
}

BRANCH_NOT_FOUND = {
    "branch_name": "",
    "commit_sha": "",
    "created_date": "",
    "last_commit_date": "",
    "author_email": "",
}

BRANCH_STALE = {
    "branch_name": "feature/FSC-2417-old-work",
    "commit_sha": "oldsha",
    "created_date": "2026-01-01T09:00:00+00:00",  # very old
    "last_commit_date": "2026-01-05T14:00:00+00:00",
    "author_email": "dev@firm.com",
}

MOCK_TRACE_PASS = {
    "narrative": "Branch feature/FSC-2417-suitability-assessment is correctly named and traces to FSC-2417.",
    "traceability_risk": "low",
}

MOCK_TRACE_FAIL = {
    "narrative": "No branch was found for FSC-2417. The developer must create a branch before proceeding.",
    "traceability_risk": "high",
}


# ── Deterministic analysis tests ──────────────────────────────────────────────

class TestBranchAnalysis:
    def test_valid_branch_all_checks_pass(self):
        found, naming, story_in, stale, age = _analyse_branch(BRANCH_VALID, "FSC-2417")
        assert found is True
        assert naming is True
        assert story_in is True

    def test_branch_not_found_all_false(self):
        found, naming, story_in, stale, age = _analyse_branch(BRANCH_NOT_FOUND, "FSC-2417")
        assert found is False
        assert naming is False
        assert story_in is False

    def test_bad_naming_convention_detected(self):
        _, naming, _, _, _ = _analyse_branch(BRANCH_BAD_NAME, "FSC-2417")
        assert naming is False

    def test_wrong_story_id_in_branch(self):
        found, naming, story_in, _, _ = _analyse_branch(BRANCH_NO_STORY_ID, "FSC-2417")
        assert found is True
        assert naming is True
        assert story_in is False  # FSC-9999, not FSC-2417

    def test_stale_branch_detected(self):
        _, _, _, stale, age = _analyse_branch(BRANCH_STALE, "FSC-2417")
        assert stale is True
        assert age > 14

    def test_recent_branch_not_stale(self):
        _, _, _, stale, _ = _analyse_branch(BRANCH_VALID, "FSC-2417")
        assert stale is False

    def test_bugfix_prefix_accepted(self):
        bugfix_branch = {**BRANCH_VALID, "branch_name": "bugfix/FSC-2417-fix-suitability"}
        _, naming, _, _, _ = _analyse_branch(bugfix_branch, "FSC-2417")
        assert naming is True

    def test_hotfix_prefix_accepted(self):
        hotfix_branch = {**BRANCH_VALID, "branch_name": "hotfix/FSC-2417-urgent-fix"}
        _, naming, _, _, _ = _analyse_branch(hotfix_branch, "FSC-2417")
        assert naming is True

    def test_story_id_check_is_case_insensitive(self):
        lower_branch = {**BRANCH_VALID, "branch_name": "feature/fsc-2417-suitability"}
        _, _, story_in, _, _ = _analyse_branch(lower_branch, "FSC-2417")
        assert story_in is True

    def test_invalid_date_handled_gracefully(self):
        bad_date = {**BRANCH_VALID, "created_date": "not-a-date"}
        found, naming, story_in, stale, age = _analyse_branch(bad_date, "FSC-2417")
        assert age == 0
        assert found is True


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_valid_branch_scores_high(self):
        score, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert score >= 75

    def test_branch_not_found_heavily_penalised(self):
        score_found, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        score_missing, _ = _compute_confidence(BRANCH_NOT_FOUND, False, False, False, False)
        assert score_found > score_missing

    def test_invalid_naming_reduces_confidence(self):
        score_valid, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        score_invalid, _ = _compute_confidence(BRANCH_BAD_NAME, True, False, False, False)
        assert score_valid > score_invalid

    def test_story_id_missing_penalised(self):
        score_present, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        score_missing, _ = _compute_confidence(BRANCH_NO_STORY_ID, True, True, False, False)
        assert score_present > score_missing

    def test_stale_branch_reduces_confidence(self):
        score_fresh, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        score_stale, _ = _compute_confidence(BRANCH_STALE, True, True, True, True)
        assert score_fresh > score_stale

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(BRANCH_NOT_FOUND, False, False, False, False)
        assert score >= 20

    def test_branch_not_found_signal_recorded(self):
        _, signals = _compute_confidence(BRANCH_NOT_FOUND, False, False, False, False)
        assert "branch_not_found" in signals


# ── Integration tests — full agent run ───────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_11_branch_tracer.get_branch_for_story",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_11_branch_tracer.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = BRANCH_VALID
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 11
        assert result.agent_name == "Story-to-Branch Tracer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_11_branch_tracer.get_branch_for_story",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_11_branch_tracer.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = BRANCH_VALID
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["branch_name", "commit_sha", "branch_found",
                    "branch_naming_valid", "story_id_in_branch"]:
            assert key in result.data

    async def test_valid_branch_produces_correct_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_11_branch_tracer.get_branch_for_story",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_11_branch_tracer.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = BRANCH_VALID
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["branch_found"] is True
        assert result.data["branch_naming_valid"] is True
        assert result.data["story_id_in_branch"] is True
        assert result.data["commit_sha"] == "a1b2c3d4e5f6"

    async def test_no_branch_data_degrades_gracefully(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_11_branch_tracer.get_branch_for_story",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_11_branch_tracer.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = BRANCH_NOT_FOUND
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.agent_id == 11
        assert result.data["branch_found"] is False
        assert result.data["branch_name"] == ""

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_11_branch_tracer.get_branch_for_story",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_11_branch_tracer.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = BRANCH_VALID
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
