"""
Tests for Agent 11 — Story-to-Branch Tracer (Augmented Script).

The deterministic _analyse_branch() function is the primary test target.
Integration tests mock the Copado call and the Haiku narrative call.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_11_branch_tracer import (
    _analyse_branch,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
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

    def test_empty_created_date_gives_zero_age(self):
        _, _, _, _, age = _analyse_branch(BRANCH_NOT_FOUND, "FSC-2417")
        assert age == 0


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

    def test_branch_found_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert "branch_found" in signals

    def test_naming_convention_valid_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert "naming_convention_valid" in signals

    def test_naming_convention_invalid_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_BAD_NAME, True, False, False, False)
        assert "naming_convention_invalid" in signals

    def test_story_id_in_branch_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert "story_id_in_branch" in signals

    def test_story_id_missing_from_branch_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_NO_STORY_ID, True, True, False, False)
        assert "story_id_missing_from_branch" in signals

    def test_commit_sha_present_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_VALID, True, True, True, False)
        assert "commit_sha_present" in signals

    def test_branch_stale_key_in_signals(self):
        _, signals = _compute_confidence(BRANCH_STALE, True, True, True, True)
        assert "branch_stale" in signals


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

    async def test_branch_not_found_causes_escalation(self):
        # branch_not_found (-20): base=65-20=45 < 60 → escalated
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

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
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

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
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

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
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

        assert isinstance(result.data["narrative"], str)


# ── Trace message tests ───────────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "FSC-2417" in msg

    def test_includes_branch_name_when_present(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "feature/FSC-2417-suitability-assessment" in msg

    def test_shows_none_found_when_no_branch(self):
        msg = _build_trace_message("FSC-2417", BRANCH_NOT_FOUND, False, False, False, 0)
        assert "(none found)" in msg

    def test_includes_commit_sha_when_present(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "a1b2c3d4e5f6" in msg

    def test_includes_branch_found_status(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "Branch found: True" in msg

    def test_includes_naming_convention_hint(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "feature|bugfix|hotfix" in msg

    def test_includes_branch_age(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert "Branch age (days):" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", BRANCH_VALID, True, True, True, 8)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "traceability_risk"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_traceability_risk_enum_has_three_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["traceability_risk"]["enum"] == ["low", "medium", "high"]
