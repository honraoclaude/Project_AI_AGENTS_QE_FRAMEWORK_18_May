"""Tests for Agent 42 — Dry-Run Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_42_dry_run import (
    _build_trace_message,
    _compute_confidence,
    _simulate_dry_run,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT25_READY = {
    "env_ready": True,
    "env_verdict": "READY",
    "crt_connected": True,
}

AGENT25_BLOCKED = {
    "env_ready": False,
    "env_verdict": "BLOCKED",
    "crt_connected": False,
}

AGENT41_PASS = {
    "integrity_valid": True,
    "integrity_verdict": "PASS",
    "destructive_changes_present": False,
}

AGENT41_WARN = {
    "integrity_valid": True,
    "integrity_verdict": "WARN",
    "destructive_changes_present": True,
}

AGENT41_FAIL = {
    "integrity_valid": False,
    "integrity_verdict": "FAIL",
    "destructive_changes_present": False,
}

MOCK_TRACE_PASS = {
    "narrative": "Dry-run deployment successful. All components validated without errors. Smoke tests can proceed.",
    "dry_run_concern": "none",
}

MOCK_TRACE_FAIL_ENV = {
    "narrative": "Dry-run FAILED — staging environment is not ready. Environment must be provisioned before deployment can proceed.",
    "dry_run_concern": "env_not_ready",
}

MOCK_TRACE_FAIL_INTEGRITY = {
    "narrative": "Dry-run FAILED — change set integrity check failed. Missing dependencies must be resolved first.",
    "dry_run_concern": "change_set_invalid",
}

MOCK_TRACE_SKIPPED = {
    "narrative": "Dry-run skipped — no environment or change set data available.",
    "dry_run_concern": "none",
}


# ── Deterministic dry-run simulation tests ────────────────────────────────────

class TestSimulateDryRun:
    def test_ready_env_and_valid_change_set_gives_pass(self):
        success, errors, verdict = _simulate_dry_run(AGENT25_READY, AGENT41_PASS)
        assert success is True
        assert verdict == "PASS"
        assert len(errors) == 0

    def test_blocked_env_gives_fail(self):
        success, errors, verdict = _simulate_dry_run(AGENT25_BLOCKED, AGENT41_PASS)
        assert success is False
        assert verdict == "FAIL"
        assert any("environment" in e.lower() for e in errors)

    def test_integrity_fail_gives_dry_run_fail(self):
        success, errors, verdict = _simulate_dry_run(AGENT25_READY, AGENT41_FAIL)
        assert success is False
        assert verdict == "FAIL"
        assert any("integrity" in e.lower() for e in errors)

    def test_warn_integrity_does_not_block_dry_run(self):
        # WARN integrity = valid=True, dry-run should pass
        success, _, verdict = _simulate_dry_run(AGENT25_READY, AGENT41_WARN)
        assert success is True
        assert verdict == "PASS"

    def test_no_upstream_data_gives_skipped(self):
        success, _, verdict = _simulate_dry_run(None, None)
        assert verdict == "SKIPPED"
        assert success is False

    def test_both_env_and_integrity_fail_both_errors_collected(self):
        _, errors, _ = _simulate_dry_run(AGENT25_BLOCKED, AGENT41_FAIL)
        assert len(errors) >= 2

    def test_env_verdict_blocked_alone_triggers_fail(self):
        # env_ready=True but env_verdict="BLOCKED" — string check fires independently
        agent25_blocked_str = {"env_ready": True, "env_verdict": "BLOCKED", "crt_connected": True}
        success, errors, verdict = _simulate_dry_run(agent25_blocked_str, AGENT41_PASS)
        assert success is False
        assert verdict == "FAIL"
        assert any("environment" in e.lower() for e in errors)

    def test_integrity_verdict_fail_alone_triggers_fail(self):
        # integrity_valid=True but integrity_verdict="FAIL" — string check fires independently
        agent41_fail_str = {"integrity_valid": True, "integrity_verdict": "FAIL", "destructive_changes_present": False}
        success, errors, verdict = _simulate_dry_run(AGENT25_READY, agent41_fail_str)
        assert success is False
        assert verdict == "FAIL"
        assert any("integrity" in e.lower() for e in errors)

    def test_one_source_absent_not_skipped(self):
        # only agent25 present → not SKIPPED (requires BOTH to be None)
        success, _, verdict = _simulate_dry_run(AGENT25_READY, None)
        assert verdict != "SKIPPED"


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_both_sources_available_scores_well(self):
        score, _ = _compute_confidence(AGENT25_READY, AGENT41_PASS, True)
        assert score >= 65

    def test_no_env_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT25_READY, AGENT41_PASS, True)
        score_without, _ = _compute_confidence(None, AGENT41_PASS, True)
        assert score_with > score_without

    def test_dry_run_failure_reduces_confidence(self):
        score_pass, _ = _compute_confidence(AGENT25_READY, AGENT41_PASS, True)
        score_fail, _ = _compute_confidence(AGENT25_BLOCKED, AGENT41_FAIL, False)
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT25_READY, AGENT41_PASS, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, False)
        assert score >= 20

    def test_env_data_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, None, True)
        assert "env_data_available" in signals

    def test_no_env_data_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT41_PASS, True)
        assert "no_env_data" in signals

    def test_integrity_data_available_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT41_PASS, True)
        assert "integrity_data_available" in signals

    def test_no_integrity_data_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, None, True)
        assert "no_integrity_data" in signals

    def test_dry_run_succeeded_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT41_PASS, True)
        assert "dry_run_succeeded" in signals

    def test_dry_run_failed_key_in_signals(self):
        _, signals = _compute_confidence(None, None, False)
        assert "dry_run_failed" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 42
        assert result.agent_name == "Dry-Run Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["dry_run_success", "dry_run_errors", "dry_run_verdict"]:
            assert key in result.data

    async def test_pass_when_env_ready_and_integrity_valid(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["dry_run_verdict"] == "PASS"
        assert result.data["dry_run_success"] is True

    async def test_fail_when_env_blocked(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_BLOCKED}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL_ENV
            result = await run(state)

        assert result.data["dry_run_verdict"] == "FAIL"
        assert result.data["dry_run_success"] is False

    async def test_skipped_with_no_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["dry_run_verdict"] == "SKIPPED"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=58, no_env_data→-8=50, no_integrity_data→-5=45, dry_run_failed→-8=37 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_42_dry_run.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_success_flag(self):
        msg = _build_trace_message("FSC-001", True, [], "PASS")
        assert "True" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-001", False, [], "FAIL")
        assert "FAIL" in msg

    def test_errors_sentinel_when_no_errors(self):
        msg = _build_trace_message("FSC-001", True, [], "PASS")
        assert "['none']" in msg

    def test_includes_error_text_when_errors_present(self):
        errors = ["Staging environment not ready: BLOCKED"]
        msg = _build_trace_message("FSC-001", False, errors, "FAIL")
        assert "BLOCKED" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-001", True, [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "dry_run_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_dry_run_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["dry_run_concern"]["enum"] == [
            "none", "deployment_error", "env_not_ready",
            "change_set_invalid", "validation_timeout",
        ]
