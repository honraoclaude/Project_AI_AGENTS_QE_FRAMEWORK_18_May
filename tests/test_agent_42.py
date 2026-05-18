"""Tests for Agent 42 — Dry-Run Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_42_dry_run import (
    _compute_confidence,
    _simulate_dry_run,
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
