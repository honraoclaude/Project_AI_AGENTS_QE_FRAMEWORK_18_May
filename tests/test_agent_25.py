"""Tests for Agent 25 — Test Environment Provisioner (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_25_test_env_provisioner import (
    _check_environment,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT21_PASS = {
    "data_verdict": "PASS",
    "seed_record_count": 3,
    "vulnerable_profiles": ["VCI_01"],
}

AGENT21_INCOMPLETE = {
    "data_verdict": "INCOMPLETE",
    "seed_record_count": 0,
    "vulnerable_profiles": [],
}

AGENT22_READY = {
    "sandbox_ready": True,
    "sandbox_verdict": "READY",
    "sandbox_health_score": 95,
    "sandbox_blockers": [],
}

AGENT22_BLOCKED = {
    "sandbox_ready": False,
    "sandbox_verdict": "BLOCKED",
    "sandbox_health_score": 30,
    "sandbox_blockers": ["No story branch found"],
}

AGENT22_DEGRADED = {
    "sandbox_ready": True,
    "sandbox_verdict": "DEGRADED",
    "sandbox_health_score": 55,
    "sandbox_blockers": [],
}

MOCK_TRACE_READY = {
    "narrative": "Environment ready. Sandbox healthy, test data seeded, CRT connected.",
    "provisioning_concern": "none",
}

MOCK_TRACE_BLOCKED = {
    "narrative": "Environment BLOCKED. Sandbox not ready and test data strategy incomplete.",
    "provisioning_concern": "multiple",
}


# ── Deterministic environment check tests ─────────────────────────────────────

class TestEnvironmentCheck:
    def test_all_ready_gives_ready_verdict(self):
        env_ready, verdict, blockers, crt = _check_environment(AGENT21_PASS, AGENT22_READY)
        assert verdict == "READY"
        assert env_ready is True
        assert crt is True
        assert len(blockers) == 0

    def test_blocked_sandbox_gives_blocked_verdict(self):
        env_ready, verdict, blockers, crt = _check_environment(AGENT21_PASS, AGENT22_BLOCKED)
        assert env_ready is False
        assert verdict == "BLOCKED"
        assert crt is False
        assert any("not ready" in b for b in blockers)

    def test_degraded_sandbox_gives_degraded_verdict(self):
        env_ready, verdict, blockers, crt = _check_environment(AGENT21_PASS, AGENT22_DEGRADED)
        assert verdict == "DEGRADED"
        assert env_ready is True  # degraded but not blocked

    def test_incomplete_test_data_adds_blocker(self):
        _, _, blockers, _ = _check_environment(AGENT21_INCOMPLETE, AGENT22_READY)
        assert any("INCOMPLETE" in b for b in blockers)

    def test_incomplete_data_with_blocked_sandbox_is_blocked(self):
        env_ready, verdict, _, _ = _check_environment(AGENT21_INCOMPLETE, AGENT22_BLOCKED)
        assert env_ready is False
        assert verdict == "BLOCKED"

    def test_no_upstream_data_defaults_to_ready(self):
        env_ready, verdict, blockers, crt = _check_environment(None, None)
        assert isinstance(env_ready, bool)
        assert verdict in ("READY", "DEGRADED", "BLOCKED")

    def test_crt_connected_when_sandbox_ready(self):
        _, _, _, crt = _check_environment(AGENT21_PASS, AGENT22_READY)
        assert crt is True

    def test_crt_not_connected_when_sandbox_blocked(self):
        _, _, _, crt = _check_environment(AGENT21_PASS, AGENT22_BLOCKED)
        assert crt is False


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(AGENT21_PASS, AGENT22_READY, True)
        assert score >= 65

    def test_no_sandbox_signal_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT21_PASS, AGENT22_READY, True)
        score_without, _ = _compute_confidence(AGENT21_PASS, None, True)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT21_PASS, AGENT22_READY, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["21"] = {"data": AGENT21_PASS}
        state["agent_results"]["22"] = {"data": AGENT22_READY}

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.agent_id == 25
        assert result.agent_name == "Test Environment Provisioner"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        for key in ["env_ready", "env_verdict", "env_blockers", "crt_connected"]:
            assert key in result.data

    async def test_ready_when_all_signals_healthy(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["21"] = {"data": AGENT21_PASS}
        state["agent_results"]["22"] = {"data": AGENT22_READY}

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.data["env_ready"] is True
        assert result.data["env_verdict"] == "READY"
        assert result.data["crt_connected"] is True

    async def test_blocked_when_sandbox_not_ready(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["22"] = {"data": AGENT22_BLOCKED}

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.data["env_ready"] is False
        assert result.data["env_verdict"] == "BLOCKED"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
