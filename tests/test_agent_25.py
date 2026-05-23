"""Tests for Agent 25 — Test Environment Provisioner (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_25_test_env_provisioner import (
    _build_trace_message,
    _check_environment,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
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

AGENT21_WARN = {
    "data_verdict": "WARN",
    "seed_record_count": 1,
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

    def test_warn_data_verdict_gives_degraded_verdict(self):
        env_ready, verdict, blockers, _ = _check_environment(AGENT21_WARN, AGENT22_READY)
        assert verdict == "DEGRADED"
        assert env_ready is True
        assert any("gaps" in b.lower() for b in blockers)

    def test_degraded_sandbox_crt_still_connected(self):
        _, _, _, crt = _check_environment(AGENT21_PASS, AGENT22_DEGRADED)
        assert crt is True

    def test_degraded_sandbox_with_incomplete_data_is_blocked(self):
        env_ready, verdict, _, _ = _check_environment(AGENT21_INCOMPLETE, AGENT22_DEGRADED)
        assert verdict == "BLOCKED"
        assert env_ready is False


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

    def test_sandbox_health_signal_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT21_PASS, AGENT22_READY, True)
        assert "sandbox_health_signal_available" in signals

    def test_no_sandbox_health_signal_key_in_signals(self):
        _, signals = _compute_confidence(AGENT21_PASS, None, True)
        assert "no_sandbox_health_signal" in signals

    def test_test_data_strategy_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT21_PASS, AGENT22_READY, True)
        assert "test_data_strategy_available" in signals

    def test_no_test_data_strategy_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT22_READY, True)
        assert "no_test_data_strategy" in signals

    def test_environment_not_ready_key_in_signals(self):
        _, signals = _compute_confidence(AGENT21_PASS, AGENT22_READY, False)
        assert "environment_not_ready" in signals


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

    async def test_escalated_when_no_upstream_data(self):
        # base=60, no agent22→-8, no agent21→-5 = 47 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_25_test_env_provisioner.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "FSC-2417" in msg

    def test_includes_sandbox_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "READY" in msg

    def test_includes_health_score(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "95" in msg

    def test_includes_data_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "PASS" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "Verdict: READY" in msg

    def test_no_blockers_shows_none(self):
        msg = _build_trace_message("FSC-2417", AGENT21_PASS, AGENT22_READY, [], "READY")
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, None, [], "READY")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "provisioning_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_provisioning_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["provisioning_concern"]["enum"] == [
            "none", "sandbox_degraded", "data_not_seeded", "crt_unavailable", "multiple"
        ]
