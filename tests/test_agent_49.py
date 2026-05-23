"""Tests for Agent 49 — Post-Release Monitor (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_49_post_release_monitor import (
    _build_trace_message,
    _check_monitoring,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT46_HEALTHY = {"production_healthy": True, "prod_verdict": "HEALTHY"}
AGENT46_SKIPPED = {"production_healthy": False, "prod_verdict": "SKIPPED"}
AGENT46_UNHEALTHY = {"production_healthy": False, "prod_verdict": "DEGRADED"}

MOCK_TRACE_MONITORING = {
    "narrative": "Post-release monitoring active. All 5 health checks passing. No alerts triggered; system nominal.",
    "monitor_concern": "none",
}
MOCK_TRACE_SKIPPED = {
    "narrative": "Post-release monitoring skipped — production validation was not approved or deployment did not proceed.",
    "monitor_concern": "deployment_not_done",
}


# ── Deterministic monitoring check tests ─────────────────────────────────────

class TestCheckMonitoring:
    def test_no_data_gives_skipped(self):
        active, alerts, health, verdict = _check_monitoring(None)
        assert active is False
        assert verdict == "SKIPPED"
        assert health == "UNKNOWN"
        assert alerts == []

    def test_healthy_production_activates_monitoring(self):
        active, alerts, health, verdict = _check_monitoring(AGENT46_HEALTHY)
        assert active is True
        assert verdict == "MONITORING"
        assert health == "NOMINAL"

    def test_skipped_production_gives_skipped(self):
        active, alerts, health, verdict = _check_monitoring(AGENT46_SKIPPED)
        assert active is False
        assert verdict == "SKIPPED"

    def test_unhealthy_production_gives_skipped(self):
        active, alerts, health, verdict = _check_monitoring(AGENT46_UNHEALTHY)
        assert active is False
        assert verdict == "SKIPPED"

    def test_prod_verdict_skipped_gives_skipped_regardless_of_flag(self):
        # prod_verdict="SKIPPED" overrides production_healthy=True
        data = {"production_healthy": True, "prod_verdict": "SKIPPED"}
        active, _, _, verdict = _check_monitoring(data)
        assert active is False
        assert verdict == "SKIPPED"

    def test_nominal_health_when_monitoring_active(self):
        _, alerts, health, _ = _check_monitoring(AGENT46_HEALTHY)
        assert health == "NOMINAL"
        assert alerts == []

    def test_no_alerts_in_stub(self):
        _, alerts, _, _ = _check_monitoring(AGENT46_HEALTHY)
        assert alerts == []

    def test_returns_tuple_of_four(self):
        result = _check_monitoring(AGENT46_HEALTHY)
        assert len(result) == 4
        active, alerts, health, verdict = result
        assert isinstance(active, bool)
        assert isinstance(alerts, list)
        assert isinstance(health, str)
        assert isinstance(verdict, str)

    def test_missing_prod_healthy_flag_gives_skipped(self):
        data = {"prod_verdict": "HEALTHY"}  # production_healthy key absent
        active, _, _, verdict = _check_monitoring(data)
        assert active is False
        assert verdict == "SKIPPED"


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_production_data_and_active_scores_well(self):
        score, _ = _compute_confidence(AGENT46_HEALTHY, True)
        assert score >= 60

    def test_no_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT46_HEALTHY, True)
        score_without, _ = _compute_confidence(None, False)
        assert score_with > score_without

    def test_monitoring_active_adds_confidence(self):
        score_active, _ = _compute_confidence(AGENT46_HEALTHY, True)
        score_inactive, _ = _compute_confidence(AGENT46_HEALTHY, False)
        assert score_active > score_inactive

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT46_HEALTHY, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, False)
        assert score >= 20

    def test_returns_signals_dict(self):
        _, signals = _compute_confidence(AGENT46_HEALTHY, True)
        assert isinstance(signals, dict)

    def test_production_data_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT46_HEALTHY, True)
        assert "production_data_available" in signals

    def test_no_production_data_key_in_signals(self):
        _, signals = _compute_confidence(None, False)
        assert "no_production_data" in signals

    def test_monitoring_active_key_in_signals(self):
        _, signals = _compute_confidence(AGENT46_HEALTHY, True)
        assert "monitoring_active" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_HEALTHY}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert result.agent_id == 49
        assert result.agent_name == "Post-Release Monitor"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["monitoring_active", "alerts_triggered", "health_status", "monitor_verdict"]:
            assert key in result.data

    async def test_monitoring_when_production_healthy(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_HEALTHY}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert result.data["monitor_verdict"] == "MONITORING"
        assert result.data["monitoring_active"] is True
        assert result.data["health_status"] == "NOMINAL"

    async def test_skipped_when_no_production_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["monitor_verdict"] == "SKIPPED"
        assert result.data["monitoring_active"] is False

    async def test_skipped_when_production_skipped(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_SKIPPED}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["monitor_verdict"] == "SKIPPED"
        assert result.data["monitoring_active"] is False

    async def test_no_alerts_in_nominal_state(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_HEALTHY}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert result.data["alerts_triggered"] == []

    async def test_monitor_concern_propagated(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["monitor_concern"] == "deployment_not_done"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_narrative_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_HEALTHY}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert result.data["narrative"] != ""

    async def test_escalation_flag_set_correctly(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["46"] = {"data": AGENT46_HEALTHY}

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MONITORING
            result = await run(state)

        assert isinstance(result.confidence.escalated, bool)

    async def test_escalated_when_no_upstream_data(self):
        # base=50, no_production_data→-10=40, active=False→no delta → 40 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_49_post_release_monitor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, [], "NOMINAL", "MONITORING")
        assert "FSC-2417" in msg

    def test_includes_active_flag(self):
        msg = _build_trace_message("FSC-001", False, [], "UNKNOWN", "SKIPPED")
        assert "False" in msg

    def test_includes_health_status(self):
        msg = _build_trace_message("FSC-001", True, [], "NOMINAL", "MONITORING")
        assert "NOMINAL" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-001", False, [], "UNKNOWN", "SKIPPED")
        assert "SKIPPED" in msg

    def test_alerts_sentinel_when_empty(self):
        msg = _build_trace_message("FSC-001", True, [], "NOMINAL", "MONITORING")
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-001", True, [], "NOMINAL", "MONITORING")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "monitor_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_monitor_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["monitor_concern"]["enum"] == [
            "none", "apex_exceptions", "governor_breach",
            "degraded_performance", "deployment_not_done",
        ]
