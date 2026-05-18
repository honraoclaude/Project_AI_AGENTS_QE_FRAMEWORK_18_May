"""Tests for Agent 51 — Agent Health Monitor (Augmented Script)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.monitoring.agent_51_health import (
    AGENT_NAMES,
    _compute_status,
    run,
    run_scheduled,
)
from src.core.schemas import AgentHealthMetric, initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_metric(agent_id: int, status: str, runs: int = 5, errors: int = 0) -> AgentHealthMetric:
    return AgentHealthMetric(
        agent_id=agent_id,
        agent_name=AGENT_NAMES.get(agent_id, f"Agent {agent_id}"),
        last_run_at=datetime.now(timezone.utc),
        runs_last_hour=runs,
        errors_last_hour=errors,
        avg_latency_ms=500.0,
        avg_confidence=0.0,
        false_positive_rate_30d=None,
        status=status,
    )


MOCK_ALL_HEALTHY = [
    _make_metric(1, "HEALTHY"),
    _make_metric(2, "HEALTHY"),
    _make_metric(3, "HEALTHY"),
]

MOCK_ONE_DEGRADED = [
    _make_metric(1, "HEALTHY"),
    _make_metric(2, "DEGRADED", runs=10, errors=3),
    _make_metric(3, "HEALTHY"),
]

MOCK_ONE_DOWN = [
    _make_metric(1, "DOWN", runs=10, errors=8),
]


# ── _compute_status tests ─────────────────────────────────────────────────────

class TestComputeStatus:
    def test_no_runs_gives_healthy(self):
        # idle agent — not degraded, just no activity
        assert _compute_status(0, 0, 0.0) == "HEALTHY"

    def test_low_error_rate_gives_healthy(self):
        assert _compute_status(10, 1, 500.0) == "HEALTHY"  # 10% < 20% threshold

    def test_error_rate_above_threshold_gives_degraded(self):
        # 25% error rate, < 50% → DEGRADED
        assert _compute_status(100, 25, 500.0) == "DEGRADED"

    def test_high_error_rate_gives_down(self):
        # 60% error rate → DOWN
        assert _compute_status(10, 6, 500.0) == "DOWN"

    def test_exact_threshold_gives_healthy(self):
        # exactly 20% — not > 20% so still HEALTHY
        assert _compute_status(10, 2, 500.0) == "HEALTHY"

    def test_high_latency_gives_degraded(self):
        # latency > 30s threshold
        assert _compute_status(5, 0, 35_000.0) == "DEGRADED"

    def test_normal_latency_healthy(self):
        assert _compute_status(5, 0, 1_000.0) == "HEALTHY"

    def test_error_at_50_percent_boundary(self):
        # exactly 50% — not < 0.5 so → DOWN
        assert _compute_status(10, 5, 500.0) == "DOWN"

    def test_error_just_below_50_gives_degraded(self):
        # 40% error rate → DEGRADED (< 0.5)
        assert _compute_status(10, 4, 500.0) == "DEGRADED"


# ── AGENT_NAMES coverage ──────────────────────────────────────────────────────

class TestAgentNames:
    def test_all_50_agents_covered(self):
        expected_ids = set(range(1, 51))
        assert expected_ids.issubset(set(AGENT_NAMES.keys()))

    def test_no_agent_51_in_names(self):
        # health monitor does not monitor itself
        assert 51 not in AGENT_NAMES

    def test_all_names_are_strings(self):
        for agent_id, name in AGENT_NAMES.items():
            assert isinstance(name, str), f"Agent {agent_id} name is not a string"
            assert len(name) > 0

    def test_spot_check_names(self):
        assert AGENT_NAMES[1] == "Story Intent"
        assert AGENT_NAMES[45] == "Go/No-Go Coordinator"
        assert AGENT_NAMES[50] == "Release Retrospective"


# ── run() integration tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ALL_HEALTHY
            mock_summary.return_value = "All 3 monitored agents are healthy."
            result = await run(state)

        assert result.agent_id == 51
        assert result.agent_name == "Agent Health Monitor"
        assert result.confidence.tier == "A"

    async def test_confidence_is_97_deterministic(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ALL_HEALTHY
            mock_summary.return_value = "All healthy."
            result = await run(state)

        assert result.confidence.final_score == 97
        assert result.confidence.escalated is False

    async def test_data_includes_metrics_and_degraded(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ONE_DEGRADED
            mock_summary.return_value = "Agent 2 is degraded."
            result = await run(state)

        assert "metrics" in result.data
        assert "degraded_agents" in result.data
        assert 2 in result.data["degraded_agents"]

    async def test_no_degraded_when_all_healthy(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ALL_HEALTHY
            mock_summary.return_value = "All healthy."
            result = await run(state)

        assert result.data["degraded_agents"] == []

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ALL_HEALTHY
            mock_summary.return_value = "All healthy."
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_what_reports_degraded_count(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary:
            mock_collect.return_value = MOCK_ONE_DEGRADED
            mock_summary.return_value = "Agent 2 degraded."
            result = await run(state)

        assert "1" in result.what  # 1 of 3 degraded


# ── run_scheduled() tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRunScheduled:
    async def test_returns_metrics_list(self):
        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary, \
             patch("src.agents.monitoring.agent_51_health._send_alert",
                   new_callable=AsyncMock):
            mock_collect.return_value = MOCK_ALL_HEALTHY
            mock_summary.return_value = "All healthy."
            metrics = await run_scheduled()

        assert isinstance(metrics, list)
        assert len(metrics) == 3

    async def test_does_not_send_alert_when_all_healthy(self):
        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._send_alert",
                   new_callable=AsyncMock) as mock_alert:
            mock_collect.return_value = MOCK_ALL_HEALTHY
            await run_scheduled()

        mock_alert.assert_not_called()

    async def test_sends_alert_when_degraded(self):
        with patch("src.agents.monitoring.agent_51_health._collect_metrics",
                   new_callable=AsyncMock) as mock_collect, \
             patch("src.agents.monitoring.agent_51_health._generate_summary",
                   new_callable=AsyncMock) as mock_summary, \
             patch("src.agents.monitoring.agent_51_health._send_alert",
                   new_callable=AsyncMock) as mock_alert:
            mock_collect.return_value = MOCK_ONE_DEGRADED
            mock_summary.return_value = "Agent 2 is degraded."
            await run_scheduled()

        mock_alert.assert_called_once()
