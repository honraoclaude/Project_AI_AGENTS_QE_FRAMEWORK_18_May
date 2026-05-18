"""Tests for Agent 53 — Incident Response Agent (True AI — Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.monitoring.agent_53_incident_response import (
    _compute_confidence,
    run,
    run_incident,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT48_FEASIBLE = {"rollback_feasible": True, "rollback_risk": "LOW"}
AGENT48_NOT_FEASIBLE = {"rollback_feasible": False, "rollback_risk": "HIGH"}
AGENT49_NOMINAL = {"monitor_verdict": "MONITORING", "health_status": "NOMINAL", "alerts_triggered": []}
AGENT49_ALERT = {
    "monitor_verdict": "ALERT",
    "health_status": "DEGRADED",
    "alerts_triggered": ["apex_exceptions"],
}
AGENT51_HEALTHY = {"degraded_agents": []}
AGENT51_DEGRADED = {"degraded_agents": [12, 14]}

MOCK_TRIAGE_P2 = {
    "incident_severity": "P2",
    "rollback_recommended": False,
    "triage_steps": [
        "Check Salesforce Event Log Files for Apex exceptions",
        "Review recent deployment in Copado for failed components",
        "Notify TECH_LEAD and QE_LEAD",
    ],
    "escalate_to": ["TECH_LEAD", "QE_LEAD"],
    "estimated_resolution": "1 hour",
    "incident_verdict": "MONITORING",
    "narrative": "P2 degradation detected post-deployment. Apex exceptions in event logs suggest a regression. TECH_LEAD notified.",
}

MOCK_TRIAGE_P1 = {
    "incident_severity": "P1",
    "rollback_recommended": True,
    "triage_steps": [
        "Initiate rollback via Copado immediately",
        "Restore previous change set version",
        "Verify data integrity after rollback",
        "Notify CO and TECH_LEAD",
    ],
    "escalate_to": ["CO", "TECH_LEAD", "QE_LEAD"],
    "estimated_resolution": "30 minutes",
    "incident_verdict": "ESCALATING",
    "narrative": "Critical P1 incident. Production is down. Immediate rollback initiated via Copado.",
}

MOCK_TRIAGE_P3 = {
    "incident_severity": "P3",
    "rollback_recommended": False,
    "triage_steps": ["Monitor error logs for recurrence"],
    "escalate_to": ["QE_LEAD"],
    "estimated_resolution": "Next business day",
    "incident_verdict": "MONITORING",
    "narrative": "Minor P3 issue observed. System stable. Monitoring for recurrence.",
}


# ── _compute_confidence tests ─────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_p1_reduces_confidence(self):
        score_p1, _ = _compute_confidence("P1", 3)
        score_p2, _ = _compute_confidence("P2", 3)
        assert score_p2 > score_p1

    def test_p3_adds_confidence_bonus(self):
        score_p3, _ = _compute_confidence("P3", 3)
        score_p2, _ = _compute_confidence("P2", 3)
        assert score_p3 > score_p2

    def test_detailed_triage_plan_adds_bonus(self):
        score_detailed, _ = _compute_confidence("P2", 5)
        score_empty, _ = _compute_confidence("P2", 0)
        assert score_detailed > score_empty

    def test_no_triage_steps_penalised(self):
        score_none, _ = _compute_confidence("P2", 0)
        score_some, _ = _compute_confidence("P2", 3)
        assert score_some > score_none

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence("P3", 10)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence("P1", 0)
        assert score >= 20

    def test_returns_signals_dict(self):
        _, signals = _compute_confidence("P2", 3)
        assert isinstance(signals, dict)


# ── run() integration tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["48"] = {"data": AGENT48_FEASIBLE}
        state["agent_results"]["49"] = {"data": AGENT49_ALERT}

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        assert result.agent_id == 53
        assert result.agent_name == "Incident Response Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        for key in ["incident_severity", "rollback_recommended", "triage_steps",
                    "escalate_to", "estimated_resolution", "incident_verdict"]:
            assert key in result.data

    async def test_p1_incident_with_rollback(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["48"] = {"data": AGENT48_FEASIBLE}
        state["agent_results"]["49"] = {"data": AGENT49_ALERT}
        state["agent_results"]["51"] = {"data": AGENT51_DEGRADED}

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P1
            result = await run(state)

        assert result.data["incident_severity"] == "P1"
        assert result.data["rollback_recommended"] is True
        assert result.data["incident_verdict"] == "ESCALATING"

    async def test_p2_monitoring_verdict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        assert result.data["incident_verdict"] == "MONITORING"
        assert result.data["rollback_recommended"] is False

    async def test_escalate_to_is_list(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        assert isinstance(result.data["escalate_to"], list)
        assert len(result.data["escalate_to"]) >= 1

    async def test_triage_steps_ordered_list(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P1
            result = await run(state)

        assert isinstance(result.data["triage_steps"], list)
        assert len(result.data["triage_steps"]) >= 1

    async def test_narrative_used_as_why(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        assert result.why == MOCK_TRIAGE_P2["narrative"]

    async def test_what_includes_story_id_and_verdict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run(state)

        assert "FSC-2417" in result.what
        assert "MONITORING" in result.what

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P3
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_no_upstream_data_still_runs(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P3
            result = await run(state)

        assert result.agent_id == 53


# ── run_incident() webhook entry point tests ──────────────────────────────────

@pytest.mark.asyncio
class TestRunIncident:
    async def test_webhook_trigger_returns_agent_result(self):
        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P1
            result = await run_incident(
                story_id="FSC-2417",
                incident_type="apex_exception",
                severity_hint="P1",
                error_details="GoalService.execute: System.LimitException: Too many SOQL queries",
                rollback_feasible=True,
            )

        assert result.agent_id == 53
        assert result.data["incident_severity"] == "P1"

    async def test_webhook_trigger_with_health_snapshot(self):
        snapshot = {"degraded_agents": [12], "healthy_agents": 49}

        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P2
            result = await run_incident(
                story_id="FSC-2417",
                incident_type="degraded_performance",
                severity_hint="P2",
                error_details="Governor limits approaching",
                health_snapshot=snapshot,
            )

        assert result.agent_id == 53

    async def test_p1_always_escalates_to_tech_lead(self):
        with patch("src.agents.monitoring.agent_53_incident_response.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_TRIAGE_P1
            result = await run_incident(
                story_id="FSC-2417",
                incident_type="production_down",
                severity_hint="P1",
                error_details="All Apex classes failing",
            )

        assert "TECH_LEAD" in result.data["escalate_to"]
