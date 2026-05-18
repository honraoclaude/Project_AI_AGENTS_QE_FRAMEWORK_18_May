"""Tests for Agent 46 — Production Validation Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_46_production_validation import (
    _compute_confidence,
    _validate_production,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT45_GO = {"go_decision": True, "coordinator_verdict": "GO"}
AGENT45_CONDITIONAL = {"go_decision": True, "coordinator_verdict": "CONDITIONAL"}
AGENT45_NO_GO = {"go_decision": False, "coordinator_verdict": "NO_GO"}

MOCK_TRACE_HEALTHY = {
    "narrative": "Production health checks passed 5/5. Apex execution is normal, governor limits within bounds, critical flows accessible.",
    "prod_concern": "none",
}
MOCK_TRACE_SKIPPED = {
    "narrative": "Production validation skipped — deployment was not approved (NO_GO decision).",
    "prod_concern": "deployment_skipped",
}


# ── Deterministic production validation tests ─────────────────────────────────

class TestValidateProduction:
    def test_go_decision_gives_healthy(self):
        healthy, passed, total, verdict = _validate_production(AGENT45_GO)
        assert healthy is True
        assert verdict == "HEALTHY"
        assert passed == total

    def test_conditional_decision_gives_healthy(self):
        healthy, passed, total, verdict = _validate_production(AGENT45_CONDITIONAL)
        assert healthy is True
        assert verdict == "HEALTHY"

    def test_no_go_decision_gives_skipped(self):
        healthy, passed, _, verdict = _validate_production(AGENT45_NO_GO)
        assert healthy is False
        assert verdict == "SKIPPED"
        assert passed == 0

    def test_no_upstream_data_gives_skipped(self):
        healthy, passed, _, verdict = _validate_production(None)
        assert healthy is False
        assert verdict == "SKIPPED"
        assert passed == 0

    def test_all_checks_pass_when_go(self):
        _, passed, total, _ = _validate_production(AGENT45_GO)
        assert passed == total
        assert total == 5  # _HEALTH_CHECKS has 5 items


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_go_no_go_data_available_scores_well(self):
        score, _ = _compute_confidence(AGENT45_GO, True)
        assert score >= 65

    def test_no_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT45_GO, True)
        score_without, _ = _compute_confidence(None, False)
        assert score_with > score_without

    def test_unhealthy_reduces_confidence(self):
        score_healthy, _ = _compute_confidence(AGENT45_GO, True)
        score_sick, _ = _compute_confidence(AGENT45_GO, False)
        assert score_healthy > score_sick

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT45_GO, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["45"] = {"data": AGENT45_GO}

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HEALTHY
            result = await run(state)

        assert result.agent_id == 46
        assert result.agent_name == "Production Validation Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["production_healthy", "validation_checks_passed",
                    "validation_checks_total", "prod_verdict"]:
            assert key in result.data

    async def test_healthy_when_go_approved(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["45"] = {"data": AGENT45_GO}

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HEALTHY
            result = await run(state)

        assert result.data["prod_verdict"] == "HEALTHY"
        assert result.data["production_healthy"] is True
        assert result.data["validation_checks_passed"] == result.data["validation_checks_total"]

    async def test_skipped_when_no_go(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["45"] = {"data": AGENT45_NO_GO}

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["prod_verdict"] == "SKIPPED"
        assert result.data["production_healthy"] is False

    async def test_skipped_with_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["prod_verdict"] == "SKIPPED"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_46_production_validation.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HEALTHY
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
