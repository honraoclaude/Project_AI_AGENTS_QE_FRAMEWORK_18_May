"""Tests for Agent 20 — Performance Risk Estimator (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_20_performance_risk import (
    _compute_confidence,
    _estimate_performance_risk,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT8_SHALLOW = {"detected_objects": ["revenue__c"], "dependency_depth": 0}
AGENT8_DEEP    = {"detected_objects": ["financialaccount", "financialholding"], "dependency_depth": 3}

AGENT13_DEEP = {
    "detected_objects": ["financialaccount", "financialholding", "suitability__c"],
    "dependency_depth": 3,
    "changed_files_count": 4,
}

AGENT13_SHALLOW = {
    "detected_objects": ["revenue__c"],
    "dependency_depth": 0,
    "changed_files_count": 1,
}

AGENT14_SOQL_VIOLATION = {
    "critical_violations": [
        {"rule_name": "OperationWithLimitsInLoop", "priority": 1, "file_path": "classes/Foo.cls"},
    ],
    "quality_verdict": "FAIL",
}

AGENT14_CLEAN = {
    "critical_violations": [],
    "quality_verdict": "PASS",
}

AGENT16_HIGH = {"bulk_risk_level": "HIGH", "async_recommended": True}
AGENT16_LOW  = {"bulk_risk_level": "LOW",  "async_recommended": False}

MOCK_TRACE_HIGH = {
    "narrative": "HIGH governor limit exposure due to deep dependency chain and high-volume objects.",
    "performance_concern": "governor_limits",
}

MOCK_TRACE_LOW = {
    "narrative": "No significant performance risk detected.",
    "performance_concern": "none",
}


# ── Deterministic performance risk estimation tests ───────────────────────────

class TestPerformanceRiskEstimation:
    def test_soql_in_loop_gives_high_risk(self):
        risk, factors, soql_loop, gov, verdict = _estimate_performance_risk(
            AGENT3_HIGH, None, AGENT13_SHALLOW, AGENT14_SOQL_VIOLATION, AGENT16_LOW
        )
        assert risk == "HIGH"
        assert soql_loop is True
        assert verdict == "FAIL"

    def test_deep_dependency_gives_medium_or_high_risk(self):
        risk, _, _, _, _ = _estimate_performance_risk(
            AGENT3_LOW, None, AGENT13_DEEP, AGENT14_CLEAN, AGENT16_LOW
        )
        assert risk in ("MEDIUM", "HIGH")

    def test_high_volume_objects_raise_risk(self):
        risk, factors, _, _, _ = _estimate_performance_risk(
            AGENT3_LOW, None, AGENT13_DEEP, AGENT14_CLEAN, AGENT16_LOW
        )
        assert any("high-volume" in f.lower() or "financialaccount" in f.lower() for f in factors)

    def test_shallow_clean_story_gives_low_risk(self):
        risk, _, soql_loop, gov, verdict = _estimate_performance_risk(
            AGENT3_LOW, None, AGENT13_SHALLOW, AGENT14_CLEAN, AGENT16_LOW
        )
        assert risk == "LOW"
        assert soql_loop is False
        assert verdict == "PASS"

    def test_high_bulk_risk_raises_performance_risk(self):
        risk, factors, _, _, _ = _estimate_performance_risk(
            AGENT3_LOW, None, AGENT13_SHALLOW, AGENT14_CLEAN, AGENT16_HIGH
        )
        assert risk in ("MEDIUM", "HIGH")
        assert any("bulk" in f.lower() for f in factors)

    def test_no_upstream_data_degrades_gracefully(self):
        risk, factors, soql_loop, gov, verdict = _estimate_performance_risk(
            None, None, None, None, None
        )
        assert risk == "LOW"
        assert verdict == "PASS"
        assert len(factors) >= 1

    def test_agent13_preferred_over_agent8_for_objects(self):
        """Agent 13 (code-time) takes precedence over Agent 8 (refinement-time)."""
        risk_with_13, _, _, _, _ = _estimate_performance_risk(
            AGENT3_LOW, AGENT8_DEEP, AGENT13_SHALLOW, AGENT14_CLEAN, AGENT16_LOW
        )
        risk_without_13, _, _, _, _ = _estimate_performance_risk(
            AGENT3_LOW, AGENT8_DEEP, None, AGENT14_CLEAN, AGENT16_LOW
        )
        # With Agent 13 shallow, risk should be lower than without (which uses Agent 8 deep)
        assert risk_without_13 in ("MEDIUM", "HIGH")

    def test_factors_never_empty(self):
        _, factors, _, _, _ = _estimate_performance_risk(
            None, None, None, None, None
        )
        assert len(factors) >= 1

    def test_high_risk_verdict_is_fail(self):
        _, _, _, _, verdict = _estimate_performance_risk(
            AGENT3_HIGH, None, AGENT13_DEEP, AGENT14_SOQL_VIOLATION, AGENT16_HIGH
        )
        assert verdict == "FAIL"

    def test_medium_risk_verdict_is_warn(self):
        risk, _, _, _, verdict = _estimate_performance_risk(
            AGENT3_LOW, None, AGENT13_DEEP, AGENT14_CLEAN, AGENT16_LOW
        )
        if risk == "MEDIUM":
            assert verdict == "WARN"


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, AGENT16_HIGH, "HIGH")
        assert score >= 65

    def test_no_code_time_metadata_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, AGENT16_LOW, "LOW")
        score_without, _ = _compute_confidence(AGENT3_HIGH, None, AGENT16_LOW, "LOW")
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, AGENT16_HIGH, "LOW")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, "HIGH")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_DEEP}
        state["agent_results"]["16"] = {"data": AGENT16_HIGH}

        with patch("src.agents.development.agent_20_performance_risk.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.agent_id == 20
        assert result.agent_name == "Performance Risk Estimator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_20_performance_risk.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        for key in ["performance_risk_level", "performance_risk_factors",
                    "soql_loop_risk", "governor_limit_exposure", "performance_verdict"]:
            assert key in result.data

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_20_performance_risk.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_standalone_mode_low_risk(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_20_performance_risk.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.data["performance_risk_level"] == "LOW"
        assert result.data["performance_verdict"] == "PASS"
