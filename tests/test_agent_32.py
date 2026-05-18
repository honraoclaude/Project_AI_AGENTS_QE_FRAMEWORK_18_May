"""Tests for Agent 32 — Regression Risk Assessor (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_32_regression_risk_assessor import (
    _assess_regression_risk,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT13_HIGH_BLAST = {
    "detected_objects": ["financialaccount", "financialholding", "financialgoal"],
    "dependency_depth": 3,
}

AGENT13_LOW_RISK = {
    "detected_objects": ["revenue__c"],
    "dependency_depth": 0,
}

AGENT18_REGULATED = {
    "regulated_components": ["SuitabilityService", "RiskProfileCalculator"],
    "merge_risk_components": ["SharedHelper"],
    "component_verdict": "REVIEW_REQUIRED",
}

AGENT18_CLEAN = {
    "regulated_components": [],
    "merge_risk_components": [],
    "component_verdict": "PASS",
}

MOCK_TRACE_HIGH = {
    "narrative": "HIGH regression risk: FinancialAccount and FinancialHolding have wide downstream consumers. Full regression suite required.",
    "regression_concern": "high_blast_radius",
}

MOCK_TRACE_LOW = {
    "narrative": "LOW regression risk. Single non-core object with no downstream consumers. Smoke suite sufficient.",
    "regression_concern": "none",
}


# ── Deterministic regression risk tests ──────────────────────────────────────

class TestRegressionRiskAssessment:
    def test_high_blast_objects_give_high_risk(self):
        risk, factors, _, suite, verdict = _assess_regression_risk(
            AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_CLEAN, None
        )
        assert risk == "HIGH"
        assert suite == "FULL"
        assert verdict == "FAIL"

    def test_single_low_risk_object_gives_low_risk(self):
        risk, _, _, suite, verdict = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_CLEAN, None
        )
        assert risk == "LOW"
        assert suite == "SMOKE"
        assert verdict == "PASS"

    def test_regulated_components_add_risk(self):
        risk, factors, _, _, _ = _assess_regression_risk(
            AGENT3_HIGH, AGENT13_LOW_RISK, AGENT18_REGULATED, None
        )
        assert any("regulated" in f.lower() or "FCA" in f for f in factors)

    def test_shared_components_detected(self):
        _, _, shared, _, _ = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_REGULATED, None
        )
        assert "SharedHelper" in shared

    def test_medium_risk_gives_regression_suite(self):
        agent13_med = {"detected_objects": ["financialaccount"], "dependency_depth": 2}
        risk, _, _, suite, _ = _assess_regression_risk(
            AGENT3_LOW, agent13_med, AGENT18_CLEAN, None
        )
        assert risk in ("MEDIUM", "HIGH")
        assert suite in ("REGRESSION", "FULL")

    def test_factors_never_empty(self):
        _, factors, _, _, _ = _assess_regression_risk(None, None, None, None)
        assert len(factors) >= 1

    def test_no_upstream_data_gives_low_risk(self):
        risk, _, _, suite, verdict = _assess_regression_risk(None, None, None, None)
        assert risk == "LOW"
        assert suite == "SMOKE"
        assert verdict == "PASS"

    def test_deep_dependency_chain_adds_factor(self):
        _, factors, _, _, _ = _assess_regression_risk(
            AGENT3_LOW, AGENT13_HIGH_BLAST, AGENT18_CLEAN, None
        )
        assert any("depth" in f.lower() or "chain" in f.lower() for f in factors)


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_REGULATED, "HIGH")
        assert score >= 65

    def test_no_metadata_scope_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_CLEAN, "LOW")
        score_without, _ = _compute_confidence(AGENT3_HIGH, None, AGENT18_CLEAN, "LOW")
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_REGULATED, "LOW")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, "HIGH")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_HIGH_BLAST}
        state["agent_results"]["18"] = {"data": AGENT18_REGULATED}

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.agent_id == 32
        assert result.agent_name == "Regression Risk Assessor"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        for key in ["regression_risk_level", "regression_risk_factors",
                    "shared_components", "recommended_regression_suite",
                    "regression_verdict"]:
            assert key in result.data

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
