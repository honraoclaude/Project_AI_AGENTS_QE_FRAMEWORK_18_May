"""Tests for Agent 32 — Regression Risk Assessor (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_32_regression_risk_assessor import (
    _assess_regression_risk,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
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

    def test_medium_risk_gives_warn_verdict_exactly(self):
        agent13_med = {"detected_objects": ["financialaccount"], "dependency_depth": 2}
        risk, _, _, suite, verdict = _assess_regression_risk(
            AGENT3_LOW, agent13_med, AGENT18_CLEAN, None
        )
        assert risk == "MEDIUM"
        assert verdict == "WARN"
        assert suite == "REGRESSION"

    def test_depth_2_gives_moderate_dependency_chain_factor(self):
        agent13_mod = {"detected_objects": ["revenue__c"], "dependency_depth": 2}
        _, factors, _, _, _ = _assess_regression_risk(
            AGENT3_LOW, agent13_mod, AGENT18_CLEAN, None
        )
        assert any("Moderate" in f for f in factors)

    def test_no_risk_factors_fallback_message(self):
        _, factors, _, _, _ = _assess_regression_risk(None, None, None, None)
        assert factors[0] == "No significant regression risk indicators detected"

    def test_single_blast_object_no_other_factors_gives_low(self):
        # 1 blast object → +1 only; score=1 < 2 → LOW
        agent13_single = {"detected_objects": ["financialaccount"], "dependency_depth": 0}
        risk, _, _, suite, verdict = _assess_regression_risk(
            AGENT3_LOW, agent13_single, AGENT18_CLEAN, None
        )
        assert risk == "LOW"
        assert suite == "SMOKE"
        assert verdict == "PASS"

    def test_dev_verdict_fail_gives_medium_risk_and_warn_verdict(self):
        # FAIL → +2, no other factors → score=2 → MEDIUM / WARN
        risk, _, _, suite, verdict = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_CLEAN, AGENT23_FAIL
        )
        assert risk == "MEDIUM"
        assert verdict == "WARN"
        assert suite == "REGRESSION"


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

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_CLEAN, "LOW")
        assert "metadata_scope_available" in signals

    def test_no_metadata_scope_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, AGENT18_CLEAN, "LOW")
        assert "no_metadata_scope" in signals

    def test_component_attribution_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_REGULATED, "LOW")
        assert "component_attribution_available" in signals

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_CLEAN, "LOW")
        assert "fca_classification_available" in signals

    def test_high_risk_detected_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_HIGH_BLAST, AGENT18_CLEAN, "HIGH")
        assert "high_risk_detected" in signals


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

    async def test_escalated_when_no_upstream_data(self):
        # base=63, no_metadata_scope→-8 = 55 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── REQ-23: Development verdict wired + shared_components_stub ────────────────

AGENT23_PARTIAL = {"development_verdict": "PARTIAL", "critical_failures": ["BDD Gherkin: INCOMPLETE"]}
AGENT23_FAIL    = {"development_verdict": "FAIL",    "critical_failures": ["Coverage below threshold"]}
AGENT23_PASS    = {"development_verdict": "PASS",    "critical_failures": []}


class TestDevelopmentVerdictWiredREQ23:
    def test_development_verdict_partial_elevates_risk(self):
        risk, factors, _, _, _ = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_CLEAN, AGENT23_PARTIAL
        )
        assert any("PARTIAL" in f or "development" in f.lower() for f in factors)

    def test_development_verdict_fail_elevates_risk(self):
        risk, _, _, _, _ = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_CLEAN, AGENT23_FAIL
        )
        assert risk in ("MEDIUM", "HIGH")

    def test_development_verdict_pass_no_extra_factor(self):
        _, factors, _, _, _ = _assess_regression_risk(
            AGENT3_LOW, AGENT13_LOW_RISK, AGENT18_CLEAN, AGENT23_PASS
        )
        assert not any("PARTIAL" in f or "FAIL" in f for f in factors)

    def test_high_blast_objects_plus_depth_3_gives_full_suite(self):
        agent13_high = {**AGENT13_HIGH_BLAST, "dependency_depth": 3}
        _, _, _, suite, _ = _assess_regression_risk(
            AGENT3_HIGH, agent13_high, AGENT18_CLEAN, None
        )
        assert suite == "FULL"


@pytest.mark.asyncio
class TestSharedComponentsStubREQ23:
    async def test_shared_components_stub_true_in_output(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_HIGH_BLAST}

        with patch("src.agents.testing.agent_32_regression_risk_assessor.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.data.get("shared_components_stub") is True


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "HIGH", [], [], "FULL", "FAIL")
        assert "FSC-2417" in msg

    def test_includes_risk_level(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "HIGH", [], [], "FULL", "FAIL")
        assert "HIGH" in msg

    def test_includes_suite(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "HIGH", [], [], "FULL", "FAIL")
        assert "FULL" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "HIGH", [], [], "FULL", "FAIL")
        assert "Verdict: FAIL" in msg

    def test_includes_objects_from_agent13(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "HIGH", [], [], "FULL", "FAIL")
        assert "financialaccount" in msg

    def test_no_agent13_shows_unknown(self):
        msg = _build_trace_message("FSC-2417", None, "LOW", [], [], "SMOKE", "PASS")
        assert "unknown" in msg

    def test_no_shared_shows_none(self):
        msg = _build_trace_message("FSC-2417", AGENT13_HIGH_BLAST, "LOW", [], [], "SMOKE", "PASS")
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, "LOW", [], [], "SMOKE", "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "regression_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_regression_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["regression_concern"]["enum"] == [
            "none", "shared_components", "high_blast_radius",
            "deep_dependency_chain", "multiple",
        ]
