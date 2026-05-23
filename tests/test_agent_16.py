"""Tests for Agent 16 — Bulk/Async Quality (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_16_bulk_quality import (
    _analyse_bulk_risk,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT8_DEPTH_2 = {"detected_objects": ["suitability__c"], "dependency_depth": 2}
AGENT8_DEPTH_0 = {"detected_objects": [], "dependency_depth": 0}

AGENT13_DEEP = {
    "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
    "dependency_depth": 3,
    "changed_files_count": 4,
}

AGENT13_MEDIUM = {
    "detected_objects": ["financialaccount", "financialholding"],
    "dependency_depth": 1,
    "changed_files_count": 2,
}

AGENT13_SHALLOW = {
    "detected_objects": ["revenue__c"],
    "dependency_depth": 0,
    "changed_files_count": 1,
}

MOCK_TRACE_HIGH = {
    "narrative": "Dependency depth 3 across Suitability, RiskProfile, and FinancialAccount creates HIGH bulk risk.",
    "bulk_risk_concern": "high",
}

MOCK_TRACE_LOW = {
    "narrative": "Single FSC object with no dependency chain. LOW governor limit risk.",
    "bulk_risk_concern": "low",
}


# ── Deterministic bulk risk analysis tests ────────────────────────────────────

class TestBulkRiskAnalysis:
    def test_high_fca_with_depth_2_gives_high_risk(self):
        risk, _, async_rec = _analyse_bulk_risk(AGENT3_HIGH, AGENT8_DEPTH_2, None)
        assert risk == "HIGH"
        assert async_rec is True

    def test_depth_3_gives_high_risk_regardless_of_fca(self):
        risk, _, _ = _analyse_bulk_risk(AGENT3_LOW, None, AGENT13_DEEP)
        assert risk == "HIGH"

    def test_two_fsc_objects_gives_medium_risk(self):
        risk, _, _ = _analyse_bulk_risk(AGENT3_LOW, None, AGENT13_MEDIUM)
        assert risk == "MEDIUM"

    def test_single_object_no_depth_gives_low_risk(self):
        risk, _, async_rec = _analyse_bulk_risk(AGENT3_LOW, AGENT8_DEPTH_0, AGENT13_SHALLOW)
        assert risk == "LOW"
        assert async_rec is False

    def test_factors_not_empty(self):
        _, factors, _ = _analyse_bulk_risk(AGENT3_HIGH, AGENT8_DEPTH_2, AGENT13_DEEP)
        assert len(factors) >= 1

    def test_factors_not_empty_for_low_risk(self):
        _, factors, _ = _analyse_bulk_risk(AGENT3_LOW, AGENT8_DEPTH_0, AGENT13_SHALLOW)
        assert len(factors) >= 1  # at least the "no significant risk" message

    def test_async_recommended_for_high_risk(self):
        _, _, async_rec = _analyse_bulk_risk(AGENT3_HIGH, None, AGENT13_DEEP)
        assert async_rec is True

    def test_no_upstream_data_degrades_gracefully(self):
        risk, factors, _ = _analyse_bulk_risk(None, None, None)
        assert risk == "LOW"
        assert len(factors) >= 1

    def test_agent13_depth_preferred_over_agent8(self):
        """Agent 13 (code-time depth) takes precedence over Agent 8 (refinement-time depth)."""
        agent13_shallow = {**AGENT13_DEEP, "dependency_depth": 0}
        risk, _, _ = _analyse_bulk_risk(AGENT3_HIGH, AGENT8_DEPTH_2, agent13_shallow)
        # Agent 13 depth=0 — HIGH FCA but depth < 2, so not HIGH risk
        assert risk == "MEDIUM"

    def test_medium_fca_depth_2_triggers_soql_factor(self):
        # MEDIUM FCA + depth=2 + <3 objects → factors 1/2/3 all False → factor 4 fires
        agent13_one = {"detected_objects": ["financialaccount"], "dependency_depth": 2}
        _, factors, _ = _analyse_bulk_risk(AGENT3_MEDIUM, None, agent13_one)
        assert any("soql" in f.lower() for f in factors)

    def test_three_fsc_objects_low_depth_gives_medium_and_fires_factor3(self):
        # 3 detected objects but depth < 3 → Factor 3 fires in isolation (not Factor 1)
        agent13_three = {"detected_objects": ["a__c", "b__c", "c__c"], "dependency_depth": 1}
        risk, factors, _ = _analyse_bulk_risk(AGENT3_LOW, None, agent13_three)
        assert risk == "MEDIUM"
        assert any("FSC objects in scope" in f for f in factors)

    def test_single_object_depth_1_gives_medium_risk(self):
        # 1 detected + depth >= 1 → MEDIUM via the depth >= 1 branch
        agent13_one = {"detected_objects": ["financialaccount"], "dependency_depth": 1}
        risk, _, async_rec = _analyse_bulk_risk(AGENT3_LOW, None, agent13_one)
        assert risk == "MEDIUM"
        assert async_rec is False

    def test_medium_fca_depth_2_not_high_risk(self):
        # Only HIGH FCA + depth >= 2 triggers HIGH; MEDIUM FCA + depth=2 stays MEDIUM
        agent13_d2 = {"detected_objects": ["financialaccount"], "dependency_depth": 2}
        risk, _, _ = _analyse_bulk_risk(AGENT3_MEDIUM, None, agent13_d2)
        assert risk == "MEDIUM"


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_both_agents_available_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        assert score >= 70

    def test_no_metadata_scope_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        score_without, _ = _compute_confidence(AGENT3_HIGH, None, 0)
        assert score_with > score_without

    def test_known_depth_boosts_confidence(self):
        score_depth, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        score_zero, _ = _compute_confidence(AGENT3_HIGH, AGENT13_SHALLOW, 0)
        assert score_depth >= score_zero

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, 0)
        assert score >= 20

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        assert "metadata_scope_available" in signals

    def test_metadata_scope_unavailable_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, 0)
        assert "metadata_scope_unavailable" in signals

    def test_fca_class_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        assert "fca_class_available" in signals

    def test_dependency_depth_known_stores_depth(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_DEEP, 3)
        assert signals["dependency_depth_known"] == 3

    def test_zero_depth_confirmed_key_in_signals(self):
        # depth=0 AND agent13 present → zero_depth_confirmed fires
        _, signals = _compute_confidence(AGENT3_LOW, AGENT13_SHALLOW, 0)
        assert "zero_depth_confirmed" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_DEEP}

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.agent_id == 16
        assert result.agent_name == "Bulk/Async Quality"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        for key in ["bulk_risk_level", "bulk_risk_factors", "async_recommended"]:
            assert key in result.data

    async def test_high_bulk_risk_for_deep_dependency(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_DEEP}

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.data["bulk_risk_level"] == "HIGH"
        assert result.data["async_recommended"] is True

    async def test_low_bulk_risk_for_shallow_story(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_LOW}
        state["agent_results"]["13"] = {"data": AGENT13_SHALLOW}

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.data["bulk_risk_level"] == "LOW"

    async def test_standalone_mode(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.agent_id == 16

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=65, metadata_scope_unavailable=-10 → 55 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_DEEP}

        with patch("src.agents.development.agent_16_bulk_quality.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], 3, "HIGH", ["bulk factor"])
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], 3, "HIGH", ["bulk factor"])
        assert "HIGH" in msg

    def test_includes_depth(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], 3, "HIGH", ["bulk factor"])
        assert "Dependency depth: 3" in msg

    def test_empty_detected_shows_none(self):
        msg = _build_trace_message("FSC-2417", "LOW", [], 0, "LOW", ["no risk"])
        assert "Detected FSC objects: ['none']" in msg

    def test_detected_objects_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", "HIGH", ["suitability__c"], 2, "HIGH", ["factor"])
        assert "suitability__c" in msg

    def test_risk_level_shown(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], 3, "HIGH", ["bulk factor"])
        assert "Bulk risk level: HIGH" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", "LOW", [], 0, "LOW", ["no risk"])
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "bulk_risk_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_bulk_risk_concern_enum_has_four_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["bulk_risk_concern"]["enum"] == ["none", "low", "medium", "high"]
