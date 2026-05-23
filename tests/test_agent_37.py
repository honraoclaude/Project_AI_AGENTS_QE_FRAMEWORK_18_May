"""Tests for Agent 37 — Performance Test Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_37_performance_test import (
    _assess_performance,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT20_HIGH = {
    "performance_risk_level": "HIGH",
    "soql_loop_risk": True,
    "governor_limit_exposure": "HIGH",
    "performance_verdict": "FAIL",
}

AGENT20_LOW = {
    "performance_risk_level": "LOW",
    "soql_loop_risk": False,
    "governor_limit_exposure": "LOW",
    "performance_verdict": "PASS",
}

AGENT20_MEDIUM = {
    "performance_risk_level": "MEDIUM",
    "soql_loop_risk": False,
    "governor_limit_exposure": "MEDIUM",
    "performance_verdict": "WARN",
}

AGENT27_PASS = {"crt_execution_verdict": "PASS", "tests_executed": 3}
AGENT27_SKIP = {"crt_execution_verdict": "SKIPPED", "tests_executed": 0}

MOCK_TRACE_SKIPPED = {
    "narrative": "Performance test not required for LOW-risk story. SKIPPED.",
    "performance_concern": "none",
}

MOCK_TRACE_FAIL = {
    "narrative": "Performance test REQUIRED for HIGH-risk story. SOQL loop detected — governor limit breach expected.",
    "performance_concern": "governor_limit_breach",
}


# ── Deterministic performance assessment tests ────────────────────────────────

class TestPerformanceAssessment:
    def test_low_risk_not_required_gives_skipped(self):
        required, resp_ok, gov_ok, verdict = _assess_performance(AGENT20_LOW, AGENT27_PASS)
        assert required is False
        assert verdict == "SKIPPED"

    def test_high_risk_required_gives_fail(self):
        required, resp_ok, gov_ok, verdict = _assess_performance(AGENT20_HIGH, AGENT27_PASS)
        assert required is True
        assert verdict == "FAIL"
        assert resp_ok is False
        assert gov_ok is False

    def test_soql_loop_makes_test_required(self):
        agent20_soql = {**AGENT20_LOW, "soql_loop_risk": True}
        required, _, _, _ = _assess_performance(agent20_soql, AGENT27_PASS)
        assert required is True

    def test_no_upstream_data_not_required(self):
        required, _, _, verdict = _assess_performance(None, None)
        assert required is False
        assert verdict == "SKIPPED"

    def test_medium_risk_not_required(self):
        required, _, _, verdict = _assess_performance(AGENT20_MEDIUM, AGENT27_PASS)
        assert required is False
        assert verdict == "SKIPPED"

    def test_response_time_ok_for_low_risk(self):
        _, resp_ok, _, _ = _assess_performance(AGENT20_LOW, None)
        assert resp_ok is True

    def test_governor_limits_ok_for_low_risk(self):
        _, _, gov_ok, _ = _assess_performance(AGENT20_LOW, None)
        assert gov_ok is True

    def test_soql_loop_only_gives_fail(self):
        # soql_loop=True with MEDIUM risk: required=True (soql), resp_ok=True (not HIGH), gov_ok=False (soql)
        agent20_soql_medium = {"performance_risk_level": "MEDIUM", "soql_loop_risk": True, "governor_limit_exposure": "LOW"}
        required, resp_ok, gov_ok, verdict = _assess_performance(agent20_soql_medium, None)
        assert required is True
        assert resp_ok is True
        assert gov_ok is False
        assert verdict == "FAIL"

    def test_high_risk_without_soql_loop_gives_fail(self):
        # HIGH risk + no soql_loop: resp_ok=False (HIGH), gov_ok=True (LOW exposure, no soql)
        agent20_high_no_soql = {"performance_risk_level": "HIGH", "soql_loop_risk": False, "governor_limit_exposure": "LOW"}
        required, resp_ok, gov_ok, verdict = _assess_performance(agent20_high_no_soql, None)
        assert required is True
        assert resp_ok is False
        assert gov_ok is True
        assert verdict == "FAIL"

    def test_high_gov_exposure_without_soql_gives_gov_not_ok(self):
        # gov_exposure=HIGH without soql_loop → gov_ok=False (not required since LOW perf_risk)
        agent20_gov = {"performance_risk_level": "LOW", "soql_loop_risk": False, "governor_limit_exposure": "HIGH"}
        _, _, gov_ok, verdict = _assess_performance(agent20_gov, None)
        assert gov_ok is False
        assert verdict == "SKIPPED"  # not required since LOW + no soql_loop


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_perf_risk_signal_available_scores_well(self):
        score, _ = _compute_confidence(AGENT20_LOW, AGENT27_PASS, "SKIPPED")
        assert score >= 60

    def test_no_perf_risk_signal_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT20_LOW, AGENT27_PASS, "SKIPPED")
        score_without, _ = _compute_confidence(None, AGENT27_PASS, "SKIPPED")
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT20_LOW, AGENT27_PASS, "PASS")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, "FAIL")
        assert score >= 20

    def test_performance_risk_signal_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT20_LOW, None, "SKIPPED")
        assert "performance_risk_signal_available" in signals

    def test_no_performance_risk_signal_key_in_signals(self):
        _, signals = _compute_confidence(None, None, "SKIPPED")
        assert "no_performance_risk_signal" in signals

    def test_crt_passed_key_in_signals(self):
        _, signals = _compute_confidence(AGENT20_LOW, AGENT27_PASS, "SKIPPED")
        assert "crt_passed" in signals

    def test_performance_fail_key_in_signals(self):
        _, signals = _compute_confidence(AGENT20_HIGH, None, "FAIL")
        assert "performance_fail" in signals

    def test_performance_test_not_required_key_in_signals(self):
        _, signals = _compute_confidence(AGENT20_LOW, None, "SKIPPED")
        assert "performance_test_not_required" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["20"] = {"data": AGENT20_LOW}
        state["agent_results"]["27"] = {"data": AGENT27_PASS}

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.agent_id == 37
        assert result.agent_name == "Performance Test Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["perf_test_required", "perf_test_verdict",
                    "response_time_ok", "governor_limits_ok"]:
            assert key in result.data

    async def test_skipped_for_low_risk(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["20"] = {"data": AGENT20_LOW}

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["perf_test_required"] is False
        assert result.data["perf_test_verdict"] == "SKIPPED"

    async def test_fail_for_high_risk_with_soql_loop(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["20"] = {"data": AGENT20_HIGH}

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["perf_test_required"] is True
        assert result.data["perf_test_verdict"] == "FAIL"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=58, no_performance_risk_signal→-10=48, verdict=SKIPPED→+3=51 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_37_performance_test.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "FSC-2417" in msg

    def test_includes_performance_risk(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "HIGH" in msg

    def test_includes_soql_loop_risk(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "True" in msg

    def test_includes_governor_exposure(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "HIGH" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "FAIL" in msg

    def test_includes_required_flag(self):
        msg = _build_trace_message("FSC-2417", AGENT20_HIGH, True, False, False, "FAIL")
        assert "True" in msg

    def test_unknown_risk_when_no_agent20(self):
        msg = _build_trace_message("FSC-2417", None, False, True, True, "SKIPPED")
        assert "UNKNOWN" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, False, True, True, "SKIPPED")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "performance_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_performance_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["performance_concern"]["enum"] == [
            "none", "response_time_breach", "governor_limit_breach",
            "test_skipped_for_high_risk", "multiple",
        ]
