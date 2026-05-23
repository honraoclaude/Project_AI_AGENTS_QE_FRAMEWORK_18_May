"""Tests for Agent 27 — CRT Execution Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_27_crt_execution import (
    _build_trace_message,
    _compute_confidence,
    _simulate_execution,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT25_READY = {
    "env_ready": True,
    "env_verdict": "READY",
    "crt_connected": True,
    "env_blockers": [],
}

AGENT25_BLOCKED = {
    "env_ready": False,
    "env_verdict": "BLOCKED",
    "crt_connected": False,
    "env_blockers": ["Sandbox not ready"],
}

AGENT26_TESTS = {
    "crt_test_cases": [
        {"test_id": "CRT-001", "title": "Suitability check fails for HIGH-risk",
         "tags": ["@fca"], "steps": [], "data_references": []},
        {"test_id": "CRT-002", "title": "Portfolio rebalancing",
         "tags": ["@smoke"], "steps": [], "data_references": []},
    ],
    "crt_test_count": 2,
    "crt_design_verdict": "PASS",
    "automation_coverage": 100.0,
}

AGENT26_EMPTY = {
    "crt_test_cases": [],
    "crt_test_count": 0,
    "crt_design_verdict": "INCOMPLETE",
    "automation_coverage": 0.0,
}

MOCK_TRACE_PASS = {
    "narrative": "All 2 CRT tests executed and passed. No failures detected.",
    "execution_concern": "none",
}

MOCK_TRACE_SKIPPED = {
    "narrative": "CRT execution skipped — environment not ready for test run.",
    "execution_concern": "execution_skipped",
}


# ── Deterministic execution simulation tests ──────────────────────────────────

class TestExecutionSimulation:
    def test_env_ready_with_tests_gives_pass(self):
        results, executed, passed, failed, verdict = _simulate_execution(
            AGENT25_READY, AGENT26_TESTS
        )
        assert verdict == "PASS"
        assert executed == 2
        assert passed == 2
        assert failed == 0

    def test_env_not_ready_gives_skipped(self):
        results, executed, passed, failed, verdict = _simulate_execution(
            AGENT25_BLOCKED, AGENT26_TESTS
        )
        assert verdict == "SKIPPED"
        assert executed == 0

    def test_no_crt_not_connected_gives_skipped(self):
        agent25_no_crt = {**AGENT25_READY, "crt_connected": False}
        _, executed, _, _, verdict = _simulate_execution(agent25_no_crt, AGENT26_TESTS)
        assert verdict == "SKIPPED"

    def test_no_test_cases_gives_skipped(self):
        _, executed, _, _, verdict = _simulate_execution(AGENT25_READY, AGENT26_EMPTY)
        assert verdict == "SKIPPED"
        assert executed == 0

    def test_no_upstream_data_gives_skipped(self):
        _, executed, _, _, verdict = _simulate_execution(None, None)
        assert verdict == "SKIPPED"

    def test_results_contain_test_ids(self):
        results, _, _, _, _ = _simulate_execution(AGENT25_READY, AGENT26_TESTS)
        assert any(r["test_id"] == "CRT-001" for r in results)
        assert any(r["test_id"] == "CRT-002" for r in results)

    def test_all_results_passed_in_stub(self):
        results, _, _, failed, _ = _simulate_execution(AGENT25_READY, AGENT26_TESTS)
        assert failed == 0
        assert all(r["status"] == "PASSED" for r in results)

    def test_incomplete_design_verdict_skips_even_with_cases(self):
        agent26_incomplete = {**AGENT26_TESTS, "crt_design_verdict": "INCOMPLETE"}
        _, executed, _, _, verdict = _simulate_execution(AGENT25_READY, agent26_incomplete)
        assert verdict == "SKIPPED"
        assert executed == 0

    def test_result_dict_has_all_expected_fields(self):
        results, _, _, _, _ = _simulate_execution(AGENT25_READY, AGENT26_TESTS)
        r = results[0]
        for field in ("test_id", "title", "tags", "status", "duration_ms",
                      "error_message", "self_healed"):
            assert field in r


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_ready_env_with_tests_scores_well(self):
        score, _ = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "PASS")
        assert score >= 70

    def test_blocked_env_reduces_confidence(self):
        score_ready, _ = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "PASS")
        score_blocked, _ = _compute_confidence(AGENT25_BLOCKED, AGENT26_TESTS, 0, "SKIPPED")
        assert score_ready > score_blocked

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 4, "PASS")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, 0, "SKIPPED")
        assert score >= 20

    def test_env_ready_for_execution_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "PASS")
        assert "env_ready_for_execution" in signals

    def test_env_not_ready_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_BLOCKED, AGENT26_TESTS, 0, "SKIPPED")
        assert "env_not_ready" in signals

    def test_crt_test_cases_available_key_and_value(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "PASS")
        assert signals["crt_test_cases_available"] == 2

    def test_no_crt_test_cases_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT26_EMPTY, 0, "SKIPPED")
        assert "no_crt_test_cases" in signals

    def test_tests_executed_key_and_value(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "PASS")
        assert signals["tests_executed"] == 2

    def test_execution_skipped_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_BLOCKED, AGENT26_EMPTY, 0, "SKIPPED")
        assert "execution_skipped" in signals

    def test_tests_failed_key_in_signals(self):
        _, signals = _compute_confidence(AGENT25_READY, AGENT26_TESTS, 2, "FAIL")
        assert "tests_failed" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_READY}
        state["agent_results"]["26"] = {"data": AGENT26_TESTS}

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 27
        assert result.agent_name == "CRT Execution Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["crt_results", "crt_pass_count", "crt_fail_count",
                    "crt_execution_verdict", "tests_executed"]:
            assert key in result.data

    async def test_pass_verdict_when_ready(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_READY}
        state["agent_results"]["26"] = {"data": AGENT26_TESTS}

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["crt_execution_verdict"] == "PASS"
        assert result.data["tests_executed"] == 2

    async def test_skipped_when_env_blocked(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["25"] = {"data": AGENT25_BLOCKED}
        state["agent_results"]["26"] = {"data": AGENT26_TESTS}

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["crt_execution_verdict"] == "SKIPPED"
        assert result.data["tests_executed"] == 0

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=62, env_not_ready→-15, no_crt_test_cases→-10, execution_skipped→-10 = 27 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_27_crt_execution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── REQ-19: Tags carried into execution result dicts ──────────────────────────

class TestTagsCarriedREQ19:
    def test_test_case_tags_carried_into_result_dict(self):
        results, _, _, _, _ = _simulate_execution(AGENT25_READY, AGENT26_TESTS)
        fca_result = next((r for r in results if r.get("test_id") == "CRT-001"), None)
        assert fca_result is not None
        assert "@fca" in fca_result.get("tags", [])

    def test_smoke_tags_carried_into_result_dict(self):
        results, _, _, _, _ = _simulate_execution(AGENT25_READY, AGENT26_TESTS)
        smoke_result = next((r for r in results if r.get("test_id") == "CRT-002"), None)
        assert smoke_result is not None
        assert "@smoke" in smoke_result.get("tags", [])

    def test_results_without_tags_have_empty_tags_list(self):
        agent26_no_tags = {
            "crt_test_cases": [
                {"test_id": "CRT-001", "title": "Test with no tags", "steps": [], "data_references": []},
            ],
            "crt_test_count": 1,
            "crt_design_verdict": "PASS",
            "automation_coverage": 100.0,
        }
        results, _, _, _, _ = _simulate_execution(AGENT25_READY, agent26_no_tags)
        assert results[0].get("tags", []) == []


# ── REQ-18: Gate G5 SKIPPED test (integration) ───────────────────────────────
# (Gate G5 test lives in test_testing_phase.py — see tests added alongside REQ-18)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", 2, 2, 0, "PASS")
        assert "FSC-2417" in msg

    def test_includes_executed_count(self):
        msg = _build_trace_message("FSC-2417", 2, 2, 0, "PASS")
        assert "Tests executed: 2" in msg

    def test_includes_passed_count(self):
        msg = _build_trace_message("FSC-2417", 2, 2, 0, "PASS")
        assert "Tests passed: 2" in msg

    def test_includes_failed_count(self):
        msg = _build_trace_message("FSC-2417", 2, 1, 1, "PARTIAL")
        assert "Tests failed: 1" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", 2, 2, 0, "PASS")
        assert "Verdict: PASS" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", 0, 0, 0, "SKIPPED")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "execution_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_execution_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["execution_concern"]["enum"] == [
            "none", "test_failures", "env_instability", "execution_skipped", "partial_run"
        ]
