"""Tests for Agent 38 — Flaky Test Hunter (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_38_flaky_test_hunter import (
    _build_trace_message,
    _compute_confidence,
    _detect_flaky_tests,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT27_CLEAN = {
    "tests_executed": 3,
    "crt_execution_verdict": "PASS",
    "crt_results": [
        {"test_id": "CRT-001", "title": "Suitability check",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
        {"test_id": "CRT-002", "title": "Portfolio rebalancing",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
        {"test_id": "CRT-003", "title": "Risk profile update",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
    ],
}

AGENT27_ONE_FLAKY = {
    "tests_executed": 3,
    "crt_execution_verdict": "PASS",
    "crt_results": [
        {"test_id": "CRT-001", "title": "Suitability check",
         "status": "PASSED", "self_healed": True, "retry_passed": False},
        {"test_id": "CRT-002", "title": "Portfolio rebalancing",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
        {"test_id": "CRT-003", "title": "Risk profile update",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
    ],
}

AGENT27_TWO_FLAKY = {
    "tests_executed": 3,
    "crt_execution_verdict": "PASS",
    "crt_results": [
        {"test_id": "CRT-001", "title": "Check A",
         "status": "PASSED", "self_healed": True, "retry_passed": False},
        {"test_id": "CRT-002", "title": "Check B",
         "status": "PASSED", "self_healed": False, "retry_passed": True},
        {"test_id": "CRT-003", "title": "Check C",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
    ],
}

AGENT27_MANY_FLAKY = {
    "tests_executed": 4,
    "crt_execution_verdict": "PASS",
    "crt_results": [
        {"test_id": f"CRT-{i:03d}", "title": f"Test {i}",
         "status": "PASSED", "self_healed": True, "retry_passed": False}
        for i in range(1, 5)  # 4 flaky → exceeds quarantine threshold of 3
    ],
}

AGENT27_INTERMITTENT = {
    "tests_executed": 2,
    "crt_execution_verdict": "PASS",
    "crt_results": [
        {"test_id": "CRT-001", "title": "Intermittent check",
         "status": "INTERMITTENT", "self_healed": False, "retry_passed": False},
        {"test_id": "CRT-002", "title": "Stable check",
         "status": "PASSED", "self_healed": False, "retry_passed": False},
    ],
}

AGENT27_EMPTY = {
    "tests_executed": 0,
    "crt_results": [],
}

MOCK_TRACE_CLEAN = {
    "narrative": "No flaky tests detected. All 3 CRT tests ran deterministically without locator changes.",
    "flaky_concern": "none",
}

MOCK_TRACE_FLAKY = {
    "narrative": "CRT-001 self-healed — likely locator drift from UI change. QE must verify the selector is stable.",
    "flaky_concern": "locator_drift",
}

MOCK_TRACE_QUARANTINE = {
    "narrative": "4 tests flagged as flaky — excessive self-healing detected. QE must quarantine all 4 tests and investigate root cause before release.",
    "flaky_concern": "excessive_flakiness",
}


# ── Deterministic flaky detection tests ───────────────────────────────────────

class TestDetectFlakyTests:
    def test_no_heals_no_retries_gives_pass(self):
        flaky, quarantine, verdict = _detect_flaky_tests(AGENT27_CLEAN)
        assert len(flaky) == 0
        assert verdict == "PASS"
        assert len(quarantine) == 0

    def test_self_healed_test_is_flagged_as_flaky(self):
        flaky, _, _ = _detect_flaky_tests(AGENT27_ONE_FLAKY)
        assert "CRT-001" in flaky

    def test_retry_passed_test_is_flagged_as_flaky(self):
        flaky, _, _ = _detect_flaky_tests(AGENT27_TWO_FLAKY)
        assert "CRT-002" in flaky  # retry_passed=True

    def test_one_flaky_gives_warn(self):
        _, _, verdict = _detect_flaky_tests(AGENT27_ONE_FLAKY)
        assert verdict == "WARN"

    def test_two_flaky_gives_warn(self):
        _, _, verdict = _detect_flaky_tests(AGENT27_TWO_FLAKY)
        assert verdict == "WARN"

    def test_three_or_more_flaky_gives_quarantine_required(self):
        flaky, _, verdict = _detect_flaky_tests(AGENT27_MANY_FLAKY)
        assert verdict == "QUARANTINE_REQUIRED"
        assert len(flaky) >= 3

    def test_quarantine_list_populated_when_threshold_exceeded(self):
        _, quarantine, verdict = _detect_flaky_tests(AGENT27_MANY_FLAKY)
        assert verdict == "QUARANTINE_REQUIRED"
        assert len(quarantine) >= 3

    def test_quarantine_empty_when_below_threshold(self):
        _, quarantine, _ = _detect_flaky_tests(AGENT27_ONE_FLAKY)
        assert len(quarantine) == 0

    def test_intermittent_status_flagged_as_flaky(self):
        flaky, _, verdict = _detect_flaky_tests(AGENT27_INTERMITTENT)
        assert "CRT-001" in flaky
        assert verdict == "WARN"

    def test_empty_results_gives_pass(self):
        flaky, quarantine, verdict = _detect_flaky_tests(AGENT27_EMPTY)
        assert len(flaky) == 0
        assert verdict == "PASS"

    def test_no_upstream_data_gives_pass(self):
        flaky, quarantine, verdict = _detect_flaky_tests(None)
        assert verdict == "PASS"
        assert len(flaky) == 0

    def test_stable_tests_not_flagged(self):
        flaky, _, _ = _detect_flaky_tests(AGENT27_CLEAN)
        assert "CRT-001" not in flaky
        assert "CRT-002" not in flaky

    def test_exactly_three_flaky_gives_quarantine_required(self):
        agent27_three = {
            "tests_executed": 3,
            "crt_results": [
                {"test_id": f"CRT-{i:03d}", "status": "PASSED",
                 "self_healed": True, "retry_passed": False}
                for i in range(1, 4)
            ],
        }
        flaky, quarantine, verdict = _detect_flaky_tests(agent27_three)
        assert verdict == "QUARANTINE_REQUIRED"
        assert len(quarantine) == 3


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_crt_results_available_scores_well(self):
        score, _ = _compute_confidence(AGENT27_CLEAN, 0)
        assert score >= 65

    def test_no_crt_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT27_CLEAN, 0)
        score_without, _ = _compute_confidence(None, 0)
        assert score_with > score_without

    def test_no_flaky_tests_boosts_confidence(self):
        score_clean, _ = _compute_confidence(AGENT27_CLEAN, 0)
        score_flaky, _ = _compute_confidence(AGENT27_MANY_FLAKY, 4)
        assert score_clean > score_flaky

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT27_CLEAN, 0)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, 10)
        assert score >= 20

    def test_no_tests_executed_reduces_score(self):
        score_with, _ = _compute_confidence(AGENT27_CLEAN, 0)
        score_empty, _ = _compute_confidence(AGENT27_EMPTY, 0)
        assert score_with > score_empty

    def test_crt_results_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT27_CLEAN, 0)
        assert "crt_results_available" in signals

    def test_crt_results_available_stores_count(self):
        _, signals = _compute_confidence(AGENT27_CLEAN, 0)
        assert signals["crt_results_available"] == 3

    def test_no_tests_executed_key_in_signals(self):
        _, signals = _compute_confidence(AGENT27_EMPTY, 0)
        assert "no_tests_executed" in signals

    def test_no_crt_data_key_in_signals(self):
        _, signals = _compute_confidence(None, 0)
        assert "no_crt_data" in signals

    def test_no_flaky_tests_key_in_signals(self):
        _, signals = _compute_confidence(AGENT27_CLEAN, 0)
        assert "no_flaky_tests" in signals

    def test_high_flakiness_key_in_signals(self):
        _, signals = _compute_confidence(AGENT27_MANY_FLAKY, 4)
        assert "high_flakiness" in signals

    def test_high_flakiness_stores_count(self):
        _, signals = _compute_confidence(AGENT27_MANY_FLAKY, 4)
        assert signals["high_flakiness"] == 4


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_CLEAN}

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert result.agent_id == 38
        assert result.agent_name == "Flaky Test Hunter"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        for key in ["flaky_tests", "flaky_count", "quarantine_recommended", "flaky_verdict"]:
            assert key in result.data

    async def test_pass_when_no_flaky_tests(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_CLEAN}

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert result.data["flaky_verdict"] == "PASS"
        assert result.data["flaky_count"] == 0

    async def test_warn_when_one_flaky_test(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_ONE_FLAKY}

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FLAKY
            result = await run(state)

        assert result.data["flaky_verdict"] == "WARN"
        assert result.data["flaky_count"] == 1

    async def test_quarantine_required_when_many_flaky(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_MANY_FLAKY}

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_QUARANTINE
            result = await run(state)

        assert result.data["flaky_verdict"] == "QUARANTINE_REQUIRED"
        assert len(result.data["quarantine_recommended"]) >= 3

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_graceful_with_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert result.agent_id == 38
        assert result.data["flaky_verdict"] == "PASS"

    async def test_escalated_when_no_upstream_data(self):
        # base=62, no_crt_data→-15=47, flaky_count=0→no_flaky_tests+5=52 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_38_flaky_test_hunter.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_CLEAN
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", ["CRT-001"], [], "WARN")
        assert "FSC-2417" in msg

    def test_includes_flaky_tests(self):
        msg = _build_trace_message("FSC-2417", ["CRT-001"], [], "WARN")
        assert "CRT-001" in msg

    def test_no_flaky_shows_none_sentinel(self):
        msg = _build_trace_message("FSC-2417", [], [], "PASS")
        assert "['none']" in msg

    def test_includes_quarantine_list(self):
        msg = _build_trace_message("FSC-2417", ["CRT-001", "CRT-002", "CRT-003"],
                                   ["CRT-001", "CRT-002", "CRT-003"], "QUARANTINE_REQUIRED")
        assert "CRT-001" in msg

    def test_no_quarantine_shows_none_sentinel(self):
        msg = _build_trace_message("FSC-2417", ["CRT-001"], [], "WARN")
        assert "['none']" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", [], [], "QUARANTINE_REQUIRED")
        assert "QUARANTINE_REQUIRED" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", [], [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "flaky_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_flaky_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["flaky_concern"]["enum"] == [
            "none", "locator_drift", "timing_issue", "data_dependency", "excessive_flakiness",
        ]
