"""Tests for Agent 27 — CRT Execution Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_27_crt_execution import (
    _compute_confidence,
    _simulate_execution,
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
