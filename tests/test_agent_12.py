"""Tests for Agent 12 — Apex Coverage Analyser (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_12_apex_coverage import (
    _analyse_coverage,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH", "ensemble_agreement": True}
AGENT3_LOW  = {"fca_classification": "LOW",  "ensemble_agreement": True}
AGENT6_85   = {"coverage_target_pct": 85}
AGENT6_75   = {"coverage_target_pct": 75}

RESULTS_PASSING_HIGH = {"test_run_id": "run-001", "tests_run": 12, "tests_passed": 12, "tests_failed": 0, "coverage_pct": 90}
RESULTS_FAILING_HIGH = {"test_run_id": "run-002", "tests_run": 12, "tests_passed": 10, "tests_failed": 2, "coverage_pct": 82}
RESULTS_LOW_COV      = {"test_run_id": "run-003", "tests_run": 8,  "tests_passed": 8,  "tests_failed": 0, "coverage_pct": 70}
RESULTS_NO_TESTS     = {"test_run_id": "",         "tests_run": 0,  "tests_passed": 0,  "tests_failed": 0, "coverage_pct": 0}

MOCK_TRACE_PASS = {"narrative": "Coverage 90% exceeds 85% threshold. All tests passing.", "coverage_concern": "none"}
MOCK_TRACE_FAIL = {"narrative": "Coverage 70% is below 85% threshold.", "coverage_concern": "critical"}


# ── Deterministic analysis tests ──────────────────────────────────────────────

class TestCoverageAnalysis:
    def test_pass_when_coverage_exceeds_threshold(self):
        pct, threshold, _, _, verdict, gap = _analyse_coverage(RESULTS_PASSING_HIGH, AGENT3_HIGH, None)
        assert verdict == "PASS"
        assert gap == 0
        assert threshold == 85

    def test_fail_when_coverage_below_threshold(self):
        pct, threshold, _, _, verdict, gap = _analyse_coverage(RESULTS_LOW_COV, AGENT3_HIGH, None)
        assert verdict == "FAIL"
        assert gap > 0

    def test_fail_when_tests_failing_despite_coverage(self):
        pct, _, _, failed, verdict, _ = _analyse_coverage(RESULTS_FAILING_HIGH, AGENT3_HIGH, None)
        assert failed == 2
        assert verdict == "FAIL"

    def test_unknown_when_no_tests_run(self):
        _, _, tests_run, _, verdict, _ = _analyse_coverage(RESULTS_NO_TESTS, AGENT3_HIGH, None)
        assert tests_run == 0
        assert verdict == "UNKNOWN"

    def test_high_fca_uses_85_threshold(self):
        _, threshold, _, _, _, _ = _analyse_coverage(RESULTS_PASSING_HIGH, AGENT3_HIGH, None)
        assert threshold == 85

    def test_low_fca_uses_75_threshold(self):
        _, threshold, _, _, _, _ = _analyse_coverage(RESULTS_PASSING_HIGH, AGENT3_LOW, None)
        assert threshold == 75

    def test_agent6_target_overrides_fca_derived_threshold(self):
        _, threshold, _, _, _, _ = _analyse_coverage(RESULTS_PASSING_HIGH, AGENT3_HIGH, AGENT6_75)
        assert threshold == 75

    def test_low_fca_76pct_passes_75_threshold(self):
        results_76 = {**RESULTS_LOW_COV, "coverage_pct": 76}
        _, threshold, _, _, verdict, _ = _analyse_coverage(results_76, AGENT3_LOW, None)
        assert threshold == 75
        assert verdict == "PASS"


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_passing_coverage_scores_high(self):
        score, _ = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        assert score >= 75

    def test_no_tests_run_heavily_penalised(self):
        score_with, _ = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        score_without, _ = _compute_confidence(RESULTS_NO_TESTS, 0, 85, "UNKNOWN", 0)
        assert score_with > score_without

    def test_fail_verdict_reduces_confidence(self):
        score_pass, _ = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        score_fail, _ = _compute_confidence(RESULTS_LOW_COV, 70, 85, "FAIL", 8)
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(RESULTS_NO_TESTS, 0, 85, "UNKNOWN", 0)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_PASSING_HIGH
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 12
        assert result.agent_name == "Apex Coverage Analyser"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_PASSING_HIGH
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["coverage_pct", "coverage_threshold", "coverage_passed",
                    "coverage_verdict", "tests_failed"]:
            assert key in result.data

    async def test_pass_verdict_for_sufficient_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_PASSING_HIGH
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["coverage_passed"] is True
        assert result.data["coverage_verdict"] == "PASS"

    async def test_fail_verdict_for_insufficient_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_LOW_COV
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["coverage_passed"] is False
        assert result.data["coverage_verdict"] == "FAIL"

    async def test_standalone_mode_uses_default_threshold(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_PASSING_HIGH
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 12
        assert result.data["coverage_threshold"] == 75  # LOW default when no Agent 3

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_PASSING_HIGH
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
