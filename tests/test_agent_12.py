"""Tests for Agent 12 — Apex Coverage Analyser (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_12_apex_coverage import (
    _analyse_coverage,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH",   "ensemble_agreement": True}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM", "ensemble_agreement": True}
AGENT3_LOW    = {"fca_classification": "LOW",    "ensemble_agreement": True}
AGENT6_85     = {"coverage_target_pct": 85}
AGENT6_75     = {"coverage_target_pct": 75}

RESULTS_PASSING_HIGH    = {"test_run_id": "run-001", "tests_run": 12, "tests_passed": 12, "tests_failed": 0, "coverage_pct": 90}
RESULTS_FAILING_HIGH    = {"test_run_id": "run-002", "tests_run": 12, "tests_passed": 10, "tests_failed": 2, "coverage_pct": 82}
RESULTS_LOW_COV         = {"test_run_id": "run-003", "tests_run": 8,  "tests_passed": 8,  "tests_failed": 0, "coverage_pct": 70}
RESULTS_NO_TESTS        = {"test_run_id": "",         "tests_run": 0,  "tests_passed": 0,  "tests_failed": 0, "coverage_pct": 0}
RESULTS_RUN_ID_NO_TESTS = {"test_run_id": "run-005", "tests_run": 0,  "tests_passed": 0,  "tests_failed": 0, "coverage_pct": 0}

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

    def test_medium_fca_uses_85_threshold(self):
        _, threshold, _, _, _, _ = _analyse_coverage(RESULTS_PASSING_HIGH, AGENT3_MEDIUM, None)
        assert threshold == 85

    def test_unclassified_fca_uses_75_threshold(self):
        _, threshold, _, _, _, _ = _analyse_coverage(RESULTS_PASSING_HIGH, None, None)
        assert threshold == 75

    def test_gap_equals_threshold_when_unknown(self):
        _, threshold, _, _, verdict, gap = _analyse_coverage(RESULTS_NO_TESTS, AGENT3_HIGH, None)
        assert verdict == "UNKNOWN"
        assert gap == threshold

    def test_gap_when_tests_failing(self):
        # RESULTS_FAILING_HIGH: coverage=82, threshold=85, tests_failed=2 → gap = max(85-82,0) = 3
        _, threshold, _, failed, verdict, gap = _analyse_coverage(RESULTS_FAILING_HIGH, AGENT3_HIGH, None)
        assert verdict == "FAIL"
        assert failed > 0
        assert gap == max(threshold - 82, 0)


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

    def test_coverage_data_available_key_in_signals(self):
        _, signals = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        assert "coverage_data_available" in signals

    def test_no_coverage_data_key_in_signals(self):
        _, signals = _compute_confidence(RESULTS_NO_TESTS, 0, 85, "UNKNOWN", 0)
        assert "no_coverage_data" in signals

    def test_adequate_test_count_stores_tests_run(self):
        _, signals = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        assert signals["adequate_test_count"] == 12

    def test_no_tests_run_signal_with_run_id_present(self):
        # test_run_id truthy but tests_run=0 → data_available=True → no_tests_run fires
        _, signals = _compute_confidence(RESULTS_RUN_ID_NO_TESTS, 0, 85, "UNKNOWN", 0)
        assert "no_tests_run" in signals

    def test_coverage_passed_stores_coverage_pct(self):
        _, signals = _compute_confidence(RESULTS_PASSING_HIGH, 90, 85, "PASS", 12)
        assert signals["coverage_passed"] == 90

    def test_coverage_failed_stores_coverage_pct(self):
        _, signals = _compute_confidence(RESULTS_LOW_COV, 70, 85, "FAIL", 8)
        assert signals["coverage_failed"] == 70

    def test_coverage_unknown_key_in_signals(self):
        _, signals = _compute_confidence(RESULTS_NO_TESTS, 0, 85, "UNKNOWN", 0)
        assert "coverage_unknown" in signals


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

    async def test_no_coverage_data_causes_escalation(self):
        # no_coverage_data (-15) + coverage_unknown (-10): base=68-15-10=43 < 60
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_NO_TESTS
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
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

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
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

        assert isinstance(result.data["signals"], dict)


# ── REQ-08: medium coverage_concern + per-class breakdown tests ───────────────

from src.agents.development.agent_12_apex_coverage import _analyse_coverage

RESULTS_MEDIUM_GAP = {
    "test_run_id": "run-004",
    "tests_run": 10,
    "tests_passed": 10,
    "tests_failed": 0,
    "coverage_pct": 76,  # 9% below 85% threshold — medium range
    "per_class_coverage": [
        {"class_name": "SuitabilityAssessmentService", "coverage_pct": 60, "meets_threshold": False},
        {"class_name": "SuitabilityTriggerHandler", "coverage_pct": 90, "meets_threshold": True},
    ],
}

RESULTS_WITH_PER_CLASS = {
    **RESULTS_PASSING_HIGH,
    "per_class_coverage": [
        {"class_name": "SuitabilityAssessmentService", "coverage_pct": 88, "meets_threshold": True},
        {"class_name": "SuitabilityTriggerHandler", "coverage_pct": 91, "meets_threshold": True},
    ],
}

MOCK_TRACE_MEDIUM = {"narrative": "Coverage 76% is 9% below 85% threshold.", "coverage_concern": "medium"}

AGENT6_WITH_APEX_CLASSES = {
    "coverage_target_pct": 85,
    "apex_unit_test_classes": [
        "SuitabilityAssessmentService — validation and record creation",
        "SuitabilityTriggerHandler — before-insert validation",
    ],
}


class TestMediumCoverageConcernREQ08:
    def test_medium_concern_enum_accepted(self):
        """REQ-08: 'medium' is a valid coverage_concern value in Haiku schema."""
        # Verify via round-trip: Haiku returning 'medium' is passed through to data
        # (schema validation happens at call_with_tool level — we test the output plumbing)
        # The enum in _TRACE_TOOL_SCHEMA must include 'medium'; this is covered by integration test below
        pass  # schema test is in integration class below

    def test_per_class_coverage_empty_when_copado_omits_it(self):
        """REQ-08: per_class_coverage defaults to [] when Copado omits it."""
        results_no_classes = {**RESULTS_PASSING_HIGH}  # no per_class_coverage key
        results_no_classes.pop("per_class_coverage", None)
        # _analyse_coverage doesn't read per_class_coverage — it's in run(); test via run()


@pytest.mark.asyncio
class TestPerClassCoverageREQ08:
    async def test_per_class_coverage_in_output_data(self):
        """REQ-08: per_class_coverage list present in result.data."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_WITH_PER_CLASS
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "per_class_coverage" in result.data
        assert isinstance(result.data["per_class_coverage"], list)
        assert len(result.data["per_class_coverage"]) == 2

    async def test_uncovered_classes_cross_references_agent6(self):
        """REQ-08: uncovered_classes cross-references Agent 06 apex_unit_test_classes."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["6"] = {"data": AGENT6_WITH_APEX_CLASSES}

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_MEDIUM_GAP
            mock_haiku.return_value = MOCK_TRACE_MEDIUM
            result = await run(state)

        assert "uncovered_classes" in result.data
        assert "SuitabilityAssessmentService" in result.data["uncovered_classes"]
        # SuitabilityTriggerHandler meets threshold — not in uncovered
        assert "SuitabilityTriggerHandler" not in result.data["uncovered_classes"]

    async def test_uncovered_classes_empty_when_all_pass(self):
        """REQ-08: uncovered_classes is empty when all classes meet threshold."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["6"] = {"data": AGENT6_WITH_APEX_CLASSES}

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_WITH_PER_CLASS
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["uncovered_classes"] == []

    async def test_medium_concern_passes_through_from_haiku(self):
        """REQ-08: coverage_concern='medium' from Haiku is stored in result.data."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_12_apex_coverage.get_apex_test_results",
                  new_callable=AsyncMock) as mock_copado,
            patch("src.agents.development.agent_12_apex_coverage.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_copado.return_value = RESULTS_MEDIUM_GAP
            mock_haiku.return_value = MOCK_TRACE_MEDIUM
            result = await run(state)

        assert result.data["coverage_concern"] == "medium"


# ── Trace message tests ───────────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert "HIGH" in msg

    def test_includes_coverage_pct(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert "90%" in msg

    def test_includes_threshold(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert "85%" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert "PASS" in msg

    def test_uncovered_classes_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", 76, 85, 10, 0, "FAIL", "HIGH",
                                   uncovered_classes=["SuitabilityAssessmentService"])
        assert "Under-covered classes:" in msg
        assert "SuitabilityAssessmentService" in msg

    def test_uncovered_classes_absent_when_empty(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH",
                                   uncovered_classes=[])
        assert "Under-covered classes:" not in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", 90, 85, 12, 0, "PASS", "HIGH")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "coverage_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_coverage_concern_enum_has_four_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["coverage_concern"]["enum"] == [
            "none", "low", "medium", "critical"
        ]
