"""Tests for Agent 43 — Smoke-on-Staging (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_43_smoke_on_staging import (
    _build_trace_message,
    _compute_confidence,
    _run_smoke_tests,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT32_HIGH = {"regression_risk_level": "HIGH", "recommended_regression_suite": "FULL"}
AGENT32_MEDIUM = {"regression_risk_level": "MEDIUM", "recommended_regression_suite": "REGRESSION"}
AGENT32_LOW = {"regression_risk_level": "LOW", "recommended_regression_suite": "SMOKE"}

AGENT42_PASS = {"dry_run_success": True, "dry_run_verdict": "PASS"}
AGENT42_FAIL = {"dry_run_success": False, "dry_run_verdict": "FAIL"}
AGENT42_SKIPPED = {"dry_run_success": False, "dry_run_verdict": "SKIPPED"}

MOCK_TRACE_PASS = {
    "narrative": "SMOKE suite ran 5 tests on staging. All passed. Story cleared for production deployment.",
    "smoke_concern": "none",
}
MOCK_TRACE_SKIPPED = {
    "narrative": "Smoke tests skipped — dry-run did not complete successfully.",
    "smoke_concern": "dry_run_not_done",
}
MOCK_TRACE_FULL = {
    "narrative": "FULL suite ran 20 tests on staging due to HIGH regression risk. All passed.",
    "smoke_concern": "none",
}


# ── Deterministic smoke test simulation tests ─────────────────────────────────

class TestRunSmokeTests:
    def test_dry_run_pass_low_risk_gives_smoke_suite(self):
        passed, count, failed, suite, verdict = _run_smoke_tests(AGENT32_LOW, AGENT42_PASS)
        assert suite == "SMOKE"
        assert count == 5
        assert verdict == "PASS"
        assert passed is True

    def test_dry_run_pass_medium_risk_gives_regression_suite(self):
        _, count, _, suite, verdict = _run_smoke_tests(AGENT32_MEDIUM, AGENT42_PASS)
        assert suite == "REGRESSION"
        assert count == 10
        assert verdict == "PASS"

    def test_dry_run_pass_high_risk_gives_full_suite(self):
        _, count, _, suite, verdict = _run_smoke_tests(AGENT32_HIGH, AGENT42_PASS)
        assert suite == "FULL"
        assert count == 20
        assert verdict == "PASS"

    def test_dry_run_fail_gives_skipped(self):
        passed, count, _, _, verdict = _run_smoke_tests(AGENT32_LOW, AGENT42_FAIL)
        assert verdict == "SKIPPED"
        assert passed is False
        assert count == 0

    def test_dry_run_skipped_gives_skipped(self):
        _, _, _, _, verdict = _run_smoke_tests(AGENT32_LOW, AGENT42_SKIPPED)
        assert verdict == "SKIPPED"

    def test_no_upstream_data_gives_skipped(self):
        passed, count, _, _, verdict = _run_smoke_tests(None, None)
        assert verdict == "SKIPPED"
        assert passed is False

    def test_no_failures_in_passing_suite(self):
        _, _, failed, _, _ = _run_smoke_tests(AGENT32_LOW, AGENT42_PASS)
        assert failed == 0

    def test_regression_risk_level_key_high_gives_full_suite(self):
        """REQ-27: Agent 43 reads regression_risk_level (not regression_risk)."""
        _, count, _, suite, _ = _run_smoke_tests(AGENT32_HIGH, AGENT42_PASS)
        assert suite == "FULL"
        assert count == 20

    def test_regression_risk_level_key_medium_gives_regression_suite(self):
        """REQ-27: Agent 43 reads recommended_regression_suite (not recommended_suite)."""
        _, count, _, suite, _ = _run_smoke_tests(AGENT32_MEDIUM, AGENT42_PASS)
        assert suite == "REGRESSION"
        assert count == 10

    def test_old_wrong_keys_default_to_smoke(self):
        """REQ-27: Old keys regression_risk / recommended_suite must not be read."""
        old_keys_data = {"regression_risk": "HIGH", "recommended_suite": "FULL"}
        _, count, _, suite, _ = _run_smoke_tests(old_keys_data, AGENT42_PASS)
        # Old keys not recognised → defaults to LOW/SMOKE
        assert suite == "SMOKE"
        assert count == 5

    def test_suite_type_full_alone_gives_full_suite(self):
        # regression_risk=LOW but suite_type=FULL → suite_type arm fires independently
        agent32 = {"regression_risk_level": "LOW", "recommended_regression_suite": "FULL"}
        _, count, _, suite, _ = _run_smoke_tests(agent32, AGENT42_PASS)
        assert suite == "FULL"
        assert count == 20

    def test_regression_risk_high_alone_gives_full_suite(self):
        # regression_risk=HIGH but suite_type=SMOKE → regression_risk arm fires independently
        agent32 = {"regression_risk_level": "HIGH", "recommended_regression_suite": "SMOKE"}
        _, count, _, suite, _ = _run_smoke_tests(agent32, AGENT42_PASS)
        assert suite == "FULL"
        assert count == 20

    def test_suite_type_regression_alone_gives_regression_suite(self):
        # regression_risk=LOW but suite_type=REGRESSION → suite_type MEDIUM arm fires independently
        agent32 = {"regression_risk_level": "LOW", "recommended_regression_suite": "REGRESSION"}
        _, count, _, suite, _ = _run_smoke_tests(agent32, AGENT42_PASS)
        assert suite == "REGRESSION"
        assert count == 10


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_dry_run_data_available_scores_well(self):
        score, _ = _compute_confidence(AGENT32_LOW, AGENT42_PASS, True)
        assert score >= 65

    def test_no_dry_run_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT32_LOW, AGENT42_PASS, True)
        score_without, _ = _compute_confidence(None, None, False)
        assert score_with > score_without

    def test_smoke_failed_reduces_confidence(self):
        score_pass, _ = _compute_confidence(AGENT32_LOW, AGENT42_PASS, True)
        score_fail, _ = _compute_confidence(AGENT32_LOW, AGENT42_FAIL, False)
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT32_HIGH, AGENT42_PASS, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, False)
        assert score >= 20

    def test_dry_run_data_available_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT42_PASS, True)
        assert "dry_run_data_available" in signals

    def test_no_dry_run_data_key_in_signals(self):
        _, signals = _compute_confidence(None, None, False)
        assert "no_dry_run_data" in signals

    def test_regression_risk_data_key_in_signals(self):
        _, signals = _compute_confidence(AGENT32_LOW, AGENT42_PASS, True)
        assert "regression_risk_data" in signals

    def test_smoke_tests_passed_key_in_signals(self):
        _, signals = _compute_confidence(AGENT32_LOW, AGENT42_PASS, True)
        assert "smoke_tests_passed" in signals

    def test_smoke_tests_failed_or_skipped_key_in_signals(self):
        _, signals = _compute_confidence(None, None, False)
        assert "smoke_tests_failed_or_skipped" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["32"] = {"data": AGENT32_LOW}
        state["agent_results"]["42"] = {"data": AGENT42_PASS}

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 43
        assert result.agent_name == "Smoke-on-Staging"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        for key in ["smoke_tests_passed", "smoke_test_count", "smoke_failed",
                    "smoke_suite", "smoke_verdict"]:
            assert key in result.data

    async def test_pass_with_dry_run_success_and_low_risk(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["32"] = {"data": AGENT32_LOW}
        state["agent_results"]["42"] = {"data": AGENT42_PASS}

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["smoke_verdict"] == "PASS"
        assert result.data["smoke_tests_passed"] is True
        assert result.data["smoke_suite"] == "SMOKE"

    async def test_full_suite_for_high_risk(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["32"] = {"data": AGENT32_HIGH}
        state["agent_results"]["42"] = {"data": AGENT42_PASS}

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FULL
            result = await run(state)

        assert result.data["smoke_suite"] == "FULL"
        assert result.data["smoke_test_count"] == 20

    async def test_skipped_when_dry_run_failed(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["42"] = {"data": AGENT42_FAIL}

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.data["smoke_verdict"] == "SKIPPED"
        assert result.data["smoke_tests_passed"] is False

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=60, no_dry_run_data→-10=50, smoke_tests_failed_or_skipped→-5=45 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_43_smoke_on_staging.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_SKIPPED
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, 5, 0, "SMOKE", "PASS")
        assert "FSC-2417" in msg

    def test_includes_suite(self):
        msg = _build_trace_message("FSC-001", True, 20, 0, "FULL", "PASS")
        assert "FULL" in msg

    def test_includes_test_count(self):
        msg = _build_trace_message("FSC-001", True, 10, 0, "REGRESSION", "PASS")
        assert "10" in msg

    def test_includes_failed_count(self):
        msg = _build_trace_message("FSC-001", False, 5, 2, "SMOKE", "FAIL")
        assert "2" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-001", False, 0, 0, "SMOKE", "SKIPPED")
        assert "SKIPPED" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-001", True, 5, 0, "SMOKE", "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "smoke_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_smoke_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["smoke_concern"]["enum"] == [
            "none", "smoke_failures", "full_suite_failures",
            "dry_run_not_done", "regression_risk_unresolved",
        ]
