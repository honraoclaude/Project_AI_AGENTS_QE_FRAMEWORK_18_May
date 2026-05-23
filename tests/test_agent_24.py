"""Tests for Agent 24 — Test Strategy Validator (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_24_test_strategy_validator import (
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    _validate_strategy,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT19_FULL = {
    "scenario_count": 5,
    "fca_coverage_present": True,
    "gherkin_verdict": "PASS",
}

AGENT19_MINIMAL = {
    "scenario_count": 1,
    "fca_coverage_present": False,
    "gherkin_verdict": "WARN",
}

AGENT19_EMPTY = {
    "scenario_count": 0,
    "fca_coverage_present": False,
    "gherkin_verdict": "INCOMPLETE",
}

AGENT21_PASS = {
    "data_verdict": "PASS",
    "seed_record_count": 3,
    "vulnerable_profiles": ["VCI_01"],
}

AGENT21_INCOMPLETE = {
    "data_verdict": "INCOMPLETE",
    "seed_record_count": 0,
    "vulnerable_profiles": [],
}

AGENT6_MANUAL_TEST = {
    "test_tools": ["CRT", "ManualTest"],
    "coverage_target_pct": 75,
    "crt_recommended_count": 5,
}

AGENT6_HIGH_COVERAGE = {
    "test_tools": ["CRT"],
    "coverage_target_pct": 90,
    "crt_recommended_count": 8,
}

AGENT19_FULL_WITH_VC = {
    "scenario_count": 5,
    "fca_coverage_present": True,
    "gherkin_verdict": "PASS",
    "vulnerable_customer_coverage_present": True,
}

AGENT23_PASS = {"development_verdict": "PASS", "critical_failures": []}
AGENT23_FAIL = {"development_verdict": "FAIL", "critical_failures": ["Coverage FAIL"]}

MOCK_TRACE_PASS = {
    "narrative": "Test strategy is adequate. 5 scenarios cover all ACs with FCA negative tests.",
    "strategy_concern": "none",
}

MOCK_TRACE_FAIL = {
    "narrative": "Insufficient scenarios for HIGH-FCA story. FCA negative tests missing.",
    "strategy_concern": "missing_fca_coverage",
}


# ── Deterministic validation tests ────────────────────────────────────────────

class TestStrategyValidation:
    def test_full_strategy_passes(self):
        valid, verdict, gaps, fca_covered = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_FULL, AGENT21_PASS, AGENT23_PASS
        )
        assert verdict == "PASS"
        assert valid is True
        assert len(gaps) == 0

    def test_insufficient_scenarios_for_high_fca_fails(self):
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_MINIMAL, AGENT21_PASS, AGENT23_PASS
        )
        assert any("Insufficient" in g for g in gaps)

    def test_high_fca_missing_fca_scenarios_fails(self):
        agent19_no_fca = {**AGENT19_FULL, "fca_coverage_present": False}
        valid, verdict, gaps, fca_covered = _validate_strategy(
            AGENT3_HIGH, None, agent19_no_fca, AGENT21_PASS, AGENT23_PASS
        )
        assert fca_covered is False
        assert any("negative" in g or "FCA" in g for g in gaps)
        assert verdict == "FAIL"

    def test_low_fca_2_scenarios_passes(self):
        agent19_low = {"scenario_count": 2, "fca_coverage_present": False, "gherkin_verdict": "PASS"}
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_LOW, None, agent19_low, AGENT21_PASS, AGENT23_PASS
        )
        assert verdict == "PASS"

    def test_incomplete_test_data_adds_gap(self):
        _, _, gaps, _ = _validate_strategy(
            AGENT3_LOW, None, AGENT19_FULL, AGENT21_INCOMPLETE, AGENT23_PASS
        )
        assert any("INCOMPLETE" in g for g in gaps)

    def test_dev_phase_fail_adds_critical_gap(self):
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_FULL, AGENT21_PASS, AGENT23_FAIL
        )
        assert any("Development phase FAILED" in g for g in gaps)
        assert verdict == "FAIL"
        assert valid is False

    def test_high_fca_missing_vulnerable_profiles_adds_gap(self):
        agent21_no_vci = {**AGENT21_PASS, "vulnerable_profiles": []}
        _, _, gaps, _ = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_FULL, agent21_no_vci, AGENT23_PASS
        )
        assert any("Vulnerable Customer" in g for g in gaps)

    def test_no_upstream_data_degrades_gracefully(self):
        valid, verdict, gaps, fca_covered = _validate_strategy(
            None, None, None, None, None
        )
        assert isinstance(verdict, str)
        assert isinstance(valid, bool)

    def test_warn_strategy_is_still_valid(self):
        agent21_warn = {**AGENT21_PASS, "data_verdict": "WARN"}
        valid, verdict, _, _ = _validate_strategy(
            AGENT3_LOW, None, AGENT19_FULL, agent21_warn, AGENT23_PASS
        )
        assert verdict == "WARN"
        assert valid is True

    def test_medium_fca_insufficient_scenarios_fails(self):
        agent19_few = {"scenario_count": 2, "fca_coverage_present": True, "gherkin_verdict": "WARN"}
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_MEDIUM, None, agent19_few, AGENT21_PASS, AGENT23_PASS
        )
        assert verdict == "FAIL"
        assert valid is False
        assert any("Insufficient" in g for g in gaps)

    def test_medium_fca_missing_fca_scenarios_fails(self):
        agent19_no_fca = {**AGENT19_FULL, "fca_coverage_present": False}
        _, verdict, gaps, fca_covered = _validate_strategy(
            AGENT3_MEDIUM, None, agent19_no_fca, AGENT21_PASS, AGENT23_PASS
        )
        assert verdict == "FAIL"
        assert fca_covered is False
        assert any("FCA" in g for g in gaps)

    def test_incomplete_test_data_is_critical_failure(self):
        valid, verdict, _, _ = _validate_strategy(
            AGENT3_LOW, None, AGENT19_FULL, AGENT21_INCOMPLETE, AGENT23_PASS
        )
        assert verdict == "FAIL"
        assert valid is False

    def test_warn_data_verdict_adds_non_critical_gap(self):
        agent21_warn = {**AGENT21_PASS, "data_verdict": "WARN", "vulnerable_profiles": ["VCI_01"]}
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_LOW, None, AGENT19_FULL, agent21_warn, AGENT23_PASS
        )
        assert verdict == "WARN"
        assert valid is True
        assert any("WARN" in g or "gaps" in g.lower() for g in gaps)

    def test_high_fca_missing_vulnerable_profiles_verdict_is_warn(self):
        agent21_no_vci = {**AGENT21_PASS, "vulnerable_profiles": []}
        valid, verdict, _, _ = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_FULL, agent21_no_vci, AGENT23_PASS
        )
        assert verdict == "WARN"
        assert valid is True


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        assert score >= 70

    def test_no_gherkin_scenarios_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        score_without, _ = _compute_confidence(AGENT3_HIGH, AGENT19_EMPTY, AGENT21_PASS, True)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, False)
        assert score >= 20

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        assert "fca_classification_available" in signals

    def test_gherkin_scenarios_available_key_and_value(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        assert signals["gherkin_scenarios_available"] == 5

    def test_no_gherkin_scenarios_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT19_EMPTY, AGENT21_PASS, True)
        assert "no_gherkin_scenarios" in signals

    def test_no_gherkin_agent_data_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, AGENT21_PASS, True)
        assert "no_gherkin_agent_data" in signals

    def test_test_data_strategy_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, True)
        assert "test_data_strategy_available" in signals

    def test_strategy_invalid_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT19_FULL, AGENT21_PASS, False)
        assert "strategy_invalid" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["19"] = {"data": AGENT19_FULL}
        state["agent_results"]["21"] = {"data": AGENT21_PASS}

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 24
        assert result.agent_name == "Test Strategy Validator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["strategy_valid", "strategy_verdict", "strategy_gaps",
                    "fca_scenario_coverage"]:
            assert key in result.data

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=65, no agent19→-8, valid=False→-8 = 49 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_24_test_strategy_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── REQ-16: Agent 06 wiring + VC coverage check ───────────────────────────────

class TestAgentSixAndVCCoverageREQ16:
    def test_manual_test_in_agent6_gives_informational_note_not_fail(self):
        """ManualTest in test_tools → informational gap, strategy still PASS if otherwise clean."""
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_LOW, AGENT6_MANUAL_TEST, AGENT19_FULL, AGENT21_PASS, AGENT23_PASS
        )
        manual_notes = [g for g in gaps if "ManualTest" in g or "manual" in g.lower()]
        assert len(manual_notes) >= 1
        # ManualTest note is informational only — no FAIL
        assert verdict != "FAIL"
        assert valid is True

    def test_high_coverage_target_with_few_scenarios_adds_gap(self):
        """coverage_target ≥ 85 + <3 scenarios → gap added."""
        agent19_few = {"scenario_count": 2, "fca_coverage_present": True, "gherkin_verdict": "PASS",
                       "vulnerable_customer_coverage_present": False}
        _, _, gaps, _ = _validate_strategy(
            AGENT3_LOW, AGENT6_HIGH_COVERAGE, agent19_few, AGENT21_PASS, AGENT23_PASS
        )
        coverage_gaps = [g for g in gaps if "coverage" in g.lower() and "target" in g.lower()]
        assert len(coverage_gaps) >= 1

    def test_high_fca_vc_impact_vc_coverage_missing_is_critical(self):
        """HIGH-FCA + vulnerable_customer_impact + vc_coverage_present=False → FAIL."""
        agent19_no_vc = {
            "scenario_count": 5,
            "fca_coverage_present": True,
            "gherkin_verdict": "PASS",
            "vulnerable_customer_coverage_present": False,
        }
        agent21_with_vc_profiles = {**AGENT21_PASS, "vulnerable_profiles": ["VCI_01"]}
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_HIGH, None, agent19_no_vc, agent21_with_vc_profiles, AGENT23_PASS
        )
        assert verdict == "FAIL"
        assert valid is False
        vc_gaps = [g for g in gaps if "Vulnerable Customer" in g and "scenario" in g.lower()]
        assert len(vc_gaps) >= 1

    def test_vc_coverage_present_does_not_add_vc_gap(self):
        """HIGH-FCA + vc_coverage_present=True → no VC scenario gap."""
        agent21_with_vc = {**AGENT21_PASS, "vulnerable_profiles": ["VCI_01"]}
        _, _, gaps, _ = _validate_strategy(
            AGENT3_HIGH, None, AGENT19_FULL_WITH_VC, agent21_with_vc, AGENT23_PASS
        )
        vc_scenario_gaps = [g for g in gaps if "FG21/1" in g]
        assert len(vc_scenario_gaps) == 0

    def test_low_fca_vc_impact_not_critical(self):
        """LOW-FCA: VC coverage missing is not a FAIL (FCA tier doesn't require it)."""
        agent19_no_vc = {
            "scenario_count": 2, "fca_coverage_present": False,
            "gherkin_verdict": "PASS", "vulnerable_customer_coverage_present": False,
        }
        # LOW FCA doesn't trigger VC check
        valid, verdict, gaps, _ = _validate_strategy(
            AGENT3_LOW, None, agent19_no_vc, AGENT21_PASS, AGENT23_PASS
        )
        assert verdict != "FAIL"


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "HIGH" in msg

    def test_includes_scenario_count(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "5" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "Verdict: PASS" in msg

    def test_fca_coverage_in_message(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "True" in msg

    def test_no_gaps_shows_none(self):
        msg = _build_trace_message("FSC-2417", AGENT3_HIGH, AGENT19_FULL, [], "PASS")
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, None, [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "strategy_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_strategy_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["strategy_concern"]["enum"] == [
            "none", "insufficient_scenarios", "missing_fca_coverage", "no_test_data", "multiple"
        ]
