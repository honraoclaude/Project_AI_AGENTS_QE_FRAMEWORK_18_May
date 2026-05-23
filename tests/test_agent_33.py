"""Tests for Agent 33 — Test Coverage Analyser (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_33_test_coverage_analyser import (
    _analyse_coverage,
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

AGENT5_FOUR_ACS = {"ac_count": 4}
AGENT5_TWO_ACS  = {"ac_count": 2}

AGENT19_FIVE   = {"scenario_count": 5}   # → 100% gherkin
AGENT19_THREE  = {"scenario_count": 3}   # → 60% gherkin
AGENT19_ZERO   = {"scenario_count": 0}

AGENT26_FULL   = {"automation_coverage": 100.0}
AGENT26_PARTIAL = {"automation_coverage": 60.0}
AGENT26_NONE   = {"automation_coverage": 0.0}

AGENT27_MANY   = {"tests_executed": 5}
AGENT27_FEW    = {"tests_executed": 1}

AGENT29_FOUR   = {"uat_test_count": 4}
AGENT29_TWO    = {"uat_test_count": 2}
AGENT29_ZERO   = {"uat_test_count": 0}

AGENT30_PRESENT = {"fca_scenario_count": 3}
AGENT30_ZERO    = {"fca_scenario_count": 0}

MOCK_TRACE_PASS = {
    "narrative": "Test coverage is 93.3% overall with full Gherkin, CRT, and UAT coverage. All ACs are covered.",
    "coverage_concern": "none",
}

MOCK_TRACE_FAIL = {
    "narrative": "Test coverage is 65.0% — below the HIGH-risk threshold of 90%. UAT and FCA scenarios are missing.",
    "coverage_concern": "below_threshold",
}

MOCK_TRACE_WARN = {
    "narrative": "Test coverage is 80.0% but 2 ACs remain uncovered. QE engineer must add UAT tests for AC3 and AC4.",
    "coverage_concern": "fca_scenarios_uncovered",
}


# ── Deterministic coverage analysis tests ─────────────────────────────────────

class TestAnalyseCoverage:
    def test_full_coverage_all_sources_gives_pass(self):
        _, _, _, verdict = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "PASS"

    def test_no_data_gives_zero_pct(self):
        pct, _, _, _ = _analyse_coverage(None, None, None, None, None, None, None, None)
        assert pct == 0.0

    def test_no_data_fails_low_threshold(self):
        _, _, _, verdict = _analyse_coverage(None, None, None, None, None, None, None, None)
        assert verdict == "FAIL"

    def test_five_gherkin_scenarios_contributes_100_pct(self):
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, None, None, None, None,
        )
        assert pct == 100.0

    def test_three_gherkin_scenarios_contributes_60_pct(self):
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_THREE, None, None, None, None,
        )
        assert abs(pct - 60.0) < 0.1

    def test_crt_automation_pct_included_in_average(self):
        pct_with, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, AGENT26_FULL, None, None, None,
        )
        pct_without, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, None, None, None, None,
        )
        assert pct_with >= pct_without

    def test_zero_crt_pct_not_included_in_average(self):
        pct_with, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, AGENT26_NONE, None, None, None,
        )
        pct_without, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, None, None, None, None,
        )
        # AGENT26_NONE (0.0) should not be added as a component
        assert pct_with == pct_without

    def test_uat_with_known_ac_count_correct_pct(self):
        # 2 UAT / 4 ACs = 50%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, None, None, None, AGENT29_TWO, None,
        )
        assert abs(pct - 50.0) < 0.1

    def test_uat_without_ac_count_uses_flat_multiplier(self):
        # uat_count=2, no ac_count → min(100, 2 * 25) = 50%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, None, None, None, AGENT29_TWO, None,
        )
        assert abs(pct - 50.0) < 0.1

    def test_uat_capped_at_100_pct(self):
        # uat_count=5, ac_count=2 → min(100, 250%) = 100%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TWO_ACS, None, None, None, {"uat_test_count": 5}, None,
        )
        assert pct == 100.0

    def test_uncovered_acs_identified(self):
        # 2 ACs expected, 0 UAT → AC1, AC2 uncovered
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TWO_ACS, None, None, None, AGENT29_ZERO, None,
        )
        assert "AC1" in uncovered
        assert "AC2" in uncovered

    def test_partial_uat_identifies_remaining_uncovered_acs(self):
        # 4 ACs, 2 UAT → AC3, AC4 uncovered
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, None, None, None, AGENT29_TWO, None,
        )
        assert "AC3" in uncovered
        assert "AC4" in uncovered
        assert "AC1" not in uncovered

    def test_all_acs_covered_no_uncovered_list(self):
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, None, None, None, AGENT29_FOUR, None,
        )
        assert len(uncovered) == 0

    def test_high_fca_below_90_threshold_gives_fail(self):
        # HIGH requires 90%; 3 gherkin = 60% → FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, None, None, AGENT19_THREE, None, None, None, None,
        )
        assert verdict == "FAIL"

    def test_high_fca_above_90_no_gaps_gives_pass(self):
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "PASS"

    def test_medium_fca_below_85_gives_fail(self):
        # MEDIUM requires 85%; 3 gherkin = 60% → FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_MEDIUM, None, None, AGENT19_THREE, None, None, None, None,
        )
        assert verdict == "FAIL"

    def test_high_fca_no_fca_scenarios_gives_fail(self):
        # REQ-24 Gap 3: HIGH-FCA with zero FCA scenarios → FAIL (compliance gap)
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, AGENT29_FOUR, AGENT30_ZERO,
        )
        assert verdict == "FAIL"

    def test_medium_fca_no_fca_scenarios_gives_warn(self):
        _, _, _, verdict = _analyse_coverage(
            AGENT3_MEDIUM, None, None, AGENT19_FIVE, AGENT26_FULL,
            None, AGENT29_FOUR, AGENT30_ZERO,
        )
        assert verdict == "WARN"

    def test_low_fca_no_fca_scenarios_is_not_warn(self):
        # LOW classification — FCA scenario absence not a warn trigger
        _, _, _, verdict = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TWO_ACS, AGENT19_FIVE, AGENT26_FULL,
            None, AGENT29_TWO, AGENT30_ZERO,
        )
        assert verdict == "PASS"

    def test_uncovered_acs_with_sufficient_pct_gives_warn(self):
        # 100% gherkin + full CRT but only 1 UAT / 4 ACs → uncovered → WARN
        _, _, _, verdict = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, {"uat_test_count": 1}, AGENT30_PRESENT,
        )
        assert verdict == "WARN"

    def test_by_type_dict_has_expected_keys(self):
        _, by_type, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, AGENT26_FULL, AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        for key in ["gherkin", "crt_automation_pct", "crt_executed", "uat", "fca_regulatory"]:
            assert key in by_type

    def test_overall_pct_capped_at_100(self):
        # Even if all components are 100, result should not exceed 100
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TWO_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, AGENT29_TWO, AGENT30_PRESENT,
        )
        assert pct <= 100.0

    def test_medium_fca_vc_impact_no_vc_coverage_gives_fail(self):
        # MEDIUM FCA + VC impact required + vc_coverage_present=False → FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_MEDIUM, AGENT4_VC_IMPACT, AGENT5_FOUR_ACS, AGENT19_WITHOUT_VC_FLAG,
            AGENT26_FULL, None, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "FAIL"

    def test_by_type_has_vulnerable_customer_covered_key(self):
        _, by_type, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, None, None, None, None,
        )
        assert "vulnerable_customer_covered" in by_type

    def test_gherkin_capped_at_100_when_scenarios_exceed_ac_count(self):
        # 5 gherkin / 2 ACs = 250% → capped at 100%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TWO_ACS, AGENT19_FIVE, None, None, None, None,
        )
        assert pct == 100.0

    def test_uncovered_ac_clause_missing_description_uses_fallback(self):
        agent5_sparse = {
            "ac_count": 3,
            "ac_clauses": [
                {"description": "AC1: Block unsuitable products"},
                {},  # no description → fallback "AC2"
                {},  # no description → fallback "AC3"
            ],
        }
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, agent5_sparse, None, None, None, AGENT29_ZERO, None,
        )
        assert "AC2" in uncovered
        assert "AC3" in uncovered


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_three_plus_agents_available_scores_well(self):
        score, _ = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
        )
        assert score >= 70

    def test_no_test_agents_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
        )
        score_without, _ = _compute_confidence(None, None, None, None, 0.0)
        assert score_with > score_without

    def test_high_coverage_adds_signal(self):
        score_high, _ = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 85.0,
        )
        score_low, _ = _compute_confidence(
            AGENT19_THREE, AGENT26_PARTIAL, AGENT29_TWO, AGENT30_PRESENT, 40.0,
        )
        assert score_high > score_low

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 100.0,
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, 0.0)
        assert score >= 20

    def test_partial_agents_gives_intermediate_score(self):
        score_full, _ = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
        )
        score_one, _ = _compute_confidence(AGENT19_FIVE, None, None, None, 60.0)
        assert score_full > score_one

    def test_comprehensive_test_data_key_and_value(self):
        _, signals = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
        )
        assert "comprehensive_test_data" in signals
        assert signals["comprehensive_test_data"] == 4

    def test_partial_test_data_key_and_value(self):
        _, signals = _compute_confidence(AGENT19_FIVE, None, None, None, 60.0)
        assert "partial_test_data" in signals
        assert signals["partial_test_data"] == 1

    def test_no_test_data_key_in_signals(self):
        _, signals = _compute_confidence(None, None, None, None, 0.0)
        assert "no_test_data" in signals

    def test_high_coverage_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 85.0,
        )
        assert "high_coverage" in signals

    def test_low_coverage_key_in_signals(self):
        _, signals = _compute_confidence(None, None, None, None, 30.0)
        assert "low_coverage" in signals

    def test_crt_scenario_truncated_key_in_signals(self):
        _, signals = _compute_confidence(
            AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
            scenarios_truncated=True, truncated_count=2,
        )
        assert "crt_scenario_truncated" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["5"]  = {"data": AGENT5_FOUR_ACS}
        state["agent_results"]["19"] = {"data": AGENT19_FIVE}
        state["agent_results"]["26"] = {"data": AGENT26_FULL}
        state["agent_results"]["27"] = {"data": AGENT27_MANY}
        state["agent_results"]["29"] = {"data": AGENT29_FOUR}
        state["agent_results"]["30"] = {"data": AGENT30_PRESENT}

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 33
        assert result.agent_name == "Test Coverage Analyser"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["overall_coverage_pct", "coverage_by_type",
                    "uncovered_acs", "coverage_verdict"]:
            assert key in result.data

    async def test_pass_with_full_coverage(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_LOW}
        state["agent_results"]["5"]  = {"data": AGENT5_FOUR_ACS}
        state["agent_results"]["19"] = {"data": AGENT19_FIVE}
        state["agent_results"]["26"] = {"data": AGENT26_FULL}
        state["agent_results"]["29"] = {"data": AGENT29_FOUR}
        state["agent_results"]["30"] = {"data": AGENT30_PRESENT}

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["coverage_verdict"] == "PASS"
        assert result.data["overall_coverage_pct"] > 75.0

    async def test_fail_with_no_test_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["coverage_verdict"] == "FAIL"
        assert result.data["overall_coverage_pct"] == 0.0

    async def test_warn_with_uncovered_acs(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_LOW}
        state["agent_results"]["5"]  = {"data": AGENT5_FOUR_ACS}
        state["agent_results"]["19"] = {"data": AGENT19_FIVE}
        state["agent_results"]["26"] = {"data": AGENT26_FULL}
        state["agent_results"]["29"] = {"data": {"uat_test_count": 1}}

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_WARN
            result = await run(state)

        assert result.data["coverage_verdict"] == "WARN"
        assert len(result.data["uncovered_acs"]) >= 1

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_narrative_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["narrative"] == MOCK_TRACE_PASS["narrative"]

    async def test_escalated_when_no_upstream_data(self):
        # base=65, no_test_data→-10, low_coverage(0%)→-5 = 50 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_33_test_coverage_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)


# ── REQ-24: New gap tests ──────────────────────────────────────────────────────

AGENT5_TEN_ACS = {"ac_count": 10}
AGENT5_WITH_CLAUSES = {
    "ac_count": 4,
    "ac_clauses": [
        {"description": "AC1: Block unsuitable products", "scenario_type": "regulatory"},
        {"description": "AC2: Alert compliance team",    "scenario_type": "happy_path"},
        {"description": "AC3: Vulnerable customer check", "scenario_type": "vulnerable_customer"},
        {"description": "AC4: Audit trail created",       "scenario_type": "regulatory"},
    ],
}
AGENT4_VC_IMPACT    = {"fca_classification": "HIGH", "vulnerable_customer_impact": True}
AGENT4_NO_VC_IMPACT = {"fca_classification": "HIGH", "vulnerable_customer_impact": False}
AGENT19_WITH_VC_FLAG = {
    "scenario_count": 5,
    "vulnerable_customer_coverage_present": True,
}
AGENT19_WITHOUT_VC_FLAG = {
    "scenario_count": 5,
    "vulnerable_customer_coverage_present": False,
}
AGENT26_TRUNCATED = {
    "automation_coverage": 100.0,
    "scenarios_truncated": True,
    "truncated_scenario_count": 2,
}


class TestGherkinNormalisedREQ24:
    def test_5_gherkin_scenarios_10_acs_gives_50_pct_not_100(self):
        # REQ-24 Gap 1: 5 scenarios / 10 ACs = 50%, not 100%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_TEN_ACS, AGENT19_FIVE, None, None, None, None,
        )
        assert abs(pct - 50.0) < 0.1

    def test_gherkin_fallback_when_no_ac_count(self):
        # No AC count → fallback: scenario_count * 20 = 5 * 20 = 100%
        pct, _, _, _ = _analyse_coverage(
            AGENT3_LOW, None, None, AGENT19_FIVE, None, None, None, None,
        )
        assert pct == 100.0


class TestUncoveredAcsEnrichedREQ24:
    def test_uncovered_acs_use_real_descriptions_when_clauses_present(self):
        # REQ-24 Gap 2: with ac_clauses, uncovered list uses real descriptions
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_WITH_CLAUSES, None, None, None, AGENT29_TWO, None,
        )
        # 4 ACs, 2 UAT → clauses[2:] → AC3, AC4 descriptions
        assert any("Vulnerable customer" in u for u in uncovered)
        assert any("Audit trail" in u for u in uncovered)

    def test_uncovered_acs_fallback_without_clauses(self):
        # No ac_clauses → synthetic "AC3", "AC4"
        _, _, uncovered, _ = _analyse_coverage(
            AGENT3_LOW, None, AGENT5_FOUR_ACS, None, None, None, AGENT29_TWO, None,
        )
        assert "AC3" in uncovered
        assert "AC4" in uncovered


class TestHighFcaFcaCountFailREQ24:
    def test_high_fca_zero_fca_scenarios_above_threshold_gives_fail(self):
        # REQ-24 Gap 3: HIGH-FCA + fca_count=0 + coverage ≥ 90% → FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            AGENT27_MANY, AGENT29_FOUR, AGENT30_ZERO,
        )
        assert verdict == "FAIL"

    def test_medium_fca_zero_fca_scenarios_gives_warn(self):
        # MEDIUM-FCA + zero FCA scenarios → WARN (not promoted to FAIL)
        _, _, _, verdict = _analyse_coverage(
            AGENT3_MEDIUM, None, AGENT5_FOUR_ACS, AGENT19_FIVE, AGENT26_FULL,
            None, AGENT29_FOUR, AGENT30_ZERO,
        )
        assert verdict == "WARN"


class TestVulnerableCustomerCoverageREQ24:
    def test_vc_impact_true_no_vc_coverage_high_fca_gives_fail(self):
        # REQ-24 Gap 4: VC impact + no VC coverage + HIGH FCA → FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, AGENT4_VC_IMPACT, AGENT5_FOUR_ACS, AGENT19_WITHOUT_VC_FLAG,
            AGENT26_FULL, AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "FAIL"

    def test_vc_impact_true_vc_coverage_present_does_not_fail(self):
        # VC coverage present → no VC-driven FAIL
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, AGENT4_VC_IMPACT, AGENT5_FOUR_ACS, AGENT19_WITH_VC_FLAG,
            AGENT26_FULL, AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "PASS"

    def test_vc_impact_false_no_vc_coverage_is_not_fail(self):
        # VC impact=False → VC coverage absence doesn't matter
        _, _, _, verdict = _analyse_coverage(
            AGENT3_HIGH, AGENT4_NO_VC_IMPACT, AGENT5_FOUR_ACS, AGENT19_WITHOUT_VC_FLAG,
            AGENT26_FULL, AGENT27_MANY, AGENT29_FOUR, AGENT30_PRESENT,
        )
        assert verdict == "PASS"


class TestTruncationConfidencePenaltyREQ24:
    def test_scenarios_truncated_reduces_confidence(self):
        from src.agents.testing.agent_33_test_coverage_analyser import _compute_confidence
        score_clean, _   = _compute_confidence(AGENT19_FIVE, AGENT26_FULL, AGENT29_FOUR, AGENT30_PRESENT, 95.0)
        score_trunc, _   = _compute_confidence(AGENT19_FIVE, AGENT26_TRUNCATED, AGENT29_FOUR, AGENT30_PRESENT, 95.0,
                                               scenarios_truncated=True, truncated_count=2)
        assert score_clean > score_trunc


# ── Trace message unit tests ──────────────────────────────────────────────────

_SAMPLE_BY_TYPE = {
    "gherkin": 5,
    "crt_automation_pct": 100.0,
    "crt_executed": 3,
    "uat": 4,
    "fca_regulatory": 2,
    "vulnerable_customer_covered": True,
}


class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_overall_pct(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert "85.0%" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert "Verdict: PASS" in msg

    def test_includes_gherkin_count(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert "Gherkin scenarios: 5" in msg

    def test_no_uncovered_shows_none(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert "['none']" in msg

    def test_includes_uncovered_acs(self):
        msg = _build_trace_message("FSC-2417", 65.0, _SAMPLE_BY_TYPE, ["AC3", "AC4"], "WARN")
        assert "AC3" in msg
        assert "AC4" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", 85.0, _SAMPLE_BY_TYPE, [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "coverage_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_coverage_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["coverage_concern"]["enum"] == [
            "none", "below_threshold", "fca_scenarios_uncovered",
            "uat_uncovered", "multiple",
        ]
