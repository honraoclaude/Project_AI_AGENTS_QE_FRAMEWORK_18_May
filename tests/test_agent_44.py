"""Tests for Agent 44 — FCA Evidence Pack (True AI Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_44_fca_evidence_pack import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

_EMPTY_STATE = initial_story_state("FSC-2417")

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT4_PASS = {
    "consumer_duty_verdict": "PASS",
    "obligations_mapped": ["fair_outcomes", "consumer_understanding", "vulnerability"],
}
AGENT4_FAIL = {
    "consumer_duty_verdict": "FAIL",
    "obligations_mapped": [],
}

AGENT30_FULL = {
    "fca_scenario_verdict": "PASS",
    "fca_scenario_count": 4,
    "regulatory_gaps": [],
}
AGENT30_GAPS = {
    "fca_scenario_verdict": "WARN",
    "fca_scenario_count": 1,
    "regulatory_gaps": ["COBS 9 not covered"],
}

AGENT33_PASS = {"overall_coverage_pct": 92.0, "coverage_verdict": "PASS"}

AGENT36_SIGNED_OFF = {"uat_coordination_verdict": "SIGNED_OFF"}
AGENT36_NOT_REQUIRED = {"uat_coordination_verdict": "NOT_REQUIRED"}
AGENT36_PENDING = {"uat_coordination_verdict": "PENDING"}

MOCK_EVIDENCE_COMPLETE = {
    "evidence_items": [
        {"rule": "Consumer Duty PS22/9", "status": "COVERED", "evidence_ref": "AC4 + FCA-001"},
        {"rule": "COBS 9 Suitability", "status": "COVERED", "evidence_ref": "FCA-002, FCA-003"},
    ],
    "consumer_duty_covered": True,
    "regulatory_sign_off_ready": True,
    "evidence_verdict": "COMPLETE",
    "evidence_gaps": [],
    "narrative": "FCA evidence pack complete. Consumer Duty and COBS 9 suitability are fully evidenced. Story is ready for regulatory sign-off.",
}

MOCK_EVIDENCE_PARTIAL = {
    "evidence_items": [
        {"rule": "Consumer Duty PS22/9", "status": "COVERED", "evidence_ref": "AC4"},
        {"rule": "COBS 9 Suitability", "status": "MISSING", "evidence_ref": ""},
    ],
    "consumer_duty_covered": True,
    "regulatory_sign_off_ready": False,
    "evidence_verdict": "PARTIAL",
    "evidence_gaps": ["COBS 9 Suitability"],
    "narrative": "FCA evidence pack is PARTIAL. COBS 9 suitability scenarios are not covered — QE must add test evidence before release.",
}

MOCK_EVIDENCE_LOW_FCA = {
    "evidence_items": [],
    "consumer_duty_covered": True,
    "regulatory_sign_off_ready": True,
    "evidence_verdict": "COMPLETE",
    "evidence_gaps": [],
    "narrative": "FCA classification is LOW — minimal regulatory evidence required. Consumer Duty obligations confirmed. Story may proceed.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_fca_class_and_scenario_data_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF, "COMPLETE", _EMPTY_STATE)
        assert score >= 75

    def test_unknown_fca_class_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF, "COMPLETE", _EMPTY_STATE)
        score_without, _ = _compute_confidence(None, None, None, None, "MISSING", _EMPTY_STATE)
        assert score_with > score_without

    def test_complete_verdict_boosts_confidence(self):
        score_complete, _ = _compute_confidence(AGENT3_HIGH, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF, "COMPLETE", _EMPTY_STATE)
        score_missing, _ = _compute_confidence(AGENT3_HIGH, AGENT30_GAPS, AGENT33_PASS, AGENT36_PENDING, "MISSING", _EMPTY_STATE)
        assert score_complete > score_missing

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF, "COMPLETE", _EMPTY_STATE)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, "MISSING", _EMPTY_STATE)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["4"]  = {"data": AGENT4_PASS}
        state["agent_results"]["30"] = {"data": AGENT30_FULL}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_SIGNED_OFF}

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert result.agent_id == 44
        assert result.agent_name == "FCA Evidence Pack"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        for key in ["evidence_items", "consumer_duty_covered",
                    "regulatory_sign_off_ready", "evidence_verdict", "evidence_gaps"]:
            assert key in result.data

    async def test_complete_when_all_evidence_present(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["4"]  = {"data": AGENT4_PASS}
        state["agent_results"]["30"] = {"data": AGENT30_FULL}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_SIGNED_OFF}

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert result.data["evidence_verdict"] == "COMPLETE"
        assert result.data["regulatory_sign_off_ready"] is True
        assert result.data["consumer_duty_covered"] is True
        assert len(result.data["evidence_gaps"]) == 0

    async def test_partial_when_regulatory_gaps(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["30"] = {"data": AGENT30_GAPS}

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_PARTIAL
            result = await run(state)

        assert result.data["evidence_verdict"] == "PARTIAL"
        assert result.data["regulatory_sign_off_ready"] is False
        assert len(result.data["evidence_gaps"]) >= 1

    async def test_complete_for_low_fca_minimal_evidence(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_LOW}

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_LOW_FCA
            result = await run(state)

        assert result.data["evidence_verdict"] == "COMPLETE"

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_graceful_with_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert result.agent_id == 44
        assert result.data["evidence_verdict"] in ("COMPLETE", "PARTIAL", "MISSING")


# ── TA-enhanced Shapley tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTAEnhancedShapley:
    async def test_ta_evidence_summary_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["4"]  = {"data": AGENT4_PASS}
        state["agent_results"]["30"] = {"data": AGENT30_FULL}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_SIGNED_OFF}

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert "ta_evidence_summary" in result.data

    async def test_ta_evidence_summary_has_all_agent_ids(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        summary = result.data["ta_evidence_summary"]
        for agent_id in ["3", "4", "30", "33", "36"]:
            assert agent_id in summary

    async def test_ta_evidence_summary_values_are_ok_or_not_ok(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        for val in result.data["ta_evidence_summary"].values():
            assert val in ("OK", "NOT_OK")

    async def test_ta_evidence_ok_when_high_confidence_upstream(self):
        state = initial_story_state("FSC-2417")
        # Set agent_results with confidence >= 60 so _ta_mult returns 1.0 (OK)
        state["agent_results"]["3"] = {
            "data": AGENT3_HIGH,
            "confidence": {"final_score": 80},
        }

        with patch("src.agents.release.agent_44_fca_evidence_pack.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_EVIDENCE_COMPLETE
            result = await run(state)

        assert result.data["ta_evidence_summary"]["3"] == "OK"


# ── REQ-28: new tests ─────────────────────────────────────────────────────────

from src.agents.release.agent_44_fca_evidence_pack import _build_evidence_message


class TestREQ28CdObligationsKeyFix:
    def test_cd_obligations_key_used_not_obligations_mapped(self):
        agent4_with_obligations = {
            "consumer_duty_verdict": "PASS",
            "cd_obligations": ["fair_outcomes", "consumer_understanding"],
        }
        msg = _build_evidence_message(
            "FSC-001", AGENT3_HIGH, agent4_with_obligations,
            None, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF,
        )
        assert "fair_outcomes" in msg

    def test_obligations_mapped_key_absent_no_crash(self):
        # Old key "obligations_mapped" should not be read — should not raise
        agent4_old_key = {
            "consumer_duty_verdict": "PASS",
            "obligations_mapped": ["should_not_appear"],
        }
        msg = _build_evidence_message(
            "FSC-001", AGENT3_HIGH, agent4_old_key,
            None, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF,
        )
        # With the fix, cd_obligations key is used — old key falls through to []
        assert "should_not_appear" not in msg


class TestREQ28Agent29Fallback:
    def test_co_required_from_agent29_when_agent36_absent(self):
        agent29 = {"co_sign_off_required": True}
        msg = _build_evidence_message(
            "FSC-001", AGENT3_HIGH, AGENT4_PASS,
            agent29, AGENT30_FULL, AGENT33_PASS, None,
        )
        assert "co_required" in msg.lower() or "co required" in msg.lower() or "sign-off" in msg.lower() or "sign_off" in msg.lower()

    def test_agent36_preferred_over_agent29_when_both_present(self):
        agent29 = {"co_sign_off_required": True}
        msg = _build_evidence_message(
            "FSC-001", AGENT3_HIGH, AGENT4_PASS,
            agent29, AGENT30_FULL, AGENT33_PASS, AGENT36_SIGNED_OFF,
        )
        assert "SIGNED_OFF" in msg
