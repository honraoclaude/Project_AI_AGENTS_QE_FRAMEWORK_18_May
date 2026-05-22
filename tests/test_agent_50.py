"""Tests for Agent 50 — Release Retrospective Agent (True AI — Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_50_retrospective import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT23_DATA = {"development_verdict": "PASS"}
AGENT33_DATA = {"overall_coverage_pct": 92.0, "coverage_verdict": "PASS"}
AGENT34_DATA = {"defect_count": 0, "defect_verdict": "PASS"}
AGENT39_DATA = {"readiness_verdict": "READY"}
AGENT45_DATA = {"go_decision": True, "coordinator_verdict": "GO"}
AGENT46_DATA = {"prod_verdict": "HEALTHY"}

MOCK_RETRO_COMPLETE = {
    "lessons_learned": [
        {
            "area": "Test Coverage",
            "finding": "Coverage exceeded 90% threshold consistently across all AC scenarios.",
            "recommendation": "Set coverage floor to 90% for future FSC stories.",
        },
        {
            "area": "FCA Evidence",
            "finding": "Consumer Duty mapping completed ahead of release gate.",
            "recommendation": "Include Consumer Duty mapping as a Day 1 task in refinement.",
        },
    ],
    "process_improvements": [
        "Automate coverage threshold check in CI/CD pipeline",
        "Add Consumer Duty checklist to story template",
    ],
    "calibration_signals": {
        "coverage_above_threshold": "true",
        "defect_count": "0",
        "fca_evidence_complete": "true",
        "go_no_go": "GO",
    },
    "retrospective_verdict": "COMPLETE",
    "narrative": "Release FSC-2417 completed successfully. Coverage and FCA evidence were strong. No defects found post-release.",
}

MOCK_RETRO_PARTIAL = {
    "lessons_learned": [],
    "process_improvements": ["Gather more phase data in future sprints"],
    "calibration_signals": {"data_coverage": "insufficient"},
    "retrospective_verdict": "PARTIAL",
    "narrative": "Insufficient phase data available for a full retrospective. Partial signals recorded.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_all_three_sources_gives_high_score(self):
        score, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "COMPLETE")
        assert score >= 70

    def test_two_sources_gets_phase_coverage_bonus(self):
        score_two, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, None, "COMPLETE")
        score_zero, _ = _compute_confidence(None, None, None, "PARTIAL")
        assert score_two > score_zero

    def test_one_source_gets_minimal_bonus(self):
        score_one, _ = _compute_confidence(AGENT23_DATA, None, None, "PARTIAL")
        score_zero, _ = _compute_confidence(None, None, None, "PARTIAL")
        assert score_one > score_zero

    def test_no_sources_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "COMPLETE")
        score_without, _ = _compute_confidence(None, None, None, "PARTIAL")
        assert score_with > score_without

    def test_complete_verdict_adds_bonus(self):
        score_complete, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "COMPLETE")
        score_partial, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "PARTIAL")
        assert score_complete > score_partial

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "COMPLETE")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, "PARTIAL")
        assert score >= 20

    def test_returns_signals_dict(self):
        _, signals = _compute_confidence(AGENT23_DATA, AGENT33_DATA, AGENT45_DATA, "COMPLETE")
        assert isinstance(signals, dict)


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_DATA}
        state["agent_results"]["33"] = {"data": AGENT33_DATA}
        state["agent_results"]["45"] = {"data": AGENT45_DATA}

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert result.agent_id == 50
        assert result.agent_name == "Release Retrospective Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        for key in ["lessons_learned", "process_improvements",
                    "calibration_signals", "retrospective_verdict"]:
            assert key in result.data

    async def test_complete_retro_with_full_upstream(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_DATA}
        state["agent_results"]["33"] = {"data": AGENT33_DATA}
        state["agent_results"]["34"] = {"data": AGENT34_DATA}
        state["agent_results"]["39"] = {"data": AGENT39_DATA}
        state["agent_results"]["45"] = {"data": AGENT45_DATA}
        state["agent_results"]["46"] = {"data": AGENT46_DATA}

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert result.data["retrospective_verdict"] == "COMPLETE"
        assert len(result.data["lessons_learned"]) == 2
        assert len(result.data["process_improvements"]) == 2

    async def test_partial_retro_when_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_PARTIAL
            result = await run(state)

        assert result.data["retrospective_verdict"] == "PARTIAL"

    async def test_lessons_learned_structure(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        lessons = result.data["lessons_learned"]
        assert isinstance(lessons, list)
        for lesson in lessons:
            assert "area" in lesson
            assert "finding" in lesson
            assert "recommendation" in lesson

    async def test_calibration_signals_are_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert isinstance(result.data["calibration_signals"], dict)

    async def test_calibration_signals_contain_expected_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["45"] = {"data": AGENT45_DATA}

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        signals = result.data["calibration_signals"]
        assert "coverage_above_threshold" in signals
        assert "defect_count" in signals

    async def test_process_improvements_are_list(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert isinstance(result.data["process_improvements"], list)

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_narrative_used_as_why(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert result.why == MOCK_RETRO_COMPLETE["narrative"]

    async def test_what_includes_story_id_and_verdict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert "FSC-2417" in result.what
        assert "COMPLETE" in result.what

    async def test_escalation_flag_set_correctly(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_DATA}
        state["agent_results"]["33"] = {"data": AGENT33_DATA}
        state["agent_results"]["45"] = {"data": AGENT45_DATA}

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert isinstance(result.confidence.escalated, bool)

    async def test_signals_propagated_to_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_COMPLETE
            result = await run(state)

        assert "signals" in result.data
        assert isinstance(result.data["signals"], dict)


# ── REQ-32: new tests — fixed calibration_signals schema ─────────────────────

MOCK_RETRO_FIXED_SCHEMA = {
    "lessons_learned": [
        {"area": "Coverage", "finding": "Good.", "recommendation": "Keep threshold at 85%."},
    ],
    "process_improvements": ["Add coverage check to CI"],
    "calibration_signals": {
        "coverage_above_threshold": True,
        "defect_count": 0,
        "fca_evidence_complete": True,
        "development_verdict_clean": True,
        "go_decision_reached": True,
        "production_healthy": True,
    },
    "retrospective_verdict": "COMPLETE",
    "narrative": "Release complete. All signals nominal.",
}


@pytest.mark.asyncio
class TestREQ32CalibrationSignalsSchema:
    async def test_calibration_signals_has_required_keys(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_DATA}
        state["agent_results"]["33"] = {"data": AGENT33_DATA}
        state["agent_results"]["45"] = {"data": AGENT45_DATA}

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_FIXED_SCHEMA
            result = await run(state)

        sigs = result.data["calibration_signals"]
        for key in ["coverage_above_threshold", "defect_count", "fca_evidence_complete",
                    "development_verdict_clean", "go_decision_reached", "production_healthy"]:
            assert key in sigs, f"Missing required calibration signal key: {key}"

    async def test_coverage_above_threshold_is_bool(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_FIXED_SCHEMA
            result = await run(state)

        assert isinstance(result.data["calibration_signals"]["coverage_above_threshold"], bool)

    async def test_defect_count_is_int(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RETRO_FIXED_SCHEMA
            result = await run(state)

        assert isinstance(result.data["calibration_signals"]["defect_count"], int)

    async def test_fca_evidence_complete_when_evidence_verdict_complete(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["44"] = {"data": {"evidence_verdict": "COMPLETE"}}

        retro_with_evidence = {**MOCK_RETRO_FIXED_SCHEMA,
                               "calibration_signals": {**MOCK_RETRO_FIXED_SCHEMA["calibration_signals"],
                                                       "fca_evidence_complete": True}}
        with patch("src.agents.release.agent_50_retrospective.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = retro_with_evidence
            result = await run(state)

        assert result.data["calibration_signals"]["fca_evidence_complete"] is True
