"""Tests for Agent 34 — Defect Triage Agent (True AI Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_34_defect_triage import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT27_PASS = {
    "crt_execution_verdict": "PASS",
    "crt_pass_count": 3,
    "crt_fail_count": 0,
    "tests_executed": 3,
}

AGENT27_FAIL = {
    "crt_execution_verdict": "FAIL",
    "crt_pass_count": 1,
    "crt_fail_count": 2,
    "tests_executed": 3,
}

AGENT28_CLEAN = {
    "self_heal_verdict": "PASS",
    "suspect_self_heals": [],
}

AGENT28_SUSPECT = {
    "self_heal_verdict": "REVIEW_REQUIRED",
    "suspect_self_heals": ["CRT-001"],
}

AGENT30_PASS = {
    "fca_scenario_verdict": "PASS",
    "regulatory_gaps": [],
}

AGENT30_GAPS = {
    "fca_scenario_verdict": "WARN",
    "regulatory_gaps": ["COBS 9 suitability not covered"],
}

AGENT31_PASS = {
    "integrity_verdict": "PASS",
    "integrity_violations": [],
}

AGENT31_FAIL = {
    "integrity_verdict": "WARN",
    "integrity_violations": ["CRT execution failed — integrity unconfirmed"],
}

AGENT37_SKIPPED = {
    "perf_test_verdict": "SKIPPED",
    "performance_concern": "none",
}

AGENT37_FAIL = {
    "perf_test_verdict": "FAIL",
    "performance_concern": "governor_limit_breach",
}

MOCK_NO_DEFECTS = {
    "defects_found": [],
    "defect_count": 0,
    "critical_defects": [],
    "defect_verdict": "PASS",
    "triage_complete": True,
    "narrative": "No defects found across all test types. All CRT, FCA, and integrity checks passed.",
}

MOCK_CRITICAL_DEFECT = {
    "defects_found": [
        {"id": "DEF-001", "title": "CRT test failure: Portfolio rebalancing",
         "severity": "P2", "owner": "Developer", "source": "CRT"},
    ],
    "defect_count": 1,
    "critical_defects": ["DEF-001"],
    "defect_verdict": "FAIL",
    "triage_complete": True,
    "narrative": "P2 defect DEF-001 detected in CRT execution. Developer must fix portfolio rebalancing before release.",
}

MOCK_MINOR_DEFECTS = {
    "defects_found": [
        {"id": "DEF-002", "title": "Self-heal on FCA test",
         "severity": "P3", "owner": "QE", "source": "SelfHeal"},
    ],
    "defect_count": 1,
    "critical_defects": [],
    "defect_verdict": "WARN",
    "triage_complete": True,
    "narrative": "P3 defect DEF-002 found — QE must review self-healed FCA test locator.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_multiple_sources_available_scores_well(self):
        score, _ = _compute_confidence(AGENT27_PASS, AGENT28_CLEAN, AGENT31_PASS, AGENT37_SKIPPED, 0)
        assert score >= 70

    def test_no_sources_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT27_PASS, AGENT28_CLEAN, AGENT31_PASS, AGENT37_SKIPPED, 0)
        score_without, _ = _compute_confidence(None, None, None, None, 0)
        assert score_with > score_without

    def test_crt_pass_boosts_confidence(self):
        score_pass, _ = _compute_confidence(AGENT27_PASS, None, None, None, 0)
        score_fail, _ = _compute_confidence(AGENT27_FAIL, None, None, None, 0)
        assert score_pass > score_fail

    def test_no_defects_boosts_confidence(self):
        score_clean, _ = _compute_confidence(AGENT27_PASS, AGENT28_CLEAN, AGENT31_PASS, AGENT37_SKIPPED, 0)
        score_dirty, _ = _compute_confidence(AGENT27_PASS, AGENT28_CLEAN, AGENT31_PASS, AGENT37_SKIPPED, 5)
        assert score_clean >= score_dirty

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT27_PASS, AGENT28_CLEAN, AGENT31_PASS, AGENT37_SKIPPED, 0)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, 10)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_PASS}
        state["agent_results"]["28"] = {"data": AGENT28_CLEAN}
        state["agent_results"]["30"] = {"data": AGENT30_PASS}
        state["agent_results"]["31"] = {"data": AGENT31_PASS}
        state["agent_results"]["37"] = {"data": AGENT37_SKIPPED}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.agent_id == 34
        assert result.agent_name == "Defect Triage Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        for key in ["defects_found", "defect_count", "critical_defects", "defect_verdict",
                    "triage_complete"]:
            assert key in result.data

    async def test_pass_when_no_defects(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_PASS}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.data["defect_verdict"] == "PASS"
        assert result.data["defect_count"] == 0
        assert len(result.data["critical_defects"]) == 0

    async def test_fail_with_critical_defect(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_FAIL}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRITICAL_DEFECT
            result = await run(state)

        assert result.data["defect_verdict"] == "FAIL"
        assert len(result.data["critical_defects"]) >= 1

    async def test_warn_with_minor_defects(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["28"] = {"data": AGENT28_SUSPECT}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_MINOR_DEFECTS
            result = await run(state)

        assert result.data["defect_verdict"] == "WARN"
        assert len(result.data["critical_defects"]) == 0

    async def test_triage_complete_when_all_owned(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.data["triage_complete"] is True

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_narrative_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.data["narrative"] == MOCK_NO_DEFECTS["narrative"]

    async def test_graceful_with_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert result.agent_id == 34
        assert result.data["defect_verdict"] in ("PASS", "WARN", "FAIL")
