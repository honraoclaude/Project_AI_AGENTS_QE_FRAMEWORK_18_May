"""Tests for Agent 35 — Root Cause Analyser (True AI Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_35_root_cause_analyser import (
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT27_PASS = {
    "crt_execution_verdict": "PASS",
    "crt_pass_count": 3,
    "crt_fail_count": 0,
}

AGENT27_FAIL = {
    "crt_execution_verdict": "FAIL",
    "crt_pass_count": 1,
    "crt_fail_count": 2,
}

AGENT28_CLEAN = {
    "self_heal_verdict": "PASS",
    "suspect_self_heals": [],
}

AGENT28_SUSPECT = {
    "self_heal_verdict": "REVIEW_REQUIRED",
    "suspect_self_heals": ["CRT-001"],
}

AGENT33_PASS = {
    "overall_coverage_pct": 92.0,
    "coverage_verdict": "PASS",
    "uncovered_acs": [],
}

AGENT33_FAIL = {
    "overall_coverage_pct": 60.0,
    "coverage_verdict": "FAIL",
    "uncovered_acs": ["AC3", "AC4"],
}

AGENT34_NO_DEFECTS = {
    "defects_found": [],
    "defect_count": 0,
    "critical_defects": [],
    "defect_verdict": "PASS",
}

AGENT34_WITH_DEFECTS = {
    "defects_found": [
        {"id": "DEF-001", "title": "SOQL in loop", "severity": "P2",
         "owner": "Developer", "source": "CRT"},
    ],
    "defect_count": 1,
    "critical_defects": ["DEF-001"],
    "defect_verdict": "FAIL",
}

AGENT37_SKIPPED = {
    "perf_test_verdict": "SKIPPED",
    "performance_concern": "none",
}

AGENT38_CLEAN = {
    "flaky_tests": [],
    "flaky_verdict": "PASS",
}

AGENT38_FLAKY = {
    "flaky_tests": ["CRT-002"],
    "flaky_verdict": "WARN",
}

MOCK_RCA_NO_ACTION = {
    "root_causes": [],
    "rca_verdict": "NO_ACTION_REQUIRED",
    "fix_plan_complete": True,
    "estimated_effort": "LOW",
    "narrative": "No defects or failures detected. All tests passed and coverage thresholds met. No remediation required.",
}

MOCK_RCA_RESOLVED = {
    "root_causes": [
        {
            "defect_id": "DEF-001",
            "root_cause": "SOQL query inside AccountService loop causes governor limit breach at >200 records.",
            "fix_action": "Move query outside the loop using Map-based bulk pattern.",
            "owner": "Developer",
            "effort": "MEDIUM",
        }
    ],
    "rca_verdict": "RESOLVED_PLAN",
    "fix_plan_complete": True,
    "estimated_effort": "MEDIUM",
    "narrative": "DEF-001 root cause identified as SOQL-in-loop governor breach. Developer must refactor AccountService with Map-based bulk query pattern before release.",
}

MOCK_RCA_INCOMPLETE = {
    "root_causes": [
        {
            "defect_id": "DEF-001",
            "root_cause": "Unknown — insufficient signal from test output.",
            "fix_action": "Investigate with Developer — enable debug logging.",
            "owner": "Developer",
            "effort": "HIGH",
        }
    ],
    "rca_verdict": "INCOMPLETE",
    "fix_plan_complete": False,
    "estimated_effort": "HIGH",
    "narrative": "Root cause for DEF-001 could not be determined from available signals. Developer must investigate with debug logging enabled.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_defect_data_available_with_failures_scores_well(self):
        score, _ = _compute_confidence(AGENT34_WITH_DEFECTS, AGENT38_CLEAN, 1, "RESOLVED_PLAN")
        assert score >= 70

    def test_no_defect_triage_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT34_NO_DEFECTS, AGENT38_CLEAN, 0, "NO_ACTION_REQUIRED")
        score_without, _ = _compute_confidence(None, None, 0, "NO_ACTION_REQUIRED")
        assert score_with > score_without

    def test_resolved_plan_boosts_confidence(self):
        score_resolved, _ = _compute_confidence(AGENT34_WITH_DEFECTS, AGENT38_CLEAN, 1, "RESOLVED_PLAN")
        score_incomplete, _ = _compute_confidence(AGENT34_WITH_DEFECTS, AGENT38_CLEAN, 1, "INCOMPLETE")
        assert score_resolved > score_incomplete

    def test_incomplete_rca_penalises_confidence(self):
        score, _ = _compute_confidence(AGENT34_WITH_DEFECTS, AGENT38_FLAKY, 2, "INCOMPLETE")
        score_clean, _ = _compute_confidence(AGENT34_NO_DEFECTS, AGENT38_CLEAN, 0, "NO_ACTION_REQUIRED")
        assert score_clean > score

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT34_NO_DEFECTS, AGENT38_CLEAN, 0, "NO_ACTION_REQUIRED")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, 0, "INCOMPLETE")
        assert score >= 20

    def test_flaky_test_data_adds_signal(self):
        score_with, _ = _compute_confidence(AGENT34_NO_DEFECTS, AGENT38_CLEAN, 0, "NO_ACTION_REQUIRED")
        score_without, _ = _compute_confidence(AGENT34_NO_DEFECTS, None, 0, "NO_ACTION_REQUIRED")
        assert score_with >= score_without


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_PASS}
        state["agent_results"]["28"] = {"data": AGENT28_CLEAN}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_NO_DEFECTS}
        state["agent_results"]["37"] = {"data": AGENT37_SKIPPED}
        state["agent_results"]["38"] = {"data": AGENT38_CLEAN}

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_NO_ACTION
            result = await run(state)

        assert result.agent_id == 35
        assert result.agent_name == "Root Cause Analyser"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_NO_ACTION
            result = await run(state)

        for key in ["root_causes", "rca_verdict", "fix_plan_complete", "estimated_effort"]:
            assert key in result.data

    async def test_no_action_when_all_tests_pass(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["34"] = {"data": AGENT34_NO_DEFECTS}

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_NO_ACTION
            result = await run(state)

        assert result.data["rca_verdict"] == "NO_ACTION_REQUIRED"
        assert len(result.data["root_causes"]) == 0

    async def test_resolved_plan_when_defects_present(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["34"] = {"data": AGENT34_WITH_DEFECTS}

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_RESOLVED
            result = await run(state)

        assert result.data["rca_verdict"] == "RESOLVED_PLAN"
        assert result.data["fix_plan_complete"] is True
        assert len(result.data["root_causes"]) >= 1

    async def test_incomplete_rca_when_cause_unknown(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["34"] = {"data": AGENT34_WITH_DEFECTS}

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_INCOMPLETE
            result = await run(state)

        assert result.data["rca_verdict"] == "INCOMPLETE"
        assert result.data["fix_plan_complete"] is False

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_NO_ACTION
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_graceful_with_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_35_root_cause_analyser.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_RCA_NO_ACTION
            result = await run(state)

        assert result.agent_id == 35
        assert result.data["rca_verdict"] in ("NO_ACTION_REQUIRED", "RESOLVED_PLAN", "INCOMPLETE")
