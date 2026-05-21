"""Tests for Agent 34 — Defect Triage Agent (True AI Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_34_defect_triage import (
    _build_severity_votes,
    _compute_confidence,
    _infer_severity,
    _resolve_severity_votes,
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


# ── Severity voting unit tests ────────────────────────────────────────────────

class TestInferSeverity:
    def test_fail_with_count_produces_p1(self):
        assert _infer_severity("FAIL", 2) == "P1"

    def test_warn_with_count_produces_p2(self):
        assert _infer_severity("WARN", 1) == "P2"

    def test_fail_without_count_produces_p2(self):
        assert _infer_severity("FAIL", 0) == "P2"

    def test_pass_produces_p3(self):
        assert _infer_severity("PASS", 0) == "P3"

    def test_pass_with_count_still_p3(self):
        assert _infer_severity("PASS", 5) == "P3"


class TestBuildSeverityVotes:
    def test_returns_five_source_keys(self):
        votes = _build_severity_votes(
            AGENT27_PASS, AGENT28_CLEAN, AGENT30_PASS, AGENT31_PASS, AGENT37_SKIPPED,
        )
        assert set(votes.keys()) == {"crt", "fca_scenario", "financial", "performance", "self_heal"}

    def test_crt_fail_produces_p1(self):
        votes = _build_severity_votes(
            AGENT27_FAIL, AGENT28_CLEAN, AGENT30_PASS, AGENT31_PASS, AGENT37_SKIPPED,
        )
        assert votes["crt"] == "P1"

    def test_all_pass_produces_all_p3(self):
        votes = _build_severity_votes(
            AGENT27_PASS, AGENT28_CLEAN, AGENT30_PASS, AGENT31_PASS, AGENT37_SKIPPED,
        )
        assert all(v == "P3" for v in votes.values())

    def test_none_data_handled_gracefully(self):
        votes = _build_severity_votes(None, None, None, None, None)
        assert len(votes) == 5
        assert all(v in ("P1", "P2", "P3", "P4") for v in votes.values())


class TestResolveSeverityVotes:
    def test_any_p1_vote_forces_p1_final(self):
        votes = {"crt": "P1", "fca_scenario": "P3", "financial": "P3", "performance": "P3", "self_heal": "P3"}
        severity, minimax_escalated = _resolve_severity_votes(votes)
        assert severity == "P1"

    def test_p1_escalated_when_not_unanimous(self):
        votes = {"crt": "P1", "fca_scenario": "P3", "financial": "P3", "performance": "P3", "self_heal": "P3"}
        severity, minimax_escalated = _resolve_severity_votes(votes)
        assert minimax_escalated is True

    def test_all_p1_not_minimax_escalated(self):
        votes = {"crt": "P1", "fca_scenario": "P1", "financial": "P1", "performance": "P1", "self_heal": "P1"}
        severity, minimax_escalated = _resolve_severity_votes(votes)
        assert severity == "P1"
        assert minimax_escalated is False

    def test_majority_p3_wins_no_escalation(self):
        votes = {"crt": "P3", "fca_scenario": "P3", "financial": "P2", "performance": "P3", "self_heal": "P3"}
        severity, minimax_escalated = _resolve_severity_votes(votes)
        assert severity == "P3"
        assert minimax_escalated is False


# ── Coalition integration tests ───────────────────────────────────────────────

@pytest.mark.asyncio
class TestCoalitionSeverityVoting:
    async def test_severity_votes_in_data_with_five_keys(self):
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

        assert "severity_votes" in result.data
        votes = result.data["severity_votes"]
        assert set(votes.keys()) == {"crt", "fca_scenario", "financial", "performance", "self_heal"}

    async def test_coalition_severity_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_PASS}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert "coalition_severity" in result.data
        assert result.data["coalition_severity"] in ("P1", "P2", "P3", "P4")

    async def test_minimax_escalated_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert "minimax_escalated" in result.data
        assert isinstance(result.data["minimax_escalated"], bool)

    async def test_coalition_dissent_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_NO_DEFECTS
            result = await run(state)

        assert "coalition_dissent" in result.data
        assert isinstance(result.data["coalition_dissent"], list)

    async def test_crt_fail_triggers_p1_coalition_severity(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_FAIL}
        state["agent_results"]["28"] = {"data": AGENT28_CLEAN}
        state["agent_results"]["30"] = {"data": AGENT30_PASS}
        state["agent_results"]["31"] = {"data": AGENT31_PASS}
        state["agent_results"]["37"] = {"data": AGENT37_SKIPPED}

        with patch("src.agents.testing.agent_34_defect_triage.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_CRITICAL_DEFECT
            result = await run(state)

        assert result.data["coalition_severity"] == "P1"
        assert result.data["minimax_escalated"] is True

    async def test_all_pass_no_dissent_p3_severity(self):
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

        assert result.data["coalition_severity"] == "P3"
        assert result.data["coalition_dissent"] == []
