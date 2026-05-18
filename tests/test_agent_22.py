"""Tests for Agent 22 — Sandbox State Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_22_sandbox_state import (
    _assess_sandbox_state,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT11_GOOD = {
    "branch_found": True,
    "naming_convention_valid": True,
    "branch_stale": False,
    "branch_name": "feature/FSC-2417-suitability-fix",
}

AGENT11_STALE = {
    "branch_found": True,
    "naming_convention_valid": True,
    "branch_stale": True,
    "branch_name": "feature/FSC-2417-suitability-fix",
}

AGENT11_MISSING = {
    "branch_found": False,
    "naming_convention_valid": False,
    "branch_stale": False,
}

AGENT13_CLEAN = {
    "detected_objects": ["suitability__c"],
    "dependency_depth": 1,
    "scope_delta": [],
}

AGENT13_SCOPE_DRIFT = {
    "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
    "dependency_depth": 2,
    "scope_delta": ["riskprofile__c", "financialaccount"],
}

AGENT17_PASS = {
    "sfdx_format_valid": True,
    "sfdx_verdict": "PASS",
    "invalid_files": [],
}

AGENT17_FAIL = {
    "sfdx_format_valid": False,
    "sfdx_verdict": "FAIL",
    "invalid_files": ["src/classes/OldApex.cls", "src/classes/Another.cls", "metadata/Foo.object"],
}

MOCK_TRACE_READY = {
    "narrative": "Sandbox is healthy: branch valid, SFDX format compliant, no scope drift.",
    "deployment_risk": "low",
}

MOCK_TRACE_BLOCKED = {
    "narrative": "Sandbox BLOCKED: no branch found and legacy-format files present.",
    "deployment_risk": "high",
}


# ── Deterministic sandbox state assessment tests ──────────────────────────────

class TestSandboxStateAssessment:
    def test_all_signals_healthy_gives_ready(self):
        score, blockers, ready, verdict = _assess_sandbox_state(
            AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS
        )
        assert verdict == "READY"
        assert ready is True
        assert score >= 70

    def test_missing_branch_gives_blocked(self):
        score, blockers, ready, verdict = _assess_sandbox_state(
            AGENT11_MISSING, AGENT13_CLEAN, AGENT17_PASS
        )
        assert ready is False
        assert any("branch" in b.lower() for b in blockers)
        assert score < 70

    def test_sfdx_fail_reduces_health_score(self):
        score_pass, _, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS)
        score_fail, _, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_FAIL)
        assert score_pass > score_fail

    def test_sfdx_fail_adds_blocker(self):
        _, blockers, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_FAIL)
        assert any("sfdx" in b.lower() or "legacy" in b.lower() for b in blockers)

    def test_stale_branch_adds_blocker(self):
        _, blockers, _, _ = _assess_sandbox_state(AGENT11_STALE, AGENT13_CLEAN, AGENT17_PASS)
        assert any("stale" in b.lower() for b in blockers)

    def test_scope_drift_adds_blocker(self):
        _, blockers, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_SCOPE_DRIFT, AGENT17_PASS)
        assert any("scope" in b.lower() or "metadata" in b.lower() for b in blockers)

    def test_no_upstream_data_degrades_gracefully(self):
        score, blockers, ready, verdict = _assess_sandbox_state(None, None, None)
        assert verdict in ("READY", "DEGRADED", "BLOCKED")
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_health_score_bounded(self):
        score, _, _, _ = _assess_sandbox_state(
            AGENT11_MISSING, AGENT13_SCOPE_DRIFT, AGENT17_FAIL
        )
        assert 0 <= score <= 100

    def test_missing_branch_and_sfdx_fail_gives_blocked(self):
        score, _, ready, verdict = _assess_sandbox_state(
            AGENT11_MISSING, AGENT13_CLEAN, AGENT17_FAIL
        )
        assert ready is False
        assert score < 70


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        score, _ = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        assert score >= 65

    def test_no_branch_context_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        score_without, _ = _compute_confidence(None, AGENT13_CLEAN, AGENT17_PASS, True)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["11"] = {"data": AGENT11_GOOD}
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["17"] = {"data": AGENT17_PASS}

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.agent_id == 22
        assert result.agent_name == "Sandbox State Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        for key in ["sandbox_ready", "sandbox_health_score",
                    "sandbox_blockers", "sandbox_verdict"]:
            assert key in result.data

    async def test_ready_when_all_signals_healthy(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["11"] = {"data": AGENT11_GOOD}
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["17"] = {"data": AGENT17_PASS}

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.data["sandbox_ready"] is True
        assert result.data["sandbox_verdict"] == "READY"

    async def test_not_ready_when_branch_missing(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["11"] = {"data": AGENT11_MISSING}

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.data["sandbox_ready"] is False

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
