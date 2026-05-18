"""Tests for Agent 41 — Change Set Integrity (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_41_change_set_integrity import (
    _check_integrity,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT13_CLEAN = {
    "changed_files_count": 4,
    "missing_dependencies": [],
    "has_destructive_changes": False,
    "dependency_depth": 1,
}

AGENT13_DESTRUCTIVE = {
    "changed_files_count": 3,
    "missing_dependencies": [],
    "has_destructive_changes": True,
    "dependency_depth": 1,
}

AGENT13_MISSING_DEPS = {
    "changed_files_count": 5,
    "missing_dependencies": ["FinancialAccount__c.Rating__c"],
    "has_destructive_changes": False,
    "dependency_depth": 2,
}

AGENT13_LARGE = {
    "changed_files_count": 25,
    "missing_dependencies": [],
    "has_destructive_changes": False,
    "dependency_depth": 3,
}

AGENT40_COMPOSED = {
    "release_name": "FSC-2417-minor-release",
    "component_count": 4,
    "composer_verdict": "COMPOSED",
    "release_type": "MINOR",
}

AGENT40_FAILED = {
    "release_name": "FSC-2417-patch-release",
    "component_count": 0,
    "composer_verdict": "FAILED",
    "release_type": "PATCH",
}

MOCK_TRACE_PASS = {
    "narrative": "Change set FSC-2417-minor-release is valid. 4 components, no missing dependencies, no destructive changes.",
    "integrity_concern": "none",
}

MOCK_TRACE_WARN = {
    "narrative": "Change set contains destructive changes. Manual review required before deployment to ensure no data loss.",
    "integrity_concern": "destructive_changes",
}

MOCK_TRACE_FAIL = {
    "narrative": "Change set integrity FAILED — missing dependency FinancialAccount__c.Rating__c must be deployed first.",
    "integrity_concern": "missing_dependencies",
}


# ── Deterministic integrity check tests ──────────────────────────────────────

class TestCheckIntegrity:
    def test_clean_change_set_gives_pass(self):
        valid, issues, destructive, verdict = _check_integrity(AGENT13_CLEAN, AGENT40_COMPOSED)
        assert valid is True
        assert verdict == "PASS"
        assert len(issues) == 0
        assert destructive is False

    def test_destructive_changes_give_warn(self):
        valid, issues, destructive, verdict = _check_integrity(AGENT13_DESTRUCTIVE, AGENT40_COMPOSED)
        assert valid is True  # destructive = WARN, not FAIL
        assert verdict == "WARN"
        assert destructive is True

    def test_missing_dependencies_give_fail(self):
        valid, issues, destructive, verdict = _check_integrity(AGENT13_MISSING_DEPS, AGENT40_COMPOSED)
        assert valid is False
        assert verdict == "FAIL"
        assert any("dependencies" in i.lower() for i in issues)

    def test_composer_failed_gives_fail(self):
        valid, issues, _, verdict = _check_integrity(AGENT13_CLEAN, AGENT40_FAILED)
        assert valid is False
        assert verdict == "FAIL"

    def test_large_change_set_gives_warn(self):
        valid, issues, _, verdict = _check_integrity(AGENT13_LARGE, AGENT40_COMPOSED)
        assert verdict == "WARN"
        assert any("large" in i.lower() for i in issues)

    def test_no_upstream_data_gives_pass(self):
        valid, issues, destructive, verdict = _check_integrity(None, None)
        assert valid is True
        assert verdict == "PASS"
        assert len(issues) == 0

    def test_destructive_does_not_make_invalid(self):
        valid, _, _, _ = _check_integrity(AGENT13_DESTRUCTIVE, AGENT40_COMPOSED)
        assert valid is True


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_release_package_available_scores_well(self):
        score, _ = _compute_confidence(AGENT13_CLEAN, AGENT40_COMPOSED, True)
        assert score >= 68

    def test_no_release_package_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT13_CLEAN, AGENT40_COMPOSED, True)
        score_without, _ = _compute_confidence(None, None, True)
        assert score_with > score_without

    def test_invalid_reduces_confidence(self):
        score_valid, _ = _compute_confidence(AGENT13_CLEAN, AGENT40_COMPOSED, True)
        score_invalid, _ = _compute_confidence(AGENT13_MISSING_DEPS, AGENT40_COMPOSED, False)
        assert score_valid > score_invalid

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT13_CLEAN, AGENT40_COMPOSED, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["40"] = {"data": AGENT40_COMPOSED}

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 41
        assert result.agent_name == "Change Set Integrity"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["integrity_valid", "integrity_issues",
                    "destructive_changes_present", "integrity_verdict"]:
            assert key in result.data

    async def test_pass_when_clean_change_set(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["40"] = {"data": AGENT40_COMPOSED}

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["integrity_verdict"] == "PASS"
        assert result.data["integrity_valid"] is True

    async def test_warn_when_destructive_changes(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_DESTRUCTIVE}
        state["agent_results"]["40"] = {"data": AGENT40_COMPOSED}

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_WARN
            result = await run(state)

        assert result.data["integrity_verdict"] == "WARN"
        assert result.data["destructive_changes_present"] is True

    async def test_fail_when_missing_dependencies(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_MISSING_DEPS}
        state["agent_results"]["40"] = {"data": AGENT40_COMPOSED}

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["integrity_verdict"] == "FAIL"
        assert result.data["integrity_valid"] is False

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_41_change_set_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
