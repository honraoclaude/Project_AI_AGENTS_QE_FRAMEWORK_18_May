"""Tests for Agent 22 — Sandbox State Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_22_sandbox_state import (
    _assess_sandbox_state,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
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

AGENT11_BAD_NAMING = {
    "branch_found": True,
    "naming_convention_valid": False,
    "branch_stale": False,
    "branch_name": "fix-suitability",
}

AGENT13_CLEAN = {
    "detected_objects": ["suitability__c"],
    "dependency_depth": 1,
    "scope_delta": [],
}

AGENT13_SCOPE_DRIFT = {
    "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
    "dependency_depth": 2,
    "scope_delta_objects": ["riskprofile__c", "financialaccount"],
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

AGENT17_WARN = {
    "sfdx_format_valid": False,
    "sfdx_verdict": "WARN",
    "invalid_files": ["src/classes/LegacyApex.cls"],
}

AGENT13_DEEP_DEPTH = {
    "detected_objects": ["financialaccount"],
    "dependency_depth": 4,
    "scope_delta_objects": [],
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

    def test_scope_delta_objects_key_triggers_blocker(self):
        """REQ-14: Agent 22 reads scope_delta_objects (not scope_delta)."""
        agent13_with_new_key = {
            "detected_objects": ["suitability__c", "riskprofile__c", "financialaccount"],
            "dependency_depth": 2,
            "scope_delta_objects": ["riskprofile__c", "financialaccount"],
        }
        _, blockers, _, _ = _assess_sandbox_state(AGENT11_GOOD, agent13_with_new_key, AGENT17_PASS)
        assert any("scope" in b.lower() or "metadata" in b.lower() for b in blockers)

    def test_old_scope_delta_key_not_read(self):
        """REQ-14: Agent 22 must not read the old 'scope_delta' key — only scope_delta_objects."""
        agent13_old_key = {
            "detected_objects": ["suitability__c", "riskprofile__c"],
            "dependency_depth": 1,
            "scope_delta": ["riskprofile__c"],  # old key — must be ignored
        }
        _, blockers, ready, _ = _assess_sandbox_state(AGENT11_GOOD, agent13_old_key, AGENT17_PASS)
        assert ready is True  # no scope blocker because old key is not read

    def test_bad_naming_convention_adds_blocker(self):
        # branch_found=True, naming_valid=False → -15, not -35
        score, blockers, ready, verdict = _assess_sandbox_state(
            AGENT11_BAD_NAMING, AGENT13_CLEAN, AGENT17_PASS
        )
        assert any("naming" in b.lower() or "convention" in b.lower() for b in blockers)
        assert score == 85  # 100 - 15
        assert verdict == "READY"  # still ready at 85

    def test_sfdx_warn_adds_warn_blocker(self):
        # sfdx_valid=False, verdict=WARN → "SFDX format WARN" blocker, -10 not -25
        score_fail, _, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_FAIL)
        score_warn, blockers, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_WARN)
        assert any("warn" in b.lower() for b in blockers)
        assert score_warn > score_fail  # WARN penalty (-10) < FAIL penalty (-25)

    def test_depth_4_reduces_health_score(self):
        score_shallow, _, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS)
        score_deep, _, _, _ = _assess_sandbox_state(AGENT11_GOOD, AGENT13_DEEP_DEPTH, AGENT17_PASS)
        assert score_shallow > score_deep

    def test_stale_branch_and_sfdx_fail_gives_degraded(self):
        # 100 - 10 (stale) - 25 (FAIL) = 65 → DEGRADED
        score, _, ready, verdict = _assess_sandbox_state(AGENT11_STALE, AGENT13_CLEAN, AGENT17_FAIL)
        assert verdict == "DEGRADED"
        assert ready is False
        assert 40 <= score < 70


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

    def test_branch_context_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        assert "branch_context_available" in signals

    def test_no_branch_context_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT13_CLEAN, AGENT17_PASS, True)
        assert "no_branch_context" in signals

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        assert "metadata_scope_available" in signals

    def test_sfdx_format_signal_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, True)
        assert "sfdx_format_signal_available" in signals

    def test_sandbox_not_ready_key_in_signals(self):
        _, signals = _compute_confidence(AGENT11_GOOD, AGENT13_CLEAN, AGENT17_PASS, False)
        assert "sandbox_not_ready" in signals


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

    async def test_escalated_when_no_upstream_data(self):
        # base=58, no_branch_context=-8, sandbox_not_ready=-5 → 45 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["11"] = {"data": AGENT11_GOOD}
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["17"] = {"data": AGENT17_PASS}

        with patch("src.agents.development.agent_22_sandbox_state.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_PASS, 95, [], "READY")
        assert "FSC-2417" in msg

    def test_shows_branch_found(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_PASS, 95, [], "READY")
        assert "Branch found: True" in msg

    def test_shows_sfdx_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_FAIL, 75, ["SFDX FAIL"], "READY")
        assert "SFDX verdict: FAIL" in msg

    def test_shows_health_score(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_PASS, 85, [], "READY")
        assert "85/100" in msg

    def test_no_blockers_shows_none(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_PASS, 100, [], "READY")
        assert "none" in msg

    def test_shows_verdict(self):
        msg = _build_trace_message("FSC-2417", AGENT11_GOOD, AGENT13_CLEAN,
                                   AGENT17_PASS, 95, [], "READY")
        assert "Verdict: READY" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, None, None, 45, [], "BLOCKED")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "deployment_risk"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_deployment_risk_enum_has_three_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["deployment_risk"]["enum"] == [
            "low", "medium", "high"
        ]
