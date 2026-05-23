"""Tests for Agent 39 — Release Readiness Assessor (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_39_release_readiness import (
    _assess_readiness,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT23_PASS    = {"development_verdict": "PASS",    "narrative": "All development checks passed."}
AGENT23_FAIL    = {"development_verdict": "FAIL",    "narrative": "Critical failures found."}
AGENT23_PARTIAL = {"development_verdict": "PARTIAL", "narrative": "Some checks incomplete."}

AGENT33_PASS = {"coverage_verdict": "PASS", "overall_coverage_pct": 92.0}
AGENT33_FAIL = {"coverage_verdict": "FAIL", "overall_coverage_pct": 58.0}

AGENT34_PASS = {"defect_verdict": "PASS", "critical_defects": [], "defect_count": 0}
AGENT34_FAIL = {"defect_verdict": "FAIL", "critical_defects": ["DEF-001"], "defect_count": 1}

AGENT35_RESOLVED = {"rca_verdict": "RESOLVED_PLAN", "fix_plan_complete": True}
AGENT35_INCOMPLETE = {"rca_verdict": "INCOMPLETE", "fix_plan_complete": False}

AGENT36_NOT_REQUIRED = {"uat_coordination_verdict": "NOT_REQUIRED", "uat_sign_off_required": False}
AGENT36_PENDING = {"uat_coordination_verdict": "PENDING", "uat_sign_off_required": True}
AGENT36_BLOCKED = {"uat_coordination_verdict": "BLOCKED", "uat_sign_off_required": True}

MOCK_TRACE_READY = {
    "narrative": "All development and testing phases passed. Coverage is 92%. Story is ready for release.",
    "readiness_concern": "none",
}

MOCK_TRACE_BLOCKED = {
    "narrative": "Release is BLOCKED — critical defect DEF-001 unresolved. Developer must fix before release can proceed.",
    "readiness_concern": "unresolved_defects",
}

MOCK_TRACE_PARTIAL = {
    "narrative": "Story is technically ready but awaiting Compliance Officer sign-off. Release can proceed once CO approves.",
    "readiness_concern": "uat_pending",
}


# ── Deterministic readiness assessment tests ──────────────────────────────────

class TestAssessReadiness:
    def test_all_pass_gives_ready(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_NOT_REQUIRED,
        )
        assert ready is True
        assert verdict == "READY"
        assert len(blockers) == 0

    def test_dev_fail_blocks_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_FAIL, AGENT33_PASS, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_NOT_REQUIRED,
        )
        assert ready is False
        assert verdict == "BLOCKED"
        assert any("development" in b.lower() for b in blockers)

    def test_coverage_fail_blocks_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_FAIL, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_NOT_REQUIRED,
        )
        assert ready is False
        assert verdict == "BLOCKED"
        assert any("coverage" in b.lower() or "threshold" in b.lower() for b in blockers)

    def test_critical_defects_block_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_PASS, AGENT34_FAIL, AGENT35_RESOLVED, AGENT36_NOT_REQUIRED,
        )
        assert ready is False
        assert verdict == "BLOCKED"
        assert any("DEF-001" in b for b in blockers)

    def test_incomplete_rca_blocks_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT35_INCOMPLETE, AGENT36_NOT_REQUIRED,
        )
        assert ready is False
        assert verdict == "BLOCKED"

    def test_uat_blocked_blocks_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_BLOCKED,
        )
        assert ready is False
        assert verdict == "BLOCKED"

    def test_uat_pending_gives_partial(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_PENDING,
        )
        assert ready is True
        assert verdict == "PARTIAL"

    def test_no_upstream_data_gives_ready(self):
        ready, blockers, verdict = _assess_readiness(None, None, None, None, None)
        assert ready is True
        assert verdict == "READY"

    def test_multiple_blockers_collected(self):
        _, blockers, _ = _assess_readiness(
            AGENT23_FAIL, AGENT33_FAIL, AGENT34_FAIL, AGENT35_INCOMPLETE, AGENT36_BLOCKED,
        )
        assert len(blockers) >= 4

    def test_dev_partial_blocks_release(self):
        ready, blockers, verdict = _assess_readiness(
            AGENT23_PARTIAL, AGENT33_PASS, AGENT34_PASS, AGENT35_RESOLVED, AGENT36_NOT_REQUIRED,
        )
        assert ready is False
        assert verdict == "BLOCKED"
        assert any("PARTIAL" in b for b in blockers)


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_phase_data_scores_well(self):
        score, _ = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        assert score >= 70

    def test_no_phase_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        score_without, _ = _compute_confidence(None, None, None, None, "READY")
        assert score_with > score_without

    def test_ready_verdict_boosts_confidence(self):
        score_ready, _ = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        score_blocked, _ = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_FAIL, AGENT36_NOT_REQUIRED, "BLOCKED")
        assert score_ready > score_blocked

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, "BLOCKED")
        assert score >= 20

    def test_comprehensive_phase_data_key_in_signals(self):
        _, signals = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        assert "comprehensive_phase_data" in signals

    def test_comprehensive_phase_data_stores_count(self):
        _, signals = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        assert signals["comprehensive_phase_data"] == 4

    def test_partial_phase_data_key_in_signals(self):
        _, signals = _compute_confidence(AGENT23_PASS, None, None, None, "READY")
        assert "partial_phase_data" in signals

    def test_no_phase_data_key_in_signals(self):
        _, signals = _compute_confidence(None, None, None, None, "READY")
        assert "no_phase_data" in signals

    def test_all_phases_clear_key_in_signals(self):
        _, signals = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_PASS, AGENT36_NOT_REQUIRED, "READY")
        assert "all_phases_clear" in signals

    def test_release_blocked_key_in_signals(self):
        _, signals = _compute_confidence(AGENT23_PASS, AGENT33_PASS, AGENT34_FAIL, AGENT36_NOT_REQUIRED, "BLOCKED")
        assert "release_blocked" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_PASS}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_NOT_REQUIRED}

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.agent_id == 39
        assert result.agent_name == "Release Readiness Assessor"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        for key in ["release_ready", "readiness_blockers", "readiness_verdict"]:
            assert key in result.data

    async def test_ready_when_all_pass(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_PASS}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_NOT_REQUIRED}

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.data["release_ready"] is True
        assert result.data["readiness_verdict"] == "READY"

    async def test_blocked_when_defects_present(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["34"] = {"data": AGENT34_FAIL}

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.data["release_ready"] is False
        assert result.data["readiness_verdict"] == "BLOCKED"

    async def test_partial_when_uat_pending(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["23"] = {"data": AGENT23_PASS}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}
        state["agent_results"]["36"] = {"data": AGENT36_PENDING}

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PARTIAL
            result = await run(state)

        assert result.data["readiness_verdict"] == "PARTIAL"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_single_source_and_blocked(self):
        # base=63, partial_phase_data(1)→+4=67, release_blocked→-8=59 < 60
        state = initial_story_state("FSC-2417")
        state["agent_results"]["34"] = {"data": AGENT34_FAIL}

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_39_release_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_READY
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, [], "READY", AGENT33_PASS, AGENT36_NOT_REQUIRED)
        assert "FSC-2417" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", True, [], "READY", AGENT33_PASS, AGENT36_NOT_REQUIRED)
        assert "READY" in msg

    def test_includes_coverage_pct(self):
        msg = _build_trace_message("FSC-2417", True, [], "READY", AGENT33_PASS, AGENT36_NOT_REQUIRED)
        assert "92.0" in msg

    def test_includes_uat_coordination(self):
        msg = _build_trace_message("FSC-2417", True, [], "PARTIAL", AGENT33_PASS, AGENT36_PENDING)
        assert "PENDING" in msg

    def test_includes_blockers(self):
        blockers = ["Unresolved critical defects: ['DEF-001']"]
        msg = _build_trace_message("FSC-2417", False, blockers, "BLOCKED", None, None)
        assert "DEF-001" in msg

    def test_no_blockers_shows_none_sentinel(self):
        msg = _build_trace_message("FSC-2417", True, [], "READY", None, None)
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", True, [], "READY", None, None)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "readiness_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_readiness_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["readiness_concern"]["enum"] == [
            "none", "testing_incomplete", "unresolved_defects",
            "uat_pending", "coverage_below_threshold",
        ]
