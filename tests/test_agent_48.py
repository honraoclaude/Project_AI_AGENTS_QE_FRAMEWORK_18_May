"""Tests for Agent 48 — Rollback Readiness (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_48_rollback_readiness import (
    _assess_rollback,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT13_CLEAN = {"has_destructive_changes": False, "dependency_depth": 0}
AGENT13_DESTRUCTIVE = {"has_destructive_changes": True, "dependency_depth": 0}
AGENT13_DEEP_DEPS = {"has_destructive_changes": False, "dependency_depth": 4}
AGENT13_DESTRUCTIVE_SHALLOW = {"has_destructive_changes": True, "dependency_depth": 1}
AGENT13_DESTRUCTIVE_MEDIUM = {"has_destructive_changes": True, "dependency_depth": 2}
AGENT13_DESTRUCTIVE_DEEP = {"has_destructive_changes": True, "dependency_depth": 3}

AGENT41_PASS = {"integrity_verdict": "PASS"}
AGENT41_FAIL = {"integrity_verdict": "FAIL"}

MOCK_TRACE_LOW = {
    "narrative": "Rollback is straightforward. No destructive changes present. Standard Copado rollback procedure applies.",
    "rollback_concern": "none",
}
MOCK_TRACE_HIGH = {
    "narrative": "Rollback is HIGH risk. Destructive changes with deep dependencies make reversal complex. Manual restore of deleted metadata is required.",
    "rollback_concern": "destructive_changes",
}
MOCK_TRACE_MEDIUM = {
    "narrative": "Rollback is MEDIUM risk. Destructive changes present; restore from version control needed before proceeding.",
    "rollback_concern": "destructive_changes",
}


# ── Deterministic rollback assessment tests ───────────────────────────────────

class TestAssessRollback:
    def test_no_data_gives_low_risk_feasible(self):
        feasible, risk, steps, verdict = _assess_rollback(None, None, None, None)
        assert feasible is True
        assert risk == "LOW"
        assert verdict == "FEASIBLE"

    def test_clean_change_gives_feasible(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_PASS)
        assert feasible is True
        assert risk == "LOW"
        assert verdict == "FEASIBLE"

    def test_destructive_no_deps_gives_risky(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_DESTRUCTIVE, None, AGENT41_PASS)
        assert feasible is True
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_destructive_depth_1_gives_risky(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_DESTRUCTIVE_SHALLOW, None, AGENT41_PASS)
        assert feasible is True
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_destructive_depth_2_gives_not_feasible(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_DESTRUCTIVE_MEDIUM, None, AGENT41_PASS)
        assert feasible is False
        assert risk == "HIGH"
        assert verdict == "NOT_FEASIBLE"

    def test_destructive_depth_3_gives_not_feasible(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_DESTRUCTIVE_DEEP, None, AGENT41_PASS)
        assert feasible is False
        assert risk == "HIGH"
        assert verdict == "NOT_FEASIBLE"

    def test_deep_deps_no_destructive_gives_risky(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_DEEP_DEPS, None, AGENT41_PASS)
        assert feasible is True
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_depth_3_no_destructive_gives_risky(self):
        data = {"has_destructive_changes": False, "dependency_depth": 3}
        feasible, risk, steps, verdict = _assess_rollback(None, data, None, None)
        assert feasible is True
        assert verdict == "RISKY"
        assert risk == "MEDIUM"

    def test_depth_2_no_destructive_gives_feasible(self):
        data = {"has_destructive_changes": False, "dependency_depth": 2}
        feasible, risk, steps, verdict = _assess_rollback(None, data, None, None)
        assert feasible is True
        assert risk == "LOW"
        assert verdict == "FEASIBLE"

    def test_steps_always_include_base_step(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_CLEAN, None, None)
        assert any("Copado" in s for s in steps)

    def test_destructive_adds_restore_steps(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_DESTRUCTIVE, None, AGENT41_PASS)
        assert any("deleted metadata" in s.lower() for s in steps)
        assert any("data integrity" in s.lower() for s in steps)

    def test_deep_deps_adds_dependency_review_step(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_DEEP_DEPS, None, AGENT41_PASS)
        assert any("dependency" in s.lower() for s in steps)

    def test_clean_change_has_minimal_steps(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_PASS)
        assert len(steps) == 1

    def test_destructive_deep_has_most_steps(self):
        _, _, steps_max, _ = _assess_rollback(None, AGENT13_DESTRUCTIVE_DEEP, None, AGENT41_PASS)
        _, _, steps_min, _ = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_PASS)
        assert len(steps_max) > len(steps_min)

    def test_returns_tuple_of_four(self):
        result = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_PASS)
        assert len(result) == 4
        feasible, risk, steps, verdict = result
        assert isinstance(feasible, bool)
        assert isinstance(risk, str)
        assert isinstance(steps, list)
        assert isinstance(verdict, str)

    def test_feasible_is_false_only_for_high_risk(self):
        _, _, _, verdict_high = _assess_rollback(None, AGENT13_DESTRUCTIVE_MEDIUM, None, AGENT41_PASS)
        _, _, _, verdict_medium = _assess_rollback(None, AGENT13_DESTRUCTIVE, None, AGENT41_PASS)
        assert verdict_high == "NOT_FEASIBLE"
        assert verdict_medium == "RISKY"


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_both_sources_gives_high_score(self):
        score, _ = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert score >= 65

    def test_metadata_available_adds_confidence(self):
        score_with, _ = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        score_without, _ = _compute_confidence(None, None)
        assert score_with > score_without

    def test_no_metadata_reduces_confidence(self):
        score_no_meta, _ = _compute_confidence(None, AGENT41_PASS)
        score_with_meta, _ = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert score_with_meta > score_no_meta

    def test_change_set_data_adds_confidence(self):
        score_with, _ = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        score_without, _ = _compute_confidence(AGENT13_CLEAN, None)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None)
        assert score >= 20

    def test_returns_signals_dict(self):
        _, signals = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert isinstance(signals, dict)

    def test_metadata_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert "metadata_available" in signals

    def test_no_metadata_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT41_PASS)
        assert "no_metadata" in signals

    def test_change_set_data_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT13_CLEAN, AGENT41_PASS)
        assert "change_set_data_available" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.agent_id == 48
        assert result.agent_name == "Rollback Readiness"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        for key in ["rollback_feasible", "rollback_risk", "rollback_steps", "rollback_verdict"]:
            assert key in result.data

    async def test_feasible_when_clean_change(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.data["rollback_verdict"] == "FEASIBLE"
        assert result.data["rollback_feasible"] is True
        assert result.data["rollback_risk"] == "LOW"

    async def test_not_feasible_when_destructive_and_deep(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_DESTRUCTIVE_MEDIUM}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.data["rollback_verdict"] == "NOT_FEASIBLE"
        assert result.data["rollback_feasible"] is False
        assert result.data["rollback_risk"] == "HIGH"

    async def test_risky_when_destructive_shallow(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_DESTRUCTIVE}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MEDIUM
            result = await run(state)

        assert result.data["rollback_verdict"] == "RISKY"
        assert result.data["rollback_risk"] == "MEDIUM"

    async def test_rollback_concern_propagated(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_DESTRUCTIVE}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_MEDIUM
            result = await run(state)

        assert result.data["rollback_concern"] == "destructive_changes"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_steps_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["rollback_steps"], list)
        assert len(result.data["rollback_steps"]) >= 1

    async def test_no_upstream_data_gives_feasible(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.data["rollback_verdict"] == "FEASIBLE"

    async def test_escalation_flag_set_correctly(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_CLEAN}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.confidence.escalated, bool)

    async def test_escalated_when_no_upstream_data(self):
        # base=55, no_metadata→-10=45, no agent41 → no delta → 45 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_48_rollback_readiness.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── REQ-31: new tests ─────────────────────────────────────────────────────────

AGENT8_EXT_DEPS    = {"has_external_dependencies": True}
AGENT8_NO_EXT_DEPS = {"has_external_dependencies": False}
AGENT40_MAJOR      = {"release_type": "MAJOR"}
AGENT40_PATCH      = {"release_type": "PATCH"}


class TestREQ31IntegrityFailElevatesRisk:
    def test_integrity_fail_elevates_to_medium_from_low(self):
        feasible, risk, steps, verdict = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_FAIL)
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_integrity_fail_adds_step(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_FAIL)
        assert any("integrity" in s.lower() for s in steps)

    def test_integrity_pass_clean_stays_feasible(self):
        feasible, risk, _, verdict = _assess_rollback(None, AGENT13_CLEAN, None, AGENT41_PASS)
        assert risk == "LOW"
        assert verdict == "FEASIBLE"


class TestREQ31ExternalDepsElevatesRisk:
    def test_external_deps_elevates_to_medium(self):
        feasible, risk, steps, verdict = _assess_rollback(AGENT8_EXT_DEPS, AGENT13_CLEAN, None, AGENT41_PASS)
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_external_deps_adds_step_about_named_credentials(self):
        _, _, steps, _ = _assess_rollback(AGENT8_EXT_DEPS, AGENT13_CLEAN, None, AGENT41_PASS)
        assert any("external" in s.lower() or "named credential" in s.lower() for s in steps)

    def test_no_external_deps_clean_stays_feasible(self):
        _, risk, _, verdict = _assess_rollback(AGENT8_NO_EXT_DEPS, AGENT13_CLEAN, None, AGENT41_PASS)
        assert risk == "LOW"
        assert verdict == "FEASIBLE"


class TestREQ31MajorReleaseElevatesRisk:
    def test_major_release_elevates_to_medium(self):
        _, risk, _, verdict = _assess_rollback(None, AGENT13_CLEAN, AGENT40_MAJOR, AGENT41_PASS)
        assert risk == "MEDIUM"
        assert verdict == "RISKY"

    def test_major_release_adds_schema_step(self):
        _, _, steps, _ = _assess_rollback(None, AGENT13_CLEAN, AGENT40_MAJOR, AGENT41_PASS)
        assert any("schema" in s.lower() or "migration" in s.lower() for s in steps)

    def test_patch_release_stays_feasible(self):
        _, risk, _, verdict = _assess_rollback(None, AGENT13_CLEAN, AGENT40_PATCH, AGENT41_PASS)
        assert risk == "LOW"
        assert verdict == "FEASIBLE"


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, "LOW", [], "FEASIBLE")
        assert "FSC-2417" in msg

    def test_includes_feasible_flag(self):
        msg = _build_trace_message("FSC-001", False, "HIGH", [], "NOT_FEASIBLE")
        assert "False" in msg

    def test_includes_risk_level(self):
        msg = _build_trace_message("FSC-001", True, "MEDIUM", [], "RISKY")
        assert "MEDIUM" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-001", False, "HIGH", [], "NOT_FEASIBLE")
        assert "NOT_FEASIBLE" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-001", True, "LOW", [], "FEASIBLE")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "rollback_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_rollback_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["rollback_concern"]["enum"] == [
            "none", "destructive_changes", "schema_migration",
            "data_writes", "high_complexity",
        ]


class TestREQ31ExistingBehaviourPreserved:
    def test_destructive_plus_deps_2_still_not_feasible(self):
        feasible, risk, _, verdict = _assess_rollback(None, AGENT13_DESTRUCTIVE_MEDIUM, None, AGENT41_PASS)
        assert feasible is False
        assert risk == "HIGH"
        assert verdict == "NOT_FEASIBLE"
