"""Tests for Agent 15 — Apex Security Scanner (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_15_apex_security import (
    _analyse_security_risk,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT13_SUITABILITY = {
    "detected_objects": ["suitability__c", "riskprofile__c", "vulnerablecustomerindicator__c"],
    "dependency_depth": 2,
    "changed_files_count": 3,
}

AGENT13_FINANCIAL = {
    "detected_objects": ["financialaccount", "financialholding"],
    "dependency_depth": 1,
    "changed_files_count": 2,
}

AGENT13_EMPTY = {
    "detected_objects": [],
    "dependency_depth": 0,
    "changed_files_count": 1,
}

MOCK_TRACE_HIGH = {
    "narrative": "HIGH-FCA story touches Suitability__c and RiskProfile__c — CRUD/FLS and sharing model review required.",
    "security_concern": "high",
}

MOCK_TRACE_LOW = {
    "narrative": "No high-risk FSC objects detected. Standard security review applies.",
    "security_concern": "none",
}


# ── Deterministic security analysis tests ─────────────────────────────────────

class TestSecurityAnalysis:
    def test_high_fca_with_high_risk_objects_gives_high_risk(self):
        risk, flags, crud, sharing, verdict = _analyse_security_risk(
            AGENT3_HIGH, AGENT13_SUITABILITY
        )
        assert risk == "HIGH"
        assert crud is True
        assert sharing is True
        assert verdict == "REVIEW_REQUIRED"

    def test_crud_required_when_high_risk_objects_present(self):
        _, _, crud, _, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_SUITABILITY)
        assert crud is True

    def test_sharing_required_for_high_fca(self):
        _, _, _, sharing, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_FINANCIAL)
        assert sharing is True

    def test_sharing_required_for_medium_fca(self):
        _, _, _, sharing, _ = _analyse_security_risk(AGENT3_MEDIUM, AGENT13_FINANCIAL)
        assert sharing is True

    def test_sharing_not_required_for_low_fca_non_high_risk(self):
        _, _, _, sharing, _ = _analyse_security_risk(AGENT3_LOW, AGENT13_FINANCIAL)
        assert sharing is False

    def test_low_risk_when_no_fsc_objects_and_low_fca(self):
        risk, _, crud, _, verdict = _analyse_security_risk(AGENT3_LOW, AGENT13_EMPTY)
        assert risk == "LOW"
        assert crud is False
        assert verdict == "PASS"

    def test_flags_list_not_empty_for_high_risk(self):
        _, flags, _, _, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_SUITABILITY)
        assert len(flags) >= 1

    def test_no_agent13_data_degrades_gracefully(self):
        risk, flags, crud, sharing, verdict = _analyse_security_risk(AGENT3_HIGH, None)
        assert risk in ("LOW", "MEDIUM")
        assert crud is False

    def test_no_agent3_data_degrades_gracefully(self):
        risk, _, _, _, _ = _analyse_security_risk(None, AGENT13_SUITABILITY)
        assert risk in ("MEDIUM", "HIGH")  # high-risk objects still detected

    def test_deep_dependency_adds_flag(self):
        agent13_deep = {**AGENT13_SUITABILITY, "dependency_depth": 3}
        _, flags, _, _, _ = _analyse_security_risk(AGENT3_HIGH, agent13_deep)
        assert any("depth" in f.lower() or "chain" in f.lower() for f in flags)

    def test_medium_fca_no_fsc_objects_gives_pass(self):
        """REQ-09: MEDIUM-FCA + no regulated objects → PASS, not REVIEW_REQUIRED."""
        risk, _, crud, sharing, verdict = _analyse_security_risk(AGENT3_MEDIUM, AGENT13_EMPTY)
        assert risk == "LOW"
        assert verdict == "PASS"
        assert crud is False
        assert sharing is False

    def test_high_fca_no_fsc_objects_gives_pass(self):
        """REQ-09: HIGH-FCA + no regulated objects → LOW risk / PASS."""
        risk, _, crud, sharing, verdict = _analyse_security_risk(AGENT3_HIGH, AGENT13_EMPTY)
        assert risk == "LOW"
        assert verdict == "PASS"
        assert sharing is False

    def test_financialaccount_detected_gives_review_required(self):
        """REQ-09: financialaccount is now in _HIGH_RISK_OBJECTS — triggers REVIEW_REQUIRED."""
        agent13_aum = {
            "detected_objects": ["financialaccount"],
            "dependency_depth": 1,
        }
        _, _, crud, _, verdict = _analyse_security_risk(AGENT3_HIGH, agent13_aum)
        assert verdict == "REVIEW_REQUIRED"
        assert crud is True

    def test_financialholding_detected_gives_crud_required(self):
        """REQ-09: financialholding is now in _HIGH_RISK_OBJECTS — CRUD/FLS required."""
        agent13_aum = {
            "detected_objects": ["financialholding"],
            "dependency_depth": 1,
        }
        _, _, crud, _, _ = _analyse_security_risk(AGENT3_LOW, agent13_aum)
        assert crud is True

    def test_sharing_required_only_when_fca_elevated_and_objects_present(self):
        """REQ-09: sharing_required = HIGH/MEDIUM FCA AND high-risk objects — not just FCA tier."""
        # HIGH FCA but no high-risk objects → sharing NOT required
        _, _, _, sharing_no_objects, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_EMPTY)
        assert sharing_no_objects is False
        # HIGH FCA + high-risk objects → sharing required
        _, _, _, sharing_with_objects, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_SUITABILITY)
        assert sharing_with_objects is True

    def test_medium_fca_with_high_risk_objects_gives_medium_risk(self):
        risk, _, _, _, verdict = _analyse_security_risk(AGENT3_MEDIUM, AGENT13_SUITABILITY)
        assert risk == "MEDIUM"
        assert verdict == "REVIEW_REQUIRED"

    def test_low_fca_with_high_risk_objects_gives_medium_risk(self):
        risk, _, _, _, verdict = _analyse_security_risk(AGENT3_LOW, AGENT13_FINANCIAL)
        assert risk == "MEDIUM"
        assert verdict == "REVIEW_REQUIRED"

    def test_high_fca_no_fsc_objects_flag_mentions_no_review_required(self):
        _, flags, _, _, _ = _analyse_security_risk(AGENT3_HIGH, AGENT13_EMPTY)
        assert any("no regulated" in f.lower() or "not required" in f.lower() for f in flags)


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_both_agents_available_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        assert score >= 70

    def test_no_fca_class_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        score_without, _ = _compute_confidence(None, AGENT13_SUITABILITY, ["suitability__c"])
        assert score_with > score_without

    def test_no_metadata_scope_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        score_without, _ = _compute_confidence(AGENT3_HIGH, None, [])
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, [])
        assert score >= 20

    def test_fca_class_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        assert "fca_class_available" in signals

    def test_fca_class_unavailable_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT13_SUITABILITY, ["suitability__c"])
        assert "fca_class_unavailable" in signals

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        assert "metadata_scope_available" in signals

    def test_metadata_scope_unavailable_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, [])
        assert "metadata_scope_unavailable" in signals

    def test_fsc_objects_in_scope_stores_count(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_SUITABILITY, ["suitability__c"])
        assert signals["fsc_objects_in_scope"] == 1

    def test_no_fsc_objects_detected_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_EMPTY, [])
        assert "no_fsc_objects_detected" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_SUITABILITY}

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.agent_id == 15
        assert result.agent_name == "Apex Security Scanner"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        for key in ["security_verdict", "crud_fls_review_required",
                    "sharing_model_review_required", "security_flags"]:
            assert key in result.data

    async def test_high_risk_verdict_for_suitability_high_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_SUITABILITY}

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert result.data["security_verdict"] == "REVIEW_REQUIRED"
        assert result.data["crud_fls_review_required"] is True

    async def test_pass_verdict_for_low_fca_no_risk_objects(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_LOW}
        state["agent_results"]["13"] = {"data": AGENT13_EMPTY}

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.data["security_verdict"] == "PASS"

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_LOW
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_SUITABILITY}

        with patch("src.agents.development.agent_15_apex_security.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_HIGH
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], [], "HIGH", True, True)
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], [], "HIGH", True, True)
        assert "HIGH" in msg

    def test_includes_risk_level(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], [], "HIGH", True, True)
        assert "Security risk level: HIGH" in msg

    def test_empty_detected_shows_none(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], [], "HIGH", True, True)
        assert "Detected FSC objects: ['none']" in msg

    def test_detected_objects_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", "HIGH", ["suitability__c"], [], "HIGH", True, True)
        assert "suitability__c" in msg

    def test_empty_flags_shows_none(self):
        msg = _build_trace_message("FSC-2417", "LOW", [], [], "LOW", False, False)
        assert "Security flags: ['none']" in msg

    def test_flags_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", "HIGH", [], ["CRUD/FLS enforcement required"], "HIGH", True, False)
        assert "CRUD/FLS enforcement required" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", "LOW", [], [], "LOW", False, False)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "security_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_security_concern_enum_has_four_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["security_concern"]["enum"] == ["none", "low", "medium", "high"]
