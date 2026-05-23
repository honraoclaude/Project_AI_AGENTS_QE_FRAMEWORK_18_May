"""Tests for Agent 31 — Financial Data Integrity (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_31_financial_data_integrity import (
    _build_trace_message,
    _check_integrity,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT13_FINANCIAL = {
    "detected_objects": ["financialaccount", "financialholding", "suitability__c"],
    "dependency_depth": 2,
}

AGENT13_MINIMAL = {
    "detected_objects": ["revenue__c"],
    "dependency_depth": 0,
}

AGENT13_EMPTY = {
    "detected_objects": [],
    "dependency_depth": 0,
}

AGENT27_PASS = {
    "crt_execution_verdict": "PASS",
    "tests_executed": 3,
    "crt_pass_count": 3,
    "crt_fail_count": 0,
}

AGENT27_FAIL = {
    "crt_execution_verdict": "FAIL",
    "tests_executed": 3,
    "crt_pass_count": 1,
    "crt_fail_count": 2,
}

AGENT27_SKIPPED = {
    "crt_execution_verdict": "SKIPPED",
    "tests_executed": 0,
}

MOCK_TRACE_PASS = {
    "narrative": "Financial data integrity PASSED. Balance consistency and audit trail rules checked for FinancialAccount and Suitability__c.",
    "integrity_concern": "none",
}

MOCK_TRACE_FAIL = {
    "narrative": "Financial data integrity WARN. CRT tests did not pass — integrity unconfirmed.",
    "integrity_concern": "audit_gap",
}


# ── Deterministic integrity check tests ──────────────────────────────────────

class TestIntegrityCheck:
    def test_financial_objects_with_passing_crt_gives_pass(self):
        valid, violations, rules, verdict = _check_integrity(
            AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS
        )
        assert verdict == "PASS"
        assert valid is True
        assert len(violations) == 0

    def test_crt_failed_adds_violation(self):
        valid, violations, rules, verdict = _check_integrity(
            AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_FAIL
        )
        assert len(violations) >= 1
        assert valid is False

    def test_crt_skipped_still_valid(self):
        valid, violations, rules, verdict = _check_integrity(
            AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_SKIPPED
        )
        assert verdict == "PASS"  # skipped CRT = not a violation
        assert valid is True

    def test_rules_checked_for_financial_objects(self):
        _, _, rules, _ = _check_integrity(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS)
        assert len(rules) >= 2  # balance_consistency, suitability_score_range, audit_trail

    def test_no_applicable_objects_still_valid(self):
        valid, violations, rules, verdict = _check_integrity(
            AGENT3_LOW, AGENT13_EMPTY, AGENT27_PASS
        )
        assert valid is True

    def test_no_upstream_data_degrades_gracefully(self):
        valid, violations, rules, verdict = _check_integrity(None, None, None)
        assert isinstance(valid, bool)
        assert isinstance(verdict, str)

    def test_warn_verdict_when_single_crt_violation(self):
        valid, violations, rules, verdict = _check_integrity(
            AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_FAIL
        )
        assert verdict == "WARN"
        assert len(violations) == 1
        assert valid is False

    def test_no_objects_in_scope_rule_added_for_empty_objects(self):
        _, _, rules, _ = _check_integrity(AGENT3_LOW, AGENT13_EMPTY, AGENT27_SKIPPED)
        assert "no_objects_in_scope" in rules

    def test_non_applicable_objects_give_empty_rules_no_no_objects_sentinel(self):
        # revenue__c does not match any _INTEGRITY_RULES applicable_objects set
        _, _, rules, _ = _check_integrity(AGENT3_LOW, AGENT13_MINIMAL, AGENT27_SKIPPED)
        assert rules == []
        assert "no_objects_in_scope" not in rules

    def test_revenueschedule_triggers_revenue_schedule_continuity(self):
        agent13_rev = {"detected_objects": ["revenueschedule"], "dependency_depth": 0}
        _, _, rules, _ = _check_integrity(AGENT3_LOW, agent13_rev, AGENT27_PASS)
        assert "revenue_schedule_continuity" in rules

    def test_all_four_rules_checked_for_financial_objects(self):
        _, _, rules, _ = _check_integrity(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS)
        assert set(rules) == {
            "balance_consistency",
            "suitability_score_range",
            "revenue_schedule_continuity",
            "audit_trail_completeness",
        }


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_with_crt_pass_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        assert score >= 70

    def test_no_metadata_scope_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        score_without, _ = _compute_confidence(AGENT3_HIGH, None, AGENT27_PASS, True)
        assert score_with > score_without

    def test_crt_skipped_reduces_confidence_vs_pass(self):
        score_pass, _ = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        score_skip, _ = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_SKIPPED, True)
        assert score_pass > score_skip

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, False)
        assert score >= 20

    def test_metadata_scope_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        assert "metadata_scope_available" in signals

    def test_no_metadata_scope_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, AGENT27_PASS, True)
        assert "no_metadata_scope" in signals

    def test_fca_classification_available_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        assert "fca_classification_available" in signals

    def test_crt_passed_supports_integrity_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, True)
        assert "crt_passed_supports_integrity" in signals

    def test_crt_skipped_reduced_confidence_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_SKIPPED, True)
        assert "crt_skipped_reduced_confidence" in signals

    def test_integrity_violations_found_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT13_FINANCIAL, AGENT27_PASS, False)
        assert "integrity_violations_found" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_FINANCIAL}
        state["agent_results"]["27"] = {"data": AGENT27_PASS}

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 31
        assert result.agent_name == "Financial Data Integrity"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["integrity_valid", "integrity_violations",
                    "integrity_verdict", "rules_checked"]:
            assert key in result.data

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=65, no_metadata_scope→-8, crt_skipped→-5, integrity_violations_found→-8 = 44 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── REQ-22: stub_mode in output ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStubModeREQ22:
    async def test_stub_mode_true_in_output(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data.get("stub_mode") is True

    async def test_stub_mode_present_with_objects_in_scope(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": {"detected_objects": ["FinancialAccount"]}}

        with patch("src.agents.testing.agent_31_financial_data_integrity.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data.get("stub_mode") is True


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", AGENT13_FINANCIAL, [], [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_objects_from_agent13(self):
        msg = _build_trace_message("FSC-2417", AGENT13_FINANCIAL, [], [], "PASS")
        assert "financialaccount" in msg

    def test_no_agent13_shows_unknown(self):
        msg = _build_trace_message("FSC-2417", None, [], [], "PASS")
        assert "unknown" in msg

    def test_includes_rules_checked(self):
        msg = _build_trace_message("FSC-2417", AGENT13_FINANCIAL, ["balance_consistency"], [], "PASS")
        assert "balance_consistency" in msg

    def test_no_rules_shows_none_applicable(self):
        msg = _build_trace_message("FSC-2417", AGENT13_FINANCIAL, [], [], "PASS")
        assert "['none applicable']" in msg

    def test_includes_violations(self):
        msg = _build_trace_message(
            "FSC-2417", AGENT13_FINANCIAL, [],
            ["CRT tests did not pass — integrity cannot be confirmed via automated tests"],
            "WARN",
        )
        assert "CRT" in msg

    def test_no_violations_shows_none(self):
        msg = _build_trace_message("FSC-2417", AGENT13_FINANCIAL, [], [], "PASS")
        assert "['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", None, [], [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "integrity_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_integrity_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["integrity_concern"]["enum"] == [
            "none", "balance_mismatch", "suitability_invalid", "audit_gap", "multiple",
        ]
