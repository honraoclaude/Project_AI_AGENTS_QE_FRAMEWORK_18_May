"""Tests for Agent 31 — Financial Data Integrity (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_31_financial_data_integrity import (
    _check_integrity,
    _compute_confidence,
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
