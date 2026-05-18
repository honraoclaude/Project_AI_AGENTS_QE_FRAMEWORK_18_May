"""Tests for Agent 23 — Story-to-Code Tracer (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_23_story_code_tracer import (
    _classify_signals,
    _collect_agent_signals,
    _compute_confidence,
    _determine_verdict,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_state_all_pass() -> dict:
    state = initial_story_state("FSC-2417")
    state["agent_results"]["10"] = {"data": {"ac_compliance_verdict": "PASS", "coverage_verdict": "PASS"}}
    state["agent_results"]["11"] = {"data": {"branch_verdict": "PASS", "branch_found": True}}
    state["agent_results"]["12"] = {"data": {"coverage_verdict": "PASS", "coverage_passed": True, "coverage_pct": 90}}
    state["agent_results"]["13"] = {"data": {"dependency_depth": 1, "scope_delta": []}}
    state["agent_results"]["14"] = {"data": {"quality_verdict": "PASS", "critical_violations": []}}
    state["agent_results"]["15"] = {"data": {"security_verdict": "PASS", "security_flags": []}}
    state["agent_results"]["16"] = {"data": {"bulk_risk_level": "LOW"}}
    state["agent_results"]["17"] = {"data": {"sfdx_verdict": "PASS", "sfdx_format_valid": True}}
    state["agent_results"]["18"] = {"data": {"component_verdict": "PASS", "regulated_components": []}}
    state["agent_results"]["19"] = {"data": {"gherkin_verdict": "PASS", "scenario_count": 3}}
    state["agent_results"]["21"] = {"data": {"data_verdict": "PASS"}}
    state["agent_results"]["22"] = {"data": {"sandbox_verdict": "READY", "sandbox_ready": True}}
    return state


def _make_state_critical_fail() -> dict:
    state = initial_story_state("FSC-2417")
    state["agent_results"]["12"] = {"data": {"coverage_verdict": "FAIL", "coverage_passed": False, "coverage_pct": 65}}
    state["agent_results"]["14"] = {"data": {"quality_verdict": "FAIL", "critical_violations": [{"rule_name": "ApexSOQLInjection"}]}}
    state["agent_results"]["15"] = {"data": {"security_verdict": "REVIEW_REQUIRED"}}
    state["agent_results"]["22"] = {"data": {"sandbox_verdict": "READY", "sandbox_ready": True}}
    return state


MOCK_TRACE_PASS = {
    "narrative": "Development phase PASSED for FSC-2417: 90% Apex coverage, SFDX format valid, no security violations. All 12 dev agents completed successfully. Story cleared to proceed to Testing.",
    "audit_summary": "Development phase PASSED for FSC-2417 with 90% Apex coverage, SFDX valid, zero security violations.",
}

MOCK_TRACE_FAIL = {
    "narrative": "Development phase FAILED for FSC-2417: Apex coverage 65% below 85% threshold. SOQL injection violation detected. Security review required. Story must not proceed until critical failures are resolved.",
    "audit_summary": "Development phase FAILED for FSC-2417: coverage below threshold, SOQL injection, security review required.",
}


# ── Signal collection tests ───────────────────────────────────────────────────

class TestSignalCollection:
    def test_collects_signals_from_all_present_agents(self):
        state = _make_state_all_pass()
        signals = _collect_agent_signals(state)
        assert len(signals) >= 8

    def test_absent_agents_not_in_signals(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["12"] = {"data": {"coverage_verdict": "PASS", "coverage_passed": True}}
        signals = _collect_agent_signals(state)
        assert "12" in signals
        assert "14" not in signals  # not set

    def test_verdict_extracted_from_coverage_verdict_key(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["12"] = {"data": {"coverage_verdict": "PASS", "coverage_passed": True, "coverage_pct": 90}}
        signals = _collect_agent_signals(state)
        assert signals["12"]["verdict"] == "PASS"

    def test_verdict_extracted_from_security_verdict_key(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["15"] = {"data": {"security_verdict": "REVIEW_REQUIRED"}}
        signals = _collect_agent_signals(state)
        assert signals["15"]["verdict"] == "REVIEW_REQUIRED"


# ── Signal classification tests ───────────────────────────────────────────────

class TestSignalClassification:
    def test_critical_agent_fail_gives_critical_failure(self):
        signals = {
            "12": {"agent_name": "Apex Coverage", "verdict": "FAIL", "data": {}},
        }
        critical, _ = _classify_signals(signals)
        assert len(critical) >= 1
        assert any("Apex Coverage" in f for f in critical)

    def test_advisory_agent_warn_gives_advisory_warning(self):
        signals = {
            "17": {"agent_name": "SFDX Validator", "verdict": "WARN", "data": {}},
        }
        _, advisory = _classify_signals(signals)
        assert len(advisory) >= 1

    def test_all_pass_gives_no_failures(self):
        signals = {
            "12": {"agent_name": "Apex Coverage", "verdict": "PASS", "data": {}},
            "14": {"agent_name": "Code Quality", "verdict": "PASS", "data": {}},
            "15": {"agent_name": "Apex Security", "verdict": "PASS", "data": {}},
        }
        critical, advisory = _classify_signals(signals)
        assert len(critical) == 0
        assert len(advisory) == 0

    def test_review_required_from_critical_agent_is_critical_failure(self):
        signals = {
            "15": {"agent_name": "Apex Security", "verdict": "REVIEW_REQUIRED", "data": {}},
        }
        critical, _ = _classify_signals(signals)
        assert len(critical) >= 1


# ── Verdict determination tests ───────────────────────────────────────────────

class TestVerdictDetermination:
    def test_no_failures_gives_pass(self):
        verdict = _determine_verdict([], [], {})
        assert verdict == "PASS"

    def test_critical_failures_give_fail(self):
        verdict = _determine_verdict(["Coverage: FAIL"], [], {})
        assert verdict == "FAIL"

    def test_three_or_more_advisory_warnings_give_partial(self):
        verdict = _determine_verdict([], ["W1", "W2", "W3"], {})
        assert verdict == "PARTIAL"

    def test_two_advisory_warnings_give_pass(self):
        verdict = _determine_verdict([], ["W1", "W2"], {})
        assert verdict == "PASS"

    def test_blocked_sandbox_gives_fail(self):
        signals = {"22": {"verdict": "BLOCKED", "data": {"sandbox_ready": False}, "agent_name": "Sandbox State"}}
        verdict = _determine_verdict([], [], signals)
        assert verdict == "FAIL"


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_all_dev_agents_present_scores_high(self):
        state = _make_state_all_pass()
        signals = _collect_agent_signals(state)
        score, _ = _compute_confidence(signals, [], "PASS")
        assert score >= 75

    def test_critical_failures_reduce_confidence(self):
        state_pass = _make_state_all_pass()
        signals_pass = _collect_agent_signals(state_pass)
        score_pass, _ = _compute_confidence(signals_pass, [], "PASS")

        score_fail, _ = _compute_confidence({}, ["Coverage FAIL", "Security FAIL"], "FAIL")
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        state = _make_state_all_pass()
        signals = _collect_agent_signals(state)
        score, _ = _compute_confidence(signals, [], "PASS")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence({}, ["F1", "F2", "F3"], "FAIL")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 23
        assert result.agent_name == "Story-to-Code Tracer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["trace_record", "development_verdict", "gate_g4_signals", "escalation_required"]:
            assert key in result.data

    async def test_pass_verdict_with_all_agents_passing(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["development_verdict"] == "PASS"
        assert result.data["escalation_required"] is False

    async def test_fail_verdict_with_critical_failures(self):
        state = _make_state_critical_fail()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["development_verdict"] == "FAIL"
        assert result.data["escalation_required"] is True
        assert len(result.data["critical_failures"]) >= 1

    async def test_trace_record_contains_story_id(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["trace_record"]["story_id"] == "FSC-2417"

    async def test_gate_g4_signals_present(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        g4 = result.data["gate_g4_signals"]
        assert "coverage_passed" in g4
        assert "security_verdict" in g4
        assert "quality_verdict" in g4

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
