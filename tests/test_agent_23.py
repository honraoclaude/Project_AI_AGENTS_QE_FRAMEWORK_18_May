"""Tests for Agent 23 — Story-to-Code Tracer (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_23_story_code_tracer import (
    _build_gate_g4_signals,
    _build_trace_message,
    _classify_signals,
    _collect_agent_signals,
    _compute_confidence,
    _determine_verdict,
    _extract_verdict,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
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

    def test_verdict_unknown_when_no_known_key(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["22"] = {"data": {"some_unrecognised_key": "PASS"}}
        signals = _collect_agent_signals(state)
        assert signals["22"]["verdict"] == "UNKNOWN"

    def test_generic_verdict_key_used_as_last_resort(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["22"] = {"data": {"verdict": "READY"}}
        signals = _collect_agent_signals(state)
        assert signals["22"]["verdict"] == "READY"


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

    def test_partial_verdict_from_advisory_agent_gives_advisory_warning(self):
        signals = {
            "20": {"agent_name": "Performance Risk", "verdict": "PARTIAL", "data": {}},
        }
        _, advisory = _classify_signals(signals)
        assert any("PARTIAL" in a for a in advisory)

    def test_review_required_from_non_critical_agent_is_advisory(self):
        signals = {
            "13": {"agent_name": "Metadata Dependency", "verdict": "REVIEW_REQUIRED", "data": {}},
        }
        critical, advisory = _classify_signals(signals)
        assert len(critical) == 0
        assert any("REVIEW_REQUIRED" in a for a in advisory)

    def test_medium_fca_gherkin_incomplete_is_critical(self):
        signals = {
            "19": {"agent_name": "BDD Gherkin", "verdict": "INCOMPLETE", "data": {}},
            "21": {"agent_name": "Test Data Architect", "verdict": "PASS",
                   "data": {"isolation_override": False}},
        }
        critical, _ = _classify_signals(signals, fca_class="MEDIUM")
        assert any("BDD Gherkin" in f or "INCOMPLETE" in f for f in critical)


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

    def test_degraded_sandbox_does_not_cause_fail(self):
        # DEGRADED is not BLOCKED — verdict falls through to PASS
        signals = {"22": {"verdict": "DEGRADED", "data": {"sandbox_ready": False}, "agent_name": "Sandbox State"}}
        verdict = _determine_verdict([], [], signals)
        assert verdict == "PASS"


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

    def test_five_to_seven_agents_gives_some_completed(self):
        signals_6 = {str(i): {"agent_name": f"Agent{i}", "verdict": "PASS", "data": {}} for i in range(6)}
        _, signals = _compute_confidence(signals_6, [], "PASS")
        assert "some_dev_agents_completed" in signals

    def test_most_dev_agents_completed_stores_count(self):
        state = _make_state_all_pass()
        agent_signals = _collect_agent_signals(state)
        _, signals = _compute_confidence(agent_signals, [], "PASS")
        assert signals["most_dev_agents_completed"] == len(agent_signals)

    def test_few_dev_agents_completed_key_in_signals(self):
        _, signals = _compute_confidence({}, [], "PASS")
        assert "few_dev_agents_completed" in signals

    def test_all_critical_agents_present_key_in_signals(self):
        state = _make_state_all_pass()
        agent_signals = _collect_agent_signals(state)
        _, signals = _compute_confidence(agent_signals, [], "PASS")
        assert "all_critical_agents_present" in signals

    def test_some_critical_agents_missing_stores_count(self):
        _, signals = _compute_confidence({}, [], "PASS")
        assert "some_critical_agents_missing" in signals
        assert signals["some_critical_agents_missing"] == 0

    def test_critical_failures_detected_stores_count(self):
        _, signals = _compute_confidence({}, ["Coverage: FAIL"], "FAIL")
        assert "critical_failures_detected" in signals
        assert signals["critical_failures_detected"] == 1


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

    async def test_escalated_when_no_upstream_data(self):
        # base=65, few_dev_agents=-8, some_critical_missing=-5 → 52 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = _make_state_all_pass()

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["trace_record"]["narrative"], str)


# ── REQ-15: isolation_override + FCA-conditional INCOMPLETE criticality ───────

class TestIsolationOverrideAndFCAIncompleteREQ15:
    def test_high_fca_gherkin_incomplete_is_critical(self):
        """Option B: HIGH-FCA + gherkin INCOMPLETE → critical_failures."""
        signals = {
            "19": {"agent_name": "BDD Gherkin", "verdict": "INCOMPLETE", "data": {}},
            "21": {"agent_name": "Test Data Architect", "verdict": "PASS",
                   "data": {"isolation_override": False}},
        }
        critical, advisory = _classify_signals(signals, fca_class="HIGH")
        assert any("BDD Gherkin" in f or "INCOMPLETE" in f for f in critical)

    def test_high_fca_data_incomplete_is_critical(self):
        """Option B: HIGH-FCA + data INCOMPLETE → critical_failures."""
        signals = {
            "21": {"agent_name": "Test Data Architect", "verdict": "INCOMPLETE",
                   "data": {"isolation_override": False}},
        }
        critical, _ = _classify_signals(signals, fca_class="HIGH")
        assert any("Test Data Architect" in f or "INCOMPLETE" in f for f in critical)

    def test_low_fca_gherkin_incomplete_is_only_advisory(self):
        """LOW-FCA + INCOMPLETE → advisory only, not critical."""
        signals = {
            "19": {"agent_name": "BDD Gherkin", "verdict": "INCOMPLETE", "data": {}},
            "21": {"agent_name": "Test Data Architect", "verdict": "PASS",
                   "data": {"isolation_override": False}},
        }
        critical, advisory = _classify_signals(signals, fca_class="LOW")
        assert not any("BDD Gherkin" in f for f in critical)
        assert any("INCOMPLETE" in a for a in advisory)

    def test_isolation_override_gives_advisory_warning(self):
        """isolation_override=True → advisory_warnings contains override message."""
        signals = {
            "21": {
                "agent_name": "Test Data Architect",
                "verdict": "PASS",
                "data": {
                    "isolation_override": True,
                    "isolation_override_reason": "Overridden for HIGH FCA",
                },
            },
        }
        _, advisory = _classify_signals(signals, fca_class="HIGH")
        assert any("isolation" in a.lower() or "corrected" in a.lower() for a in advisory)

    def test_no_isolation_override_no_extra_advisory(self):
        signals = {
            "21": {"agent_name": "Test Data Architect", "verdict": "PASS",
                   "data": {"isolation_override": False}},
        }
        _, advisory = _classify_signals(signals, fca_class="HIGH")
        assert not any("isolation" in a.lower() for a in advisory)


@pytest.mark.asyncio
class TestIsolationOverrideInRunREQ15:
    async def test_isolation_override_in_trace_record(self):
        state = _make_state_all_pass()
        state["agent_results"]["21"] = {
            "data": {
                "data_verdict": "PASS",
                "isolation_override": True,
                "isolation_override_reason": "Overridden for HIGH FCA",
            }
        }

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["isolation_override"] is True
        assert result.data["trace_record"]["isolation_override"] is True

    async def test_isolation_override_in_gate_g4_signals(self):
        state = _make_state_all_pass()
        state["agent_results"]["21"] = {
            "data": {
                "data_verdict": "PASS",
                "isolation_override": True,
                "isolation_override_reason": "Overridden for HIGH FCA",
            }
        }

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["gate_g4_signals"].get("isolation_override") is True

    async def test_high_fca_gherkin_incomplete_triggers_fail_verdict(self):
        """REQ-15 Option B in run(): HIGH-FCA + INCOMPLETE → FAIL."""
        state = _make_state_all_pass()
        state["agent_results"]["3"] = {"data": {"fca_classification": "HIGH"}}
        state["agent_results"]["19"] = {"data": {"gherkin_verdict": "INCOMPLETE", "scenario_count": 0}}

        with patch("src.agents.development.agent_23_story_code_tracer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.data["development_verdict"] == "FAIL"
        assert result.data["escalation_required"] is True


# ── _extract_verdict unit tests ───────────────────────────────────────────────

class TestExtractVerdict:
    def test_returns_unknown_when_no_known_key(self):
        result = _extract_verdict("99", {"unrelated_key": "PASS"})
        assert result == "UNKNOWN"

    def test_generic_verdict_key_used_as_last_resort(self):
        result = _extract_verdict("99", {"verdict": "READY"})
        assert result == "READY"

    def test_quality_verdict_key_returned(self):
        result = _extract_verdict("14", {"quality_verdict": "FAIL", "other": "stuff"})
        assert result == "FAIL"


# ── _build_gate_g4_signals unit tests ────────────────────────────────────────

class TestBuildGateG4Signals:
    def test_includes_sfdx_valid(self):
        agent_signals = {
            "17": {"agent_name": "SFDX Validator", "verdict": "PASS",
                   "data": {"sfdx_format_valid": True}},
        }
        g4 = _build_gate_g4_signals(agent_signals, [], [])
        assert g4["sfdx_valid"] is True

    def test_includes_sandbox_ready(self):
        agent_signals = {
            "22": {"agent_name": "Sandbox State", "verdict": "READY",
                   "data": {"sandbox_ready": True}},
        }
        g4 = _build_gate_g4_signals(agent_signals, [], [])
        assert g4["sandbox_ready"] is True

    def test_critical_failure_count_matches(self):
        g4 = _build_gate_g4_signals({}, ["failure1", "failure2"], [])
        assert g4["critical_failure_count"] == 2

    def test_advisory_warning_count_matches(self):
        g4 = _build_gate_g4_signals({}, [], ["w1", "w2", "w3"])
        assert g4["advisory_warning_count"] == 3


# ── _build_trace_message unit tests ──────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", {}, [], [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_agent_verdicts_for_each_signal(self):
        signals = {"14": {"agent_name": "Code Quality", "verdict": "PASS", "data": {}}}
        msg = _build_trace_message("FSC-2417", signals, [], [], "PASS")
        assert "Code Quality" in msg
        assert "PASS" in msg

    def test_no_critical_failures_shows_none(self):
        msg = _build_trace_message("FSC-2417", {}, [], [], "PASS")
        assert "['none']" in msg

    def test_dev_verdict_shown(self):
        msg = _build_trace_message("FSC-2417", {}, ["Coverage: FAIL"], [], "FAIL")
        assert "Development Verdict: FAIL" in msg

    def test_agent_ids_sorted_in_verdict_lines(self):
        signals = {
            "22": {"agent_name": "Sandbox", "verdict": "READY", "data": {}},
            "10": {"agent_name": "AC Compliance", "verdict": "PASS", "data": {}},
        }
        msg = _build_trace_message("FSC-2417", signals, [], [], "PASS")
        pos_10 = msg.index("Agent 10")
        pos_22 = msg.index("Agent 22")
        assert pos_10 < pos_22

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", {}, [], [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "audit_summary"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_audit_summary_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["audit_summary"]["type"] == "string"
