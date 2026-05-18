"""Tests for Development phase Gates G2, G3, G4."""

import pytest

from src.fleet_commander.phases.development import (
    GateG2Error,
    GateG3Error,
    GateG4Error,
    _check_gate_g2,
    _check_gate_g3,
    _check_gate_g4,
)
from src.core.schemas import initial_story_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_with(agent_data: dict[str, dict]) -> dict:
    state = initial_story_state("FSC-2417")
    for agent_id, data in agent_data.items():
        state["agent_results"][agent_id] = {"data": data}
    return state


# ── Gate G2: Story Integrity ──────────────────────────────────────────────────

class TestGateG2:
    def test_passes_when_all_critical_agents_pass(self):
        state = _state_with({
            "10": {"coverage_verdict": "PASS"},
            "12": {"coverage_verdict": "PASS", "coverage_pct": 90, "coverage_threshold": 85},
            "15": {"security_verdict": "PASS"},
        })
        _check_gate_g2(state)  # no exception

    def test_fails_when_ac_compliance_fails(self):
        state = _state_with({
            "10": {"coverage_verdict": "FAIL"},
            "12": {"coverage_verdict": "PASS", "coverage_pct": 90, "coverage_threshold": 85},
            "15": {"security_verdict": "PASS"},
        })
        with pytest.raises(GateG2Error, match="AC Compliance"):
            _check_gate_g2(state)

    def test_fails_when_coverage_fails(self):
        state = _state_with({
            "10": {"coverage_verdict": "PASS"},
            "12": {"coverage_verdict": "FAIL", "coverage_pct": 70, "coverage_threshold": 85},
            "15": {"security_verdict": "PASS"},
        })
        with pytest.raises(GateG2Error, match="Apex Coverage"):
            _check_gate_g2(state)

    def test_fails_when_security_review_required(self):
        state = _state_with({
            "10": {"coverage_verdict": "PASS"},
            "12": {"coverage_verdict": "PASS", "coverage_pct": 90, "coverage_threshold": 85},
            "15": {"security_verdict": "REVIEW_REQUIRED", "security_flags": ["CRUD/FLS"]},
        })
        with pytest.raises(GateG2Error, match="Apex Security"):
            _check_gate_g2(state)

    def test_fails_with_multiple_blockers(self):
        state = _state_with({
            "10": {"coverage_verdict": "FAIL"},
            "12": {"coverage_verdict": "FAIL", "coverage_pct": 65, "coverage_threshold": 85},
            "15": {"security_verdict": "REVIEW_REQUIRED", "security_flags": []},
        })
        with pytest.raises(GateG2Error):
            _check_gate_g2(state)

    def test_passes_when_agents_absent(self):
        """Absent agents default to no violation — gate is permissive on missing data."""
        state = initial_story_state("FSC-2417")
        _check_gate_g2(state)  # no exception when no agent data

    def test_error_message_contains_story_id(self):
        state = _state_with({
            "12": {"coverage_verdict": "FAIL", "coverage_pct": 60, "coverage_threshold": 85},
        })
        with pytest.raises(GateG2Error, match="FSC-2417"):
            _check_gate_g2(state)


# ── Gate G3: Code Quality ──────────────────────────────────────────────────────

class TestGateG3:
    def test_passes_when_quality_pass_no_soql_risk(self):
        state = _state_with({
            "14": {"quality_verdict": "PASS", "critical_violations": []},
            "20": {"performance_verdict": "PASS", "soql_loop_risk": False},
        })
        _check_gate_g3(state)  # no exception

    def test_fails_when_quality_verdict_fail(self):
        state = _state_with({
            "14": {"quality_verdict": "FAIL", "critical_violations": [{"rule_name": "ApexSOQLInjection"}]},
            "20": {"soql_loop_risk": False},
        })
        with pytest.raises(GateG3Error, match="Code Quality"):
            _check_gate_g3(state)

    def test_fails_when_soql_loop_risk_detected(self):
        state = _state_with({
            "14": {"quality_verdict": "PASS", "critical_violations": []},
            "20": {"soql_loop_risk": True},
        })
        with pytest.raises(GateG3Error, match="SOQL"):
            _check_gate_g3(state)

    def test_fails_with_both_blockers(self):
        state = _state_with({
            "14": {"quality_verdict": "FAIL", "critical_violations": [{"rule_name": "ApexXSSFromURLParam"}]},
            "20": {"soql_loop_risk": True},
        })
        with pytest.raises(GateG3Error):
            _check_gate_g3(state)

    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-2417")
        _check_gate_g3(state)  # no exception when no agent data

    def test_error_message_contains_story_id(self):
        state = _state_with({
            "14": {"quality_verdict": "FAIL", "critical_violations": []},
        })
        with pytest.raises(GateG3Error, match="FSC-2417"):
            _check_gate_g3(state)


# ── Gate G4: Development Phase Go/No-Go ──────────────────────────────────────

class TestGateG4:
    def test_passes_when_verdict_pass_and_sandbox_ready(self):
        state = _state_with({
            "23": {"development_verdict": "PASS", "critical_failures": []},
            "22": {"sandbox_ready": True, "sandbox_verdict": "READY", "sandbox_blockers": []},
        })
        _check_gate_g4(state)  # no exception

    def test_fails_when_development_verdict_fail(self):
        state = _state_with({
            "23": {"development_verdict": "FAIL", "critical_failures": ["Coverage FAIL"]},
            "22": {"sandbox_ready": True, "sandbox_verdict": "READY", "sandbox_blockers": []},
        })
        with pytest.raises(GateG4Error, match="Story-to-Code Tracer"):
            _check_gate_g4(state)

    def test_fails_when_development_verdict_partial(self):
        state = _state_with({
            "23": {"development_verdict": "PARTIAL", "critical_failures": []},
            "22": {"sandbox_ready": True, "sandbox_verdict": "READY", "sandbox_blockers": []},
        })
        with pytest.raises(GateG4Error, match="PARTIAL"):
            _check_gate_g4(state)

    def test_fails_when_sandbox_not_ready(self):
        state = _state_with({
            "23": {"development_verdict": "PASS", "critical_failures": []},
            "22": {
                "sandbox_ready": False,
                "sandbox_verdict": "BLOCKED",
                "sandbox_blockers": ["No branch found"],
            },
        })
        with pytest.raises(GateG4Error, match="Sandbox State"):
            _check_gate_g4(state)

    def test_fails_with_both_blockers(self):
        state = _state_with({
            "23": {"development_verdict": "FAIL", "critical_failures": ["Coverage FAIL"]},
            "22": {"sandbox_ready": False, "sandbox_verdict": "BLOCKED", "sandbox_blockers": []},
        })
        with pytest.raises(GateG4Error):
            _check_gate_g4(state)

    def test_passes_when_agents_absent(self):
        """Absent agent 23 defaults to UNKNOWN — gate requires explicit FAIL to block."""
        state = initial_story_state("FSC-2417")
        _check_gate_g4(state)  # no exception: development_verdict defaults to UNKNOWN (not FAIL)

    def test_error_message_contains_story_id(self):
        state = _state_with({
            "23": {"development_verdict": "FAIL", "critical_failures": ["Coverage FAIL"]},
        })
        with pytest.raises(GateG4Error, match="FSC-2417"):
            _check_gate_g4(state)
