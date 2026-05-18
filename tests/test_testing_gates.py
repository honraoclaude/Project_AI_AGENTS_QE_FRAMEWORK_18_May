"""Tests for Testing Phase Gates G5 and G6."""

import pytest

from src.fleet_commander.phases.testing import (
    GateG5Error,
    GateG6Error,
    _check_gate_g5,
    _check_gate_g6,
)
from src.core.schemas import initial_story_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_with(agent_id: str, data: dict):
    state = initial_story_state("FSC-9999")
    state["agent_results"][agent_id] = {"data": data}
    return state


def _state_with_many(**kwargs):
    """_state_with_many(a33={...}, a34={...}) → state with those agents."""
    state = initial_story_state("FSC-9999")
    for key, data in kwargs.items():
        agent_id = key.lstrip("a")
        state["agent_results"][agent_id] = {"data": data}
    return state


# ── Gate G5 tests ─────────────────────────────────────────────────────────────

class TestGateG5:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g5(state)  # no exception = pass

    def test_passes_when_coverage_pass_and_no_defects(self):
        state = _state_with_many(
            a33={"coverage_verdict": "PASS", "overall_coverage_pct": 92.0},
            a34={"defect_verdict": "PASS", "critical_defects": [], "defect_count": 0},
        )
        _check_gate_g5(state)

    def test_passes_when_coverage_warn(self):
        state = _state_with_many(
            a33={"coverage_verdict": "WARN", "overall_coverage_pct": 87.0},
            a34={"defect_verdict": "PASS", "critical_defects": []},
        )
        _check_gate_g5(state)  # WARN is non-blocking at G5

    def test_fails_when_coverage_below_threshold(self):
        state = _state_with("33", {"coverage_verdict": "FAIL", "overall_coverage_pct": 60.0})
        with pytest.raises(GateG5Error) as exc_info:
            _check_gate_g5(state)
        assert "Coverage Analyser" in str(exc_info.value)
        assert "60%" in str(exc_info.value)

    def test_fails_when_critical_defects_present(self):
        state = _state_with("34", {
            "defect_verdict": "FAIL",
            "critical_defects": ["DEF-001", "DEF-002"],
            "defect_count": 2,
        })
        with pytest.raises(GateG5Error) as exc_info:
            _check_gate_g5(state)
        assert "Defect Triage" in str(exc_info.value)
        assert "DEF-001" in str(exc_info.value)

    def test_fails_when_both_coverage_and_defects_fail(self):
        state = _state_with_many(
            a33={"coverage_verdict": "FAIL", "overall_coverage_pct": 55.0},
            a34={"defect_verdict": "FAIL", "critical_defects": ["DEF-001"], "defect_count": 1},
        )
        with pytest.raises(GateG5Error) as exc_info:
            _check_gate_g5(state)
        msg = str(exc_info.value)
        assert "Coverage Analyser" in msg
        assert "Defect Triage" in msg

    def test_flaky_quarantine_does_not_block_g5(self):
        # Flaky test quarantine is informational — G5 does not check Agent 38
        state = _state_with_many(
            a33={"coverage_verdict": "PASS", "overall_coverage_pct": 92.0},
            a34={"defect_verdict": "PASS", "critical_defects": []},
            a38={"flaky_verdict": "QUARANTINE_REQUIRED", "flaky_count": 4},
        )
        _check_gate_g5(state)  # no exception = pass

    def test_defect_warn_does_not_block_g5(self):
        state = _state_with_many(
            a33={"coverage_verdict": "PASS", "overall_coverage_pct": 92.0},
            a34={"defect_verdict": "WARN", "critical_defects": [], "defect_count": 1},
        )
        _check_gate_g5(state)  # WARN is non-blocking

    def test_error_message_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["33"] = {"data": {"coverage_verdict": "FAIL", "overall_coverage_pct": 50.0}}
        with pytest.raises(GateG5Error) as exc_info:
            _check_gate_g5(state)
        assert "FSC-9999" in str(exc_info.value)


# ── Gate G6 tests ─────────────────────────────────────────────────────────────

class TestGateG6:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g6(state)  # no exception = pass

    def test_passes_when_rca_resolved_and_uat_not_required(self):
        state = _state_with_many(
            a35={"rca_verdict": "RESOLVED_PLAN", "fix_plan_complete": True, "root_causes": []},
            a36={"uat_coordination_verdict": "NOT_REQUIRED", "uat_sign_off_required": False},
        )
        _check_gate_g6(state)

    def test_passes_when_rca_no_action_and_uat_pending(self):
        # PENDING is not a hard block — story awaits async CO approval
        state = _state_with_many(
            a35={"rca_verdict": "NO_ACTION_REQUIRED", "root_causes": []},
            a36={"uat_coordination_verdict": "PENDING", "uat_sign_off_required": True},
        )
        _check_gate_g6(state)  # no exception = pass

    def test_fails_when_rca_incomplete(self):
        state = _state_with("35", {
            "rca_verdict": "INCOMPLETE",
            "root_causes": [{"defect_id": "DEF-001", "root_cause": "Unknown"}],
            "fix_plan_complete": False,
        })
        with pytest.raises(GateG6Error) as exc_info:
            _check_gate_g6(state)
        assert "Root Cause Analyser" in str(exc_info.value)
        assert "INCOMPLETE" in str(exc_info.value)

    def test_fails_when_uat_coordination_blocked(self):
        state = _state_with("36", {
            "uat_coordination_verdict": "BLOCKED",
            "uat_sign_off_required": True,
            "sign_off_request_sent": False,
        })
        with pytest.raises(GateG6Error) as exc_info:
            _check_gate_g6(state)
        assert "UAT Coordination" in str(exc_info.value)
        assert "BLOCKED" in str(exc_info.value)

    def test_fails_when_both_rca_incomplete_and_uat_blocked(self):
        state = _state_with_many(
            a35={"rca_verdict": "INCOMPLETE", "root_causes": [], "fix_plan_complete": False},
            a36={"uat_coordination_verdict": "BLOCKED", "uat_sign_off_required": True},
        )
        with pytest.raises(GateG6Error) as exc_info:
            _check_gate_g6(state)
        msg = str(exc_info.value)
        assert "Root Cause Analyser" in msg
        assert "UAT Coordination" in msg

    def test_signed_off_passes_g6(self):
        state = _state_with_many(
            a35={"rca_verdict": "RESOLVED_PLAN", "root_causes": []},
            a36={"uat_coordination_verdict": "SIGNED_OFF", "uat_sign_off_received": True},
        )
        _check_gate_g6(state)  # no exception = pass

    def test_error_message_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["35"] = {"data": {
            "rca_verdict": "INCOMPLETE",
            "root_causes": [],
            "fix_plan_complete": False,
        }}
        with pytest.raises(GateG6Error) as exc_info:
            _check_gate_g6(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_rca_resolved_plan_does_not_block(self):
        state = _state_with("35", {"rca_verdict": "RESOLVED_PLAN", "root_causes": []})
        _check_gate_g6(state)  # no exception = pass
