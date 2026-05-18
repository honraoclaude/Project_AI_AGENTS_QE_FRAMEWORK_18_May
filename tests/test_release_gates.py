"""Tests for Release Phase Gates G7–G12."""

import pytest

from src.fleet_commander.phases.release import (
    GateG7Error,
    GateG8Error,
    GateG9Error,
    GateG10Error,
    GateG11Error,
    _check_gate_g7,
    _check_gate_g8,
    _check_gate_g9,
    _check_gate_g10,
    _check_gate_g11,
    _check_gate_g12,
)
from src.core.schemas import initial_story_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_with(agent_id: str, data: dict):
    state = initial_story_state("FSC-9999")
    state["agent_results"][agent_id] = {"data": data}
    return state


def _state_with_many(**kwargs):
    """_state_with_many(a39={...}, a41={...}) → state with those agents."""
    state = initial_story_state("FSC-9999")
    for key, data in kwargs.items():
        agent_id = key.lstrip("a")
        state["agent_results"][agent_id] = {"data": data}
    return state


# ── Gate G7 — Release Readiness ───────────────────────────────────────────────

class TestGateG7:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g7(state)  # no exception = pass

    def test_passes_when_readiness_ready(self):
        state = _state_with("39", {"readiness_verdict": "READY", "readiness_blockers": []})
        _check_gate_g7(state)

    def test_passes_when_readiness_partial(self):
        # PARTIAL = UAT pending — not a hard block at G7
        state = _state_with("39", {"readiness_verdict": "PARTIAL", "readiness_blockers": []})
        _check_gate_g7(state)

    def test_fails_when_readiness_blocked(self):
        state = _state_with("39", {
            "readiness_verdict": "BLOCKED",
            "readiness_blockers": ["Coverage 60% below threshold", "RCA INCOMPLETE"],
        })
        with pytest.raises(GateG7Error) as exc_info:
            _check_gate_g7(state)
        assert "Release Readiness" in str(exc_info.value)
        assert "BLOCKED" in str(exc_info.value)

    def test_error_message_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["39"] = {"data": {
            "readiness_verdict": "BLOCKED",
            "readiness_blockers": ["test blocker"],
        }}
        with pytest.raises(GateG7Error) as exc_info:
            _check_gate_g7(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_error_message_lists_blockers(self):
        state = _state_with("39", {
            "readiness_verdict": "BLOCKED",
            "readiness_blockers": ["Coverage FAIL", "Defect FAIL"],
        })
        with pytest.raises(GateG7Error) as exc_info:
            _check_gate_g7(state)
        msg = str(exc_info.value)
        assert "Coverage FAIL" in msg or "Defect FAIL" in msg

    def test_blocker_count_in_error(self):
        state = _state_with("39", {
            "readiness_verdict": "BLOCKED",
            "readiness_blockers": ["b1", "b2"],
        })
        with pytest.raises(GateG7Error) as exc_info:
            _check_gate_g7(state)
        assert "2" in str(exc_info.value)


# ── Gate G8 — Change Set Integrity ───────────────────────────────────────────

class TestGateG8:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g8(state)

    def test_passes_when_composer_composed_and_integrity_pass(self):
        state = _state_with_many(
            a40={"composer_verdict": "COMPOSED", "component_count": 5},
            a41={"integrity_verdict": "PASS", "integrity_issues": []},
        )
        _check_gate_g8(state)

    def test_fails_when_composer_failed(self):
        state = _state_with("40", {"composer_verdict": "FAILED"})
        with pytest.raises(GateG8Error) as exc_info:
            _check_gate_g8(state)
        assert "Release Composer" in str(exc_info.value)
        assert "Agent 40" in str(exc_info.value)

    def test_fails_when_integrity_fail(self):
        state = _state_with("41", {
            "integrity_verdict": "FAIL",
            "integrity_issues": ["Missing dependency: FinancialAccount__c"],
        })
        with pytest.raises(GateG8Error) as exc_info:
            _check_gate_g8(state)
        assert "Change Set Integrity" in str(exc_info.value)
        assert "Agent 41" in str(exc_info.value)

    def test_passes_when_integrity_warn(self):
        # WARN = large/destructive changes — non-blocking
        state = _state_with_many(
            a40={"composer_verdict": "COMPOSED"},
            a41={"integrity_verdict": "WARN", "integrity_issues": ["Large change set (25 components)"]},
        )
        _check_gate_g8(state)  # no exception = pass

    def test_composer_failed_raises_before_integrity_check(self):
        # Even if integrity is absent, FAILED composer raises immediately
        state = _state_with("40", {"composer_verdict": "FAILED"})
        with pytest.raises(GateG8Error):
            _check_gate_g8(state)

    def test_error_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["41"] = {"data": {
            "integrity_verdict": "FAIL",
            "integrity_issues": ["bad dep"],
        }}
        with pytest.raises(GateG8Error) as exc_info:
            _check_gate_g8(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_integrity_issues_in_error_message(self):
        state = _state_with("41", {
            "integrity_verdict": "FAIL",
            "integrity_issues": ["Missing dependency: GoalProduct__c"],
        })
        with pytest.raises(GateG8Error) as exc_info:
            _check_gate_g8(state)
        assert "GoalProduct__c" in str(exc_info.value)

    def test_composer_partial_passes_g8(self):
        # PARTIAL composer = incomplete metadata but not fully FAILED
        state = _state_with_many(
            a40={"composer_verdict": "PARTIAL"},
            a41={"integrity_verdict": "PASS", "integrity_issues": []},
        )
        _check_gate_g8(state)  # no exception = pass


# ── Gate G9 — Staging ─────────────────────────────────────────────────────────

class TestGateG9:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g9(state)

    def test_passes_when_dry_run_pass_and_smoke_pass(self):
        state = _state_with_many(
            a42={"dry_run_verdict": "PASS", "dry_run_errors": []},
            a43={"smoke_verdict": "PASS", "smoke_failed": 0},
        )
        _check_gate_g9(state)

    def test_fails_when_dry_run_fails(self):
        state = _state_with("42", {
            "dry_run_verdict": "FAIL",
            "dry_run_errors": ["ApexClass GoalService: compile error"],
        })
        with pytest.raises(GateG9Error) as exc_info:
            _check_gate_g9(state)
        assert "Dry Run" in str(exc_info.value)
        assert "Agent 42" in str(exc_info.value)

    def test_fails_when_smoke_tests_fail(self):
        state = _state_with("43", {
            "smoke_verdict": "FAIL",
            "smoke_failed": 3,
        })
        with pytest.raises(GateG9Error) as exc_info:
            _check_gate_g9(state)
        assert "Smoke Tests" in str(exc_info.value)
        assert "Agent 43" in str(exc_info.value)

    def test_fails_when_both_dry_run_and_smoke_fail(self):
        state = _state_with_many(
            a42={"dry_run_verdict": "FAIL", "dry_run_errors": ["compile error"]},
            a43={"smoke_verdict": "FAIL", "smoke_failed": 2},
        )
        with pytest.raises(GateG9Error) as exc_info:
            _check_gate_g9(state)
        msg = str(exc_info.value)
        assert "Dry Run" in msg
        assert "Smoke Tests" in msg

    def test_passes_when_dry_run_skipped(self):
        state = _state_with_many(
            a42={"dry_run_verdict": "SKIPPED", "dry_run_errors": []},
            a43={"smoke_verdict": "PASS", "smoke_failed": 0},
        )
        _check_gate_g9(state)  # SKIPPED is non-blocking

    def test_passes_when_smoke_skipped(self):
        state = _state_with_many(
            a42={"dry_run_verdict": "PASS", "dry_run_errors": []},
            a43={"smoke_verdict": "SKIPPED", "smoke_failed": 0},
        )
        _check_gate_g9(state)  # SKIPPED is non-blocking

    def test_error_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["42"] = {"data": {
            "dry_run_verdict": "FAIL",
            "dry_run_errors": ["err"],
        }}
        with pytest.raises(GateG9Error) as exc_info:
            _check_gate_g9(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_smoke_failed_count_in_error(self):
        state = _state_with("43", {"smoke_verdict": "FAIL", "smoke_failed": 5})
        with pytest.raises(GateG9Error) as exc_info:
            _check_gate_g9(state)
        assert "5" in str(exc_info.value)


# ── Gate G10 — FCA Evidence ───────────────────────────────────────────────────

class TestGateG10:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g10(state)

    def test_passes_when_evidence_complete(self):
        state = _state_with("44", {
            "evidence_verdict": "COMPLETE",
            "evidence_gaps": [],
            "consumer_duty_covered": True,
        })
        _check_gate_g10(state)

    def test_passes_when_evidence_partial(self):
        # PARTIAL = some gaps remain — CO resolves externally; not a hard block
        state = _state_with("44", {
            "evidence_verdict": "PARTIAL",
            "evidence_gaps": ["COBS 9.2A"],
        })
        _check_gate_g10(state)  # no exception = pass

    def test_fails_when_evidence_missing(self):
        state = _state_with("44", {
            "evidence_verdict": "MISSING",
            "evidence_gaps": ["COBS 9", "Consumer Duty PS22/9"],
        })
        with pytest.raises(GateG10Error) as exc_info:
            _check_gate_g10(state)
        assert "FCA Evidence" in str(exc_info.value)

    def test_error_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["44"] = {"data": {
            "evidence_verdict": "MISSING",
            "evidence_gaps": ["COBS 9"],
        }}
        with pytest.raises(GateG10Error) as exc_info:
            _check_gate_g10(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_error_lists_missing_evidence_gaps(self):
        state = _state_with("44", {
            "evidence_verdict": "MISSING",
            "evidence_gaps": ["COBS 9", "MiFID II Art.25"],
        })
        with pytest.raises(GateG10Error) as exc_info:
            _check_gate_g10(state)
        msg = str(exc_info.value)
        assert "COBS 9" in msg or "MiFID II" in msg


# ── Gate G11 — Go/No-Go ───────────────────────────────────────────────────────

class TestGateG11:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g11(state)

    def test_passes_when_coordinator_go(self):
        state = _state_with("45", {
            "coordinator_verdict": "GO",
            "go_decision": True,
            "no_go_reasons": [],
        })
        _check_gate_g11(state)

    def test_passes_when_coordinator_conditional(self):
        # CONDITIONAL = GO with compliance conditions — not a hard block
        state = _state_with("45", {
            "coordinator_verdict": "CONDITIONAL",
            "go_decision": True,
            "no_go_reasons": [],
        })
        _check_gate_g11(state)  # no exception = pass

    def test_fails_when_coordinator_no_go(self):
        state = _state_with("45", {
            "coordinator_verdict": "NO_GO",
            "go_decision": False,
            "no_go_reasons": ["Smoke tests failed", "Coverage below threshold"],
        })
        with pytest.raises(GateG11Error) as exc_info:
            _check_gate_g11(state)
        assert "Go/No-Go" in str(exc_info.value)
        assert "NO_GO" in str(exc_info.value)

    def test_error_contains_story_id(self):
        state = initial_story_state("FSC-9999")
        state["agent_results"]["45"] = {"data": {
            "coordinator_verdict": "NO_GO",
            "go_decision": False,
            "no_go_reasons": ["reason"],
        }}
        with pytest.raises(GateG11Error) as exc_info:
            _check_gate_g11(state)
        assert "FSC-9999" in str(exc_info.value)

    def test_error_lists_no_go_reasons(self):
        state = _state_with("45", {
            "coordinator_verdict": "NO_GO",
            "go_decision": False,
            "no_go_reasons": ["Integrity FAIL", "Evidence MISSING"],
        })
        with pytest.raises(GateG11Error) as exc_info:
            _check_gate_g11(state)
        msg = str(exc_info.value)
        assert "Integrity FAIL" in msg or "Evidence MISSING" in msg

    def test_raises_gate_g11_error_type(self):
        state = _state_with("45", {
            "coordinator_verdict": "NO_GO",
            "go_decision": False,
            "no_go_reasons": [],
        })
        with pytest.raises(GateG11Error):
            _check_gate_g11(state)


# ── Gate G12 — Production Validation (informational v1) ───────────────────────

class TestGateG12:
    def test_passes_when_agents_absent(self):
        state = initial_story_state("FSC-9999")
        _check_gate_g12(state)  # no exception = pass

    def test_passes_when_production_healthy(self):
        state = _state_with("46", {
            "production_healthy": True,
            "prod_verdict": "HEALTHY",
        })
        _check_gate_g12(state)  # no exception = pass

    def test_passes_when_production_skipped(self):
        state = _state_with("46", {
            "production_healthy": False,
            "prod_verdict": "SKIPPED",
        })
        _check_gate_g12(state)  # SKIPPED (NO_GO path) is accepted — G11 already enforced

    def test_always_passes_in_v1(self):
        # G12 is informational only in v1 — no scenario should raise an exception
        for prod_verdict in ("HEALTHY", "SKIPPED", "UNKNOWN", ""):
            state = _state_with("46", {"prod_verdict": prod_verdict})
            _check_gate_g12(state)  # no exception for any value
