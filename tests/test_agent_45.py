"""Tests for Agent 45 — Go/No-Go Coordinator (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_45_go_no_go import (
    _compute_confidence,
    _make_decision,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT36_NOT_REQUIRED = {"uat_coordination_verdict": "NOT_REQUIRED"}
AGENT36_PENDING = {"uat_coordination_verdict": "PENDING"}
AGENT36_BLOCKED = {"uat_coordination_verdict": "BLOCKED"}
AGENT36_SIGNED_OFF = {"uat_coordination_verdict": "SIGNED_OFF"}

AGENT39_READY = {"readiness_verdict": "READY", "readiness_blockers": []}
AGENT39_BLOCKED = {"readiness_verdict": "BLOCKED", "readiness_blockers": ["Coverage 60% below threshold"]}

AGENT41_PASS = {"integrity_verdict": "PASS", "integrity_issues": []}
AGENT41_FAIL = {"integrity_verdict": "FAIL", "integrity_issues": ["Missing dependency"]}
AGENT41_WARN = {"integrity_verdict": "WARN", "integrity_issues": ["Destructive changes"]}

AGENT43_PASS = {"smoke_verdict": "PASS", "smoke_failed": 0}
AGENT43_FAIL = {"smoke_verdict": "FAIL", "smoke_failed": 3}
AGENT43_SKIPPED = {"smoke_verdict": "SKIPPED", "smoke_failed": 0}

AGENT44_COMPLETE = {"evidence_verdict": "COMPLETE", "evidence_gaps": []}
AGENT44_MISSING = {"evidence_verdict": "MISSING", "evidence_gaps": ["COBS 9"]}
AGENT44_PARTIAL = {"evidence_verdict": "PARTIAL", "evidence_gaps": ["COBS 9"]}

MOCK_GO = {
    "narrative": "All release gates passed. Change set valid, smoke tests passed, FCA evidence complete. Production deployment approved.",
    "coordinator_concern": "none",
}
MOCK_NO_GO = {
    "narrative": "Release BLOCKED — smoke tests failed with 3 failures. Deployment cannot proceed until tests pass.",
    "coordinator_concern": "smoke_failure",
}
MOCK_CONDITIONAL = {
    "narrative": "All technical gates passed. GO is conditional on Compliance Officer sign-off which is currently pending.",
    "coordinator_concern": "uat_pending",
}


# ── Deterministic go/no-go logic tests ───────────────────────────────────────

class TestMakeDecision:
    def test_all_clear_gives_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is True
        assert verdict == "GO"
        assert len(reasons) == 0

    def test_readiness_blocked_gives_no_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_BLOCKED, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is False
        assert verdict == "NO_GO"
        assert any("readiness" in r.lower() for r in reasons)

    def test_integrity_fail_gives_no_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_FAIL, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is False
        assert verdict == "NO_GO"
        assert any("integrity" in r.lower() for r in reasons)

    def test_integrity_warn_does_not_block(self):
        go, _, verdict, _, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_WARN, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is True
        assert verdict == "GO"

    def test_smoke_fail_gives_no_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_FAIL, AGENT44_COMPLETE,
        )
        assert go is False
        assert verdict == "NO_GO"
        assert any("smoke" in r.lower() for r in reasons)

    def test_smoke_skipped_does_not_block(self):
        go, _, verdict, _, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_SKIPPED, AGENT44_COMPLETE,
        )
        assert go is True

    def test_fca_evidence_missing_gives_no_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_MISSING,
        )
        assert go is False
        assert any("evidence" in r.lower() for r in reasons)

    def test_fca_evidence_partial_does_not_block(self):
        go, _, _, _, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_PARTIAL,
        )
        assert go is True

    def test_uat_pending_gives_conditional(self):
        go, _, verdict, _, _, _ = _make_decision(None, 
            AGENT36_PENDING, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is True
        assert verdict == "CONDITIONAL"

    def test_uat_blocked_gives_no_go(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_BLOCKED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert go is False
        assert verdict == "NO_GO"

    def test_no_upstream_data_gives_go(self):
        go, _, verdict, _, _, _ = _make_decision(None, None, None, None, None, None)
        assert go is True
        assert verdict == "GO"

    def test_multiple_failures_all_collected(self):
        _, reasons, _, _, _, _ = _make_decision(None, 
            AGENT36_BLOCKED, AGENT39_BLOCKED, AGENT41_FAIL, AGENT43_FAIL, AGENT44_MISSING,
        )
        assert len(reasons) >= 4


# ── Minimax loss analysis + coalition tests ───────────────────────────────────

class TestMinimaxAndCoalition:
    def test_all_clear_no_minimax_loss(self):
        go, reasons, verdict, minimax_loss, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert minimax_loss == []
        assert go is True

    def test_smoke_fail_produces_minimax_loss_entry(self):
        _, _, _, minimax_loss, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_FAIL, AGENT44_COMPLETE,
        )
        assert len(minimax_loss) >= 1
        loss_types = [m["loss_type"] for m in minimax_loss]
        assert "PRODUCTION_INCIDENT" in loss_types

    def test_fca_evidence_missing_maps_to_regulatory_breach(self):
        _, _, _, minimax_loss, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_MISSING,
        )
        loss_types = [m["loss_type"] for m in minimax_loss]
        assert "REGULATORY_BREACH" in loss_types

    def test_all_clear_unanimous_go(self):
        _, _, _, _, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert coalition_verdict == "UNANIMOUS_GO"
        assert coalition_dissent == []

    def test_smoke_fail_produces_dissent_no_go(self):
        _, _, _, _, coalition_verdict, coalition_dissent = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS, AGENT43_FAIL, AGENT44_COMPLETE,
        )
        assert coalition_verdict == "DISSENT_NO_GO"
        assert "smoke" in coalition_dissent

    def test_minimax_loss_entry_has_required_keys(self):
        _, _, _, minimax_loss, _, _ = _make_decision(None, 
            AGENT36_NOT_REQUIRED, AGENT39_BLOCKED, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        for entry in minimax_loss:
            assert "gate" in entry
            assert "loss_type" in entry
            assert "severity" in entry

    def test_multiple_failures_produce_multiple_loss_entries(self):
        _, reasons, _, minimax_loss, _, _ = _make_decision(None, 
            AGENT36_BLOCKED, AGENT39_BLOCKED, AGENT41_FAIL, AGENT43_FAIL, AGENT44_MISSING,
        )
        assert len(minimax_loss) == len(reasons)

    def test_uat_pending_gives_dissent_no_go(self):
        # REQ-29 Gap 3: PENDING removed from coalition pass set — outstanding CO is not unanimous
        _, _, _, _, coalition_verdict, dissent = _make_decision(None,
            AGENT36_PENDING, AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert coalition_verdict == "DISSENT_NO_GO"
        assert "uat" in dissent


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_comprehensive_gate_data_scores_well(self):
        score, _ = _compute_confidence(AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE, True)
        assert score >= 75

    def test_no_gate_data_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE, True)
        score_without, _ = _compute_confidence(None, None, None, None, False)
        assert score_with > score_without

    def test_no_go_reduces_confidence(self):
        score_go, _ = _compute_confidence(AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE, True)
        score_no_go, _ = _compute_confidence(AGENT39_BLOCKED, AGENT41_FAIL, AGENT43_FAIL, AGENT44_MISSING, False)
        assert score_go > score_no_go

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT39_READY, AGENT41_PASS, AGENT43_PASS, AGENT44_COMPLETE, True)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, False)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["39"] = {"data": AGENT39_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}
        state["agent_results"]["43"] = {"data": AGENT43_PASS}
        state["agent_results"]["44"] = {"data": AGENT44_COMPLETE}
        state["agent_results"]["36"] = {"data": AGENT36_NOT_REQUIRED}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_GO
            result = await run(state)

        assert result.agent_id == 45
        assert result.agent_name == "Go/No-Go Coordinator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_GO
            result = await run(state)

        for key in ["go_decision", "no_go_reasons", "coordinator_verdict",
                    "minimax_loss_analysis", "coalition_verdict", "coalition_dissent"]:
            assert key in result.data

    async def test_go_when_all_gates_clear(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["39"] = {"data": AGENT39_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}
        state["agent_results"]["43"] = {"data": AGENT43_PASS}
        state["agent_results"]["44"] = {"data": AGENT44_COMPLETE}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_GO
            result = await run(state)

        assert result.data["go_decision"] is True
        assert result.data["coordinator_verdict"] == "GO"

    async def test_no_go_when_smoke_fails(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["43"] = {"data": AGENT43_FAIL}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_NO_GO
            result = await run(state)

        assert result.data["go_decision"] is False
        assert result.data["coordinator_verdict"] == "NO_GO"

    async def test_conditional_when_uat_pending(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["36"] = {"data": AGENT36_PENDING}
        state["agent_results"]["39"] = {"data": AGENT39_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}
        state["agent_results"]["43"] = {"data": AGENT43_PASS}
        state["agent_results"]["44"] = {"data": AGENT44_COMPLETE}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_CONDITIONAL
            result = await run(state)

        assert result.data["coordinator_verdict"] == "CONDITIONAL"
        assert result.data["go_decision"] is True

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_GO
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_go_produces_unanimous_coalition_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["39"] = {"data": AGENT39_READY}
        state["agent_results"]["41"] = {"data": AGENT41_PASS}
        state["agent_results"]["43"] = {"data": AGENT43_PASS}
        state["agent_results"]["44"] = {"data": AGENT44_COMPLETE}
        state["agent_results"]["36"] = {"data": AGENT36_NOT_REQUIRED}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_GO
            result = await run(state)

        assert result.data["coalition_verdict"] == "UNANIMOUS_GO"
        assert result.data["coalition_dissent"] == []
        assert result.data["minimax_loss_analysis"] == []

    async def test_no_go_produces_dissent_and_loss_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["43"] = {"data": AGENT43_FAIL}

        with patch("src.agents.release.agent_45_go_no_go.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_NO_GO
            result = await run(state)

        assert result.data["coalition_verdict"] == "DISSENT_NO_GO"
        assert len(result.data["minimax_loss_analysis"]) >= 1


# ── REQ-29: new tests ─────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}


class TestREQ29PartialEvidenceEscalation:
    def test_high_fca_partial_evidence_gives_no_go(self):
        go, reasons, verdict, _, _, _ = _make_decision(
            AGENT3_HIGH, AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_PARTIAL,
        )
        assert go is False
        assert verdict == "NO_GO"
        assert any("partial" in r.lower() for r in reasons)

    def test_medium_fca_partial_evidence_gives_no_go(self):
        agent3_medium = {"fca_classification": "MEDIUM"}
        go, reasons, verdict, _, _, _ = _make_decision(
            agent3_medium, AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_PARTIAL,
        )
        assert go is False
        assert verdict == "NO_GO"

    def test_low_fca_partial_evidence_does_not_block(self):
        go, _, verdict, _, _, _ = _make_decision(
            AGENT3_LOW, AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_PARTIAL,
        )
        assert go is True

    def test_partial_evidence_loss_type_regulatory_gap(self):
        _, _, _, minimax_loss, _, _ = _make_decision(
            AGENT3_HIGH, AGENT36_NOT_REQUIRED, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_PARTIAL,
        )
        loss_types = [m["loss_type"] for m in minimax_loss]
        assert "REGULATORY_GAP" in loss_types


class TestREQ29CoalitionPendingGap:
    def test_uat_pending_gives_dissent_coalition(self):
        _, _, _, _, coalition_verdict, dissent = _make_decision(
            None, AGENT36_PENDING, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert coalition_verdict == "DISSENT_NO_GO"
        assert "uat" in dissent

    def test_uat_signed_off_gives_unanimous_go(self):
        _, _, _, _, coalition_verdict, dissent = _make_decision(
            None, AGENT36_SIGNED_OFF, AGENT39_READY, AGENT41_PASS,
            AGENT43_PASS, AGENT44_COMPLETE,
        )
        assert coalition_verdict == "UNANIMOUS_GO"
        assert dissent == []
