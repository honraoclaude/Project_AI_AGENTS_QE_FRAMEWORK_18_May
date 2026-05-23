"""Tests for Agent 36 — UAT Coordination Agent (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_36_uat_coordination import (
    _build_trace_message,
    _compute_confidence,
    _coordinate_uat,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH   = {"fca_classification": "HIGH"}
AGENT3_MEDIUM = {"fca_classification": "MEDIUM"}
AGENT3_LOW    = {"fca_classification": "LOW"}

AGENT29_CO_REQUIRED = {
    "uat_test_count": 4,
    "uat_verdict": "PASS",
    "co_sign_off_required": True,
}

AGENT29_NO_CO = {
    "uat_test_count": 2,
    "uat_verdict": "PASS",
    "co_sign_off_required": False,
}

AGENT29_FAIL = {
    "uat_test_count": 2,
    "uat_verdict": "FAIL",
    "co_sign_off_required": True,
}

AGENT33_PASS = {
    "overall_coverage_pct": 92.0,
    "coverage_verdict": "PASS",
}

AGENT33_FAIL = {
    "overall_coverage_pct": 60.0,
    "coverage_verdict": "FAIL",
}

AGENT34_PASS = {
    "defect_verdict": "PASS",
    "defect_count": 0,
}

AGENT34_FAIL = {
    "defect_verdict": "FAIL",
    "defect_count": 1,
    "critical_defects": ["DEF-001"],
}

AGENT35_RESOLVED = {
    "rca_verdict": "RESOLVED_PLAN",
    "fix_plan_complete": True,
}

MOCK_TRACE_NOT_REQUIRED = {
    "narrative": "UAT coordination complete. FCA classification is LOW — Compliance Officer sign-off not required. Story may proceed to release.",
    "coordination_concern": "none",
}

MOCK_TRACE_PENDING = {
    "narrative": "FCA classification is HIGH — CO sign-off required. HMAC-signed approval email sent to Compliance Officer. Story is blocked pending receipt.",
    "coordination_concern": "sign_off_pending",
}

MOCK_TRACE_BLOCKED = {
    "narrative": "CO sign-off cannot be requested — active P2 defects must be resolved first. Developer must fix DEF-001 before UAT sign-off can proceed.",
    "coordination_concern": "uat_failures_block_sign_off",
}


# ── Deterministic coordination logic tests ────────────────────────────────────

class TestCoordinateUAT:
    def test_low_fca_no_co_required_gives_not_required(self):
        required, received, sent, verdict = _coordinate_uat(
            AGENT3_LOW, AGENT29_NO_CO, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert required is False
        assert verdict == "NOT_REQUIRED"
        assert sent is False

    def test_high_fca_requires_co_sign_off(self):
        required, received, sent, verdict = _coordinate_uat(
            AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert required is True

    def test_medium_fca_requires_co_sign_off(self):
        required, _, _, _ = _coordinate_uat(
            AGENT3_MEDIUM, AGENT29_NO_CO, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert required is True

    def test_co_required_flag_from_agent29_triggers_requirement(self):
        required, _, _, _ = _coordinate_uat(
            AGENT3_LOW, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert required is True

    def test_high_fca_no_defects_gives_pending(self):
        _, _, sent, verdict = _coordinate_uat(
            AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert verdict == "PENDING"
        assert sent is True

    def test_sign_off_not_received_in_ci_stub(self):
        # Stub: sign-off is never pre-received — always False in CI
        _, received, _, _ = _coordinate_uat(
            AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert received is False

    def test_defect_fail_blocks_sign_off_request(self):
        _, _, sent, verdict = _coordinate_uat(
            AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_FAIL, None,
        )
        assert verdict == "BLOCKED"
        assert sent is False

    def test_coverage_fail_blocks_sign_off_request(self):
        _, _, sent, verdict = _coordinate_uat(
            AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_FAIL, AGENT34_PASS, None,
        )
        assert verdict == "BLOCKED"
        assert sent is False

    def test_no_upstream_data_gives_not_required(self):
        required, _, _, verdict = _coordinate_uat(None, None, None, None, None)
        assert required is False
        assert verdict == "NOT_REQUIRED"

    def test_medium_fca_no_defects_gives_pending(self):
        _, _, sent, verdict = _coordinate_uat(
            AGENT3_MEDIUM, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert verdict == "PENDING"
        assert sent is True

    def test_low_fca_co_required_true_gives_pending(self):
        # co_required overrides NOT_REQUIRED — must still send sign-off request
        _, _, sent, verdict = _coordinate_uat(
            AGENT3_LOW, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, None,
        )
        assert verdict == "PENDING"
        assert sent is True


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_fca_class_and_uat_data_scores_well(self):
        score, _ = _compute_confidence(AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, "PENDING")
        assert score >= 65

    def test_unknown_fca_class_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, "PENDING")
        score_without, _ = _compute_confidence(None, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, "PENDING")
        assert score_with > score_without

    def test_blocked_verdict_penalises_confidence(self):
        score_pending, _ = _compute_confidence(AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_PASS, "PENDING")
        score_blocked, _ = _compute_confidence(AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_FAIL, "BLOCKED")
        assert score_pending > score_blocked

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT3_LOW, AGENT29_NO_CO, AGENT33_PASS, AGENT34_PASS, "NOT_REQUIRED")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, None, "BLOCKED")
        assert score >= 20

    def test_fca_classification_known_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, None, None, None, "PENDING")
        assert "fca_classification_known" in signals

    def test_fca_classification_unknown_key_in_signals(self):
        _, signals = _compute_confidence(None, None, None, None, "NOT_REQUIRED")
        assert "fca_classification_unknown" in signals

    def test_uat_test_results_available_key_in_signals(self):
        _, signals = _compute_confidence(None, AGENT29_CO_REQUIRED, None, None, "NOT_REQUIRED")
        assert "uat_test_results_available" in signals

    def test_no_uat_results_key_in_signals(self):
        _, signals = _compute_confidence(None, None, None, None, "NOT_REQUIRED")
        assert "no_uat_results" in signals

    def test_coverage_data_available_key_in_signals(self):
        _, signals = _compute_confidence(None, None, AGENT33_PASS, None, "NOT_REQUIRED")
        assert "coverage_data_available" in signals

    def test_sign_off_blocked_by_defects_key_in_signals(self):
        _, signals = _compute_confidence(AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS, AGENT34_FAIL, "BLOCKED")
        assert "sign_off_blocked_by_defects" in signals


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_LOW}
        state["agent_results"]["29"] = {"data": AGENT29_NO_CO}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert result.agent_id == 36
        assert result.agent_name == "UAT Coordination Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        for key in ["uat_sign_off_required", "uat_sign_off_received",
                    "uat_coordination_verdict", "sign_off_request_sent"]:
            assert key in result.data

    async def test_not_required_for_low_fca(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_LOW}
        state["agent_results"]["29"] = {"data": AGENT29_NO_CO}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert result.data["uat_coordination_verdict"] == "NOT_REQUIRED"
        assert result.data["uat_sign_off_required"] is False

    async def test_pending_for_high_fca_no_defects(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["29"] = {"data": AGENT29_CO_REQUIRED}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_PASS}

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PENDING
            result = await run(state)

        assert result.data["uat_coordination_verdict"] == "PENDING"
        assert result.data["uat_sign_off_required"] is True
        assert result.data["sign_off_request_sent"] is True

    async def test_blocked_when_critical_defects_present(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}
        state["agent_results"]["29"] = {"data": AGENT29_CO_REQUIRED}
        state["agent_results"]["33"] = {"data": AGENT33_PASS}
        state["agent_results"]["34"] = {"data": AGENT34_FAIL}

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_BLOCKED
            result = await run(state)

        assert result.data["uat_coordination_verdict"] == "BLOCKED"
        assert result.data["sign_off_request_sent"] is False

    async def test_sign_off_never_received_in_ci_stub(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"]  = {"data": AGENT3_HIGH}

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PENDING
            result = await run(state)

        assert result.data["uat_sign_off_received"] is False

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_escalated_when_no_upstream_data(self):
        # base=60, fca_unknown→-8, no_uat_results→-5 = 47 < 60
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_36_uat_coordination.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_NOT_REQUIRED
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "HIGH" in msg

    def test_includes_uat_count(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "4" in msg

    def test_includes_coverage_pct(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "92.0" in msg

    def test_includes_coordination_verdict(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "PENDING" in msg

    def test_includes_sign_off_required_true(self):
        msg = _build_trace_message("FSC-2417", True, False, True, "PENDING",
                                   AGENT3_HIGH, AGENT29_CO_REQUIRED, AGENT33_PASS)
        assert "True" in msg

    def test_unknown_fca_class_when_no_agent3(self):
        msg = _build_trace_message("FSC-2417", False, False, False, "NOT_REQUIRED",
                                   None, None, None)
        assert "UNKNOWN" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", False, False, False, "NOT_REQUIRED",
                                   None, None, None)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "coordination_concern"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_coordination_concern_enum_has_five_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["coordination_concern"]["enum"] == [
            "none", "sign_off_pending", "uat_failures_block_sign_off",
            "coverage_below_threshold", "no_uat_tests",
        ]
