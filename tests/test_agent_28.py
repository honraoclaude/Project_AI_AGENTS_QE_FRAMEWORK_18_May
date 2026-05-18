"""Tests for Agent 28 — CRT Self-Heal Reviewer (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.testing.agent_28_crt_self_heal_reviewer import (
    _compute_confidence,
    _review_self_heals,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT27_NO_HEALS = {
    "crt_results": [
        {"test_id": "CRT-001", "title": "Suitability check", "status": "PASSED",
         "self_healed": False, "tags": ["@fca"]},
        {"test_id": "CRT-002", "title": "Portfolio rebalancing", "status": "PASSED",
         "self_healed": False, "tags": ["@smoke"]},
    ],
    "tests_executed": 2,
    "crt_execution_verdict": "PASS",
}

AGENT27_FCA_HEAL = {
    "crt_results": [
        {"test_id": "CRT-001", "title": "FCA suitability regulatory check",
         "status": "PASSED", "self_healed": True, "tags": ["@fca"]},
        {"test_id": "CRT-002", "title": "Portfolio rebalancing",
         "status": "PASSED", "self_healed": False, "tags": ["@smoke"]},
    ],
    "tests_executed": 2,
    "crt_execution_verdict": "PASS",
}

AGENT27_EXCESSIVE_HEALS = {
    "crt_results": [
        {"test_id": f"CRT-{i:03d}", "title": f"Test {i}",
         "status": "PASSED", "self_healed": True, "tags": ["@smoke"]}
        for i in range(1, 5)  # 4 heals > threshold of 2
    ],
    "tests_executed": 4,
    "crt_execution_verdict": "PASS",
}

AGENT27_EMPTY = {
    "crt_results": [],
    "tests_executed": 0,
    "crt_execution_verdict": "SKIPPED",
}

MOCK_TRACE_PASS = {
    "narrative": "No self-healed tests detected. All CRT tests ran without locator changes.",
    "heal_concern": "none",
}

MOCK_TRACE_REVIEW = {
    "narrative": "FCA-tagged test CRT-001 self-healed. Manual review required to confirm correct locator.",
    "heal_concern": "fca_test_healed",
}


# ── Deterministic self-heal review tests ──────────────────────────────────────

class TestSelfHealReview:
    def test_no_heals_gives_pass(self):
        healed, suspect, risk, verdict = _review_self_heals(AGENT27_NO_HEALS)
        assert healed == 0
        assert verdict == "PASS"
        assert risk == "LOW"
        assert len(suspect) == 0

    def test_fca_test_healed_gives_review_required(self):
        healed, suspect, risk, verdict = _review_self_heals(AGENT27_FCA_HEAL)
        assert healed == 1
        assert verdict == "REVIEW_REQUIRED"
        assert "CRT-001" in suspect

    def test_excessive_heals_gives_warn(self):
        healed, suspect, risk, verdict = _review_self_heals(AGENT27_EXCESSIVE_HEALS)
        assert healed == 4
        assert risk == "HIGH"
        assert verdict in ("WARN", "REVIEW_REQUIRED")

    def test_no_crt_results_gives_pass(self):
        healed, suspect, risk, verdict = _review_self_heals(AGENT27_EMPTY)
        assert healed == 0
        assert verdict == "PASS"
        assert risk == "LOW"

    def test_no_upstream_data_gives_pass(self):
        healed, suspect, risk, verdict = _review_self_heals(None)
        assert verdict == "PASS"
        assert healed == 0

    def test_suitability_keyword_in_title_flagged(self):
        agent27 = {
            "crt_results": [
                {"test_id": "CRT-001", "title": "Suitability score validation",
                 "status": "PASSED", "self_healed": True, "tags": ["@smoke"]},
            ],
            "tests_executed": 1,
        }
        _, suspect, _, verdict = _review_self_heals(agent27)
        assert "CRT-001" in suspect
        assert verdict == "REVIEW_REQUIRED"


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_crt_results_available_scores_well(self):
        score, _ = _compute_confidence(AGENT27_NO_HEALS, 0, "PASS")
        assert score >= 65

    def test_no_crt_results_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT27_NO_HEALS, 0, "PASS")
        score_without, _ = _compute_confidence(None, 0, "PASS")
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT27_NO_HEALS, 0, "PASS")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, 5, "REVIEW_REQUIRED")
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_NO_HEALS}

        with patch("src.agents.testing.agent_28_crt_self_heal_reviewer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 28
        assert result.agent_name == "CRT Self-Heal Reviewer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_28_crt_self_heal_reviewer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["self_healed_count", "suspect_self_heals",
                    "self_heal_verdict", "auto_heal_risk"]:
            assert key in result.data

    async def test_pass_when_no_heals(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_NO_HEALS}

        with patch("src.agents.testing.agent_28_crt_self_heal_reviewer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["self_heal_verdict"] == "PASS"
        assert result.data["self_healed_count"] == 0

    async def test_review_required_when_fca_test_healed(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["27"] = {"data": AGENT27_FCA_HEAL}

        with patch("src.agents.testing.agent_28_crt_self_heal_reviewer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_REVIEW
            result = await run(state)

        assert result.data["self_heal_verdict"] == "REVIEW_REQUIRED"
        assert len(result.data["suspect_self_heals"]) >= 1

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.testing.agent_28_crt_self_heal_reviewer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"
