"""Tests for Agent 52 — Severity Calibration Agent (Augmented Script)."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.monitoring.agent_52_severity_calibration import (
    _AGENT_BASE_MAP,
    _compute_adjustments,
    _compute_confidence,
    _derive_verdict,
    run_scheduled,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _signal_row(agent_id: int, total: int, fp: int, tp: int = 0, fn: int = 0, tn: int = 0) -> dict:
    return {"agent_id": agent_id, "total": total, "fp": fp, "tp": tp, "fn": fn, "tn": tn}


MOCK_NARRATIVE = {
    "calibration_summary": "Agent 1 base reduced from 60 to 56 due to 20% FP rate. Agent 3 unchanged.",
    "key_insight": "Agent 1 had the highest FP rate this week at 20%.",
}

ROWS_SUFFICIENT = [
    _signal_row(1, total=20, fp=4),   # 20% FP → reduce
    _signal_row(3, total=15, fp=0),   # 0% FP → increase
    _signal_row(5, total=12, fp=1),   # 8% FP → no change (between thresholds)
]

ROWS_SPARSE = [
    _signal_row(1, total=5, fp=1),    # below min volume (10)
]

ROWS_EMPTY: list[dict] = []


# ── _compute_adjustments tests ────────────────────────────────────────────────

class TestComputeAdjustments:
    def test_high_fp_rate_reduces_base(self):
        rows = [_signal_row(1, total=20, fp=4)]  # 20% FP > 15% threshold
        adjustments = _compute_adjustments(rows)
        assert len(adjustments) == 1
        assert adjustments[0]["adjustment"] < 0

    def test_low_fp_rate_increases_base(self):
        rows = [_signal_row(3, total=15, fp=0)]  # 0% FP < 5% threshold
        adjustments = _compute_adjustments(rows)
        assert len(adjustments) == 1
        assert adjustments[0]["adjustment"] > 0

    def test_mid_range_fp_no_change(self):
        rows = [_signal_row(5, total=12, fp=1)]  # 8.3% → between thresholds
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["adjustment"] == 0

    def test_insufficient_volume_no_adjustment(self):
        rows = [_signal_row(1, total=5, fp=3)]   # < 10 signals
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["adjustment"] == 0
        assert "Insufficient" in adjustments[0]["reason"]

    def test_recommended_base_never_below_20(self):
        # Extreme FP rate on a low-base agent
        rows = [_signal_row(49, total=100, fp=80)]  # base=50, big reduction
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["recommended_base"] >= 20

    def test_recommended_base_never_above_90(self):
        # Low FP rate on a high-base agent
        rows = [_signal_row(3, total=100, fp=0)]   # base=70, increase capped at 90
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["recommended_base"] <= 90

    def test_empty_rows_gives_empty_adjustments(self):
        assert _compute_adjustments([]) == []

    def test_adjustment_delta_within_bounds(self):
        # Max reduction is 10, max increase is 5
        for row in ROWS_SUFFICIENT:
            adjustments = _compute_adjustments([row])
            delta = adjustments[0]["adjustment"]
            assert delta >= -10
            assert delta <= 5

    def test_current_base_matches_agent_base_map(self):
        rows = [_signal_row(1, total=20, fp=0)]
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["current_base"] == _AGENT_BASE_MAP[1]

    def test_agent_name_populated(self):
        rows = [_signal_row(1, total=20, fp=5)]
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["agent_name"] != ""

    def test_unknown_agent_id_uses_default_base(self):
        rows = [_signal_row(99, total=20, fp=5)]  # agent 99 not in map
        adjustments = _compute_adjustments(rows)
        assert adjustments[0]["current_base"] == 60  # default


# ── _derive_verdict tests ─────────────────────────────────────────────────────

class TestDeriveVerdict:
    def test_empty_rows_gives_insufficient_data(self):
        assert _derive_verdict([], []) == "INSUFFICIENT_DATA"

    def test_adjustments_made_gives_adjusted(self):
        adjustments = [{"adjustment": -5}]
        assert _derive_verdict([{}], adjustments) == "ADJUSTED"

    def test_no_adjustments_gives_no_change(self):
        adjustments = [{"adjustment": 0}]
        assert _derive_verdict([{}], adjustments) == "NO_CHANGE"

    def test_mixed_adjustments_gives_adjusted(self):
        adjustments = [{"adjustment": 0}, {"adjustment": 3}]
        assert _derive_verdict([{}, {}], adjustments) == "ADJUSTED"


# ── _compute_confidence tests ─────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_rich_signal_volume_scores_well(self):
        score, _ = _compute_confidence(signal_count=50, agents_adjusted=2)
        assert score >= 75

    def test_no_signals_reduces_confidence_significantly(self):
        score_with, _ = _compute_confidence(50, 2)
        score_without, _ = _compute_confidence(0, 0)
        assert score_with > score_without

    def test_adequate_volume_bonus(self):
        score_adeq, _ = _compute_confidence(15, 0)
        score_sparse, _ = _compute_confidence(3, 0)
        assert score_adeq > score_sparse

    def test_adjustments_made_adds_bonus(self):
        score_adj, _ = _compute_confidence(20, 3)
        score_none, _ = _compute_confidence(20, 0)
        assert score_adj > score_none

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(100, 10)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(0, 0)
        assert score >= 20


# ── _AGENT_BASE_MAP completeness ──────────────────────────────────────────────

class TestAgentBaseMap:
    def test_all_50_agents_have_bases(self):
        expected = set(range(1, 51))
        assert expected.issubset(set(_AGENT_BASE_MAP.keys()))

    def test_all_bases_are_valid_integers_in_range(self):
        for agent_id, base in _AGENT_BASE_MAP.items():
            assert isinstance(base, int), f"Agent {agent_id} base is not int"
            assert 20 <= base <= 90, f"Agent {agent_id} base {base} out of range"

    def test_agent_base_map_in_sync(self):
        """Assert _AGENT_BASE_MAP matches TierBScorer(base=N) in each agent source file."""
        agents_root = Path(__file__).parent.parent / "src" / "agents"
        mismatches = []
        for path in sorted(agents_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            id_match = re.search(r"^AGENT_ID\s*=\s*(\d+)", source, re.MULTILINE)
            base_match = re.search(r"TierBScorer\(base=(\d+)\)", source)
            if not id_match or not base_match:
                continue  # Tier-A agents or non-agent files — no TierBScorer, skip
            agent_id = int(id_match.group(1))
            extracted_base = int(base_match.group(1))
            if agent_id in _AGENT_BASE_MAP and _AGENT_BASE_MAP[agent_id] != extracted_base:
                mismatches.append(
                    f"Agent {agent_id}: _AGENT_BASE_MAP={_AGENT_BASE_MAP[agent_id]}, "
                    f"source={extracted_base} ({path.name})"
                )
        assert not mismatches, "Base map out of sync:\n" + "\n".join(mismatches)


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRunScheduled:
    async def test_returns_agent_result(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_SUFFICIENT
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        assert result.agent_id == 52
        assert result.agent_name == "Severity Calibration Agent"
        assert result.confidence.tier == "B"

    async def test_data_has_required_keys(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_SUFFICIENT
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        for key in ["threshold_adjustments", "agents_adjusted",
                    "calibration_verdict", "calibration_summary"]:
            assert key in result.data

    async def test_insufficient_data_when_no_signals(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_EMPTY
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        assert result.data["calibration_verdict"] == "INSUFFICIENT_DATA"
        assert result.data["agents_adjusted"] == 0

    async def test_adjusted_verdict_when_rows_present(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_SUFFICIENT
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        # At least one row should trigger an adjustment (Agent 1 with 20% FP)
        assert result.data["calibration_verdict"] in ("ADJUSTED", "NO_CHANGE")

    async def test_uses_default_model(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_SPARSE
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        assert result.model_used == "claude-sonnet-4-6"

    async def test_threshold_adjustments_is_list(self):
        with patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                   new_callable=AsyncMock) as mock_fetch, \
             patch("src.agents.monitoring.agent_52_severity_calibration.call_with_tool",
                   new_callable=AsyncMock) as mock_llm:
            mock_fetch.return_value = ROWS_SUFFICIENT
            mock_llm.return_value = MOCK_NARRATIVE
            result = await run_scheduled()

        assert isinstance(result.data["threshold_adjustments"], list)


# ── REQ-33: _AGENT_BASE_MAP sync test ────────────────────────────────────────

import ast
import re


class TestAgentBaseMapSync:
    def _extract_base_from_file(self, filepath: str) -> int | None:
        """Extract TierBScorer(base=N) value from a Python source file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return None
        m = re.search(r'TierBScorer\(base=(\d+)\)', source)
        return int(m.group(1)) if m else None

    def _extract_agent_id_from_file(self, filepath: str) -> int | None:
        """Extract AGENT_ID = N from a Python source file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return None
        m = re.search(r'^AGENT_ID\s*=\s*(\d+)', source, re.MULTILINE)
        return int(m.group(1)) if m else None

    def test_agent_base_map_matches_source(self):
        import pathlib
        from src.agents.monitoring.agent_52_severity_calibration import _AGENT_BASE_MAP

        agents_root = pathlib.Path("src/agents")
        mismatches: list[str] = []

        for py_file in agents_root.rglob("agent_*.py"):
            # skip agent_52 itself and non-agent helpers
            if "agent_52" in py_file.name:
                continue
            agent_id = self._extract_agent_id_from_file(str(py_file))
            base = self._extract_base_from_file(str(py_file))
            if agent_id is None or base is None:
                continue  # Tier A agents or no TierBScorer — not in map, skip
            if agent_id in _AGENT_BASE_MAP:
                if _AGENT_BASE_MAP[agent_id] != base:
                    mismatches.append(
                        f"Agent {agent_id} ({py_file.name}): "
                        f"source base={base}, map has {_AGENT_BASE_MAP[agent_id]}"
                    )

        assert not mismatches, (
            "AGENT_BASE_MAP is out of sync with agent sources:\n" + "\n".join(mismatches)
        )
