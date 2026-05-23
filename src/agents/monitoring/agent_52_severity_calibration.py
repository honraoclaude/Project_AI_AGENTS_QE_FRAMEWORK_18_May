"""
Agent 52 — Severity Calibration Agent
Phase       : Cross-Phase (scheduled weekly)
PACT        : Proactive
Classification: Augmented Script (deterministic DB reads + Sonnet narrative)
Confidence  : Tier B (base=70)

Runs on a weekly schedule independent of story pipelines.
Also invokable from the Fleet Commander after a release retrospective batch.

Purpose:
  Reads accumulated learning signals (QE Lead overrides flagged as
  TRUE_POSITIVE / FALSE_POSITIVE / FALSE_NEGATIVE / TRUE_NEGATIVE) from
  the Learning Repository and recommends per-agent confidence base adjustments.

  If an agent's false-positive rate over the past 30 days exceeds the alert
  threshold its confidence base is recommended down by up to 10 points.
  If the false-positive rate is near zero with sufficient signal volume the
  base is recommended up by up to 5 points.  These are recommendations only —
  a QE Lead must approve threshold changes before they take effect.

Output data keys consumed by downstream (Agent 50 feeds calibration_signals here):
  threshold_adjustments  → list  ({agent_id, agent_name, current_base,
                                   recommended_base, adjustment, reason})
  agents_adjusted        → int
  calibration_verdict    → str   (ADJUSTED / NO_CHANGE / INSUFFICIENT_DATA)
  calibration_summary    → str   (narrative from Sonnet)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.database import async_session_factory
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 52
AGENT_NAME = "Severity Calibration Agent"

# Calibration thresholds
_FP_RATE_REDUCE_THRESHOLD = 0.15    # FP rate > 15% → recommend base down
_FP_RATE_INCREASE_THRESHOLD = 0.05  # FP rate < 5% with good volume → recommend base up
_MIN_SIGNAL_VOLUME = 10             # minimum signals before making any adjustment
_MAX_REDUCTION = 10                 # max base reduction per cycle
_MAX_INCREASE = 5                   # max base increase per cycle

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_CALIB_TOOL_NAME = "generate_calibration_summary"
_CALIB_TOOL_SCHEMA = {
    "type": "object",
    "required": ["calibration_summary", "key_insight"],
    "properties": {
        "calibration_summary": {
            "type": "string",
            "description": (
                "2–3 sentences summarising this week's calibration run: "
                "which agents were adjusted, why, and what the expected impact is."
            ),
        },
        "key_insight": {
            "type": "string",
            "description": "One sentence naming the most important signal observed this week.",
        },
    },
}

_CALIB_INSTRUCTIONS = """
You are the Severity Calibration Agent for the FSC Agentic QE Framework.
You receive a summary of per-agent false-positive rates and recommended threshold adjustments.
Write a concise calibration summary for the QE Lead explaining:
1. Which agents had their confidence base adjusted and why.
2. The overall signal quality this week.
3. Any agents that need monitoring attention due to high false-positive rates.
Be specific and reference agent IDs and rates.
""".strip()


# ── Main entry point (story-pipeline compatible) ──────────────────────────────

async def run(state: StoryState) -> AgentResult:
    """Called by the Fleet Commander worker if dispatched within a pipeline."""
    return await _run_calibration()


async def run_scheduled() -> AgentResult:
    """Entry point for the weekly scheduled calibration run."""
    return await _run_calibration()


# ── Core calibration logic ────────────────────────────────────────────────────

async def _run_calibration() -> AgentResult:
    signal_rows = await _fetch_signal_summary()
    adjustments = _compute_adjustments(signal_rows)
    agents_adjusted = sum(1 for a in adjustments if a["adjustment"] != 0)
    verdict = _derive_verdict(signal_rows, adjustments)

    narrative_data = await _generate_narrative(adjustments, signal_rows, verdict)
    summary = narrative_data.get("calibration_summary", "Calibration complete.")
    key_insight = narrative_data.get("key_insight", "")

    confidence_score, signals = _compute_confidence(len(signal_rows), agents_adjusted)
    escalated = confidence_score < settings.confidence_escalation_threshold

    data: dict[str, Any] = {
        "threshold_adjustments": adjustments,
        "agents_adjusted": agents_adjusted,
        "calibration_verdict": verdict,
        "calibration_summary": summary,
        "key_insight": key_insight,
        "signal_rows_analysed": len(signal_rows),
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=f"Weekly calibration: {agents_adjusted} agent(s) adjusted — verdict={verdict}",
        why=summary,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.default_model,
    )


# ── DB signal fetch ───────────────────────────────────────────────────────────

async def _fetch_signal_summary() -> list[dict]:
    """
    Returns one row per agent_id with signal counts over the past 30 days.
    Returns an empty list when the DB is unavailable (e.g., tests).
    """
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        de.agent_id,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE ls.outcome = 'FALSE_POSITIVE') AS fp,
                        COUNT(*) FILTER (WHERE ls.outcome = 'TRUE_POSITIVE')  AS tp,
                        COUNT(*) FILTER (WHERE ls.outcome = 'FALSE_NEGATIVE') AS fn,
                        COUNT(*) FILTER (WHERE ls.outcome = 'TRUE_NEGATIVE')  AS tn
                    FROM learning_signals ls
                    JOIN decision_events de ON ls.decision_event_id = de.id
                    WHERE ls.recorded_at >= :since
                      AND de.agent_id IS NOT NULL
                    GROUP BY de.agent_id
                    ORDER BY de.agent_id
                """),
                {"since": thirty_days_ago},
            )
            return [dict(row._mapping) for row in result]
    except Exception:
        return []


# ── Deterministic adjustment computation ──────────────────────────────────────

def _compute_adjustments(signal_rows: list[dict]) -> list[dict]:
    adjustments = []

    for row in signal_rows:
        agent_id = row["agent_id"]
        total = row["total"]
        fp = row["fp"]

        current_base = _AGENT_BASE_MAP.get(agent_id, 60)

        if total < _MIN_SIGNAL_VOLUME:
            adjustments.append({
                "agent_id": agent_id,
                "agent_name": _agent_name(agent_id),
                "current_base": current_base,
                "recommended_base": current_base,
                "adjustment": 0,
                "reason": f"Insufficient signal volume ({total} < {_MIN_SIGNAL_VOLUME})",
            })
            continue

        fp_rate = fp / total
        if fp_rate > _FP_RATE_REDUCE_THRESHOLD:
            delta = -min(_MAX_REDUCTION, round((fp_rate - _FP_RATE_REDUCE_THRESHOLD) * 40))
            reason = f"FP rate {fp_rate:.1%} exceeds {_FP_RATE_REDUCE_THRESHOLD:.0%} threshold"
        elif fp_rate < _FP_RATE_INCREASE_THRESHOLD:
            delta = min(_MAX_INCREASE, round((_FP_RATE_INCREASE_THRESHOLD - fp_rate) * 40))
            reason = f"FP rate {fp_rate:.1%} well below {_FP_RATE_INCREASE_THRESHOLD:.0%} threshold"
        else:
            delta = 0
            reason = f"FP rate {fp_rate:.1%} within acceptable range"

        recommended = max(20, min(90, current_base + delta))
        adjustments.append({
            "agent_id": agent_id,
            "agent_name": _agent_name(agent_id),
            "current_base": current_base,
            "recommended_base": recommended,
            "adjustment": recommended - current_base,
            "reason": reason,
        })

    return adjustments


def _derive_verdict(signal_rows: list[dict], adjustments: list[dict]) -> str:
    if not signal_rows:
        return "INSUFFICIENT_DATA"
    if any(a["adjustment"] != 0 for a in adjustments):
        return "ADJUSTED"
    return "NO_CHANGE"


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(signal_count: int, agents_adjusted: int) -> tuple[int, dict]:
    scorer = TierBScorer(base=70)

    if signal_count >= 50:
        scorer.add("rich_signal_volume", signal_count, +10)
    elif signal_count >= 10:
        scorer.add("adequate_signal_volume", signal_count, +5)
    elif signal_count == 0:
        scorer.add("no_signals", 0, -20)
    else:
        scorer.add("sparse_signals", signal_count, -5)

    if agents_adjusted > 0:
        scorer.add("adjustments_made", agents_adjusted, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet narrative ──────────────────────────────────────────────────────────

async def _generate_narrative(
    adjustments: list[dict],
    signal_rows: list[dict],
    verdict: str,
) -> dict:
    adj_summary = "\n".join(
        f"  Agent {a['agent_id']} ({a['agent_name']}): "
        f"base {a['current_base']} → {a['recommended_base']} ({a['adjustment']:+d}) — {a['reason']}"
        for a in adjustments
    ) or "  No adjustments this cycle."

    user_message = (
        f"Calibration run complete. Verdict: {verdict}\n"
        f"Signal rows analysed: {len(signal_rows)}\n"
        f"Adjustments:\n{adj_summary}\n\n"
        f"Generate a calibration summary using the {_CALIB_TOOL_NAME} tool."
    )
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_CALIB_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_CALIB_TOOL_NAME,
        tool_description="Generate a calibration summary narrative.",
        tool_schema=_CALIB_TOOL_SCHEMA,
        max_tokens=400,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_name(agent_id: int) -> str:
    from src.agents.monitoring.agent_51_health import AGENT_NAMES
    return AGENT_NAMES.get(agent_id, f"Agent {agent_id}")


# Known confidence bases per agent (mirrors TierBScorer base= values in each agent file).
# Used to compute recommended_base. Kept here rather than in each agent to avoid
# circular imports — updated manually when agent bases change.
_AGENT_BASE_MAP: dict[int, int] = {
    1: 55, 2: 60, 3: 70, 4: 55, 5: 58, 6: 58, 7: 55, 8: 72, 9: 60,
    10: 68, 11: 65, 12: 68, 13: 65, 14: 58, 15: 65, 16: 65, 17: 65,
    18: 62, 19: 70, 20: 60, 21: 68, 22: 58, 23: 65,
    24: 65, 25: 60, 26: 68, 27: 62, 28: 60, 29: 67, 30: 70, 31: 65,
    32: 63, 33: 65, 34: 60, 35: 62, 36: 60, 37: 58, 38: 62,
    39: 63, 40: 60, 41: 65, 42: 58, 43: 60, 44: 65, 45: 65, 46: 58,
    47: 62, 48: 55, 49: 50, 50: 60,
    52: 70, 53: 65, 54: 60, 55: 62,
}
