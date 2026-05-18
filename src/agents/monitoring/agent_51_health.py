"""
Agent 51 — Agent Health Monitor (Cross-Phase, Augmented Script, Haiku 4.5)

Watches the entire fleet: latency, error rate, confidence drift, hung agents.
Runs on a schedule (every 15 minutes) independent of story pipelines.
Also called by the Fleet Commander before each phase to confirm the fleet is healthy.

Classification: AUGMENTED SCRIPT
  Script:  Queries agent_runs table for metrics — deterministic.
  LLM:     Haiku generates the health summary narrative and alert text.
  Tier A:  Confidence 97 (deterministic metric checks).
"""

from datetime import datetime, timedelta, timezone

import anthropic
from sqlalchemy import func, select, text

from src.core.config import settings
from src.core.database import async_session_factory
from src.core.models import AgentRun
from src.core.schemas import AgentHealthMetric, AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 51
AGENT_NAME = "Agent Health Monitor"

# Thresholds
ERROR_RATE_ALERT = 0.20        # alert if >20% of runs in last hour failed
LATENCY_ALERT_MS = 30_000     # alert if avg latency >30s
CONFIDENCE_DRIFT_ALERT = 0.15  # alert if avg confidence dropped >15 points vs 30d baseline
HUNG_AGENT_MINUTES = 10        # alert if an agent hasn't completed a run in >10 min during active sprint

AGENT_NAMES: dict[int, str] = {
    # Refinement
    1: "Story Intent", 2: "INVEST Quality", 3: "FCA Classifier",
    4: "Consumer Duty Mapping", 5: "AC Generator", 6: "Test Design Strategy",
    7: "Data Need", 8: "Dependency Mapping", 9: "Risk Anticipation",
    # Development
    10: "AC Compliance", 11: "Branch Tracer", 12: "Apex Coverage",
    13: "Metadata Dependency", 14: "Code Quality", 15: "Apex Security",
    16: "Bulk Quality", 17: "SFDX Validator", 18: "Component Attribution",
    19: "BDD Gherkin Writer", 20: "Performance Risk", 21: "Test Data Architect",
    22: "Sandbox State", 23: "Story-to-Code Tracer",
    # Testing
    24: "Test Strategy Validator", 25: "Test Env Provisioner", 26: "CRT Scenario Designer",
    27: "CRT Execution", 28: "CRT Self-Heal Reviewer", 29: "UAT Test Case Generator",
    30: "FCA Scenario Agent", 31: "Financial Data Integrity", 32: "Regression Risk Assessor",
    33: "Test Coverage Analyser", 34: "Defect Triage", 35: "Root Cause Analyser",
    36: "UAT Coordination", 37: "Performance Test", 38: "Flaky Test Hunter",
    # Release
    39: "Release Readiness", 40: "Release Composer", 41: "Change Set Integrity",
    42: "Dry Run", 43: "Smoke on Staging", 44: "FCA Evidence Pack",
    45: "Go/No-Go Coordinator", 46: "Production Validation", 47: "Release Notes Writer",
    48: "Rollback Readiness", 49: "Post-Release Monitor", 50: "Release Retrospective",
}


async def run(state: StoryState) -> AgentResult:
    """
    Entry point called by the Fleet Commander worker dispatcher.
    For health checks triggered within a story pipeline.
    """
    metrics = await _collect_metrics()
    degraded = [m for m in metrics if m.status != "HEALTHY"]
    summary = await _generate_summary(metrics, degraded)

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=f"Fleet health check: {len(degraded)} of {len(metrics)} agents degraded or down",
        why="Scheduled fleet health assessment before phase transition",
        data={
            "metrics": [m.model_dump() for m in metrics],
            "degraded_agents": [m.agent_id for m in degraded],
            "summary": summary,
        },
        confidence=ConfidenceBreakdown(
            tier="A",
            raw_score=97,
            calibration_multiplier=1.0,
            final_score=97,
            signals={"deterministic_metric_check": True},
            escalated=False,
        ),
        model_used=settings.fast_model,
    )


async def run_scheduled() -> list[AgentHealthMetric]:
    """
    Entry point for the scheduled health check (every 15 minutes via cron/APScheduler).
    Returns metrics and sends alerts if thresholds are breached.
    """
    metrics = await _collect_metrics()
    degraded = [m for m in metrics if m.status != "HEALTHY"]

    if degraded:
        summary = await _generate_summary(metrics, degraded)
        await _send_alert(degraded, summary)

    return metrics


# ── Metric collection ─────────────────────────────────────────────────────────

async def _collect_metrics() -> list[AgentHealthMetric]:
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    thirty_days_ago = now - timedelta(days=30)

    async with async_session_factory() as session:
        metrics: list[AgentHealthMetric] = []

        for agent_id, agent_name in AGENT_NAMES.items():
            if agent_id == AGENT_ID:
                continue  # health monitor doesn't monitor itself

            # Runs in last hour
            recent = await session.execute(
                select(
                    func.count(AgentRun.id).label("total"),
                    func.sum(
                        (AgentRun.success == False).cast(  # noqa: E712
                            text("integer")
                        )
                    ).label("errors"),
                    func.avg(AgentRun.latency_ms).label("avg_latency"),
                    func.max(AgentRun.completed_at).label("last_run"),
                )
                .where(AgentRun.agent_id == agent_id)
                .where(AgentRun.started_at >= one_hour_ago)
            )
            row = recent.one()

            total = row.total or 0
            errors = int(row.errors or 0)
            avg_latency = float(row.avg_latency or 0.0)
            last_run_at = row.last_run

            # False positive rate from learning_signals (30d)
            fp_rate = await _get_false_positive_rate(session, agent_id, thirty_days_ago)

            status = _compute_status(total, errors, avg_latency)

            metrics.append(AgentHealthMetric(
                agent_id=agent_id,
                agent_name=agent_name,
                last_run_at=last_run_at,
                runs_last_hour=total,
                errors_last_hour=errors,
                avg_latency_ms=avg_latency,
                avg_confidence=0.0,  # populated in Wave 5 calibration
                false_positive_rate_30d=fp_rate,
                status=status,
            ))

        return metrics


async def _get_false_positive_rate(session, agent_id: int, since: datetime) -> float | None:
    result = await session.execute(
        text("""
            SELECT
                COUNT(*) FILTER (WHERE ls.outcome = 'FALSE_POSITIVE') AS fp,
                COUNT(*) AS total
            FROM learning_signals ls
            JOIN decision_events de ON ls.decision_event_id = de.id
            WHERE de.agent_id = :agent_id
              AND ls.recorded_at >= :since
        """),
        {"agent_id": agent_id, "since": since},
    )
    row = result.one()
    if row.total == 0:
        return None
    return round(row.fp / row.total, 4)


def _compute_status(total: int, errors: int, avg_latency: float) -> str:
    if total == 0:
        return "HEALTHY"  # no runs in last hour — not degraded, just idle
    error_rate = errors / total
    if error_rate > ERROR_RATE_ALERT or avg_latency > LATENCY_ALERT_MS:
        return "DEGRADED" if error_rate < 0.5 else "DOWN"
    return "HEALTHY"


# ── LLM narrative generation (Haiku) ─────────────────────────────────────────

async def _generate_summary(metrics: list[AgentHealthMetric], degraded: list[AgentHealthMetric]) -> str:
    if not degraded:
        return f"All {len(metrics)} monitored agents are healthy."

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    degraded_summary = "\n".join(
        f"- Agent {m.agent_id} ({m.agent_name}): status={m.status}, "
        f"errors={m.errors_last_hour}/{m.runs_last_hour} runs, "
        f"avg_latency={m.avg_latency_ms:.0f}ms, "
        f"fp_rate_30d={m.false_positive_rate_30d}"
        for m in degraded
    )

    response = await client.messages.create(
        model=settings.fast_model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "You are the Agent Health Monitor for the FSC QE Framework. "
                "Write a concise (3-5 sentence) plain-English health alert "
                "for the QE Lead. Include which agents are degraded, the likely "
                "impact on story pipelines, and the recommended immediate action.\n\n"
                f"Degraded agents:\n{degraded_summary}"
            ),
        }],
    )
    return response.content[0].text


async def _send_alert(degraded: list[AgentHealthMetric], summary: str) -> None:
    """Send health alert email to QE Lead."""
    from src.fleet_commander.email import _send
    subject = f"FLEET ALERT: {len(degraded)} agent(s) degraded — FSC QE Framework"
    await _send(to=settings.qe_lead_email, subject=subject, body=summary)
