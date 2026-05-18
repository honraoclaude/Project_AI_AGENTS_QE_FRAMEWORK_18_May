"""
QDS MCP Server — Quality Data Store tools exposed to all Claude agents.

Agents call these tools to read/write gate state, emit decision events,
and record learning signals. This server is the only write path into the
audit ledger — all other code uses it rather than touching the DB directly.
"""

import hashlib
import json
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import settings
from src.core.models import (
    AgentRun,
    DecisionEvent,
    GateState,
    LearningSignal,
    PendingApproval,
)

mcp = FastMCP("qds-mcp", description="Quality Data Store — audit ledger and gate state")

_engine = create_async_engine(settings.database_url, pool_size=5)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


# ── Hash chain ────────────────────────────────────────────────────────────────

async def _get_last_hash(session: AsyncSession) -> str:
    row = await session.execute(
        text("SELECT row_hash FROM decision_events ORDER BY id DESC LIMIT 1")
    )
    result = row.scalar_one_or_none()
    return result or "GENESIS"


def _compute_row_hash(event_data: dict, previous_hash: str) -> str:
    payload = {
        "story_id":    event_data.get("story_id"),
        "agent_id":    event_data.get("agent_id"),
        "event_type":  event_data["event_type"],
        "what":        event_data["what"],
        "why":         event_data["why"],
        "data":        json.dumps(event_data["data"], sort_keys=True),
        "final_score": event_data.get("final_score"),
        "prev_hash":   previous_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def emit_decision_event(
    event_type: str,
    what: str,
    why: str,
    data: dict,
    story_id: str | None = None,
    release_id: str | None = None,
    agent_id: int | None = None,
    gate_id: str | None = None,
    confidence_tier: str | None = None,
    raw_score: int | None = None,
    calibration_multiplier: float | None = None,
    final_score: int | None = None,
    model_used: str | None = None,
    actor_email: str | None = None,
    approval_ip: str | None = None,
    parent_event_id: str | None = None,
) -> dict:
    """
    Write an immutable decision event to the QDS audit ledger.
    Every agent decision, gate transition, human sign-off, and waiver
    must be recorded here. Returns the created event's id and event_id.
    """
    async with _session_factory() as session:
        async with session.begin():
            prev_hash = await _get_last_hash(session)
            event_data = {
                "story_id": story_id, "agent_id": agent_id,
                "event_type": event_type, "what": what, "why": why,
                "data": data, "final_score": final_score,
            }
            row_hash = _compute_row_hash(event_data, prev_hash)

            event = DecisionEvent(
                story_id=story_id,
                release_id=release_id,
                agent_id=agent_id,
                gate_id=gate_id,
                event_type=event_type,
                what=what,
                why=why,
                data=data,
                confidence_tier=confidence_tier,
                raw_score=raw_score,
                calibration_multiplier=calibration_multiplier,
                final_score=final_score,
                model_used=model_used,
                actor_email=actor_email,
                approval_ip=approval_ip,
                parent_event_id=parent_event_id,
                previous_event_hash=prev_hash,
                row_hash=row_hash,
            )
            session.add(event)
            await session.flush()
            return {"id": event.id, "event_id": str(event.event_id)}


@mcp.tool()
async def get_gate_state(story_id: str, gate_id: str) -> dict:
    """Return the current gate state for a story and gate combination."""
    async with _session_factory() as session:
        row = await session.get(GateState, (story_id, gate_id))
        if row is None:
            return {"story_id": story_id, "gate_id": gate_id, "status": "PENDING"}
        return {
            "story_id": row.story_id,
            "gate_id": row.gate_id,
            "status": row.status,
            "decided_at": row.decided_at.isoformat() if row.decided_at else None,
            "decided_by": row.decided_by,
        }


@mcp.tool()
async def set_gate_state(
    story_id: str,
    gate_id: str,
    status: str,
    decided_by: str | None = None,
) -> dict:
    """
    Update the materialised gate state for a story.
    Valid statuses: PENDING, OPEN, CLOSED, WAIVED, ESCALATED, BLOCKED.
    """
    async with _session_factory() as session:
        async with session.begin():
            row = await session.get(GateState, (story_id, gate_id))
            if row is None:
                row = GateState(story_id=story_id, gate_id=gate_id)
                session.add(row)
            row.status = status
            row.decided_by = decided_by
            row.decided_at = datetime.now(timezone.utc)
            return {"story_id": story_id, "gate_id": gate_id, "status": status}


@mcp.tool()
async def get_story_trace(story_id: str) -> list[dict]:
    """
    Return all decision events for a story, ordered by event_time.
    Used by the FCA Evidence Pack Agent and the inspector audit interface.
    """
    async with _session_factory() as session:
        result = await session.execute(
            select(DecisionEvent)
            .where(DecisionEvent.story_id == story_id)
            .order_by(DecisionEvent.event_time)
        )
        events = result.scalars().all()
        return [
            {
                "id": e.id,
                "event_id": str(e.event_id),
                "event_time": e.event_time.isoformat(),
                "agent_id": e.agent_id,
                "gate_id": e.gate_id,
                "event_type": e.event_type,
                "what": e.what,
                "why": e.why,
                "data": e.data,
                "final_score": e.final_score,
                "confidence_tier": e.confidence_tier,
                "actor_email": e.actor_email,
            }
            for e in events
        ]


@mcp.tool()
async def record_pending_approval(
    approval_id: str,
    story_id: str,
    gate_id: str,
    approver_email: str,
    approver_role: str,
    action_type: str,
    expires_at: str,
    release_id: str | None = None,
) -> dict:
    """
    Record an email approval link that has been sent.
    approval_id is the UUID embedded in the HMAC-signed link.
    expires_at is an ISO-format datetime string.
    """
    async with _session_factory() as session:
        async with session.begin():
            approval = PendingApproval(
                approval_id=approval_id,
                story_id=story_id,
                release_id=release_id,
                gate_id=gate_id,
                approver_email=approver_email,
                approver_role=approver_role,
                action_type=action_type,
                expires_at=datetime.fromisoformat(expires_at),
            )
            session.add(approval)
            return {"approval_id": approval_id, "recorded": True}


@mcp.tool()
async def mark_approval_used(
    approval_id: str,
    actor_email: str,
    approval_ip: str,
    decision: str,
    reason: str | None = None,
) -> dict:
    """
    Mark an approval link as used after the approver has clicked and submitted.
    Returns the parent decision_event_id to allow override chaining.
    """
    async with _session_factory() as session:
        async with session.begin():
            approval = await session.get(PendingApproval, approval_id)
            if approval is None:
                return {"error": "approval_not_found"}
            if approval.used:
                return {"error": "already_used"}
            approval.used = True
            approval.used_at = datetime.now(timezone.utc)

            # Emit the sign-off as a decision event
            prev_hash = await _get_last_hash(session)
            event_data = {
                "story_id": approval.story_id, "agent_id": None,
                "event_type": "HUMAN_SIGNOFF",
                "what": f"{approval.approver_role} {decision} on {approval.gate_id} for {approval.story_id}",
                "why": reason or "No reason provided",
                "data": {
                    "approval_id": approval_id,
                    "gate_id": approval.gate_id,
                    "story_id": approval.story_id,
                    "decision": decision,
                    "role": approval.approver_role,
                },
                "final_score": 99,
            }
            row_hash = _compute_row_hash(event_data, prev_hash)
            event = DecisionEvent(
                story_id=approval.story_id,
                gate_id=approval.gate_id,
                event_type="HUMAN_SIGNOFF",
                what=event_data["what"],
                why=event_data["why"],
                data=event_data["data"],
                final_score=99,
                confidence_tier="A",
                actor_email=actor_email,
                approval_ip=approval_ip,
                previous_event_hash=prev_hash,
                row_hash=row_hash,
            )
            session.add(event)
            await session.flush()
            return {"approval_id": approval_id, "decision_event_id": event.id}


@mcp.tool()
async def record_learning_signal(
    decision_event_id: int,
    outcome: str,
    override_by_email: str,
    override_reason: str,
) -> dict:
    """
    Record a QE Lead override as a learning signal for the calibration loop.
    outcome must be one of: TRUE_POSITIVE, FALSE_POSITIVE, FALSE_NEGATIVE, TRUE_NEGATIVE.
    """
    async with _session_factory() as session:
        async with session.begin():
            signal = LearningSignal(
                decision_event_id=decision_event_id,
                outcome=outcome,
                override_by_email=override_by_email,
                override_reason=override_reason,
            )
            session.add(signal)
            return {"recorded": True, "decision_event_id": decision_event_id, "outcome": outcome}


@mcp.tool()
async def record_agent_run(
    agent_id: int,
    started_at: str,
    success: bool,
    story_id: str | None = None,
    completed_at: str | None = None,
    latency_ms: int | None = None,
    error_message: str | None = None,
) -> dict:
    """
    Record an agent execution for health monitoring (Agent 51).
    Called by the Fleet Commander after each agent worker completes.
    """
    async with _session_factory() as session:
        async with session.begin():
            run = AgentRun(
                agent_id=agent_id,
                story_id=story_id,
                started_at=datetime.fromisoformat(started_at),
                completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message,
            )
            session.add(run)
            await session.flush()
            return {"run_id": run.id}


@mcp.tool()
async def verify_hash_chain(story_id: str | None = None) -> dict:
    """
    Verify the integrity of the hash chain.
    If story_id is provided, verifies only that story's events.
    Returns VERIFIED or the first tampered event_id.
    """
    async with _session_factory() as session:
        query = select(DecisionEvent).order_by(DecisionEvent.id)
        if story_id:
            query = query.where(DecisionEvent.story_id == story_id)
        result = await session.execute(query)
        events = result.scalars().all()

        prev_hash = "GENESIS"
        for event in events:
            expected = _compute_row_hash(
                {
                    "story_id": event.story_id,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type,
                    "what": event.what,
                    "why": event.why,
                    "data": event.data,
                    "final_score": event.final_score,
                },
                prev_hash,
            )
            if event.row_hash != expected:
                return {
                    "status": "TAMPERED",
                    "first_failing_event_id": str(event.event_id),
                    "at_id": event.id,
                }
            prev_hash = event.row_hash

        return {"status": "VERIFIED", "events_checked": len(events)}


if __name__ == "__main__":
    mcp.run()
