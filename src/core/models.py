"""
SQLAlchemy ORM models.
decision_events is append-only — enforced at the PostgreSQL role level.
No application code should UPDATE or DELETE from this table.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class DecisionEvent(Base):
    """Immutable audit ledger. INSERT only — never UPDATE or DELETE."""

    __tablename__ = "decision_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(UUID(as_uuid=False), server_default=func.gen_random_uuid(), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Context
    story_id: Mapped[Optional[str]] = mapped_column(String(50))
    release_id: Mapped[Optional[str]] = mapped_column(String(50))
    agent_id: Mapped[Optional[int]] = mapped_column(Integer)
    gate_id: Mapped[Optional[str]] = mapped_column(String(10))

    # Classification
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Explainability trace
    what: Mapped[str] = mapped_column(Text, nullable=False)
    why: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Confidence breakdown
    confidence_tier: Mapped[Optional[str]] = mapped_column(String(1))
    raw_score: Mapped[Optional[int]] = mapped_column(Integer)
    calibration_multiplier: Mapped[Optional[float]] = mapped_column(Numeric(5, 3))
    final_score: Mapped[Optional[int]] = mapped_column(Integer)
    model_used: Mapped[Optional[str]] = mapped_column(String(100))

    # Human attribution
    actor_email: Mapped[Optional[str]] = mapped_column(String(255))
    approval_ip: Mapped[Optional[str]] = mapped_column(INET)

    # Chain integrity
    parent_event_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False))
    previous_event_hash: Mapped[Optional[str]] = mapped_column(String(64))
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_de_story_id", "story_id"),
        Index("idx_de_event_type", "event_type"),
        Index("idx_de_gate_id", "gate_id"),
        Index("idx_de_event_time", "event_time"),
        Index("idx_de_agent_id", "agent_id"),
        Index("idx_de_actor_email", "actor_email"),
        # Partial index — fast lookup of escalations
        Index("idx_de_escalated", "final_score", postgresql_where="final_score < 60"),
    )


class GateState(Base):
    """Materialised current gate status per story. Updated on each gate transition."""

    __tablename__ = "gate_state"

    story_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    gate_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[Optional[str]] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PendingApproval(Base):
    """Email approval links awaiting click. Marked used=True on first click."""

    __tablename__ = "pending_approvals"

    approval_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    story_id: Mapped[Optional[str]] = mapped_column(String(50))
    release_id: Mapped[Optional[str]] = mapped_column(String(50))
    gate_id: Mapped[str] = mapped_column(String(10), nullable=False)
    approver_email: Mapped[str] = mapped_column(String(255), nullable=False)
    approver_role: Mapped[str] = mapped_column(String(20), nullable=False)
    action_type: Mapped[str] = mapped_column(String(20), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_pa_story_gate", "story_id", "gate_id"),
        Index("idx_pa_used", "used"),
    )


class LearningSignal(Base):
    """QE Lead overrides — the calibration input for the Severity Calibration Agent."""

    __tablename__ = "learning_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    decision_event_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("decision_events.id"), nullable=False
    )
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    override_by_email: Mapped[str] = mapped_column(String(255), nullable=False)
    override_reason: Mapped[str] = mapped_column(Text, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_ls_event_id", "decision_event_id"),)


class AgentCalibration(Base):
    """Weekly calibration multipliers per agent — reviewed and activated by QE Lead."""

    __tablename__ = "agent_calibration"

    agent_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calibration_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    multiplier: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False, default=1.000)
    false_positive_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    false_negative_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="PROPOSED")
    reviewed_by_email: Mapped[Optional[str]] = mapped_column(String(255))
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class AgentRun(Base):
    """Execution log — latency and error tracking for Agent 51 (Health Monitor)."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    story_id: Mapped[Optional[str]] = mapped_column(String(50))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_ar_agent_id", "agent_id"),
        Index("idx_ar_started_at", "started_at"),
    )
