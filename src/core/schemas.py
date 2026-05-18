"""
Pydantic schemas — the What/Why/Data/Confidence trace and all agent I/O contracts.
Every agent input and output is typed here. No freeform dicts cross agent boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

# ── Confidence tiers ──────────────────────────────────────────────────────────

ConfidenceTier = Literal["A", "B", "C", "D"]

FCAClassification = Literal["HIGH", "MEDIUM", "LOW", "UNCLASSIFIED"]

GateStatusValue = Literal["PENDING", "OPEN", "CLOSED", "WAIVED", "ESCALATED", "BLOCKED"]

Phase = Literal["REFINEMENT", "DEVELOPMENT", "TESTING", "RELEASE", "COMPLETE", "BLOCKED"]

EventType = Literal[
    "AGENT_DECISION",
    "GATE_TRANSITION",
    "HUMAN_SIGNOFF",
    "WAIVER",
    "CONFLICT_RESOLVED",
    "ESCALATION",
    "CALIBRATION_CHANGE",
    "CRT_SELF_HEAL",
    "DAILY_EXPORT",
    "GONOGO_VOTE",
]

ActionType = Literal["SIGNOFF", "WAIVER", "GONOGO"]

ApproverRole = Literal["CO", "PO", "BUSINESS", "QE_LEAD", "TECH_LEAD"]


# ── Core explainability trace ─────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    tier: ConfidenceTier
    raw_score: Annotated[int, Field(ge=0, le=100)]
    calibration_multiplier: float = 1.0
    final_score: Annotated[int, Field(ge=0, le=100)]
    signals: dict[str, Any] = Field(default_factory=dict)
    escalated: bool = False


class AgentResult(BaseModel):
    agent_id: int
    agent_name: str
    what: str
    why: str
    data: dict[str, Any]
    confidence: ConfidenceBreakdown
    model_used: str
    completed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def final_score(self) -> int:
        return self.confidence.final_score

    @property
    def escalated(self) -> bool:
        return self.confidence.escalated


# ── Gate state ────────────────────────────────────────────────────────────────

class GateState(BaseModel):
    gate_id: str
    status: GateStatusValue = "PENDING"
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None


# ── Pending human approvals ───────────────────────────────────────────────────

class PendingApproval(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    story_id: str
    release_id: Optional[str] = None
    gate_id: str
    approver_email: str
    approver_role: ApproverRole
    action_type: ActionType
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    used: bool = False


# ── Conflict resolution ───────────────────────────────────────────────────────

class ConflictResolution(BaseModel):
    agent_a_id: int
    agent_b_id: int
    agent_a_recommendation: str
    agent_b_recommendation: str
    resolution: str
    reason: str
    winning_agent_id: Optional[int] = None
    resolved_at: datetime = Field(default_factory=datetime.utcnow)


# ── LangGraph story state ─────────────────────────────────────────────────────
# TypedDict so LangGraph can checkpoint it to PostgreSQL via JSON serialisation.
# Nested objects are plain dicts (model_dump()) — validated at agent boundaries.

from typing import TypedDict  # noqa: E402


class StoryState(TypedDict):
    story_id: str
    fca_classification: str                  # FCAClassification
    current_phase: str                       # Phase
    gate_states: dict[str, dict]             # {gate_id: GateState.model_dump()}
    agent_results: dict[str, dict]           # {str(agent_id): AgentResult.model_dump()}
    pending_approvals: list[dict]            # list of PendingApproval.model_dump()
    conflicts_resolved: list[dict]           # list of ConflictResolution.model_dump()
    block_reason: Optional[str]
    phase_errors: list[str]


def initial_story_state(story_id: str) -> StoryState:
    return StoryState(
        story_id=story_id,
        fca_classification="UNCLASSIFIED",
        current_phase="REFINEMENT",
        gate_states={
            gate: GateState(gate_id=gate).model_dump()
            for gate in ["G1", "G2", "G3", "G4", "G5", "G6",
                         "G7", "G8", "G9", "G10", "G11", "G12"]
        },
        agent_results={},
        pending_approvals=[],
        conflicts_resolved=[],
        block_reason=None,
        phase_errors=[],
    )


# ── Decision event (written to audit ledger) ──────────────────────────────────

class DecisionEventCreate(BaseModel):
    story_id: Optional[str] = None
    release_id: Optional[str] = None
    agent_id: Optional[int] = None
    gate_id: Optional[str] = None
    event_type: EventType
    what: str
    why: str
    data: dict[str, Any]
    confidence_tier: Optional[ConfidenceTier] = None
    raw_score: Optional[int] = None
    calibration_multiplier: Optional[float] = None
    final_score: Optional[int] = None
    model_used: Optional[str] = None
    actor_email: Optional[str] = None
    approval_ip: Optional[str] = None
    parent_event_id: Optional[str] = None

    @classmethod
    def from_agent_result(cls, result: AgentResult, story_id: str) -> "DecisionEventCreate":
        return cls(
            story_id=story_id,
            agent_id=result.agent_id,
            event_type="AGENT_DECISION",
            what=result.what,
            why=result.why,
            data=result.data,
            confidence_tier=result.confidence.tier,
            raw_score=result.confidence.raw_score,
            calibration_multiplier=result.confidence.calibration_multiplier,
            final_score=result.confidence.final_score,
            model_used=result.model_used,
        )


class DecisionEventRead(DecisionEventCreate):
    id: int
    event_id: str
    event_time: datetime
    previous_event_hash: Optional[str]
    row_hash: str


# ── Gate state update ─────────────────────────────────────────────────────────

class GateStateUpdate(BaseModel):
    story_id: str
    gate_id: str
    status: GateStatusValue
    decided_by: Optional[str] = None


# ── Learning signal (QE Lead override → calibration input) ───────────────────

class LearningSignalCreate(BaseModel):
    decision_event_id: int
    outcome: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE", "TRUE_NEGATIVE"]
    override_by_email: str
    override_reason: str


# ── Agent health metric ───────────────────────────────────────────────────────

class AgentHealthMetric(BaseModel):
    agent_id: int
    agent_name: str
    last_run_at: Optional[datetime]
    runs_last_hour: int
    errors_last_hour: int
    avg_latency_ms: float
    avg_confidence: float
    false_positive_rate_30d: Optional[float]
    status: Literal["HEALTHY", "DEGRADED", "DOWN"]
