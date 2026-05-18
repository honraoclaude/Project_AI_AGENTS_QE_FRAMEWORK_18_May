"""Initial schema — all Wave 1 tables + role grants.

Revision ID: 001
Revises:
Create Date: 2026-05-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── decision_events — immutable audit ledger ──────────────────────────────
    op.create_table(
        "decision_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("event_id", UUID(as_uuid=False), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("story_id", sa.String(50)),
        sa.Column("release_id", sa.String(50)),
        sa.Column("agent_id", sa.Integer),
        sa.Column("gate_id", sa.String(10)),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("what", sa.Text, nullable=False),
        sa.Column("why", sa.Text, nullable=False),
        sa.Column("data", JSONB, nullable=False),
        sa.Column("confidence_tier", sa.String(1)),
        sa.Column("raw_score", sa.Integer),
        sa.Column("calibration_multiplier", sa.Numeric(5, 3)),
        sa.Column("final_score", sa.Integer),
        sa.Column("model_used", sa.String(100)),
        sa.Column("actor_email", sa.String(255)),
        sa.Column("approval_ip", INET),
        sa.Column("parent_event_id", UUID(as_uuid=False)),
        sa.Column("previous_event_hash", sa.String(64)),
        sa.Column("row_hash", sa.String(64), nullable=False),
    )
    op.create_index("idx_de_story_id", "decision_events", ["story_id"])
    op.create_index("idx_de_event_type", "decision_events", ["event_type"])
    op.create_index("idx_de_gate_id", "decision_events", ["gate_id"])
    op.create_index("idx_de_event_time", "decision_events", ["event_time"])
    op.create_index("idx_de_agent_id", "decision_events", ["agent_id"])
    op.create_index("idx_de_actor_email", "decision_events", ["actor_email"])
    op.create_index(
        "idx_de_escalated", "decision_events", ["final_score"],
        postgresql_where=sa.text("final_score < 60"),
    )

    # ── gate_state — materialised current state per story ─────────────────────
    op.create_table(
        "gate_state",
        sa.Column("story_id", sa.String(50), primary_key=True),
        sa.Column("gate_id", sa.String(10), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.Column("decided_by", sa.String(255)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ── pending_approvals — email links awaiting click ────────────────────────
    op.create_table(
        "pending_approvals",
        sa.Column("approval_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("story_id", sa.String(50)),
        sa.Column("release_id", sa.String(50)),
        sa.Column("gate_id", sa.String(10), nullable=False),
        sa.Column("approver_email", sa.String(255), nullable=False),
        sa.Column("approver_role", sa.String(20), nullable=False),
        sa.Column("action_type", sa.String(20), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("used_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_pa_story_gate", "pending_approvals", ["story_id", "gate_id"])
    op.create_index("idx_pa_used", "pending_approvals", ["used"])

    # ── learning_signals — QE Lead override → calibration input ──────────────
    op.create_table(
        "learning_signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("decision_event_id", sa.BigInteger, sa.ForeignKey("decision_events.id"), nullable=False),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("override_by_email", sa.String(255), nullable=False),
        sa.Column("override_reason", sa.Text, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_ls_event_id", "learning_signals", ["decision_event_id"])

    # ── agent_calibration — weekly multipliers ────────────────────────────────
    op.create_table(
        "agent_calibration",
        sa.Column("agent_id", sa.Integer, primary_key=True),
        sa.Column("calibration_date", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("multiplier", sa.Numeric(5, 3), nullable=False, server_default="1.000"),
        sa.Column("false_positive_rate", sa.Numeric(5, 4)),
        sa.Column("false_negative_rate", sa.Numeric(5, 4)),
        sa.Column("status", sa.String(10), nullable=False, server_default="PROPOSED"),
        sa.Column("reviewed_by_email", sa.String(255)),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
    )

    # ── agent_runs — latency + error tracking for Agent 51 ───────────────────
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("agent_id", sa.Integer, nullable=False),
        sa.Column("story_id", sa.String(50)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("success", sa.Boolean, nullable=False),
        sa.Column("error_message", sa.Text),
    )
    op.create_index("idx_ar_agent_id", "agent_runs", ["agent_id"])
    op.create_index("idx_ar_started_at", "agent_runs", ["started_at"])

    # ── Role grants — least privilege ─────────────────────────────────────────
    # qe_agent_writer: INSERT only on decision_events, full access on state tables
    op.execute("GRANT INSERT ON decision_events TO qe_agent_writer")
    op.execute("GRANT SELECT ON decision_events TO qe_agent_writer")
    op.execute("GRANT SELECT, INSERT, UPDATE ON gate_state TO qe_agent_writer")
    op.execute("GRANT SELECT, INSERT, UPDATE ON pending_approvals TO qe_agent_writer")
    op.execute("GRANT SELECT, INSERT ON learning_signals TO qe_agent_writer")
    op.execute("GRANT SELECT, INSERT, UPDATE ON agent_calibration TO qe_agent_writer")
    op.execute("GRANT SELECT, INSERT ON agent_runs TO qe_agent_writer")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO qe_agent_writer")

    # qe_audit_reader: SELECT only on all tables
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO qe_audit_reader")


def downgrade() -> None:
    op.drop_table("agent_runs")
    op.drop_table("agent_calibration")
    op.drop_table("learning_signals")
    op.drop_table("pending_approvals")
    op.drop_table("gate_state")
    op.drop_table("decision_events")
