"""
Email service — HMAC-signed approval links for human-in-the-loop sign-offs.
Sends via Azure Communication Services. Falls back to stdout in local dev.
"""

import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone

from src.core.config import settings
from src.core.schemas import PendingApproval
from src.mcp.qds_mcp import server as qds


def _sign_payload(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(
        settings.signoff_hmac_secret.get_secret_value().encode(),
        encoded.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def _build_link(approval_id: str, story_id: str, gate_id: str, approver_email: str, action_type: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.signoff_link_expiry_hours)
    payload = {
        "aid":   approval_id,
        "sid":   story_id,
        "gate":  gate_id,
        "email": approver_email,
        "type":  action_type,
        "exp":   expires_at.isoformat(),
    }
    token = _sign_payload(payload)
    return f"{settings.signoff_base_url}/signoff/{token}", expires_at


async def send_approval_email(
    story_id: str,
    gate_id: str,
    approver_email: str,
    approver_role: str,
    action_type: str,
    context: dict,
    release_id: str | None = None,
) -> dict:
    """
    Generate a signed approval link, record it in the QDS, and send the email.
    Returns a PendingApproval dict to be stored in the story state.
    """
    approval_id = str(uuid.uuid4())
    link, expires_at = _build_link(approval_id, story_id, gate_id, approver_email, action_type)

    await qds.record_pending_approval(
        approval_id=approval_id,
        story_id=story_id,
        gate_id=gate_id,
        approver_email=approver_email,
        approver_role=approver_role,
        action_type=action_type,
        expires_at=expires_at.isoformat(),
        release_id=release_id,
    )

    subject, body = _build_email_content(
        story_id=story_id,
        gate_id=gate_id,
        approver_role=approver_role,
        action_type=action_type,
        context=context,
        link=link,
        expires_at=expires_at,
    )

    await _send(to=approver_email, subject=subject, body=body)

    return PendingApproval(
        approval_id=approval_id,
        story_id=story_id,
        release_id=release_id,
        gate_id=gate_id,
        approver_email=approver_email,
        approver_role=approver_role,
        action_type=action_type,
        expires_at=expires_at,
    ).model_dump()


def _build_email_content(
    story_id: str,
    gate_id: str,
    approver_role: str,
    action_type: str,
    context: dict,
    link: str,
    expires_at: datetime,
) -> tuple[str, str]:
    fca = context.get("fca_classification", "UNKNOWN")
    invest = context.get("invest_score", "N/A")
    confidence = context.get("agent_3_confidence", "N/A")

    subject = f"ACTION REQUIRED: {story_id} — Gate {gate_id} ({fca}-FCA)"
    body = f"""
Story {story_id} requires your sign-off at Gate {gate_id}.

Classification : {fca}-FCA
INVEST Score   : {invest}/100
Confidence     : {confidence}/100

Agent Decision
  What  : {context.get("agent_3_what", "N/A")}
  Why   : {context.get("agent_3_why", "N/A")}

This story cannot proceed until you have reviewed and responded.

  APPROVE: {link}&decision=APPROVE
  REJECT : {link}&decision=REJECT

This link expires at {expires_at.strftime("%Y-%m-%d %H:%M UTC")} and is single-use.

Full audit trace is available in the QDS at {settings.signoff_base_url}/audit/story/{story_id}
""".strip()

    return subject, body


async def _send(to: str, subject: str, body: str) -> None:
    """Send via Azure Communication Services, or print to stdout in dev."""
    conn_str = settings.azure_comm_connection_string.get_secret_value()

    if not conn_str or conn_str == "":
        # Local dev — print to stdout
        print(f"\n{'='*60}")
        print(f"TO: {to}")
        print(f"SUBJECT: {subject}")
        print(f"BODY:\n{body}")
        print(f"{'='*60}\n")
        return

    from azure.communication.email import EmailClient
    client = EmailClient.from_connection_string(conn_str)
    message = {
        "senderAddress": settings.azure_comm_sender,
        "recipients": {"to": [{"address": to}]},
        "content": {"subject": subject, "plainText": body},
    }
    client.begin_send(message)
