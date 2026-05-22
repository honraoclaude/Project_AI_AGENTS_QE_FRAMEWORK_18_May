"""
Sign-off service — FastAPI application handling all human approval flows.

Three approval types:
  SIGNOFF  — agent-triggered CO/PO/Business approval (G1, G6, G10)
  WAIVER   — QE Lead-initiated gate override (any gate)
  GONOGO   — release quorum collection (G11)

Every POST records a decision_event to the QDS and publishes a Redis
resume event so the Fleet Commander graph can continue from its interrupt.
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.core.config import settings
from src.mcp.qds_mcp import server as qds

app = FastAPI(title="FSC QE Sign-off Service", docs_url=None, redoc_url=None)

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url)
    return _redis


# ── Token verification ────────────────────────────────────────────────────────

def _verify_token(token: str) -> dict:
    """
    Verify HMAC signature and expiry. Returns the decoded payload or raises 410.
    Raises HTTPException(410) for invalid, expired, or malformed tokens.
    """
    try:
        encoded, signature = token.rsplit(".", 1)
    except ValueError:
        raise HTTPException(status_code=410, detail="Invalid link")

    expected = hmac.new(
        settings.signoff_hmac_secret.get_secret_value().encode(),
        encoded.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=410, detail="Invalid link")

    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded + "==").decode())
    except Exception:
        raise HTTPException(status_code=410, detail="Invalid link")

    exp = datetime.fromisoformat(payload["exp"])
    if datetime.now(timezone.utc) > exp:
        raise HTTPException(status_code=410, detail="Link expired")

    return payload


# ── Sign-off endpoints ────────────────────────────────────────────────────────

@app.get("/signoff/{token}", response_class=HTMLResponse)
async def render_signoff(token: str, decision: str | None = None) -> HTMLResponse:
    """Render the sign-off confirmation page."""
    payload = _verify_token(token)
    story_id = payload["sid"]
    gate_id = payload["gate"]
    action_type = payload["type"]

    trace = await qds.get_story_trace(story_id)
    gate_events = [e for e in trace if e.get("gate_id") == gate_id]
    latest = gate_events[-1] if gate_events else {}

    pre_decision = decision or ""
    return HTMLResponse(_render_signoff_page(
        story_id=story_id,
        gate_id=gate_id,
        action_type=action_type,
        what=latest.get("what", ""),
        why=latest.get("why", ""),
        confidence=latest.get("final_score"),
        token=token,
        pre_decision=pre_decision,
    ))


@app.post("/signoff/{token}", response_class=HTMLResponse)
async def submit_signoff(
    request: Request,
    token: str,
    decision: Annotated[str, Form()],
    reason: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Record the sign-off decision and resume the Fleet Commander graph."""
    payload = _verify_token(token)
    approval_id = payload["aid"]
    story_id = payload["sid"]
    gate_id = payload["gate"]

    client_ip = request.client.host if request.client else "unknown"

    result = await qds.mark_approval_used(
        approval_id=approval_id,
        actor_email=payload["email"],
        approval_ip=client_ip,
        decision=decision,
        reason=reason or None,
    )

    if "error" in result:
        if result["error"] == "already_used":
            raise HTTPException(status_code=410, detail="This link has already been used")
        raise HTTPException(status_code=404, detail="Approval not found")

    # Update gate state based on decision
    new_status = "CLOSED" if decision == "APPROVE" else "BLOCKED"
    await qds.set_gate_state(
        story_id=story_id,
        gate_id=gate_id,
        status=new_status,
        decided_by=payload["email"],
    )

    # Resume the Fleet Commander graph via Redis pub/sub
    redis = await get_redis()
    await redis.publish(
        f"resume:{story_id}",
        json.dumps({"story_id": story_id, "gate_id": gate_id, "decision": decision}),
    )

    return HTMLResponse(_render_confirmation(story_id, gate_id, decision))


# ── Waiver endpoints ──────────────────────────────────────────────────────────

@app.get("/waiver/{token}", response_class=HTMLResponse)
async def render_waiver(token: str) -> HTMLResponse:
    """Render the waiver request form for a QE Lead."""
    payload = _verify_token(token)
    story_id = payload["sid"]
    gate_id = payload["gate"]
    trace = await qds.get_story_trace(story_id)
    gate_state = await qds.get_gate_state(story_id, gate_id)
    return HTMLResponse(_render_waiver_page(story_id, gate_id, gate_state, trace, token))


@app.post("/waiver/{token}", response_class=HTMLResponse)
async def submit_waiver(
    request: Request,
    token: str,
    reason: Annotated[str, Form()],
) -> HTMLResponse:
    """Record the waiver. For HIGH-FCA gates, triggers a CO counter-sign email."""
    if not reason.strip():
        raise HTTPException(status_code=422, detail="Waiver reason is mandatory")

    payload = _verify_token(token)
    story_id = payload["sid"]
    gate_id = payload["gate"]
    client_ip = request.client.host if request.client else "unknown"
    fca = payload.get("fca", "UNKNOWN")

    await qds.emit_decision_event(
        event_type="WAIVER",
        story_id=story_id,
        gate_id=gate_id,
        what=f"QE Lead waived {gate_id} for {story_id}",
        why=reason,
        data={"gate_id": gate_id, "story_id": story_id, "fca_classification": fca},
        final_score=99,
        confidence_tier="A",
        actor_email=payload["email"],
        approval_ip=client_ip,
    )

    if fca == "HIGH":
        # Require CO counter-sign — gate stays BLOCKED until CO approves
        from src.fleet_commander.email import send_approval_email
        await send_approval_email(
            story_id=story_id,
            gate_id=gate_id,
            approver_email=settings.compliance_officer_email,
            approver_role="CO",
            action_type="WAIVER",
            context={"reason": reason, "qe_lead": payload["email"], "gate_id": gate_id},
        )
        return HTMLResponse(_render_confirmation(story_id, gate_id, "WAIVER_PENDING_CO"))

    # Non-HIGH-FCA: gate is waived immediately
    await qds.set_gate_state(story_id=story_id, gate_id=gate_id, status="WAIVED", decided_by=payload["email"])
    redis = await get_redis()
    await redis.publish(f"resume:{story_id}", json.dumps({"story_id": story_id, "gate_id": gate_id, "decision": "WAIVE"}))
    return HTMLResponse(_render_confirmation(story_id, gate_id, "WAIVED"))


# ── Audit read endpoints ──────────────────────────────────────────────────────

@app.get("/audit/story/{story_id}")
async def get_story_audit(story_id: str) -> dict:
    """Full event trail for a story — for FCA inspector queries."""
    events = await qds.get_story_trace(story_id)
    return {"story_id": story_id, "event_count": len(events), "events": events}


@app.get("/audit/verify")
async def verify_integrity(story_id: str | None = None) -> dict:
    """Verify the hash chain integrity."""
    return await qds.verify_hash_chain(story_id=story_id)


# ── HTML templates (minimal, no external deps) ────────────────────────────────

def _render_signoff_page(
    story_id: str, gate_id: str, action_type: str,
    what: str, why: str, confidence: int | None,
    token: str, pre_decision: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Sign-off: {story_id} — {gate_id}</title>
<style>body{{font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px}}
.trace{{background:#f5f5f5;padding:16px;border-radius:4px;margin:16px 0}}
.confidence{{font-weight:bold;color:{"#c00" if confidence and confidence<60 else "#060"}}}
button{{padding:10px 24px;font-size:16px;cursor:pointer;border:none;border-radius:4px}}
.approve{{background:#0a0;color:#fff;margin-right:12px}}.reject{{background:#c00;color:#fff}}
textarea{{width:100%;height:80px;margin:8px 0;padding:8px;box-sizing:border-box}}</style>
</head><body>
<h2>Sign-off Required: {story_id} — Gate {gate_id}</h2>
<div class="trace">
  <p><strong>What:</strong> {what}</p>
  <p><strong>Why:</strong> {why}</p>
  <p><strong>Confidence:</strong> <span class="confidence">{confidence}/100</span></p>
</div>
<form method="POST">
  <p>Optional reason:</p>
  <textarea name="reason" placeholder="Add context for the audit trail (optional)"></textarea>
  <br>
  <button class="approve" type="submit" name="decision" value="APPROVE">Approve</button>
  <button class="reject" type="submit" name="decision" value="REJECT">Reject</button>
</form>
</body></html>"""


def _render_waiver_page(story_id: str, gate_id: str, gate_state: dict, trace: list, token: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Waiver: {story_id} — {gate_id}</title>
<style>body{{font-family:sans-serif;max-width:700px;margin:40px auto;padding:0 20px}}
.warning{{background:#fff3cd;border:1px solid #ffc107;padding:12px;border-radius:4px}}
textarea{{width:100%;height:100px;margin:8px 0;padding:8px;box-sizing:border-box}}
button{{padding:10px 24px;font-size:16px;cursor:pointer;background:#c00;color:#fff;border:none;border-radius:4px}}</style>
</head><body>
<h2>Gate Waiver Request: {story_id} — {gate_id}</h2>
<div class="warning">
  <strong>Warning:</strong> Waiving a hard gate is an audited action.
  Your name and reason will be permanently recorded in the FCA audit ledger.
  HIGH-FCA gates require Compliance Officer counter-sign.
</div>
<p>Current gate status: <strong>{gate_state.get("status")}</strong></p>
<form method="POST">
  <p><strong>Reason (mandatory):</strong></p>
  <textarea name="reason" placeholder="Explain why this gate is being waived and what manual checks have been completed..." required></textarea>
  <br>
  <button type="submit">Submit Waiver Request</button>
</form>
</body></html>"""


def _render_confirmation(story_id: str, gate_id: str, decision: str) -> str:
    messages = {
        "APPROVE": f"Gate {gate_id} approved for {story_id}. The pipeline will continue.",
        "REJECT": f"Gate {gate_id} rejected for {story_id}. The story has been blocked.",
        "WAIVED": f"Gate {gate_id} waived for {story_id}. Pipeline resuming.",
        "WAIVER_PENDING_CO": f"Waiver submitted. Compliance Officer counter-sign required before gate {gate_id} can be waived.",
    }
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Confirmed</title>
<style>body{{font-family:sans-serif;max-width:600px;margin:80px auto;text-align:center}}
.msg{{font-size:18px;margin:24px 0}}p{{color:#555}}</style>
</head><body>
<h2>Decision Recorded</h2>
<div class="msg">{messages.get(decision, f"Decision '{decision}' recorded.")}</div>
<p>This action has been written to the FCA audit ledger.</p>
<p>You may close this window.</p>
</body></html>"""
