"""
Agent 11 — Story-to-Branch Tracer
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Development Batch 1 (parallel with Agent 10, Agent 13).

Purpose:
  Deterministically validates that a git branch exists for the story, that its
  name follows the FSC naming convention (feature|bugfix|hotfix/FSC-XXXX-*),
  and that the story ID is embedded in the branch name for full traceability.

  If Copado is not configured, the agent degrades gracefully and reports
  branch data as unavailable — lowers confidence but does not block G2.

  Haiku generates the narrative — analysis is pure Python.

Output data keys consumed by downstream:
  branch_name          → str  (Agent 23 Story-to-Code Tracer — audit reference)
  commit_sha           → str  (Agent 23 — immutable code reference for FCA ledger)
  branch_naming_valid  → bool (Gate G2 — naming convention enforcement)
  story_id_in_branch   → bool (Gate G2 — traceability check)
  branch_found         → bool (Gate G2 — basic existence check)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.copado import get_branch_for_story

AGENT_ID = 11
AGENT_NAME = "Story-to-Branch Tracer"

# Accepted branch prefixes: feature/, bugfix/, hotfix/  followed by FSC-<digits>-
_BRANCH_PATTERN = re.compile(r"^(feature|bugfix|hotfix)/FSC-\d+-", re.IGNORECASE)
_STALE_DAYS_THRESHOLD = 14

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_branch_trace_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "traceability_risk"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining the branch traceability status for this story. "
                "Note whether the branch was found, naming conventions are met, and "
                "any traceability gaps the developer should fix."
            ),
        },
        "traceability_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low: Branch found, naming valid, story ID in branch name. "
                "medium: Branch found but naming issues or story ID absent from branch. "
                "high: No branch found, or branch has no story reference at all."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Story-to-Branch traceability check.
You will receive the branch information for a Jira user story and the validation result.
Write a clear 2–3 sentence narrative explaining whether the branch is properly linked
to the story and whether naming conventions are met. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    branch_info = await get_branch_for_story(story_id)

    # ── Deterministic analysis ────────────────────────────────────────────────
    branch_found, naming_valid, story_in_branch, branch_stale, age_days = (
        _analyse_branch(branch_info, story_id)
    )

    # ── Haiku trace generation ────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, branch_info, branch_found, naming_valid, story_in_branch, age_days,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        branch_info, branch_found, naming_valid, story_in_branch, branch_stale,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    branch_name = branch_info.get("branch_name", "")
    what = (
        f"Branch trace for {story_id}: "
        f"branch={'found' if branch_found else 'NOT FOUND'}, "
        f"naming={'valid' if naming_valid else 'INVALID'}, "
        f"story_id_in_branch={story_in_branch}"
    )
    why = trace.get(
        "narrative",
        "Story-to-Branch Tracer validated branch existence and naming convention.",
    )

    data = {
        "branch_name": branch_name,
        "commit_sha": branch_info.get("commit_sha", ""),
        "branch_found": branch_found,
        "branch_naming_valid": naming_valid,
        "story_id_in_branch": story_in_branch,
        "branch_stale": branch_stale,
        "branch_age_days": age_days,
        "author_email": branch_info.get("author_email", ""),
        "traceability_risk": trace.get("traceability_risk", "high"),
        "narrative": trace.get("narrative", ""),
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=why,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.fast_model,
    )


# ── Deterministic branch analysis ─────────────────────────────────────────────

def _analyse_branch(
    branch_info: dict,
    story_id: str,
) -> tuple[bool, bool, bool, bool, int]:
    """
    Validate branch existence, naming convention, and story ID embedding.
    Returns (branch_found, naming_valid, story_id_in_branch, branch_stale, age_days).
    Pure Python — no LLM involved.
    """
    branch_name = branch_info.get("branch_name", "")

    branch_found = bool(branch_name)
    naming_valid = bool(_BRANCH_PATTERN.match(branch_name)) if branch_found else False
    story_in_branch = story_id.upper() in branch_name.upper() if branch_found else False

    # Branch staleness from created_date
    age_days = 0
    created = branch_info.get("created_date", "")
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).days
        except (ValueError, TypeError):
            age_days = 0

    branch_stale = age_days > _STALE_DAYS_THRESHOLD

    return branch_found, naming_valid, story_in_branch, branch_stale, age_days


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    branch_info: dict,
    branch_found: bool,
    naming_valid: bool,
    story_in_branch: bool,
    branch_stale: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    # Signal 1: branch found — fundamental requirement
    if branch_found:
        scorer.add("branch_found", True, +10)
    else:
        scorer.add("branch_not_found", True, -20)

    # Signal 2: naming convention compliance
    if naming_valid:
        scorer.add("naming_convention_valid", True, +8)
    elif branch_found:
        scorer.add("naming_convention_invalid", True, -8)

    # Signal 3: story ID traceable in branch name
    if story_in_branch:
        scorer.add("story_id_in_branch", True, +8)
    elif branch_found:
        scorer.add("story_id_missing_from_branch", True, -10)

    # Signal 4: commit SHA present (evidence of actual code)
    if branch_info.get("commit_sha"):
        scorer.add("commit_sha_present", True, +5)

    # Signal 5: stale branch may indicate abandoned or long-lived work
    if branch_stale:
        scorer.add("branch_stale", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a branch traceability narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    branch_info: dict,
    branch_found: bool,
    naming_valid: bool,
    story_in_branch: bool,
    age_days: int,
) -> str:
    return (
        f"Story ID: {story_id}\n\n"
        f"Branch name: {branch_info.get('branch_name') or '(none found)'}\n"
        f"Commit SHA: {branch_info.get('commit_sha') or '(none)'}\n"
        f"Branch found: {branch_found}\n"
        f"Naming convention valid (feature|bugfix|hotfix/FSC-XXXX-*): {naming_valid}\n"
        f"Story ID present in branch name: {story_in_branch}\n"
        f"Branch age (days): {age_days}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )
