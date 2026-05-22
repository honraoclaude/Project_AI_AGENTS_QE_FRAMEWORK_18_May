"""
Agent 17 — SFDX Source-Format Validator
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs sequentially after Batch 2 (after Agents 12, 14, 15, 16).
Has access to Agent 13 (changed files list).

Purpose:
  Validates that all changed metadata files use SFDX source format
  (force-app/main/default/...) rather than the legacy metadata format
  (src/classes/, src/objects/, metadata/).

  Old-format files indicate the repository has not fully migrated to
  SFDX source tracking, which breaks Copado deployment reliability.

  Haiku generates the narrative — file path analysis is pure Python.

Output data keys consumed by downstream:
  sfdx_format_valid  → bool (Agent 22 Sandbox State — health check)
  invalid_files      → list (Agent 23 audit trail — files needing migration)
  sfdx_verdict       → str  (Gate G2 — WARN on invalid, does not block)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 17
AGENT_NAME = "SFDX Source-Format Validator"

_SFDX_ROOT = "force-app/"
_LEGACY_ROOTS = ("src/classes/", "src/objects/", "src/triggers/", "metadata/", "unpackaged/")

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_sfdx_validation_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "migration_urgency"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining the SFDX format validation result. "
                "Note how many files were checked, whether any are in legacy format, "
                "and what the developer must do to fix format issues."
            ),
        },
        "migration_urgency": {
            "type": "string",
            "enum": ["none", "low", "high"],
            "description": (
                "none: All files in SFDX format or no files changed. "
                "low: 1–2 legacy-format files detected. "
                "high: 3+ legacy-format files or all changed files are in legacy format."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for an SFDX source-format validation.
You will receive a list of changed metadata files and the validation result. Write
a clear 2–3 sentence narrative explaining whether the code uses SFDX source format,
what the consequences are if files are in legacy format, and what the developer
must do to address any issues. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent13_data = _get_agent_data(state, "13")

    changed_files = _get_changed_files(agent13_data)

    # ── Deterministic analysis ────────────────────────────────────────────────
    valid_count, invalid_files, sfdx_valid, verdict = _validate_sfdx_format(changed_files)

    # ── Haiku trace generation ────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, len(changed_files), valid_count, invalid_files, verdict,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(changed_files, sfdx_valid, invalid_files)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"SFDX validation for {story_id}: {len(changed_files)} file(s) checked, "
        f"{len(invalid_files)} invalid — verdict={verdict}"
    )
    why = trace.get(
        "narrative",
        "SFDX Validator checked all changed metadata files for source-format compliance.",
    )

    data = {
        "sfdx_format_valid": sfdx_valid,
        "sfdx_verdict": verdict,
        "valid_file_count": valid_count,
        "invalid_files": invalid_files,
        "total_files_checked": len(changed_files),
        "migration_urgency": trace.get("migration_urgency", "none"),
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


# ── Deterministic SFDX validation ────────────────────────────────────────────

def _validate_sfdx_format(
    changed_files: list[dict],
) -> tuple[int, list[str], bool, str]:
    """
    Validate changed file paths against SFDX source format.
    Returns (valid_count, invalid_files, all_valid, verdict).
    Pure Python — no LLM involved.
    """
    if not changed_files:
        return 0, [], True, "PASS"

    invalid: list[str] = []
    for f in changed_files:
        path = f.get("file_path", "").lower()
        if not path:
            continue
        is_sfdx = path.startswith(_SFDX_ROOT)
        is_legacy = any(path.startswith(r) for r in _LEGACY_ROOTS)
        if is_legacy or (not is_sfdx and path):
            invalid.append(f.get("file_path", path))

    valid_count = len(changed_files) - len(invalid)
    all_valid = len(invalid) == 0

    if all_valid:
        verdict = "PASS"
    elif len(invalid) <= 2:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return valid_count, invalid, all_valid, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    changed_files: list[dict],
    sfdx_valid: bool,
    invalid_files: list[str],
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if changed_files:
        scorer.add("files_available_for_check", len(changed_files), +8)
    else:
        scorer.add("no_files_to_check", 0, -5)

    if sfdx_valid and changed_files:
        scorer.add("all_files_sfdx_format", True, +10)
    elif invalid_files:
        penalty = min(len(invalid_files) * 4, 16)
        scorer.add("legacy_format_files_found", len(invalid_files), -penalty)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an SFDX format validation narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    total: int,
    valid: int,
    invalid: list[str],
    verdict: str,
) -> str:
    return (
        f"Story: {story_id}\n"
        f"Total files checked: {total}\n"
        f"Valid SFDX format: {valid}\n"
        f"Invalid (legacy format): {len(invalid)}\n"
        f"Invalid files: {invalid or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_changed_files(agent13_data: dict | None) -> list[dict]:
    if not agent13_data:
        return []
    return agent13_data.get("changed_files", [])


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
