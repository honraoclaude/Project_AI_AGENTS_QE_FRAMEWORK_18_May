"""
Agent 41 — Change Set Integrity
Phase       : Release
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs sequentially after Agent 40.
Has access to Agents 13, 40.

Purpose:
  Validates the integrity of the Salesforce change set assembled by the Release
  Composer. Checks for destructive changes, profile-only deployments, component
  conflicts, and missing dependencies.

  Gate G8 depends on this verdict.

Output data keys consumed by downstream:
  integrity_valid              → bool   (Gate G8 — must be True)
  integrity_issues             → list   (specific problems found)
  destructive_changes_present  → bool   (informational — not a hard block)
  integrity_verdict            → str    (PASS / WARN / FAIL)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 41
AGENT_NAME = "Change Set Integrity"

_LARGE_CHANGE_SET_THRESHOLD = 20

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_integrity_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "integrity_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising change set integrity. "
                "State whether the change set is valid, any issues found, "
                "and what must be resolved before dry-run can proceed."
            ),
        },
        "integrity_concern": {
            "type": "string",
            "enum": ["none", "destructive_changes", "missing_dependencies",
                     "profile_only", "oversized_change_set"],
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce change set integrity check
in a regulated FSC Wealth Management deployment pipeline.
You will receive the change set composition, any detected issues, and the verdict.
Write a clear 2–3 sentence narrative explaining the integrity status and what the
release engineer must do if problems are found.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent8_data  = _get_agent_data(state, "8")
    agent13_data = _get_agent_data(state, "13")
    agent40_data = _get_agent_data(state, "40")

    valid, issues, destructive, verdict = _check_integrity(agent8_data, agent13_data, agent40_data)

    trace_msg = _build_trace_message(story_id, valid, issues, destructive, verdict, agent40_data)
    trace = await _generate_trace(trace_msg)

    confidence_score, signals = _compute_confidence(agent13_data, agent40_data, valid)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Change set integrity for {story_id}: valid={valid}, "
        f"{len(issues)} issue(s) — verdict={verdict}"
    )
    why = trace.get("narrative", "Change Set Integrity validated the release package.")

    data = {
        "integrity_valid": valid,
        "integrity_issues": issues,
        "destructive_changes_present": destructive,
        "integrity_verdict": verdict,
        "integrity_concern": trace.get("integrity_concern", "none"),
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


# ── Deterministic integrity check ─────────────────────────────────────────────

def _check_integrity(
    agent8_data: dict | None,
    agent13_data: dict | None,
    agent40_data: dict | None,
) -> tuple[bool, list[str], bool, str]:
    """Returns (valid, issues, destructive_changes_present, verdict)."""
    issues: list[str] = []
    # REQ-26 Gap 3: typed flags prevent verdict from breaking when issue text changes
    hard_issue_flags: set[str] = set()

    component_count  = (agent40_data or {}).get("component_count", 0)
    # effective_count: component_count (sum of type counts, REQ-25) vs file count — same unit after REQ-25
    changed_files    = (agent13_data or {}).get("changed_files_count", 0)
    composer_verdict = (agent40_data or {}).get("composer_verdict", "COMPOSED")
    missing_deps     = (agent13_data or {}).get("missing_dependencies", [])
    destructive      = (agent13_data or {}).get("has_destructive_changes", False)
    ext_deps         = (agent8_data or {}).get("has_external_dependencies", False)
    components_summary = (agent40_data or {}).get("components_summary", {})

    if composer_verdict == "FAILED":
        issues.append("Release composer failed — no valid change set to validate")
        hard_issue_flags.add("composer_failed")

    if missing_deps:
        issues.append(f"Missing dependencies detected: {missing_deps}")
        hard_issue_flags.add("missing_deps")

    effective_count = max(component_count, changed_files)
    if effective_count > _LARGE_CHANGE_SET_THRESHOLD:
        issues.append(
            f"Change set is large ({effective_count} components/files) — "
            "risk of deployment timeout; consider splitting"
        )

    # Destructive changes are a warn, not a fail — but must be noted
    if destructive:
        issues.append("Destructive changes present — manual review required before deployment")

    # REQ-26 Gap 2: external dependencies not in package → WARN
    if ext_deps and "ExternalService" not in components_summary:
        issues.append(
            "External service dependencies detected but not in change set — "
            "verify Named Credentials/Connected Apps are deployed to target org"
        )

    verdict = "FAIL" if hard_issue_flags else ("WARN" if issues else "PASS")
    valid = not hard_issue_flags

    return valid, issues, destructive, verdict


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent13_data: dict | None,
    agent40_data: dict | None,
    valid: bool,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if agent40_data:
        scorer.add("release_package_available", True, +8)
    else:
        scorer.add("no_release_package", 0, -10)

    if agent13_data:
        scorer.add("metadata_available", True, +5)

    if not valid:
        scorer.add("integrity_check_failed", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace ───────────────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a change set integrity narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    valid: bool,
    issues: list[str],
    destructive: bool,
    verdict: str,
    agent40_data: dict | None,
) -> str:
    release_name    = (agent40_data or {}).get("release_name", "unknown")
    component_count = (agent40_data or {}).get("component_count", 0)
    return (
        f"Story: {story_id}\n"
        f"Release name: {release_name}\n"
        f"Component count: {component_count}\n"
        f"Integrity valid: {valid}\n"
        f"Destructive changes: {destructive}\n"
        f"Issues: {issues or ['none']}\n"
        f"Verdict: {verdict}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
