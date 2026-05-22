"""
Agent 18 — Component Attribution Tracer
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=62)

Runs sequentially after Batch 3 (after Agent 17).
Has access to Agents 11, 13.

Purpose:
  Traces which Salesforce components (Apex classes, triggers, LWC, flows,
  objects) have changed and maps them to their owners/authors. Flags
  components with multiple simultaneous editors (merge-risk) and
  identifies FSC-regulated components touched by the story.

  Haiku generates the attribution narrative; component detection is
  pure Python path parsing.

Output data keys consumed by downstream:
  changed_components    → list (Agent 23 story-to-code tracer)
  regulated_components  → list (Gate G2 — FCA-regulated component audit)
  merge_risk_components → list (informational — developer warning)
  component_verdict     → str  (PASS / WARN / REVIEW_REQUIRED)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 18
AGENT_NAME = "Component Attribution Tracer"

# FSC regulated component name fragments (lower-cased)
_REGULATED_FRAGMENTS = frozenset({
    "suitability", "riskprofile", "appropriateness", "vulnerablecustomer",
    "consumerduty", "financialaccount", "financialholding", "financialgoal",
    "investmentaccount", "revenueschedule",
})

# Component type inferred from file extension / path segment
_COMPONENT_TYPE_MAP = {
    ".cls": "ApexClass",
    ".trigger": "ApexTrigger",
    ".js": "LWC",
    ".html": "LWC",
    ".css": "LWC",
    ".flow-meta.xml": "Flow",
    ".object-meta.xml": "CustomObject",
    ".field-meta.xml": "CustomField",
    ".permissionset-meta.xml": "PermissionSet",
    ".profile-meta.xml": "Profile",
    ".page": "VisualforcePage",
    ".component": "VisualforceComponent",
}

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_attribution_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "attribution_concern"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising which Salesforce components were changed, "
                "whether any are FCA-regulated, and whether merge risk exists. "
                "Be factual and actionable."
            ),
        },
        "attribution_concern": {
            "type": "string",
            "enum": ["none", "merge_risk", "regulated_components", "both"],
            "description": (
                "none: No concerns. "
                "merge_risk: Multiple editors on same component. "
                "regulated_components: FCA-sensitive components touched. "
                "both: Regulated components AND merge risk."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Salesforce component attribution analysis.
You will receive a list of changed components with their types, regulated status, and
author information. Write a clear 2–3 sentence narrative explaining which components
were changed, whether FCA-regulated components were touched, and whether merge risk
exists due to multiple editors. Be factual and actionable.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent11_data = _get_agent_data(state, "11")
    agent13_data = _get_agent_data(state, "13")

    changed_files = _get_changed_files(agent13_data)

    # ── Deterministic analysis ────────────────────────────────────────────────
    components, regulated, merge_risk, verdict, author_data_available = _analyse_components(changed_files)

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, components, regulated, merge_risk, verdict,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        changed_files, agent11_data, agent13_data, regulated, merge_risk, author_data_available,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Component attribution for {story_id}: {len(components)} component(s) identified, "
        f"{len(regulated)} regulated, {len(merge_risk)} merge-risk — verdict={verdict}"
    )
    why = trace.get(
        "narrative",
        "Component Attribution Tracer mapped changed files to Salesforce component types.",
    )

    data = {
        "changed_components": components,
        "regulated_components": regulated,
        "merge_risk_components": merge_risk,
        "component_verdict": verdict,
        "total_components": len(components),
        "author_data_available": author_data_available,
        "attribution_concern": trace.get("attribution_concern", "none"),
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


# ── Deterministic component analysis ─────────────────────────────────────────

def _analyse_components(
    changed_files: list[dict],
) -> tuple[list[dict], list[str], list[str], str, bool]:
    """
    Map file paths to Salesforce components. Returns:
    (components, regulated_names, merge_risk_names, verdict, author_data_available).
    """
    if not changed_files:
        return [], [], [], "PASS", False

    components: list[dict] = []
    component_to_authors: dict[str, list[str]] = {}
    files_with_author = 0

    for f in changed_files:
        path = f.get("file_path", "")
        author = f.get("author_email", "")
        if author:
            files_with_author += 1
        name = _extract_component_name(path)
        ctype = _infer_component_type(path)

        component = {
            "name": name,
            "type": ctype,
            "file_path": path,
            "author_email": author,
            "is_regulated": _is_regulated(name),
        }
        components.append(component)

        if name:
            if name not in component_to_authors:
                component_to_authors[name] = []
            if author and author not in component_to_authors[name]:
                component_to_authors[name].append(author)

    author_data_available = files_with_author >= 1
    regulated = [c["name"] for c in components if c["is_regulated"] and c["name"]]
    merge_risk = [
        name for name, authors in component_to_authors.items() if len(authors) > 1
    ]

    flags: list[str] = []
    if not author_data_available:
        flags.append(
            "Author data unavailable from Copado — merge risk detection not possible for this story"
        )

    if regulated and merge_risk:
        verdict = "REVIEW_REQUIRED"
    elif regulated:
        verdict = "WARN"
    elif merge_risk:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return components, list(set(regulated)), merge_risk, verdict, author_data_available


def _extract_component_name(path: str) -> str:
    if not path:
        return ""
    filename = path.split("/")[-1].split("\\")[-1]
    # Strip known meta suffixes
    for suffix in (".cls", ".trigger", ".js", ".html", ".css",
                   ".object-meta.xml", ".field-meta.xml",
                   ".flow-meta.xml", ".permissionset-meta.xml",
                   ".profile-meta.xml", ".page", ".component"):
        if filename.lower().endswith(suffix):
            filename = filename[: -len(suffix)]
            break
    return filename


def _infer_component_type(path: str) -> str:
    lower = path.lower()
    for ext, ctype in _COMPONENT_TYPE_MAP.items():
        if lower.endswith(ext):
            return ctype
    if "/lwc/" in lower:
        return "LWC"
    if "/aura/" in lower:
        return "AuraComponent"
    if "/flows/" in lower:
        return "Flow"
    return "Unknown"


def _is_regulated(name: str) -> bool:
    lower = name.lower()
    return any(frag in lower for frag in _REGULATED_FRAGMENTS)


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    changed_files: list[dict],
    agent11_data: dict | None,
    agent13_data: dict | None,
    regulated: list[str],
    merge_risk: list[str],
    author_data_available: bool = True,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    if changed_files:
        scorer.add("files_available", len(changed_files), +8)
    else:
        scorer.add("no_files_available", 0, -10)

    if agent11_data and agent11_data.get("branch_found"):
        scorer.add("branch_context_available", True, +5)

    if agent13_data:
        scorer.add("metadata_scope_available", True, +7)
    else:
        scorer.add("no_metadata_scope", 0, -5)

    if regulated:
        scorer.add("regulated_components_detected", len(regulated), -5)

    if merge_risk:
        scorer.add("merge_risk_detected", len(merge_risk), -5)

    if changed_files and not author_data_available:
        scorer.add("author_data_unavailable", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a component attribution narrative.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    components: list[dict],
    regulated: list[str],
    merge_risk: list[str],
    verdict: str,
) -> str:
    comp_summary = [
        f"  - {c['name']} ({c['type']}) — regulated={c['is_regulated']}"
        for c in components[:10]
    ]
    return (
        f"Story: {story_id}\n"
        f"Total components: {len(components)}\n"
        f"Changed components:\n" + "\n".join(comp_summary or ["  none"]) + "\n"
        f"Regulated components: {regulated or ['none']}\n"
        f"Merge-risk components: {merge_risk or ['none']}\n"
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
