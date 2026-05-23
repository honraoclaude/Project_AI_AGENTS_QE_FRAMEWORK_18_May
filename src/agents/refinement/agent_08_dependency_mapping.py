"""
Agent 8 — Dependency Mapping
Phase       : Refinement
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (high base — deterministic analysis)

Runs in Batch 1 (parallel with Agent 1) — no upstream agent data available.

Purpose:
  Deterministically scans the story text for known FSC object names, then
  applies a hardcoded FSC dependency graph to map implied parent records.
  Haiku generates the explainability narrative — the analysis itself is
  pure Python rule evaluation (no LLM needed for correctness).

  Example: story mentions Suitability__c → dependency map implies:
    RiskProfile__c → FinancialAccount → Individual (Account)

  Output is consumed by Agent 9 (Risk Anticipation), which uses dependency
  depth and cross-phase dependencies to compute deployment risk.

Output data keys consumed by downstream:
  detected_objects    → list (Agent 9 — risk scope)
  dependency_graph    → dict (Agent 9 — full dependency tree)
  implied_objects     → list (Agent 9 — additional objects tests must cover)
  dependency_depth    → int  (Agent 9 — risk proxy: deeper = higher risk)
  cross_object_count  → int  (Agent 6 Test Design — integration test scope)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 8
AGENT_NAME = "Dependency Mapping"

# ── Hardcoded FSC dependency graph (deterministic) ────────────────────────────
# Maps each object to the list of parent/related objects it requires.
# Used to compute implied (transitively required) objects not stated in the story.

_FSC_DEPENDENCY_MAP: dict[str, list[str]] = {
    "suitability__c":               ["riskprofile__c", "financialaccount", "individual"],
    "suitabilityassessment":        ["riskprofile__c", "financialaccount", "individual"],
    "riskprofile__c":               ["financialaccount", "individual"],
    "appropriateness__c":           ["financialaccount", "individual"],
    "vulnerablecustomerindicator__c": ["individual"],
    "financialaccount":             ["household", "individual"],
    "financialgoal":                ["financialaccount", "individual"],
    "goal__c":                      ["financialaccount", "individual"],
    "financialholding":             ["financialaccount"],
    "assetsandliabilities":         ["financialaccount", "individual"],
    "revenue__c":                   ["financialaccount"],
    "financialaccounttransaction":  ["financialaccount"],
    "individualpplication":         ["individual"],
    "household":                    [],
    "individual":                   [],
}

# Object aliases → canonical name (for matching display names in story text)
_OBJECT_ALIASES: dict[str, str] = {
    "suitability":                  "suitability__c",
    "suitabilityassessment":        "suitability__c",
    "risk profile":                 "riskprofile__c",
    "riskprofile":                  "riskprofile__c",
    "appropriateness":              "appropriateness__c",
    "vulnerable customer":          "vulnerablecustomerindicator__c",
    "vulnerablecustomer":           "vulnerablecustomerindicator__c",
    "vulnerablecustomerindicator":  "vulnerablecustomerindicator__c",
    "financial account":            "financialaccount",
    "financialaccount":             "financialaccount",
    "financial goal":               "financialgoal",
    "goal":                         "goal__c",
    "financial holding":            "financialholding",
    "financialholding":             "financialholding",
    "assets and liabilities":       "assetsandliabilities",
    "assetsandliabilities":         "assetsandliabilities",
    "aum":                          "financialaccount",
    "revenue":                      "revenue__c",
    "household":                    "household",
    "individual":                   "individual",
    "client":                       "individual",
}

# Integration pattern keywords for platform events and external services
# Detected from story text — these are not FSC object-model dependencies but async/external ones.
_PLATFORM_EVENT_KEYWORDS: frozenset[str] = frozenset({
    "platform event",
    "platformevent",
    "publish event",
    "streaming api",
    "eventbus",
    "event bus",
    "publish subscribe",
    "pub/sub",
})

_EXTERNAL_SERVICE_KEYWORDS: frozenset[str] = frozenset({
    "named credential",
    "namedcredential",
    "connected app",
    "connectedapp",
    "http callout",
    "httpcallout",
    "rest callout",
    "soap callout",
    "external data source",
    "externaldatasource",
    "external service",
    "externalservice",
    "api callout",
    "callout",
    "aum provider",
    "financial data feed",
    "data feed",
})

# Haiku tool for narrative generation only
_TRACE_TOOL_NAME = "generate_dependency_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "dependency_complexity"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining what FSC objects were detected, what "
                "parent records are implied, and the deployment complexity this creates."
            ),
        },
        "dependency_complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low: 1 object, no parents. "
                "medium: 2–3 objects, 1 dependency level. "
                "high: 4+ objects or 2+ dependency levels (deep chains)."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for an automated FSC dependency analysis.
You will receive a list of detected FSC objects, their implied parent records, and
the dependency depth. Write a clear 2–3 sentence narrative explaining what was found
and why it matters for test design and deployment risk. Be factual and concise.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    story = await get_story(story_id)

    # ── Deterministic analysis ────────────────────────────────────────────────
    detected, implied, graph, depth, has_external_deps, dep_types = _analyse_dependencies(story)
    cross_object_count = len(detected) + len(implied)

    # ── Haiku trace generation ────────────────────────────────────────────────
    trace_message = _build_trace_message(story, detected, implied, depth, has_external_deps, dep_types)
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(story, detected, implied, depth)
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Dependency map for {story_id}: detected={detected}, "
        f"implied={implied}, depth={depth}"
        + (f", external_deps={dep_types}" if has_external_deps else "")
    )
    why = trace.get("narrative", "Dependency mapping applied FSC object rules deterministically.")

    story_text = (
        (story.get("description") or "") + " " + (story.get("summary") or "")
    ).lower()
    has_destructive_changes = any(
        kw in story_text for kw in ("delete", "remove field", "drop field", "deprecate", "destroy")
    )

    data = {
        "detected_objects": detected,
        "implied_objects": implied,
        "dependency_graph": graph,
        "dependency_depth": depth,
        "cross_object_count": cross_object_count,
        "dependency_complexity": trace.get("dependency_complexity", "low"),
        "has_external_dependencies": has_external_deps,
        "has_destructive_changes": has_destructive_changes,
        "detected_dependency_types": dep_types,
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


# ── Deterministic dependency analysis ────────────────────────────────────────

def _analyse_dependencies(story: dict) -> tuple[list[str], list[str], dict, int, bool, list[str]]:
    """
    Scan story text → detect FSC objects → map implied parents.
    Also detects platform event and external service integration patterns.
    Returns (detected, implied, graph, max_depth, has_external_deps, dep_types).
    Pure Python — no LLM involved.
    """
    text = (
        (story.get("description") or "") + " " +
        (story.get("summary") or "") + " " +
        " ".join(story.get("components", []))
    ).lower()

    # Detect objects mentioned in story text
    detected: set[str] = set()
    for alias, canonical in _OBJECT_ALIASES.items():
        if alias in text and canonical in _FSC_DEPENDENCY_MAP:
            detected.add(canonical)

    # Build dependency graph — BFS from detected objects
    graph: dict[str, list[str]] = {}
    visited: set[str] = set()
    queue = list(detected)
    depth = 0

    while queue:
        next_queue: list[str] = []
        for obj in queue:
            if obj in visited:
                continue
            visited.add(obj)
            parents = _FSC_DEPENDENCY_MAP.get(obj, [])
            graph[obj] = parents
            for p in parents:
                if p not in visited:
                    next_queue.append(p)
        if next_queue:
            depth += 1
        queue = next_queue

    implied = sorted(visited - detected - {"household", "individual"})
    detected_list = sorted(detected)

    # Detect integration patterns (platform events + external services)
    dep_types: list[str] = []
    if any(kw in text for kw in _PLATFORM_EVENT_KEYWORDS):
        dep_types.append("platform_event")
    if any(kw in text for kw in _EXTERNAL_SERVICE_KEYWORDS):
        dep_types.append("external_service")
    has_external_deps = bool(dep_types)

    return detected_list, implied, graph, depth, has_external_deps, dep_types


# ── Confidence scoring (Tier B, high base) ────────────────────────────────────

def _compute_confidence(
    story: dict,
    detected: list[str],
    implied: list[str],
    depth: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=72)   # High base — analysis is deterministic

    # Signal 1: number of detected objects → quality of scan
    if len(detected) >= 2:
        scorer.add("detected_objects_rich", len(detected), +10)
    elif len(detected) == 1:
        scorer.add("detected_objects_single", 1, +5)
    else:
        scorer.add("no_objects_detected", 0, -15)  # very uncertain if nothing found

    # Signal 2: description richness → quality of text scan
    word_count = len((story.get("description") or "").split())
    if word_count >= 100:
        scorer.add("description_rich", word_count, +5)
    elif word_count < 20:
        scorer.add("description_sparse", word_count, -10)

    # Signal 3: dependency chain found → confirms non-trivial analysis
    if depth >= 2:
        scorer.add("deep_dependency_chain", depth, +5)
    elif depth == 1 and implied:
        scorer.add("dependency_chain_found", depth, +3)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku narrative generation ────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an explainability narrative for a dependency analysis.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story: dict,
    detected: list[str],
    implied: list[str],
    depth: int,
    has_external_deps: bool = False,
    dep_types: list[str] | None = None,
) -> str:
    ext_line = (
        f"Integration patterns detected: {dep_types}\n"
        if has_external_deps else ""
    )
    return (
        f"Story: {story['story_id']} — {story['summary']}\n\n"
        f"Detected FSC objects: {detected or ['none']}\n"
        f"Implied parent objects: {implied or ['none']}\n"
        f"Dependency depth: {depth}\n"
        f"{ext_line}\n"
        f"Generate a 2–3 sentence narrative explaining these findings using the "
        f"generate_dependency_narrative tool."
    )


