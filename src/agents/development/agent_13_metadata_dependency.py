"""
Agent 13 — Metadata Dependency Mapper
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs in Development Batch 1 (parallel with Agent 10, Agent 11).
Has access to Agent 8 (Refinement Dependency Map) and Agent 11 (Branch Tracer).

Purpose:
  Deterministically scans the list of changed metadata files (from Copado) for
  FSC object names, then applies the same hardcoded FSC dependency graph as Agent 8
  to map implied parent records — but operating on actual code diffs rather than
  story text.

  Additionally compares the detected scope against Agent 8's refinement prediction
  to surface scope creep (objects in code not anticipated during Refinement).

  If Copado is not configured, changed_files will be empty and the agent reports
  no metadata detected — lowers confidence but does not block G2.

  Haiku generates the narrative — analysis is pure Python.

Output data keys consumed by downstream:
  detected_objects      → list (Gate G2 — verify coverage matches story scope)
  implied_objects       → list (Agent 12 Coverage Analyser — test class targets)
  dependency_depth      → int  (Gate G2 — risk proxy)
  scope_delta_objects   → list (Agent 23 Tracer — scope creep audit)
  scope_matches_refinement → bool (Gate G2 — regression check vs Agent 8)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.copado import get_branch_for_story, get_changed_files

# Reuse FSC object data from Agent 8 (single source of truth)
from src.agents.refinement.agent_08_dependency_mapping import (
    _FSC_DEPENDENCY_MAP,
    _OBJECT_ALIASES,
)

AGENT_ID = 13
AGENT_NAME = "Metadata Dependency Mapper"

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_metadata_dependency_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "dependency_complexity"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences explaining which FSC objects were detected in the changed "
                "metadata, what parent records are implied, and whether the development "
                "scope matches the Refinement prediction."
            ),
        },
        "dependency_complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low: 1 object, no parents. "
                "medium: 2–3 objects, 1 dependency level. "
                "high: 4+ objects or 2+ dependency levels."
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an explainability trace for a Metadata Dependency analysis during
the Development phase. You will receive a list of FSC objects detected in changed
metadata files, their implied parent records, and a comparison against the Refinement
phase prediction. Write a clear 2–3 sentence narrative explaining what was found and
whether the development scope is consistent with the story's stated intent.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent8_data = _get_agent_data(state, "8")
    agent11_data = _get_agent_data(state, "11")

    # Use branch from Agent 11 if available, else ask Copado directly
    branch_name = (agent11_data or {}).get("branch_name", "")
    if not branch_name:
        branch_info = await get_branch_for_story(story_id)
        branch_name = branch_info.get("branch_name", "")

    changed_files = await get_changed_files(story_id, branch_name)

    # ── Deterministic analysis ────────────────────────────────────────────────
    detected, implied, graph, depth, scope_delta = _analyse_metadata(
        changed_files, agent8_data,
    )

    # ── Haiku trace generation ────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, changed_files, detected, implied, depth, scope_delta, agent8_data,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        changed_files, detected, implied, depth, scope_delta, agent8_data,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    scope_matches = len(scope_delta) == 0

    what = (
        f"Metadata dependency map for {story_id}: "
        f"{len(changed_files)} file(s) changed, detected={detected}, depth={depth}"
    )
    why = trace.get(
        "narrative",
        "Metadata Dependency Mapper scanned changed files for FSC object dependencies.",
    )

    data = {
        "changed_files_count": len(changed_files),
        "detected_objects": detected,
        "implied_objects": implied,
        "dependency_graph": graph,
        "dependency_depth": depth,
        "scope_delta_objects": scope_delta,
        "scope_matches_refinement": scope_matches,
        "dependency_complexity": trace.get("dependency_complexity", "low"),
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


# ── Deterministic metadata analysis ──────────────────────────────────────────

def _analyse_metadata(
    changed_files: list[dict],
    agent8_data: dict | None,
) -> tuple[list[str], list[str], dict, int, list[str]]:
    """
    Scan changed file paths and component names for FSC objects.
    Run BFS to map implied parents. Compare scope against Agent 8's prediction.
    Returns (detected, implied, graph, depth, scope_delta).
    Pure Python — no LLM involved.
    """
    detected_raw = _detect_objects_from_files(changed_files)
    detected_list, implied_list, graph, depth = _run_dependency_bfs(detected_raw)

    # Scope delta: objects in development not anticipated during Refinement (Agent 8)
    refinement_objects: set[str] = set()
    if agent8_data:
        refinement_objects.update(agent8_data.get("detected_objects", []))
        refinement_objects.update(agent8_data.get("implied_objects", []))

    if refinement_objects:
        all_dev_objects = set(detected_list) | set(implied_list)
        scope_delta = sorted(all_dev_objects - refinement_objects - {"household", "individual"})
    else:
        scope_delta = []

    return detected_list, implied_list, graph, depth, scope_delta


def _detect_objects_from_files(changed_files: list[dict]) -> set[str]:
    """Scan file paths and component names for FSC object alias matches."""
    detected: set[str] = set()
    for file in changed_files:
        text = (
            file.get("file_path", "") + " " +
            file.get("object_name", "") + " " +
            file.get("object_type", "")
        ).lower()
        for alias, canonical in _OBJECT_ALIASES.items():
            if alias in text and canonical in _FSC_DEPENDENCY_MAP:
                detected.add(canonical)
    return detected


def _run_dependency_bfs(
    detected: set[str],
) -> tuple[list[str], list[str], dict[str, list[str]], int]:
    """BFS from detected objects to map implied parents. Mirrors Agent 8's algorithm."""
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
    return sorted(detected), implied, graph, depth


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    changed_files: list[dict],
    detected: list[str],
    implied: list[str],
    depth: int,
    scope_delta: list[str],
    agent8_data: dict | None,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    # Signal 1: changed files available — without these, analysis is blind
    if len(changed_files) >= 1:
        scorer.add("changed_files_present", len(changed_files), +8)
    else:
        scorer.add("no_changed_files", 0, -15)

    # Signal 2: FSC objects detected from code
    if len(detected) >= 2:
        scorer.add("detected_objects_rich", len(detected), +8)
    elif len(detected) == 1:
        scorer.add("detected_objects_single", 1, +4)
    elif changed_files:
        scorer.add("no_fsc_objects_in_changed_files", 0, -8)

    # Signal 3: refinement baseline available → scope comparison grounded
    if agent8_data:
        scorer.add("refinement_baseline_available", True, +5)
        if not scope_delta:
            scorer.add("scope_matches_refinement", True, +5)
        else:
            scorer.add("scope_delta_detected", len(scope_delta), -5)
    else:
        scorer.add("no_refinement_baseline", True, -5)

    # Signal 4: dependency chain depth
    if depth >= 2:
        scorer.add("deep_dependency_chain", depth, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate a metadata dependency narrative for the Development phase.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=300,
    )


def _build_trace_message(
    story_id: str,
    changed_files: list[dict],
    detected: list[str],
    implied: list[str],
    depth: int,
    scope_delta: list[str],
    agent8_data: dict | None,
) -> str:
    refinement_detected = (agent8_data or {}).get("detected_objects", [])
    return (
        f"Story: {story_id}\n\n"
        f"Changed files: {len(changed_files)}\n"
        f"FSC objects detected in code: {detected or ['none']}\n"
        f"Implied parent objects: {implied or ['none']}\n"
        f"Dependency depth: {depth}\n"
        f"Objects in refinement prediction: {refinement_detected or ['none']}\n"
        f"Scope delta (new objects not in refinement): {scope_delta or ['none']}\n\n"
        f"Generate a 2–3 sentence narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
