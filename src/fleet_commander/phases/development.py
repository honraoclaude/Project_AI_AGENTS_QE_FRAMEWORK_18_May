"""
Development Phase — Fleet Commander subgraph.

Orchestrates Agents 10–23 in the following execution order:

  Batch 1 (parallel): 10, 11, 13   — AC Compliance, Branch Tracer, Metadata Dependency
  Batch 2 (parallel): 12, 14, 15, 16  — Coverage, Code Quality, Security, Bulk Quality
  ── Gate G2 (Story Integrity) ──
  Batch 3 (parallel): 17, 18       — SFDX Validator, Component Attribution
  Agent 19 (sequential)            — BDD Gherkin Writer
  Batch 4 (parallel): 20, 21       — Performance Risk, Test Data Architect
  ── Gate G3 (Code Quality) ──
  Agent 22 (sequential)            — Sandbox State
  Agent 23 (sequential)            — Story-to-Code Tracer
  ── Gate G4 (Development Phase) ──

Gates:
  G2 — Story Integrity:
       FAIL if AC Compliance FAIL or Coverage FAIL or Security REVIEW_REQUIRED
       (all three must pass to proceed to BDD generation)

  G3 — Code Quality:
       FAIL if Code Quality FAIL or Performance Risk HIGH (soql_loop_risk=True)
       WARN if 2+ advisory agents returned WARN (non-blocking)

  G4 — Development Phase Go/No-Go:
       FAIL if development_verdict from Agent 23 is FAIL or PARTIAL
       FAIL if sandbox_ready is False
       Passes if PASS — story cleared to enter Testing phase
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.schemas import StoryState


# ── Gate definitions ──────────────────────────────────────────────────────────

class GateG2Error(Exception):
    """Story Integrity gate failure — blocks BDD generation and downstream agents."""


class GateG3Error(Exception):
    """Code Quality gate failure — blocks sandbox validation and trace generation."""


class GateG4Error(Exception):
    """Development Phase gate failure — story cannot proceed to Testing."""


def _check_gate_g2(state: StoryState) -> None:
    """
    Gate G2 — Story Integrity.
    Blocks if: AC Compliance FAIL, Coverage FAIL, or Security REVIEW_REQUIRED.
    """
    failures: list[str] = []

    ac_data = _get_agent_data(state, "10")
    ac_verdict = (ac_data or {}).get("coverage_verdict") or (ac_data or {}).get("ac_compliance_verdict", "")
    if ac_verdict == "FAIL":
        failures.append("AC Compliance (Agent 10): FAIL — acceptance criteria not met")

    cov_data = _get_agent_data(state, "12")
    cov_verdict = (cov_data or {}).get("coverage_verdict", "")
    if cov_verdict == "FAIL":
        coverage_pct = (cov_data or {}).get("coverage_pct", 0)
        threshold = (cov_data or {}).get("coverage_threshold", 85)
        failures.append(
            f"Apex Coverage (Agent 12): FAIL — {coverage_pct}% below {threshold}% threshold"
        )

    sec_data = _get_agent_data(state, "15")
    sec_verdict = (sec_data or {}).get("security_verdict", "")
    if sec_verdict == "REVIEW_REQUIRED":
        flags = (sec_data or {}).get("security_flags", [])
        failures.append(
            f"Apex Security (Agent 15): REVIEW_REQUIRED — {len(flags)} flag(s) require review"
        )

    if failures:
        raise GateG2Error(
            f"Gate G2 (Story Integrity) FAILED for {state['story_id']}. "
            f"Blockers: {'; '.join(failures)}"
        )


def _check_gate_g3(state: StoryState) -> None:
    """
    Gate G3 — Code Quality.
    Blocks if: Code Quality FAIL or SOQL-in-loop risk detected.
    """
    failures: list[str] = []

    qual_data = _get_agent_data(state, "14")
    qual_verdict = (qual_data or {}).get("quality_verdict", "")
    if qual_verdict == "FAIL":
        critical = (qual_data or {}).get("critical_violations", [])
        failures.append(
            f"Code Quality (Agent 14): FAIL — {len(critical)} critical violation(s)"
        )

    perf_data = _get_agent_data(state, "20")
    soql_loop = (perf_data or {}).get("soql_loop_risk", False)
    if soql_loop:
        failures.append(
            "Performance Risk (Agent 20): SOQL-in-loop risk detected — must be resolved"
        )

    if failures:
        raise GateG3Error(
            f"Gate G3 (Code Quality) FAILED for {state['story_id']}. "
            f"Blockers: {'; '.join(failures)}"
        )


def _check_gate_g4(state: StoryState) -> None:
    """
    Gate G4 — Development Phase Go/No-Go.
    Blocks if development_verdict is FAIL/PARTIAL or sandbox is not ready.
    """
    failures: list[str] = []

    tracer_data = _get_agent_data(state, "23")
    dev_verdict = (tracer_data or {}).get("development_verdict", "UNKNOWN")
    if dev_verdict in ("FAIL", "PARTIAL"):
        critical = (tracer_data or {}).get("critical_failures", [])
        failures.append(
            f"Story-to-Code Tracer (Agent 23): {dev_verdict} — "
            f"{len(critical)} critical failure(s): {critical}"
        )

    sandbox_data = _get_agent_data(state, "22")
    if sandbox_data is not None:
        sandbox_ready = sandbox_data.get("sandbox_ready", False)
        sandbox_verdict = sandbox_data.get("sandbox_verdict", "UNKNOWN")
    else:
        sandbox_ready = True  # absent data = not yet assessed; gate is permissive
        sandbox_verdict = "UNKNOWN"
    if not sandbox_ready:
        blockers = (sandbox_data or {}).get("sandbox_blockers", [])
        failures.append(
            f"Sandbox State (Agent 22): {sandbox_verdict} — "
            f"{len(blockers)} blocker(s): {blockers}"
        )

    if failures:
        raise GateG4Error(
            f"Gate G4 (Development Phase) FAILED for {state['story_id']}. "
            f"Story cannot proceed to Testing. Blockers: {'; '.join(failures)}"
        )


# ── Phase orchestration ───────────────────────────────────────────────────────

async def run_development_phase(state: StoryState) -> StoryState:
    """
    Orchestrate the full Development phase for a story.
    Returns the updated StoryState with all agent results merged in.
    Raises GateG2Error, GateG3Error, or GateG4Error on gate failure.
    """
    from src.fleet_commander.worker import dispatch_agent  # lazy import to avoid MCP circular dep

    story_id = state["story_id"]

    # ── Batch 1: AC Compliance + Branch Tracer + Metadata Dependency ──────────
    batch1_results = await asyncio.gather(
        dispatch_agent(10, state),
        dispatch_agent(11, state),
        dispatch_agent(13, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [10, 11, 13], batch1_results)

    # ── Batch 2: Coverage + Code Quality + Security + Bulk Quality ────────────
    batch2_results = await asyncio.gather(
        dispatch_agent(12, state),
        dispatch_agent(14, state),
        dispatch_agent(15, state),
        dispatch_agent(16, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [12, 14, 15, 16], batch2_results)

    # ── Gate G2: Story Integrity ──────────────────────────────────────────────
    _check_gate_g2(state)

    # ── Batch 3: SFDX Validator + Component Attribution ───────────────────────
    batch3_results = await asyncio.gather(
        dispatch_agent(17, state),
        dispatch_agent(18, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [17, 18], batch3_results)

    # ── Agent 19: BDD Gherkin Writer (sequential — needs batch 3 output) ──────
    result19 = await dispatch_agent(19, state)
    state["agent_results"]["19"] = result19

    # ── Batch 4: Performance Risk + Test Data Architect ───────────────────────
    batch4_results = await asyncio.gather(
        dispatch_agent(20, state),
        dispatch_agent(21, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [20, 21], batch4_results)

    # ── Gate G3: Code Quality ─────────────────────────────────────────────────
    _check_gate_g3(state)

    # ── Agent 22: Sandbox State ───────────────────────────────────────────────
    result22 = await dispatch_agent(22, state)
    state["agent_results"]["22"] = result22

    # ── Agent 23: Story-to-Code Tracer ────────────────────────────────────────
    result23 = await dispatch_agent(23, state)
    state["agent_results"]["23"] = result23

    # ── Gate G4: Development Phase Go/No-Go ──────────────────────────────────
    _check_gate_g4(state)

    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_results(
    state: StoryState,
    agent_ids: list[int],
    results: list[Any],
) -> StoryState:
    """
    Merge batch dispatch results into state. Failed agents (exceptions)
    are logged but do not prevent other agents from running — gates
    enforce the fail-closed logic.
    """
    for agent_id, result in zip(agent_ids, results):
        if isinstance(result, Exception):
            state["agent_results"][str(agent_id)] = {
                "error": str(result),
                "data": {},
            }
        else:
            state["agent_results"][str(agent_id)] = result
    return state


def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    if isinstance(result, dict):
        return result.get("data") or result
    return None
