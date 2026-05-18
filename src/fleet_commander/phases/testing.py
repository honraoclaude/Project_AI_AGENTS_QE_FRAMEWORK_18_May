"""
Testing Phase — Fleet Commander subgraph.

Orchestrates Agents 24–38 in the following execution order:

  Batch 1 (parallel): 24, 25, 32   — Test Strategy Validator, Env Provisioner, Regression Risk
  Batch 2 (parallel): 26, 29, 30   — CRT Scenario Designer, UAT Generator, FCA Scenarios
  Agent 27 (sequential)            — CRT Execution (needs 25, 26)
  Batch 3 (parallel): 28, 31, 37   — Self-Heal Reviewer, Financial Integrity, Performance Test
  Batch 4 (parallel): 33, 34, 38   — Coverage Analyser, Defect Triage, Flaky Test Hunter
  ── Gate G5 (Test Quality) ──
  Agent 35 (sequential)            — Root Cause Analyser
  Agent 36 (sequential)            — UAT Coordination
  ── Gate G6 (Testing Phase) ──

Gates:
  G5 — Test Quality:
       FAIL if coverage_verdict is FAIL (below FCA classification threshold)
       FAIL if defect_verdict is FAIL (critical defects found)
       WARN if flaky_verdict is QUARANTINE_REQUIRED (non-blocking, informational)

  G6 — Testing Phase Go/No-Go:
       FAIL if rca_verdict is INCOMPLETE (unresolved root causes)
       FAIL if uat_coordination_verdict is BLOCKED (sign-off blocked by defects)
       PENDING if uat_sign_off_required but not received (story awaits CO approval)
       PASS if NOT_REQUIRED or SIGNED_OFF — story cleared to enter Release phase
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.schemas import StoryState


# ── Gate definitions ──────────────────────────────────────────────────────────

class GateG5Error(Exception):
    """Test Quality gate failure — critical defects or coverage below threshold."""


class GateG6Error(Exception):
    """Testing Phase gate failure — story cannot proceed to Release."""


def _check_gate_g5(state: StoryState) -> None:
    """
    Gate G5 — Test Quality.
    Blocks if: coverage below FCA threshold, or critical defects found.
    """
    failures: list[str] = []

    cov_data = _get_agent_data(state, "33")
    cov_verdict = (cov_data or {}).get("coverage_verdict", "")
    if cov_verdict == "FAIL":
        overall_pct = (cov_data or {}).get("overall_coverage_pct", 0.0)
        failures.append(
            f"Coverage Analyser (Agent 33): FAIL — {overall_pct:.0f}% below FCA threshold"
        )

    def_data = _get_agent_data(state, "34")
    def_verdict = (def_data or {}).get("defect_verdict", "")
    if def_verdict == "FAIL":
        critical = (def_data or {}).get("critical_defects", [])
        failures.append(
            f"Defect Triage (Agent 34): FAIL — {len(critical)} critical defect(s): {critical}"
        )

    if failures:
        raise GateG5Error(
            f"Gate G5 (Test Quality) FAILED for {state['story_id']}. "
            f"Blockers: {'; '.join(failures)}"
        )


def _check_gate_g6(state: StoryState) -> None:
    """
    Gate G6 — Testing Phase Go/No-Go.
    Blocks if RCA is incomplete or UAT sign-off is blocked.
    PENDING is not a hard block — story waits for async CO approval.
    """
    failures: list[str] = []

    rca_data = _get_agent_data(state, "35")
    rca_verdict = (rca_data or {}).get("rca_verdict", "")
    if rca_verdict == "INCOMPLETE":
        causes = (rca_data or {}).get("root_causes", [])
        failures.append(
            f"Root Cause Analyser (Agent 35): INCOMPLETE — "
            f"{len(causes)} unresolved root cause(s)"
        )

    uat_data = _get_agent_data(state, "36")
    uat_verdict = (uat_data or {}).get("uat_coordination_verdict", "")
    if uat_verdict == "BLOCKED":
        failures.append(
            "UAT Coordination (Agent 36): BLOCKED — "
            "critical defects must be resolved before CO sign-off can proceed"
        )

    if failures:
        raise GateG6Error(
            f"Gate G6 (Testing Phase) FAILED for {state['story_id']}. "
            f"Story cannot proceed to Release. Blockers: {'; '.join(failures)}"
        )


# ── Phase orchestration ───────────────────────────────────────────────────────

async def run_testing_phase(state: StoryState) -> StoryState:
    """
    Orchestrate the full Testing phase for a story.
    Returns the updated StoryState with all agent results merged in.
    Raises GateG5Error or GateG6Error on gate failure.
    """
    from src.fleet_commander.worker import dispatch_agent  # lazy import to avoid MCP circular dep

    # ── Batch 1: Strategy Validator + Env Provisioner + Regression Risk ────────
    batch1_results = await asyncio.gather(
        dispatch_agent(24, state),
        dispatch_agent(25, state),
        dispatch_agent(32, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [24, 25, 32], batch1_results)

    # ── Batch 2: CRT Designer + UAT Generator + FCA Scenarios ─────────────────
    batch2_results = await asyncio.gather(
        dispatch_agent(26, state),
        dispatch_agent(29, state),
        dispatch_agent(30, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [26, 29, 30], batch2_results)

    # ── Agent 27: CRT Execution (sequential — needs 25 env + 26 test cases) ───
    result27 = await dispatch_agent(27, state)
    state["agent_results"]["27"] = result27

    # ── Batch 3: Self-Heal + Financial Integrity + Performance ────────────────
    batch3_results = await asyncio.gather(
        dispatch_agent(28, state),
        dispatch_agent(31, state),
        dispatch_agent(37, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [28, 31, 37], batch3_results)

    # ── Batch 4: Coverage Analyser + Defect Triage + Flaky Test Hunter ────────
    batch4_results = await asyncio.gather(
        dispatch_agent(33, state),
        dispatch_agent(34, state),
        dispatch_agent(38, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [33, 34, 38], batch4_results)

    # ── Gate G5: Test Quality ─────────────────────────────────────────────────
    _check_gate_g5(state)

    # ── Agent 35: Root Cause Analyser (sequential — needs G5 to pass) ─────────
    result35 = await dispatch_agent(35, state)
    state["agent_results"]["35"] = result35

    # ── Agent 36: UAT Coordination (sequential — needs 35 RCA complete) ───────
    result36 = await dispatch_agent(36, state)
    state["agent_results"]["36"] = result36

    # ── Gate G6: Testing Phase Go/No-Go ──────────────────────────────────────
    _check_gate_g6(state)

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
