"""
Release Phase — Fleet Commander subgraph.

Orchestrates Agents 39–50 in the following execution order:

  Batch 1 (parallel): 39, 47   — Release Readiness, Release Notes Writer
  ── Gate G7  (Release Readiness) ──
  Agent 40 (sequential)         — Release Composer
  Agent 41 (sequential)         — Change Set Integrity
  ── Gate G8  (Change Set) ──
  Agent 42 (sequential)         — Dry Run
  Agent 43 (sequential)         — Smoke on Staging
  ── Gate G9  (Staging) ──
  Agent 44 (sequential)         — FCA Evidence Pack
  ── Gate G10 (FCA Evidence) ──
  Agent 45 (sequential)         — Go/No-Go Coordinator
  ── Gate G11 (Go/No-Go) ──
  Agent 46 (sequential)         — Production Validation
  ── Gate G12 (Production) ──
  Final Batch (parallel): 48, 49, 50  — Rollback Readiness, Post-Release Monitor, Retrospective

Gates:
  G7  — Release Readiness:
        FAIL if readiness_verdict is BLOCKED (unresolved phase blockers)
        PARTIAL (UAT pending) is NOT a hard block

  G8  — Change Set:
        FAIL if composer_verdict is FAILED (no valid change set produced)
        FAIL if integrity_verdict is FAIL (missing dependencies or validation error)
        WARN (large/destructive) is non-blocking — handled within Agent 41

  G9  — Staging:
        FAIL if dry_run_verdict is FAIL (deployment simulation failed)
        FAIL if smoke_verdict is FAIL (critical flows broken on staging)
        SKIPPED is NOT a failure

  G10 — FCA Evidence:
        FAIL if evidence_verdict is MISSING (mandatory FCA rules unattested)
        PARTIAL is allowed — Compliance Officer handles remaining gaps post-release

  G11 — Go/No-Go:
        FAIL if coordinator_verdict is NO_GO (hard release blocker)
        CONDITIONAL is allowed — Compliance Officer conditions handled externally

  G12 — Production Validation:
        Informational in v1 — no hard failures; Phase 2 will enforce health checks
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.schemas import StoryState


# ── Gate definitions ──────────────────────────────────────────────────────────

class GateG7Error(Exception):
    """Release Readiness gate failure — unresolved phase blockers."""


class GateG8Error(Exception):
    """Change Set gate failure — invalid or unresolvable change set."""


class GateG9Error(Exception):
    """Staging gate failure — deployment simulation or smoke tests failed."""


class GateG10Error(Exception):
    """FCA Evidence gate failure — mandatory regulatory attestations missing."""


class GateG11Error(Exception):
    """Go/No-Go gate failure — release explicitly blocked by coordinator."""


def _check_gate_g7(state: StoryState) -> None:
    """
    Gate G7 — Release Readiness.
    Blocks if any upstream phase blocker remains unresolved.
    PARTIAL (UAT pending sign-off) is non-blocking — story proceeds conditionally.
    """
    readiness_data = _get_agent_data(state, "39")
    verdict = (readiness_data or {}).get("readiness_verdict", "")

    if verdict == "BLOCKED":
        blockers = (readiness_data or {}).get("readiness_blockers", [])
        raise GateG7Error(
            f"Gate G7 (Release Readiness) FAILED for {state['story_id']}. "
            f"Readiness verdict: BLOCKED — {len(blockers)} unresolved blocker(s): {'; '.join(blockers)}"
        )


def _check_gate_g8(state: StoryState) -> None:
    """
    Gate G8 — Change Set Integrity.
    Blocks if the change set cannot be composed or fails structural validation.
    WARN (large/destructive) is non-blocking.
    """
    failures: list[str] = []

    composer_data = _get_agent_data(state, "40")
    composer_verdict = (composer_data or {}).get("composer_verdict", "")
    if composer_verdict == "FAILED":
        raise GateG8Error(
            f"Gate G8 (Change Set) FAILED for {state['story_id']}. "
            "Release Composer (Agent 40): no valid change set could be produced."
        )

    integrity_data = _get_agent_data(state, "41")
    integrity_verdict = (integrity_data or {}).get("integrity_verdict", "")
    if integrity_verdict == "FAIL":
        issues = (integrity_data or {}).get("integrity_issues", [])
        failures.append(
            f"Change Set Integrity (Agent 41): FAIL — {len(issues)} issue(s): {'; '.join(issues)}"
        )

    if failures:
        raise GateG8Error(
            f"Gate G8 (Change Set) FAILED for {state['story_id']}. "
            f"Blockers: {'; '.join(failures)}"
        )


def _check_gate_g9(state: StoryState) -> None:
    """
    Gate G9 — Staging.
    Blocks if the deployment simulation failed or smoke tests report failures.
    SKIPPED results are non-blocking.
    """
    failures: list[str] = []

    dry_run_data = _get_agent_data(state, "42")
    dry_run_verdict = (dry_run_data or {}).get("dry_run_verdict", "")
    if dry_run_verdict == "FAIL":
        errors = (dry_run_data or {}).get("dry_run_errors", [])
        failures.append(
            f"Dry Run (Agent 42): FAIL — {len(errors)} error(s): {'; '.join(errors)}"
        )

    smoke_data = _get_agent_data(state, "43")
    smoke_verdict = (smoke_data or {}).get("smoke_verdict", "")
    if smoke_verdict == "FAIL":
        failed_count = (smoke_data or {}).get("smoke_failed", 0)
        failures.append(
            f"Smoke Tests (Agent 43): FAIL — {failed_count} test(s) failed on staging"
        )

    if failures:
        raise GateG9Error(
            f"Gate G9 (Staging) FAILED for {state['story_id']}. "
            f"Blockers: {'; '.join(failures)}"
        )


def _check_gate_g10(state: StoryState) -> None:
    """
    Gate G10 — FCA Evidence.
    Blocks if mandatory FCA regulatory attestations are entirely missing.
    PARTIAL evidence is allowed — Compliance Officer resolves remaining gaps.
    """
    evidence_data = _get_agent_data(state, "44")
    verdict = (evidence_data or {}).get("evidence_verdict", "")

    if verdict == "MISSING":
        gaps = (evidence_data or {}).get("evidence_gaps", [])
        raise GateG10Error(
            f"Gate G10 (FCA Evidence) FAILED for {state['story_id']}. "
            f"Mandatory FCA attestation(s) missing: {'; '.join(gaps)}"
        )


def _check_gate_g11(state: StoryState) -> None:
    """
    Gate G11 — Go/No-Go.
    Blocks if the Go/No-Go Coordinator issues a hard NO_GO decision.
    CONDITIONAL decisions proceed — conditions are managed externally by the Compliance Officer.
    """
    gng_data = _get_agent_data(state, "45")
    verdict = (gng_data or {}).get("coordinator_verdict", "")

    if verdict == "NO_GO":
        reasons = (gng_data or {}).get("no_go_reasons", [])
        raise GateG11Error(
            f"Gate G11 (Go/No-Go) FAILED for {state['story_id']}. "
            f"Coordinator verdict: NO_GO — {'; '.join(reasons)}"
        )


def _check_gate_g12(state: StoryState) -> None:
    """
    Gate G12 — Production Validation.
    Informational in v1 — production health check stubs always pass after a GO decision.
    Phase 2 will enforce health checks against Salesforce Event Log Files.
    """
    # No hard failures in v1; gate exists as an anchor for Phase 2 integration


# ── Phase orchestration ───────────────────────────────────────────────────────

async def run_release_phase(state: StoryState) -> StoryState:
    """
    Orchestrate the full Release phase for a story.
    Returns the updated StoryState with all agent results merged in.
    Raises GateG7–G11Error on gate failure.
    """
    from src.fleet_commander.worker import dispatch_agent  # lazy import to avoid MCP circular dep

    # ── Batch 1: Release Readiness + Release Notes Writer ──────────────────────
    batch1_results = await asyncio.gather(
        dispatch_agent(39, state),
        dispatch_agent(47, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [39, 47], batch1_results)

    # ── Gate G7: Release Readiness ────────────────────────────────────────────
    _check_gate_g7(state)

    # ── Agent 40: Release Composer (sequential — needs Agents 13, 17, 18) ─────
    result40 = await dispatch_agent(40, state)
    state["agent_results"]["40"] = result40

    # ── Agent 41: Change Set Integrity (sequential — needs 13, 40) ───────────
    result41 = await dispatch_agent(41, state)
    state["agent_results"]["41"] = result41

    # ── Gate G8: Change Set Integrity ─────────────────────────────────────────
    _check_gate_g8(state)

    # ── Agent 42: Dry Run (sequential — needs 25 env state, 41 integrity) ────
    result42 = await dispatch_agent(42, state)
    state["agent_results"]["42"] = result42

    # ── Agent 43: Smoke on Staging (sequential — needs 32 regression, 42 dry-run) ─
    result43 = await dispatch_agent(43, state)
    state["agent_results"]["43"] = result43

    # ── Gate G9: Staging ──────────────────────────────────────────────────────
    _check_gate_g9(state)

    # ── Agent 44: FCA Evidence Pack (sequential — needs 3, 4, 30, 33, 36) ────
    result44 = await dispatch_agent(44, state)
    state["agent_results"]["44"] = result44

    # ── Gate G10: FCA Evidence ────────────────────────────────────────────────
    _check_gate_g10(state)

    # ── Agent 45: Go/No-Go Coordinator (sequential — needs 36, 39, 41, 43, 44) ─
    result45 = await dispatch_agent(45, state)
    state["agent_results"]["45"] = result45

    # ── Gate G11: Go/No-Go ────────────────────────────────────────────────────
    _check_gate_g11(state)

    # ── Agent 46: Production Validation (sequential — needs 45 GO decision) ──
    result46 = await dispatch_agent(46, state)
    state["agent_results"]["46"] = result46

    # ── Gate G12: Production Validation ──────────────────────────────────────
    _check_gate_g12(state)

    # ── Final Batch: Rollback Readiness + Post-Release Monitor + Retrospective ─
    final_results = await asyncio.gather(
        dispatch_agent(48, state),
        dispatch_agent(49, state),
        dispatch_agent(50, state),
        return_exceptions=True,
    )
    state = _merge_results(state, [48, 49, 50], final_results)

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
