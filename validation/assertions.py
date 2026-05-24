"""
Pipeline Correctness Assertions — Finding 7 (Karpathy Principle 4)

The validation runner is a smoke test: it confirms agents run without errors
and produce non-empty output, but does not assert correctness of specific values.
This module adds a second layer: known-good assertions for the FSC-2417 story
derived from its deterministic mock data (MOCK_APEX_RESULTS, MOCK_PMD_RESULTS, etc.).

Unlike smoke tests, these assertions can only pass if agents correctly read,
compute, and propagate specific values — not just return non-empty output.

Usage:
    failures = assert_pipeline_correctness(results, story_id)
    # failures is a list of dicts: [{agent_id, key, expected, actual, note}]
"""

from __future__ import annotations

from typing import Any


def _find_result(results: list[dict], agent_id: int) -> dict | None:
    for r in results:
        if r.get("agent_id") == agent_id:
            return r
    return None


def _data(result: dict | None) -> dict:
    if result is None:
        return {}
    return result.get("data") or {}


def _fail(agent_id: int, key: str, expected: Any, actual: Any, note: str = "") -> dict:
    return {
        "agent_id": agent_id,
        "key": key,
        "expected": expected,
        "actual": actual,
        "note": note,
    }


# ── FSC-2417 known-good assertions ────────────────────────────────────────────
# Each assertion is derived from a deterministic mock input (no LLM randomness).
# Assertions that depend on LLM output are structural (non-empty, valid enum)
# rather than exact-value checks.

def _assert_fsc2417(results: list[dict]) -> list[dict]:
    failures: list[dict] = []

    # ── Agent 3: FCA Classifier ───────────────────────────────────────────────
    # Story explicitly references COBS 9.2, Consumer Duty PS22/9 — must be HIGH.
    r3 = _find_result(results, 3)
    fca_class = _data(r3).get("fca_classification")
    if fca_class != "HIGH":
        failures.append(_fail(
            3, "fca_classification", "HIGH", fca_class,
            "COBS 9.2 + Consumer Duty story must be classified HIGH FCA"
        ))

    # ── Agent 12: Apex Coverage ───────────────────────────────────────────────
    # Values come from MOCK_APEX_RESULTS (deterministic, not LLM).
    r12 = _find_result(results, 12)
    d12 = _data(r12)
    if d12.get("tests_passed") != 42:
        failures.append(_fail(
            12, "tests_passed", 42, d12.get("tests_passed"),
            "must match MOCK_APEX_RESULTS['tests_run'] = 42"
        ))
    cov_pct = d12.get("coverage_pct")
    if cov_pct != 87.3:
        failures.append(_fail(
            12, "coverage_pct", 87.3, cov_pct,
            "must match MOCK_APEX_RESULTS['coverage_pct'] = 87.3"
        ))

    # ── Agent 13: Metadata Dependency ────────────────────────────────────────
    # MOCK_CHANGED_FILES has 7 entries.
    r13 = _find_result(results, 13)
    d13 = _data(r13)
    count = d13.get("changed_files_count")
    if count != 7:
        failures.append(_fail(
            13, "changed_files_count", 7, count,
            "must match len(MOCK_CHANGED_FILES) = 7"
        ))
    # changed_files list must also be present (REQ-10)
    if not isinstance(d13.get("changed_files"), list):
        failures.append(_fail(
            13, "changed_files", "list", type(d13.get("changed_files")).__name__,
            "changed_files list must be emitted (REQ-10 — not just the count)"
        ))

    # ── Agent 14: Code Quality ────────────────────────────────────────────────
    # MOCK_PMD_RESULTS has 2 findings.
    r14 = _find_result(results, 14)
    d14 = _data(r14)
    pmd_count = d14.get("total_violation_count")
    if pmd_count != 2:
        failures.append(_fail(
            14, "total_violation_count", 2, pmd_count,
            "must match len(MOCK_PMD_RESULTS) = 2"
        ))

    # ── Agent 5: AC Generator ─────────────────────────────────────────────────
    # SAMPLE_AC has 4 acceptance criteria — ac_count must be > 0.
    r5 = _find_result(results, 5)
    d5 = _data(r5)
    ac_count = d5.get("ac_clause_count", 0)
    if not (isinstance(ac_count, int) and ac_count > 0):
        failures.append(_fail(
            5, "ac_clause_count", ">0", ac_count,
            "story has sample ACs — ac_clause_count must be a positive integer"
        ))
    if not isinstance(d5.get("ac_clauses"), list) or len(d5.get("ac_clauses", [])) == 0:
        failures.append(_fail(
            5, "ac_clauses", "non-empty list", d5.get("ac_clauses"),
            "ac_clauses must be a non-empty list"
        ))

    # ── Agent 55: 3 Amigos Facilitator ───────────────────────────────────────
    # All five required list fields must be non-empty (schema has minItems:1).
    r55 = _find_result(results, 55)
    d55 = _data(r55)
    required_list_fields = [
        "ba_discussion_points",
        "developer_discussion_points",
        "tester_discussion_points",
        "definition_of_done",
        "action_items",
    ]
    for field in required_list_fields:
        val = d55.get(field)
        if not isinstance(val, list) or len(val) == 0:
            failures.append(_fail(
                55, field, "non-empty list", val,
                f"Agent 55 required list field must be non-empty (schema minItems=1)"
            ))
    # story_ready_assessment must be a valid enum value
    sra = d55.get("story_ready_assessment")
    valid_sra = {"READY", "NEEDS_DISCUSSION", "BLOCKED"}
    if sra not in valid_sra:
        failures.append(_fail(
            55, "story_ready_assessment", f"one of {valid_sra}", sra,
            "story_ready_assessment must be a valid enum value"
        ))
    # regression fields must be non-empty (F9 — flattened schema)
    reg_areas = d55.get("regression_affected_areas")
    if not isinstance(reg_areas, list) or len(reg_areas) == 0:
        failures.append(_fail(
            55, "regression_affected_areas", "non-empty list", reg_areas,
            "regression_affected_areas must be non-empty (HIGH regression risk story)"
        ))
    reg_risk = d55.get("regression_risk_level")
    if reg_risk not in ("LOW", "MEDIUM", "HIGH"):
        failures.append(_fail(
            55, "regression_risk_level", "LOW/MEDIUM/HIGH", reg_risk,
            "regression_risk_level must be a valid enum value"
        ))

    # ── All agents: status must be ok ─────────────────────────────────────────
    for r in results:
        if r.get("status") == "error":
            failures.append(_fail(
                r["agent_id"], "status", "ok", "error",
                f"agent raised exception: {r.get('error', '')[:120]}"
            ))

    return failures


# ── FSC-3801 known-good assertions ────────────────────────────────────────────
# MEDIUM FCA story — Consumer Duty PS22/9 Outcome 1, no COBS 9.2.

def _assert_fsc3801(results: list[dict]) -> list[dict]:
    failures: list[dict] = []

    # ── Agent 3: FCA Classifier ───────────────────────────────────────────────
    # Story explicitly states MEDIUM FCA (Consumer Duty Outcome 1, no COBS 9.2).
    r3 = _find_result(results, 3)
    fca_class = _data(r3).get("fca_classification")
    if fca_class != "MEDIUM":
        failures.append(_fail(
            3, "fca_classification", "MEDIUM", fca_class,
            "Consumer Duty Outcome 1 story with no COBS 9.2 must be classified MEDIUM FCA"
        ))

    # ── Agent 12: Apex Coverage ───────────────────────────────────────────────
    r12 = _find_result(results, 12)
    d12 = _data(r12)
    if d12.get("tests_passed") != 31:
        failures.append(_fail(
            12, "tests_passed", 31, d12.get("tests_passed"),
            "must match MOCK_APEX_RESULTS_2['tests_passed'] = 31"
        ))
    cov_pct = d12.get("coverage_pct")
    if cov_pct != 84.7:
        failures.append(_fail(
            12, "coverage_pct", 84.7, cov_pct,
            "must match MOCK_APEX_RESULTS_2['coverage_pct'] = 84.7"
        ))

    # ── Agent 13: Metadata Dependency ────────────────────────────────────────
    # MOCK_CHANGED_FILES_2 has 8 entries.
    r13 = _find_result(results, 13)
    d13 = _data(r13)
    count = d13.get("changed_files_count")
    if count != 8:
        failures.append(_fail(
            13, "changed_files_count", 8, count,
            "must match len(MOCK_CHANGED_FILES_2) = 8"
        ))
    if not isinstance(d13.get("changed_files"), list):
        failures.append(_fail(
            13, "changed_files", "list", type(d13.get("changed_files")).__name__,
            "changed_files list must be emitted (REQ-10)"
        ))

    # ── Agent 14: Code Quality ────────────────────────────────────────────────
    # MOCK_PMD_RESULTS_2 has 1 finding.
    r14 = _find_result(results, 14)
    d14 = _data(r14)
    pmd_count = d14.get("total_violation_count")
    if pmd_count != 1:
        failures.append(_fail(
            14, "total_violation_count", 1, pmd_count,
            "must match len(MOCK_PMD_RESULTS_2) = 1"
        ))

    # ── Agent 5: AC Generator ─────────────────────────────────────────────────
    r5 = _find_result(results, 5)
    d5 = _data(r5)
    ac_count = d5.get("ac_clause_count", 0)
    if not (isinstance(ac_count, int) and ac_count > 0):
        failures.append(_fail(
            5, "ac_clause_count", ">0", ac_count,
            "story has 4 sample ACs — ac_clause_count must be a positive integer"
        ))
    if not isinstance(d5.get("ac_clauses"), list) or len(d5.get("ac_clauses", [])) == 0:
        failures.append(_fail(
            5, "ac_clauses", "non-empty list", d5.get("ac_clauses"),
            "ac_clauses must be a non-empty list"
        ))

    # ── Agent 55: 3 Amigos Facilitator ───────────────────────────────────────
    r55 = _find_result(results, 55)
    d55 = _data(r55)
    required_list_fields = [
        "ba_discussion_points",
        "developer_discussion_points",
        "tester_discussion_points",
        "definition_of_done",
        "action_items",
    ]
    for field in required_list_fields:
        val = d55.get(field)
        if not isinstance(val, list) or len(val) == 0:
            failures.append(_fail(
                55, field, "non-empty list", val,
                "Agent 55 required list field must be non-empty (schema minItems=1)"
            ))
    sra = d55.get("story_ready_assessment")
    valid_sra = {"READY", "NEEDS_DISCUSSION", "BLOCKED"}
    if sra not in valid_sra:
        failures.append(_fail(
            55, "story_ready_assessment", f"one of {valid_sra}", sra,
            "story_ready_assessment must be a valid enum value"
        ))

    # ── All agents: status must be ok ─────────────────────────────────────────
    for r in results:
        if r.get("status") == "error":
            failures.append(_fail(
                r["agent_id"], "status", "ok", "error",
                f"agent raised exception: {r.get('error', '')[:120]}"
            ))

    return failures


# ── Public API ────────────────────────────────────────────────────────────────

_STORY_ASSERTIONS = {
    "FSC-2417": _assert_fsc2417,
    "FSC-3801": _assert_fsc3801,
}


def assert_pipeline_correctness(results: list[dict], story_id: str) -> list[dict]:
    """
    Run correctness assertions for the given story.
    Returns a list of failure dicts (empty = all passed).
    Never raises — failures are surfaced as data so the caller can decide.
    """
    fn = _STORY_ASSERTIONS.get(story_id)
    if fn is None:
        return []  # no assertions defined for this story ID — not a failure
    return fn(results)
