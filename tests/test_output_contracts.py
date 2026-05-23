"""
Output Key Contract Tests — Finding 1 (Karpathy Principle 3)

Every key that a downstream agent reads from an upstream agent must exist in
the upstream agent's source file as an emitted data key. This catches
silent .get("wrong_key") bugs at CI time without live API calls.

Contract format:
  (upstream_agent_id, key_name, upstream_phase, upstream_filename)

The test checks both sides:
  - Upstream agent SOURCE emits the key (in its data dict literal)
  - Downstream agents that read the key use the CORRECT key name

This structural enforcement means a future renaming of any data key will
fail CI immediately, before it can silently produce wrong defaults.
"""

import re
from pathlib import Path

import pytest

AGENTS_ROOT = Path(__file__).parent.parent / "src" / "agents"

# ── Source reader ─────────────────────────────────────────────────────────────

def _src(phase: str, filename: str) -> str:
    return (AGENTS_ROOT / phase / filename).read_text(encoding="utf-8")


def _emits_key(source: str, key: str) -> bool:
    """True if the key appears as a string literal (dict key) in the source."""
    return bool(re.search(rf"""['"]{re.escape(key)}['"]""", source))


def _reads_key(source: str, key: str) -> bool:
    """True if the source reads the key via .get('key') or ['key']."""
    return bool(re.search(rf"""\.get\(['"]{re.escape(key)}['"]|['"]{re.escape(key)}['"]\]""", source))


# ── Known upstream contracts ──────────────────────────────────────────────────
# (agent_id, key_emitted, phase, filename)

UPSTREAM_KEYS = [
    ("3",  "fca_classification",           "refinement",  "agent_03_fca_classifier.py"),
    ("3",  "co_signoff_required",          "refinement",  "agent_03_fca_classifier.py"),
    ("4",  "cd_obligations",              "refinement",  "agent_04_consumer_duty.py"),
    ("4",  "vulnerable_customer_impact",   "refinement",  "agent_04_consumer_duty.py"),
    ("4",  "cd_verdict",                  "refinement",  "agent_04_consumer_duty.py"),
    ("5",  "ac_clauses",                  "refinement",  "agent_05_ac_generator.py"),
    ("5",  "ac_clause_count",              "refinement",  "agent_05_ac_generator.py"),
    ("5",  "remaining_gaps",              "refinement",  "agent_05_ac_generator.py"),
    ("7",  "required_records",            "refinement",  "agent_07_data_need.py"),
    ("7",  "data_isolation_strategy",     "refinement",  "agent_07_data_need.py"),
    ("8",  "dependency_depth",            "refinement",  "agent_08_dependency_mapping.py"),
    ("8",  "has_destructive_changes",     "refinement",  "agent_08_dependency_mapping.py"),
    ("9",  "risk_register",              "refinement",  "agent_09_risk_anticipation.py"),
    ("9",  "critical_risk_count",         "refinement",  "agent_09_risk_anticipation.py"),
    ("9",  "overall_risk_level",          "refinement",  "agent_09_risk_anticipation.py"),
    ("13", "detected_objects",            "development", "agent_13_metadata_dependency.py"),
    ("13", "dependency_depth",            "development", "agent_13_metadata_dependency.py"),
    ("13", "scope_delta_objects",         "development", "agent_13_metadata_dependency.py"),
    ("13", "changed_files",              "development", "agent_13_metadata_dependency.py"),
    ("13", "has_destructive_changes",     "development", "agent_13_metadata_dependency.py"),
    ("19", "gherkin_scenarios",           "development", "agent_19_bdd_gherkin_writer.py"),
    ("19", "scenario_count",             "development", "agent_19_bdd_gherkin_writer.py"),
    ("21", "data_verdict",              "development", "agent_21_test_data_architect.py"),
    ("23", "development_verdict",         "development", "agent_23_story_code_tracer.py"),
    ("32", "regression_risk_level",       "testing",     "agent_32_regression_risk_assessor.py"),
    ("32", "recommended_regression_suite","testing",     "agent_32_regression_risk_assessor.py"),
    ("32", "regression_verdict",          "testing",     "agent_32_regression_risk_assessor.py"),
    ("33", "coverage_verdict",            "testing",     "agent_33_test_coverage_analyser.py"),
    ("33", "overall_coverage_pct",        "testing",     "agent_33_test_coverage_analyser.py"),
    ("34", "defect_verdict",             "testing",     "agent_34_defect_triage.py"),
    ("34", "critical_defects",           "testing",     "agent_34_defect_triage.py"),
    ("40", "composer_verdict",           "release",     "agent_40_release_composer.py"),
    ("40", "release_type",              "release",     "agent_40_release_composer.py"),
    ("41", "integrity_verdict",          "release",     "agent_41_change_set_integrity.py"),
    ("41", "integrity_issues",           "release",     "agent_41_change_set_integrity.py"),
    ("42", "dry_run_verdict",            "release",     "agent_42_dry_run.py"),
    ("42", "dry_run_success",            "release",     "agent_42_dry_run.py"),
    ("43", "smoke_verdict",             "release",     "agent_43_smoke_on_staging.py"),
    ("44", "evidence_verdict",           "release",     "agent_44_fca_evidence_pack.py"),
    ("45", "coordinator_verdict",        "release",     "agent_45_go_no_go.py"),
    ("45", "no_go_reasons",             "release",     "agent_45_go_no_go.py"),
    ("48", "rollback_verdict",           "release",     "agent_48_rollback_readiness.py"),
    ("48", "rollback_feasible",          "release",     "agent_48_rollback_readiness.py"),
]

@pytest.mark.parametrize("agent_id,key,phase,filename", UPSTREAM_KEYS)
def test_upstream_agent_emits_key(agent_id, key, phase, filename):
    """Upstream agent source must contain the key as a string literal."""
    source = _src(phase, filename)
    assert _emits_key(source, key), (
        f"Agent {agent_id} ({filename}) does not emit key '{key}' in its data dict. "
        f"Either the key was renamed or was never added. Fix the agent or update this contract."
    )


# ── Known downstream reads ────────────────────────────────────────────────────
# (downstream_agent_id, key_read, downstream_phase, downstream_filename,
#  upstream_agent_id_for_context)
# These verify the downstream side reads with the CORRECT key name.

DOWNSTREAM_READS = [
    # Agent 22 reads scope_delta_objects from Agent 13
    ("22", "scope_delta_objects",          "development", "agent_22_sandbox_state.py",            "13"),
    # Agent 43 reads regression_risk_level from Agent 32
    ("43", "regression_risk_level",        "release",     "agent_43_smoke_on_staging.py",          "32"),
    # Agent 43 reads recommended_regression_suite from Agent 32
    ("43", "recommended_regression_suite", "release",     "agent_43_smoke_on_staging.py",          "32"),
    # Agent 47 reads ac_clauses from Agent 5
    ("47", "ac_clauses",                  "release",     "agent_47_release_notes_writer.py",       "5"),
    # Agent 47 reads gherkin_scenarios from Agent 19
    ("47", "gherkin_scenarios",           "release",     "agent_47_release_notes_writer.py",       "19"),
    # Agent 48 reads has_destructive_changes from Agent 13
    ("48", "has_destructive_changes",      "release",     "agent_48_rollback_readiness.py",         "13"),
    # Agent 55 reads ac_clause_count from Agent 5
    ("55", "ac_clause_count",             "refinement",  "agent_55_3_amigos_facilitator.py",       "5"),
    # Agent 55 reads cd_obligations from Agent 4
    ("55", "cd_obligations",              "refinement",  "agent_55_3_amigos_facilitator.py",       "4"),
    # Agent 55 reads required_records from Agent 7 (labelled data_vol in agent_55 but data_volume key)
    ("55", "data_isolation_strategy",     "refinement",  "agent_55_3_amigos_facilitator.py",       "7"),
    # Agent 55 reads dependency_depth from Agent 8
    ("55", "dependency_depth",            "refinement",  "agent_55_3_amigos_facilitator.py",       "8"),
    # Agent 55 reads critical_risk_count from Agent 9
    ("55", "critical_risk_count",         "refinement",  "agent_55_3_amigos_facilitator.py",       "9"),
    # Agent 32 reads development_verdict from Agent 23
    ("32", "development_verdict",          "testing",     "agent_32_regression_risk_assessor.py",   "23"),
]

@pytest.mark.parametrize("agent_id,key,phase,filename,upstream_id", DOWNSTREAM_READS)
def test_downstream_agent_reads_correct_key(agent_id, key, phase, filename, upstream_id):
    """Downstream agent must read the key using the correct string literal."""
    source = _src(phase, filename)
    assert _reads_key(source, key), (
        f"Agent {agent_id} ({filename}) does not read key '{key}' from Agent {upstream_id}. "
        f"Either the read key was renamed without updating the downstream consumer, "
        f"or the upstream agent renamed the key. Fix the mismatch."
    )


# ── Schema-level contract: required fields in tool schemas ───────────────────

def test_agent_55_schema_has_required_list_fields():
    """Agent 55 tool schema must include minItems:1 on all required list fields."""
    from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA
    required_lists = [
        "ba_discussion_points",
        "developer_discussion_points",
        "tester_discussion_points",
        "definition_of_done",
        "action_items",
    ]
    props = _AMIGOS_TOOL_SCHEMA["properties"]
    for field in required_lists:
        assert field in props, f"'{field}' missing from Agent 55 tool schema"
        assert props[field].get("minItems") == 1, (
            f"Agent 55 schema field '{field}' must have minItems=1 to prevent empty arrays"
        )


def test_agent_55_required_fields_complete():
    """Agent 55 tool schema required array must list all 10 output fields."""
    from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA
    required = set(_AMIGOS_TOOL_SCHEMA["required"])
    expected = {
        "ba_discussion_points", "developer_discussion_points",
        "tester_discussion_points", "open_questions", "recommended_decisions",
        "story_ready_assessment", "facilitator_summary",
        "definition_of_done", "action_items",
        "regression_affected_areas", "regression_risk_level", "regression_notes",
    }
    missing = expected - required
    assert not missing, f"Agent 55 schema 'required' is missing: {missing}"
