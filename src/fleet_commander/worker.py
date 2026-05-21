"""
Agent worker dispatch — the bridge between the Fleet Commander graph
and the individual agent implementations.

The Fleet Commander never runs agent logic directly. It enqueues a task,
the agent worker runs it, and the result is returned into the graph state.
In Wave 1 this runs in-process for simplicity; in Wave 2+ this moves to
Redis queue + pool of worker processes.
"""

import importlib
import time
from datetime import datetime, timezone

from src.core.schemas import AgentResult, StoryState
from src.mcp.qds_mcp import server as qds


# Map of agent_id → module path for the agent's run() function
AGENT_REGISTRY: dict[int, str] = {
    # Refinement phase
    1:  "src.agents.refinement.agent_01_story_intent",
    2:  "src.agents.refinement.agent_02_invest_quality",
    3:  "src.agents.refinement.agent_03_fca_classifier",
    4:  "src.agents.refinement.agent_04_consumer_duty",
    5:  "src.agents.refinement.agent_05_ac_generator",
    6:  "src.agents.refinement.agent_06_test_design",
    7:  "src.agents.refinement.agent_07_data_need",
    8:  "src.agents.refinement.agent_08_dependency_mapping",
    9:  "src.agents.refinement.agent_09_risk_anticipation",
    54: "src.agents.refinement.agent_05b_ac_challenger",  # AC Challenger (adversarial, runs in Batch 3)
    # Development phase
    10: "src.agents.development.agent_10_ac_compliance",
    11: "src.agents.development.agent_11_branch_tracer",
    12: "src.agents.development.agent_12_apex_coverage",
    13: "src.agents.development.agent_13_metadata_dependency",
    14: "src.agents.development.agent_14_code_quality",
    15: "src.agents.development.agent_15_apex_security",
    16: "src.agents.development.agent_16_bulk_quality",
    17: "src.agents.development.agent_17_sfdx_validator",
    18: "src.agents.development.agent_18_component_attribution",
    19: "src.agents.development.agent_19_bdd_gherkin_writer",
    20: "src.agents.development.agent_20_performance_risk",
    21: "src.agents.development.agent_21_test_data_architect",
    22: "src.agents.development.agent_22_sandbox_state",
    23: "src.agents.development.agent_23_story_code_tracer",
    # Testing phase
    24: "src.agents.testing.agent_24_test_strategy_validator",
    25: "src.agents.testing.agent_25_test_env_provisioner",
    26: "src.agents.testing.agent_26_crt_scenario_designer",
    27: "src.agents.testing.agent_27_crt_execution",
    28: "src.agents.testing.agent_28_crt_self_heal_reviewer",
    29: "src.agents.testing.agent_29_uat_test_case_generator",
    30: "src.agents.testing.agent_30_fca_scenario_agent",
    31: "src.agents.testing.agent_31_financial_data_integrity",
    32: "src.agents.testing.agent_32_regression_risk_assessor",
    33: "src.agents.testing.agent_33_test_coverage_analyser",
    34: "src.agents.testing.agent_34_defect_triage",
    35: "src.agents.testing.agent_35_root_cause_analyser",
    36: "src.agents.testing.agent_36_uat_coordination",
    37: "src.agents.testing.agent_37_performance_test",
    38: "src.agents.testing.agent_38_flaky_test_hunter",
    # Release phase
    39: "src.agents.release.agent_39_release_readiness",
    40: "src.agents.release.agent_40_release_composer",
    41: "src.agents.release.agent_41_change_set_integrity",
    42: "src.agents.release.agent_42_dry_run",
    43: "src.agents.release.agent_43_smoke_on_staging",
    44: "src.agents.release.agent_44_fca_evidence_pack",
    45: "src.agents.release.agent_45_go_no_go",
    46: "src.agents.release.agent_46_production_validation",
    47: "src.agents.release.agent_47_release_notes_writer",
    48: "src.agents.release.agent_48_rollback_readiness",
    49: "src.agents.release.agent_49_post_release_monitor",
    50: "src.agents.release.agent_50_retrospective",
    # Monitoring / cross-phase
    51: "src.agents.monitoring.agent_51_health",
    52: "src.agents.monitoring.agent_52_severity_calibration",
    53: "src.agents.monitoring.agent_53_incident_response",
}


def validate_registry() -> None:
    """
    Verify every module in AGENT_REGISTRY can be imported.
    Call at startup so missing agents are caught before any story runs.
    Raises ImportError with the list of broken entries.
    """
    failures: list[str] = []
    for agent_id, module_path in AGENT_REGISTRY.items():
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            failures.append(f"Agent {agent_id} ({module_path}): {exc}")
    if failures:
        raise ImportError(
            f"AGENT_REGISTRY has {len(failures)} unresolvable module(s):\n"
            + "\n".join(failures)
        )


async def dispatch_agent(agent_id: int, state: StoryState) -> dict:
    """
    Load the agent module, call its run() function, record the execution,
    and return the serialised AgentResult dict.

    Fails closed: if the agent raises any exception, the error is recorded
    and re-raised so the Fleet Commander can apply fail-closed gate logic.
    """
    module_path = AGENT_REGISTRY.get(agent_id)
    if not module_path:
        raise ValueError(f"Agent {agent_id} is not registered")

    started_at = datetime.now(timezone.utc)
    start_ms = time.monotonic_ns()

    try:
        module = importlib.import_module(module_path)
        result: AgentResult = await module.run(state)
        completed_at = datetime.now(timezone.utc)
        latency_ms = (time.monotonic_ns() - start_ms) // 1_000_000

        await qds.record_agent_run(
            agent_id=agent_id,
            story_id=state["story_id"],
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            latency_ms=latency_ms,
            success=True,
        )

        # Emit the agent decision to the audit ledger
        await qds.emit_decision_event(
            event_type="AGENT_DECISION",
            story_id=state["story_id"],
            agent_id=agent_id,
            what=result.what,
            why=result.why,
            data=result.data,
            confidence_tier=result.confidence.tier,
            raw_score=result.confidence.raw_score,
            calibration_multiplier=result.confidence.calibration_multiplier,
            final_score=result.confidence.final_score,
            model_used=result.model_used,
        )

        return result.model_dump()

    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        latency_ms = (time.monotonic_ns() - start_ms) // 1_000_000

        await qds.record_agent_run(
            agent_id=agent_id,
            story_id=state["story_id"],
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            latency_ms=latency_ms,
            success=False,
            error_message=str(exc),
        )
        raise
