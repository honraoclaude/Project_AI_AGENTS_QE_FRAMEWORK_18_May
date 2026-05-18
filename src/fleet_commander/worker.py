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
    1:  "src.agents.refinement.agent_01_story_intent",
    2:  "src.agents.refinement.agent_02_invest_quality",
    3:  "src.agents.refinement.agent_03_fca_classifier",
    4:  "src.agents.refinement.agent_04_consumer_duty",
    5:  "src.agents.refinement.agent_05_ac_generator",
    6:  "src.agents.refinement.agent_06_test_design",
    7:  "src.agents.refinement.agent_07_data_need",
    8:  "src.agents.refinement.agent_08_dependency_mapping",
    9:  "src.agents.refinement.agent_09_risk_anticipation",
    51: "src.agents.monitoring.agent_51_health",
}


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
