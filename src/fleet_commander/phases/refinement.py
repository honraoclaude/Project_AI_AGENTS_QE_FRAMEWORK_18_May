"""
Refinement phase subgraph — Agents 1–9, Gate G1.

Execution order derived from the dependency graph:
  Batch 1 (parallel): Agent 1 (Story Intent), Agent 8 (Dependency Mapping)
  Batch 2 (parallel): Agent 2 (INVEST Quality), Agent 3 (FCA Classifier), Agent 7 (Data Need)
  Sequential:         Agent 4 (Consumer Duty Mapping)   ← needs Agent 3 output
  Batch 3 (parallel): Agent 5 (AC Generator), Agent 6 (Test Design Strategy)
  Sequential:         Agent 9 (Risk Anticipation)       ← needs 2,3,4,8 outputs
  Gate G1 evaluation
"""

import asyncio
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.base import adaptive_threshold
from src.core.config import settings
from src.core.schemas import StoryState
from src.fleet_commander.worker import dispatch_agent


# ── Node implementations ──────────────────────────────────────────────────────

async def run_batch_1(state: StoryState) -> StoryState:
    """Parallel: Agent 1 (Story Intent) + Agent 8 (Dependency Mapping)."""
    results = await asyncio.gather(
        dispatch_agent(agent_id=1, state=state),
        dispatch_agent(agent_id=8, state=state),
        return_exceptions=True,
    )
    agent_results = dict(state["agent_results"])
    phase_errors = list(state["phase_errors"])

    for agent_id, result in zip([1, 8], results):
        if isinstance(result, Exception):
            phase_errors.append(f"Agent {agent_id} failed: {result}")
        else:
            agent_results[str(agent_id)] = result

    return {**state, "agent_results": agent_results, "phase_errors": phase_errors}


async def run_batch_2(state: StoryState) -> StoryState:
    """Parallel: Agent 2 (INVEST), Agent 3 (FCA Classifier), Agent 7 (Data Need)."""
    results = await asyncio.gather(
        dispatch_agent(agent_id=2, state=state),
        dispatch_agent(agent_id=3, state=state),
        dispatch_agent(agent_id=7, state=state),
        return_exceptions=True,
    )
    agent_results = dict(state["agent_results"])
    phase_errors = list(state["phase_errors"])
    fca_classification = state["fca_classification"]

    for agent_id, result in zip([2, 3, 7], results):
        if isinstance(result, Exception):
            phase_errors.append(f"Agent {agent_id} failed: {result}")
        else:
            agent_results[str(agent_id)] = result
            # Extract FCA classification from Agent 3's output
            if agent_id == 3 and "fca_classification" in result.get("data", {}):
                fca_classification = result["data"]["fca_classification"]

    return {
        **state,
        "agent_results": agent_results,
        "phase_errors": phase_errors,
        "fca_classification": fca_classification,
    }


async def run_agent_4(state: StoryState) -> StoryState:
    """Sequential: Agent 4 (Consumer Duty Mapping) — needs Agent 3 output."""
    result = await dispatch_agent(agent_id=4, state=state)
    return {
        **state,
        "agent_results": {**state["agent_results"], "4": result},
    }


async def run_batch_3(state: StoryState) -> StoryState:
    """Parallel: Agent 5 (AC Generator) + Agent 6 (Test Design Strategy).
    Agent 5 runs first (its output feeds Agent 5B); then 5B + 6 run in parallel."""
    # Step A: run Agent 5 to produce AC clauses
    agent5_result = await dispatch_agent(agent_id=5, state=state)
    agent_results = dict(state["agent_results"])
    phase_errors = list(state["phase_errors"])
    agent_results["5"] = agent5_result

    # Step B: Agent 5B (AC Challenger) and Agent 6 in parallel, with Agent 5 output available
    state_with_5 = {**state, "agent_results": agent_results, "phase_errors": phase_errors}
    results = await asyncio.gather(
        dispatch_agent(agent_id=54, state=state_with_5),  # AC Challenger (Agent 05B)
        dispatch_agent(agent_id=6, state=state_with_5),   # Test Design Strategy
        return_exceptions=True,
    )

    for agent_id, result in zip([54, 6], results):
        if isinstance(result, Exception):
            phase_errors.append(f"Agent {agent_id} failed: {result}")
        else:
            agent_results[str(agent_id)] = result

    return {**state, "agent_results": agent_results, "phase_errors": phase_errors}


async def run_agent_9(state: StoryState) -> StoryState:
    """Sequential: Agent 9 (Risk Anticipation) — composite of 2,3,4,8."""
    result = await dispatch_agent(agent_id=9, state=state)
    return {
        **state,
        "agent_results": {**state["agent_results"], "9": result},
    }


async def evaluate_g1(state: StoryState) -> StoryState:
    """
    Gate G1 — INVEST + FCA Classification.

    Blocks if:
      - INVEST score < 80
      - FCA classification is UNCLASSIFIED
    Interrupts (awaits human approval) if:
      - Classification is HIGH and no Compliance Officer sign-off yet
    """
    from src.fleet_commander.email import send_approval_email

    agent_results = state["agent_results"]
    gate_states = dict(state["gate_states"])
    pending_approvals = list(state["pending_approvals"])

    # Extract INVEST score from Agent 2
    invest_result = agent_results.get("2", {})
    invest_score = invest_result.get("data", {}).get("invest_score", 0)

    # Adaptive G1 threshold — stricter for HIGH-FCA, lenient for LOW-FCA
    fca_tier = state.get("fca_classification", "UNCLASSIFIED")
    invest_threshold = adaptive_threshold(80, fca_tier)

    # Hard block: INVEST score below adaptive threshold
    if invest_score < invest_threshold:
        gate_states["G1"] = {
            "gate_id": "G1", "status": "BLOCKED",
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "decided_by": "agent_2",
        }
        return {
            **state,
            "current_phase": "BLOCKED",
            "gate_states": gate_states,
            "block_reason": (
                f"INVEST score {invest_score} is below the {invest_threshold} threshold "
                f"(adaptive for {fca_tier}-FCA). Story requires rework."
            ),
        }

    # Hard block: FCA classification missing
    fca = state["fca_classification"]
    if fca == "UNCLASSIFIED":
        gate_states["G1"] = {
            "gate_id": "G1", "status": "BLOCKED",
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "decided_by": "agent_3",
        }
        return {
            **state,
            "current_phase": "BLOCKED",
            "gate_states": gate_states,
            "block_reason": "FCA classification is missing. Agent 3 could not classify this story.",
        }

    # HIGH-FCA: check for Compliance Officer sign-off
    if fca == "HIGH":
        co_signed = _has_co_signoff(state, "G1")
        if not co_signed:
            # Send approval email and interrupt graph
            approval = await send_approval_email(
                story_id=state["story_id"],
                gate_id="G1",
                approver_email=settings.compliance_officer_email,
                approver_role="CO",
                action_type="SIGNOFF",
                context=_build_g1_email_context(state),
            )
            pending_approvals.append(approval)
            gate_states["G1"] = {
                "gate_id": "G1", "status": "ESCALATED",
                "decided_at": None, "decided_by": None,
            }
            updated_state = {**state, "pending_approvals": pending_approvals, "gate_states": gate_states}
            # Graph pauses here — resumes when CO clicks the approval link
            interrupt({"reason": "awaiting_co_signoff", "gate": "G1", "story_id": state["story_id"]})
            return updated_state

    # All checks pass
    gate_states["G1"] = {
        "gate_id": "G1", "status": "CLOSED",
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "decided_by": "fleet_commander",
    }
    return {
        **state,
        "gate_states": gate_states,
        "current_phase": "DEVELOPMENT",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_g1(state: StoryState) -> str:
    phase = state["current_phase"]
    if phase == "BLOCKED":
        return "blocked"
    if phase == "DEVELOPMENT":
        return "development"
    # Still in refinement (interrupt fired and graph resumed) — re-evaluate
    return "evaluate_g1"


# ── Subgraph builder ──────────────────────────────────────────────────────────

def build_refinement_subgraph() -> StateGraph:
    graph = StateGraph(StoryState)

    graph.add_node("batch_1", run_batch_1)
    graph.add_node("batch_2", run_batch_2)
    graph.add_node("agent_4", run_agent_4)
    graph.add_node("batch_3", run_batch_3)
    graph.add_node("agent_9", run_agent_9)
    graph.add_node("evaluate_g1", evaluate_g1)
    graph.add_node("blocked", lambda s: {**s, "current_phase": "BLOCKED"})
    graph.add_node("development", lambda s: {**s, "current_phase": "DEVELOPMENT"})

    graph.add_edge(START, "batch_1")
    graph.add_edge("batch_1", "batch_2")
    graph.add_edge("batch_2", "agent_4")
    graph.add_edge("agent_4", "batch_3")
    graph.add_edge("batch_3", "agent_9")
    graph.add_edge("agent_9", "evaluate_g1")
    graph.add_conditional_edges("evaluate_g1", route_after_g1, {
        "blocked": "blocked",
        "development": "development",
        "evaluate_g1": "evaluate_g1",
    })
    graph.add_edge("blocked", END)
    graph.add_edge("development", END)

    return graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_co_signoff(state: StoryState, gate_id: str) -> bool:
    for approval in state["pending_approvals"]:
        if (
            approval.get("gate_id") == gate_id
            and approval.get("approver_role") == "CO"
            and approval.get("used") is True
        ):
            return True
    return False


def _build_g1_email_context(state: StoryState) -> dict:
    agent_2 = state["agent_results"].get("2", {})
    agent_3 = state["agent_results"].get("3", {})
    return {
        "invest_score": agent_2.get("data", {}).get("invest_score"),
        "fca_classification": state["fca_classification"],
        "agent_3_what": agent_3.get("what", ""),
        "agent_3_why": agent_3.get("why", ""),
        "agent_3_confidence": agent_3.get("confidence", {}).get("final_score"),
    }
