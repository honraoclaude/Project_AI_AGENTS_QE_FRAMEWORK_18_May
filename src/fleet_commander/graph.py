"""
Fleet Commander — top-level LangGraph stateful graph.

One graph instance per story, identified by story_id as the thread_id.
State is checkpointed to PostgreSQL after every node transition.
If the service restarts, graphs resume from their last checkpoint automatically.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from src.core.schemas import StoryState, initial_story_state
from src.fleet_commander.phases.development import run_development_phase
from src.fleet_commander.phases.refinement import build_refinement_subgraph
from src.fleet_commander.phases.release import run_release_phase
from src.fleet_commander.phases.testing import run_testing_phase


def build_fleet_commander(checkpointer: AsyncPostgresSaver) -> StateGraph:
    """
    Build the Fleet Commander graph with all phase subgraphs wired.
    Refinement runs as a LangGraph subgraph (with interrupt support for G1 sign-off).
    Development, Testing, and Release run as async node functions.
    """
    main_graph = StateGraph(StoryState)

    # Compile the refinement subgraph and add as a node
    refinement_graph = build_refinement_subgraph().compile()
    main_graph.add_node("refinement", refinement_graph)

    main_graph.add_node("development", _run_development)
    main_graph.add_node("testing", _run_testing)
    main_graph.add_node("release", _run_release)
    main_graph.add_node("complete", _mark_complete)
    main_graph.add_node("blocked", _mark_blocked)

    main_graph.add_edge(START, "refinement")
    main_graph.add_conditional_edges("refinement", _route_after_phase, {
        "development": "development",
        "blocked": "blocked",
    })
    main_graph.add_conditional_edges("development", _route_after_phase, {
        "testing": "testing",
        "blocked": "blocked",
    })
    main_graph.add_conditional_edges("testing", _route_after_phase, {
        "release": "release",
        "blocked": "blocked",
    })
    main_graph.add_conditional_edges("release", _route_after_phase, {
        "complete": "complete",
        "blocked": "blocked",
    })
    main_graph.add_edge("complete", END)
    main_graph.add_edge("blocked", END)

    return main_graph.compile(checkpointer=checkpointer)


def _route_after_phase(state: StoryState) -> str:
    return state["current_phase"].lower()


async def _run_development(state: StoryState) -> StoryState:
    """Development phase — Agents 10–23, Gates G2–G4."""
    try:
        state = await run_development_phase(state)
        return {**state, "current_phase": "TESTING"}
    except Exception as exc:
        return {**state, "current_phase": "BLOCKED", "block_reason": str(exc)}


async def _run_testing(state: StoryState) -> StoryState:
    """Testing phase — Agents 24–38, Gates G5–G6."""
    try:
        state = await run_testing_phase(state)
        return {**state, "current_phase": "RELEASE"}
    except Exception as exc:
        return {**state, "current_phase": "BLOCKED", "block_reason": str(exc)}


async def _run_release(state: StoryState) -> StoryState:
    """Release phase — Agents 39–50, Gates G7–G12."""
    try:
        state = await run_release_phase(state)
        return {**state, "current_phase": "COMPLETE"}
    except Exception as exc:
        return {**state, "current_phase": "BLOCKED", "block_reason": str(exc)}


async def _mark_complete(state: StoryState) -> StoryState:
    return {**state, "current_phase": "COMPLETE"}


async def _mark_blocked(state: StoryState) -> StoryState:
    return {**state, "current_phase": "BLOCKED"}


# ── Entry point ───────────────────────────────────────────────────────────────

async def start_story_pipeline(
    story_id: str,
    fleet_commander: StateGraph,
) -> dict:
    """
    Initialise and start the pipeline for a new story.
    Returns immediately — graph runs asynchronously and checkpoints progress.
    """
    config = {"configurable": {"thread_id": story_id}}
    initial_state = initial_story_state(story_id)
    result = await fleet_commander.ainvoke(initial_state, config)
    return {"story_id": story_id, "phase": result["current_phase"]}


async def resume_story_pipeline(
    story_id: str,
    fleet_commander: StateGraph,
) -> dict:
    """
    Resume a story that was interrupted (e.g., awaiting CO sign-off).
    Called by the Redis pub/sub listener when a sign-off email is clicked.
    """
    config = {"configurable": {"thread_id": story_id}}
    result = await fleet_commander.ainvoke(None, config)
    return {"story_id": story_id, "phase": result["current_phase"]}
