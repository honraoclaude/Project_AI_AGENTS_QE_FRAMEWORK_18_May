"""
Fleet Commander — top-level LangGraph stateful graph.

One graph instance per story, identified by story_id as the thread_id.
State is checkpointed to PostgreSQL after every node transition.
If the service restarts, graphs resume from their last checkpoint automatically.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from src.core.config import settings
from src.core.schemas import StoryState, initial_story_state
from src.fleet_commander.phases.refinement import build_refinement_subgraph


def build_fleet_commander(checkpointer: AsyncPostgresSaver) -> StateGraph:
    """
    Build the Fleet Commander graph with all phase subgraphs wired.
    Wave 1: Refinement subgraph only. Development/Testing/Release are stubs
    that will be filled in Waves 2–4.
    """
    main_graph = StateGraph(StoryState)

    # Compile the refinement subgraph and add as a node
    refinement_graph = build_refinement_subgraph().compile()
    main_graph.add_node("refinement", refinement_graph)

    # Wave 2–4 stubs — added here so the routing logic is in place
    main_graph.add_node("development", _development_stub)
    main_graph.add_node("testing", _testing_stub)
    main_graph.add_node("release", _release_stub)
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


async def _development_stub(state: StoryState) -> StoryState:
    """Wave 2 placeholder — Development phase agents and gates G2–G4."""
    return {**state, "current_phase": "TESTING"}


async def _testing_stub(state: StoryState) -> StoryState:
    """Wave 3 placeholder — Testing phase agents and gates G5–G6."""
    return {**state, "current_phase": "RELEASE"}


async def _release_stub(state: StoryState) -> StoryState:
    """Wave 4 placeholder — Release phase agents and gates G7–G12."""
    return {**state, "current_phase": "COMPLETE"}


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
