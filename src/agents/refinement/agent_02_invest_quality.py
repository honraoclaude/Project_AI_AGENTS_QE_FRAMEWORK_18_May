"""
Agent 2 — INVEST Quality Agent
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Purpose:
  Scores the user story against the INVEST framework:
  Independent, Negotiable, Valuable, Estimable, Small, Testable.
  Each dimension is scored 0–20 by Claude; total 0–120 normalised to 0–100.

  invest_score < 80 → Gate G1 blocks the story from entering development.
  Improvement suggestions are written back to Jira by the Fleet Commander
  after G1 evaluation so the Product Owner can act on them.

Output data keys consumed by downstream components:
  invest_score          → int   (Fleet Commander G1 evaluator — primary gate signal)
  invest_verdict        → str   (PASS / PASS_WITH_CONCERNS / FAIL)
  dimension_scores      → dict  (per-dimension breakdown; audit trace + Agent 9 risk)
  blocking_issues       → list  (human-readable FAIL reasons; posted to Jira on BLOCK)
  improvement_suggestions → list (action items for PO)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 2
AGENT_NAME = "INVEST Quality Agent"

# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOL_NAME = "score_invest_quality"
_TOOL_DESCRIPTION = (
    "Score a Jira user story against the INVEST quality framework. "
    "Each of the six INVEST dimensions is scored 0–20. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "independent_score", "independent_rationale",
        "negotiable_score", "negotiable_rationale",
        "valuable_score", "valuable_rationale",
        "estimable_score", "estimable_rationale",
        "small_score", "small_rationale",
        "testable_score", "testable_rationale",
        "invest_verdict", "improvement_suggestions", "blocking_issues",
    ],
    "properties": {
        "independent_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = story has no dependencies on unfinished work and can be scheduled freely. "
                "0 = story is blocked by or tightly coupled to another in-flight story."
            ),
        },
        "independent_rationale": {
            "type": "string",
            "description": "One sentence explaining the Independent score.",
        },
        "negotiable_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = written as a goal with flexible implementation options. "
                "0 = reads like a rigid spec — 'the system shall ...' with prescribed UI layout."
            ),
        },
        "negotiable_rationale": {
            "type": "string",
            "description": "One sentence explaining the Negotiable score.",
        },
        "valuable_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = clear, stated business or regulatory value (e.g., COBS 9.2 compliance, "
                "client outcome). 0 = purely cosmetic or technical with no stated business benefit."
            ),
        },
        "valuable_rationale": {
            "type": "string",
            "description": "One sentence explaining the Valuable score.",
        },
        "estimable_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = scope is clear, FSC objects are named, complexity is bounded — "
                "a team could estimate in story points confidently. "
                "0 = 'investigate' or 'look into' phrasing; open-ended scope."
            ),
        },
        "estimable_rationale": {
            "type": "string",
            "description": "One sentence explaining the Estimable score.",
        },
        "small_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = fits comfortably in one two-week sprint. "
                "0 = clearly an epic — touches many objects or flows and must be split."
            ),
        },
        "small_rationale": {
            "type": "string",
            "description": "One sentence explaining the Small score.",
        },
        "testable_score": {
            "type": "integer", "minimum": 0, "maximum": 20,
            "description": (
                "20 = all ACs are in Given/When/Then format covering happy path, "
                "error paths, and at least one edge case. "
                "0 = no acceptance criteria at all."
            ),
        },
        "testable_rationale": {
            "type": "string",
            "description": "One sentence explaining the Testable score.",
        },
        "invest_verdict": {
            "type": "string",
            "enum": ["PASS", "PASS_WITH_CONCERNS", "FAIL"],
            "description": (
                "PASS = normalised score ≥ 80. "
                "PASS_WITH_CONCERNS = score 65–79. "
                "FAIL = score < 65."
            ),
        },
        "improvement_suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific, actionable suggestions for the Product Owner. "
                "Written to be posted directly to Jira. Empty list if verdict is PASS."
            ),
        },
        "blocking_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Issues that, if unresolved, will keep Gate G1 closed. "
                "Each entry is one clear sentence. Empty list if verdict is PASS."
            ),
        },
    },
}

# ── Agent instructions ────────────────────────────────────────────────────────

_AGENT_INSTRUCTIONS = """
You are the INVEST Quality Agent for the FSC QE Framework.

Your job is to score a Jira user story against the INVEST framework and identify
what must change for the story to enter development.

Scoring guide (each dimension 0–20):

Independent (I)
  20: story is self-contained; no unresolved dependencies on other open stories.
  10: minor dependency on a shared component (e.g., permission set) but manageable.
   0: story cannot start without completing FSC-XXXX first; explicitly states a blocker.

Negotiable (N)
  20: written as a user goal — 'so that [outcome]'; implementation approach left open.
  10: outcome is clear but some implementation detail is over-specified.
   0: reads like a technical spec with exact field names, UI positions, or SQL.

Valuable (V)
  20: directly addresses FCA/COBS obligation, client outcome, or adviser efficiency with evidence.
  10: some stated value but generic ('improve experience').
   0: cosmetic change with no stated regulatory or business benefit.
  Note: for FSC stories, citing a regulation (COBS 9.2, Consumer Duty) automatically
  increases Valuable — regulatory compliance IS business value.

Estimable (E)
  20: FSC objects named, complexity bounded, flow type identified — estimable in story points.
  10: objects implied but not named; some ambiguity.
   0: 'investigate', 'explore', or 'look into' phrasing; open-ended scope.

Small (S)
  20: changes one to two FSC objects; fits one sprint.
  10: three to four objects or two flows; tight but feasible.
   0: many objects, multiple flows, or 'and also...' repeated — should be an epic.

Testable (T)
  20: two or more GWT scenarios covering happy path + at least one error/edge case.
  12: scenarios present but incomplete (missing error or edge case).
   4: vague ACs ('system should work correctly').
   0: no acceptance criteria at all.

After scoring, use the score_invest_quality tool to return your full assessment.
""".strip()

# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent1_data = _get_agent1_data(state)

    story = await get_story(story_id)
    ac_clauses = await get_acceptance_criteria(story_id)

    user_message = _build_user_message(story, ac_clauses, agent1_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=1200,
    )

    invest_score = _get_invest_score(extracted)
    confidence_score, signals = _compute_confidence(agent1_data, extracted, invest_score)
    escalated = confidence_score < settings.confidence_escalation_threshold

    dimension_scores = {
        "independent": extracted.get("independent_score", 0),
        "negotiable":  extracted.get("negotiable_score", 0),
        "valuable":    extracted.get("valuable_score", 0),
        "estimable":   extracted.get("estimable_score", 0),
        "small":       extracted.get("small_score", 0),
        "testable":    extracted.get("testable_score", 0),
        "total_raw":   sum([
            extracted.get("independent_score", 0), extracted.get("negotiable_score", 0),
            extracted.get("valuable_score", 0),    extracted.get("estimable_score", 0),
            extracted.get("small_score", 0),       extracted.get("testable_score", 0),
        ]),
    }

    what = (
        f"INVEST score for {story_id}: {invest_score}/100 "
        f"({extracted.get('invest_verdict', 'UNKNOWN')}) — "
        f"I={extracted.get('independent_score', 0)} N={extracted.get('negotiable_score', 0)} "
        f"V={extracted.get('valuable_score', 0)} E={extracted.get('estimable_score', 0)} "
        f"S={extracted.get('small_score', 0)} T={extracted.get('testable_score', 0)}"
    )
    why = (
        "INVEST Quality Agent scored the story across six readiness dimensions. "
        "A normalised score < 80 triggers a Gate G1 block. "
        f"Found {len(extracted.get('blocking_issues', []))} blocking issue(s) "
        f"and {len(extracted.get('improvement_suggestions', []))} improvement suggestion(s)."
    )

    data = {
        "invest_score": invest_score,
        "invest_total_raw": dimension_scores["total_raw"],
        "invest_verdict": extracted.get("invest_verdict", "FAIL"),
        "dimension_scores": dimension_scores,
        "dimension_rationales": {
            "independent": extracted.get("independent_rationale", ""),
            "negotiable":  extracted.get("negotiable_rationale", ""),
            "valuable":    extracted.get("valuable_rationale", ""),
            "estimable":   extracted.get("estimable_rationale", ""),
            "small":       extracted.get("small_rationale", ""),
            "testable":    extracted.get("testable_rationale", ""),
        },
        "blocking_issues": extracted.get("blocking_issues", []),
        "improvement_suggestions": extracted.get("improvement_suggestions", []),
        "ac_clause_count": len(ac_clauses),
        "agent1_available": agent1_data is not None,
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=why,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="B",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.default_model,
    )


# ── INVEST score normalisation ────────────────────────────────────────────────

def _get_invest_score(extracted: dict) -> int:
    """Normalise raw sum (0–120) to 0–100."""
    total = (
        extracted.get("independent_score", 0) + extracted.get("negotiable_score", 0) +
        extracted.get("valuable_score", 0)    + extracted.get("estimable_score", 0) +
        extracted.get("small_score", 0)       + extracted.get("testable_score", 0)
    )
    return int(total * 100 / 120)


# ── Confidence scoring (Tier B) ───────────────────────────────────────────────

def _compute_confidence(
    agent1_data: dict | None,
    extracted: dict,
    invest_score: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    # Signal 1: Distance from threshold — clear outcomes are more confident
    margin = abs(invest_score - 80)
    if margin >= 15:
        scorer.add("invest_margin", margin, +10)   # ≥95 or ≤65: unambiguous
    elif margin >= 5:
        scorer.add("invest_margin", margin, +4)    # moderate clarity
    else:
        scorer.add("invest_margin", margin, -5)    # 75–84: borderline call

    # Signal 2: Testable score is the most observable dimension (ACs either exist or they don't)
    testable = extracted.get("testable_score", 0)
    if testable >= 14:
        scorer.add("testable_high", testable, +8)  # clear ACs → reliable T score
    elif testable <= 5:
        scorer.add("testable_low", testable, +5)   # clearly absent → reliable low score

    # Signals 3–6 sourced from Agent 1's pre-analysis
    if agent1_data:
        if agent1_data.get("ac_present"):
            scorer.add("ac_present", True, +8)
        else:
            scorer.add("ac_absent", True, -8)

        word_count = agent1_data.get("description_word_count", 0)
        if word_count >= 100:
            scorer.add("description_rich", word_count, +5)
        elif word_count < 30:
            scorer.add("description_sparse", word_count, -5)

        missing = agent1_data.get("missing_elements", [])
        if not missing or missing == ["none"]:
            scorer.add("no_missing_elements", 0, +5)
        elif len(missing) >= 3:
            scorer.add("many_missing_elements", len(missing), -8)
        else:
            scorer.add("some_missing_elements", len(missing), -3)
    else:
        scorer.add("agent1_unavailable", True, -5)

    # Signal 7: Many blocking issues → clear fail → more confident
    blocking = len(extracted.get("blocking_issues", []))
    if blocking >= 3:
        scorer.add("many_blocking_issues", blocking, +5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Agent 1 data accessor ─────────────────────────────────────────────────────

def _get_agent1_data(state: StoryState) -> dict | None:
    result = state["agent_results"].get("1")
    return result.get("data") if result else None


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(story: dict, ac_clauses: list, agent1_data: dict | None) -> str:
    if ac_clauses:
        ac_text = "\n\nACCEPTANCE CRITERIA:\n"
        for i, clause in enumerate(ac_clauses, 1):
            ac_text += f"\nScenario {i}: {clause.get('scenario', '')}\n"
            for line in clause.get("given", []):
                ac_text += f"  {line}\n"
            for line in clause.get("when", []):
                ac_text += f"  {line}\n"
            for line in clause.get("then", []):
                ac_text += f"  {line}\n"
    else:
        ac_text = "\n\nACCEPTANCE CRITERIA: None provided."

    agent1_section = ""
    if agent1_data:
        fsc_objs = ", ".join(agent1_data.get("fsc_objects", [])) or "None identified"
        missing  = ", ".join(agent1_data.get("missing_elements", ["none"]))
        flags    = ", ".join(agent1_data.get("flags", ["none"]))
        agent1_section = (
            f"\n\nAGENT 1 PRE-ANALYSIS (Story Intent Agent):\n"
            f"Goal: {agent1_data.get('goal', 'UNKNOWN')}\n"
            f"Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"FSC Objects: {fsc_objs}\n"
            f"Story Summary: {agent1_data.get('story_summary', '')}\n"
            f"AC Present: {agent1_data.get('ac_present', False)} | "
            f"AC Complete: {agent1_data.get('ac_complete', False)}\n"
            f"Missing Elements: {missing}\n"
            f"Quality Flags: {flags}"
        )

    return (
        f"Score the following Jira user story against the INVEST framework.\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"STATUS: {story['status']}\n"
        f"PRIORITY: {story.get('priority', 'Unknown')}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n"
        f"LABELS: {', '.join(story.get('labels', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{ac_text}"
        f"{agent1_section}\n\n"
        f"Score each INVEST dimension 0–20 using the score_invest_quality tool."
    )
