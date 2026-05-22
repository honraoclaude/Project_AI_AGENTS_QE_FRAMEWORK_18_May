"""
Agent 1 — Story Intent Agent
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Purpose:
  Parses a Jira story's summary, description, and acceptance criteria.
  Extracts: goal, persona, FSC object scope, and AC completeness.
  Flags missing elements. Output feeds Agents 2, 3, 7 (all consume this
  structured understanding of the story's intent).

Output data keys consumed by downstream agents:
  goal         → str   (used by Agent 2 for INVEST scoring)
  persona      → str   (used by Agent 3 for FCA classification context)
  fsc_objects  → list  (used by Agent 3, Agent 7, Agent 8)
  ac_present   → bool  (used by Agent 2 for Testable score)
  ac_clauses   → list  (used by Agent 5 AC Generator as baseline)
  flags        → list  (used by Agent 9 Risk Anticipation)
"""

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 1
AGENT_NAME = "Story Intent Agent"

# ── Tool schema — defines the structured output Claude must produce ────────────

_TOOL_NAME = "extract_story_intent"
_TOOL_DESCRIPTION = (
    "Extract the structured intent of a Jira story for the FSC QE pipeline. "
    "Every field is mandatory. Use 'UNKNOWN' when information cannot be determined."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "goal", "persona", "fsc_objects", "fsc_components",
        "ac_present", "ac_complete", "missing_elements",
        "story_summary", "flags",
    ],
    "properties": {
        "goal": {
            "type": "string",
            "description": "One sentence: what business outcome this story achieves.",
        },
        "persona": {
            "type": "string",
            "description": (
                "Primary user persona. One of: Wealth Adviser, Client/Investor, "
                "Compliance Officer, Operations/Admin, QE Engineer, Product Owner, UNKNOWN."
            ),
        },
        "fsc_objects": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Salesforce FSC objects this story touches or creates. "
                "e.g. ['FinancialAccount', 'Suitability__c', 'Goal__c']. "
                "Empty list if no FSC objects are involved."
            ),
        },
        "fsc_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Salesforce platform components involved. "
                "e.g. ['Apex trigger on FinancialAccount', 'Screen Flow', 'LWC suitability-form']. "
                "Infer from description where explicit names are not given."
            ),
        },
        "ac_present": {
            "type": "boolean",
            "description": "True if acceptance criteria are present in the story.",
        },
        "ac_complete": {
            "type": "boolean",
            "description": (
                "True if all AC have Given/When/Then structure AND cover "
                "happy path, error paths, and edge cases."
            ),
        },
        "missing_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Story elements that are absent or unclear. "
                "Possible values: 'acceptance_criteria', 'persona', 'goal', "
                "'error_scenarios', 'edge_cases', 'data_requirements', 'none'."
            ),
        },
        "story_summary": {
            "type": "string",
            "description": (
                "2-3 sentence plain-English summary of the story for downstream agents "
                "that may not re-read the full Jira text."
            ),
        },
        "flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Quality concerns. Possible values: 'no_acceptance_criteria', "
                "'vague_goal', 'missing_persona', 'incomplete_gherkin', "
                "'high_fca_object_detected', 'financial_data_object_detected', "
                "'no_fsc_objects', 'none'."
            ),
        },
    },
}

# ── Agent instructions ────────────────────────────────────────────────────────

_AGENT_INSTRUCTIONS = """
You are the Story Intent Agent for the FSC QE Framework.

Your job is to parse a Jira user story and extract its structured intent.
Be precise and conservative:
- Only claim an FSC object is involved if the story text or ACs explicitly mention it
  or clearly imply a change to it.
- Flag 'high_fca_object_detected' if you see: Suitability, RiskProfile, Appropriateness,
  VulnerableCustomer, or Consumer Duty in the story text or ACs.
- Flag 'financial_data_object_detected' if you see: FinancialAccount, Goal, AUM,
  FinancialHolding, AssetsAndLiabilities in the story text or ACs.
- If the description is very short (under 30 words) or the goal is unclear, flag 'vague_goal'.
- You must call the extract_story_intent tool with your findings.
""".strip()

# ── FSC object lists for signal scoring ───────────────────────────────────────

_HIGH_FCA_OBJECTS = {
    "suitability", "suitability__c", "riskprofile", "riskprofile__c",
    "appropriateness", "appropriateness__c", "vulnerablecustomer",
    "vulnerablecustomerindicator__c", "consumer duty", "cobs",
}
_FINANCIAL_OBJECTS = {
    "financialaccount", "financialgoal", "goal__c", "financialholding",
    "assetsandliabilities", "aum", "financialaccounttransaction", "revenue__c",
}


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    # Fetch story and ACs from Jira
    story = await get_story(story_id)
    ac_clauses = await get_acceptance_criteria(story_id)

    # Build user message for Claude
    user_message = _build_user_message(story, ac_clauses)

    # Call Claude with structured output tool
    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=800,
    )

    # Compute Tier B confidence from observable signals
    confidence_score, signals = _compute_confidence(story, ac_clauses, extracted)

    # Determine escalation
    escalated = confidence_score < settings.confidence_escalation_threshold

    # Compose What/Why/Data
    what = (
        f"Extracted intent for {story_id}: goal='{extracted.get('goal', 'UNKNOWN')}', "
        f"persona='{extracted.get('persona', 'UNKNOWN')}', "
        f"fsc_objects={extracted.get('fsc_objects', [])}, "
        f"ac_present={extracted.get('ac_present', False)}"
    )
    why = (
        "Story Intent Agent parsed Jira description and acceptance criteria to produce "
        "structured intent consumed by INVEST Quality (Agent 2), FCA Classifier (Agent 3), "
        "and Data Need (Agent 7)."
    )

    data = {
        **extracted,
        "story_summary_jira": story["summary"],
        "description_word_count": len((story["description"] or "").split()),
        "ac_clause_count": len(ac_clauses),
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


# ── Confidence scoring (Tier B) ───────────────────────────────────────────────

def _compute_confidence(story: dict, ac_clauses: list, extracted: dict) -> tuple[int, dict]:
    description = story.get("description") or ""
    word_count = len(description.split())
    combined_text = (description + " " + story.get("summary", "")).lower()

    scorer = TierBScorer(base=55)

    # Signal 1: Description length
    if word_count >= 150:
        scorer.add("description_words", word_count, +15)
    elif word_count >= 50:
        scorer.add("description_words", word_count, +8)
    elif word_count >= 20:
        scorer.add("description_words", word_count, +2)
    else:
        scorer.add("description_words", word_count, -20)

    # Signal 2: Acceptance criteria
    if ac_clauses and extracted.get("ac_complete"):
        scorer.add("ac_complete", True, +15)
    elif ac_clauses:
        scorer.add("ac_present_incomplete", True, +7)
    else:
        scorer.add("ac_absent", True, -10)

    # Signal 3: FSC object identification
    fsc_objects = extracted.get("fsc_objects", [])
    if len(fsc_objects) >= 2:
        scorer.add("fsc_objects_count", len(fsc_objects), +10)
    elif len(fsc_objects) == 1:
        scorer.add("fsc_objects_count", len(fsc_objects), +5)

    # Signal 4: Persona identified
    persona = extracted.get("persona", "UNKNOWN")
    if persona != "UNKNOWN":
        scorer.add("persona_identified", persona, +5)

    # Signal 5: HIGH-FCA keyword in raw text (increases certainty of the extraction)
    high_fca_hit = any(kw in combined_text for kw in _HIGH_FCA_OBJECTS)
    if high_fca_hit:
        scorer.add("high_fca_keyword_in_text", True, +5)

    # Signal 6: Model flagged vague goal — reduce confidence
    if "vague_goal" in extracted.get("flags", []):
        scorer.add("vague_goal_flag", True, -10)

    # Signal 7: No FSC objects at all — extraction is less grounded
    if not fsc_objects:
        scorer.add("no_fsc_objects", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(story: dict, ac_clauses: list) -> str:
    ac_text = ""
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
        ac_text = "\n\nACCEPTANCE CRITERIA: None found in story."

    return f"""Analyse the following Jira story and extract its intent.

STORY ID: {story['story_id']}
SUMMARY: {story['summary']}
STATUS: {story['status']}
COMPONENTS: {', '.join(story.get('components', [])) or 'None'}
LABELS: {', '.join(story.get('labels', [])) or 'None'}

DESCRIPTION:
{story['description'] or '(empty)'}
{ac_text}

Extract the story intent using the extract_story_intent tool."""
