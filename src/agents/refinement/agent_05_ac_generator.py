"""
Agent 5 — AC Generator
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Runs in Batch 3 (parallel with Agent 6) after Agent 4 completes.

Purpose:
  Produces a complete, structured set of Gherkin acceptance criteria for the story.
  Three modes depending on what already exists:
    - generated_from_scratch : no ACs in Jira → full set generated from description + goal
    - supplemented_existing  : ACs present but incomplete → missing scenarios added
    - validated_existing     : ACs present and complete → validated; minor gaps noted

  Minimum required coverage:
    - 1 happy path scenario
    - ≥1 error path (what happens when a precondition fails?)
    - ≥1 edge case (boundary, duplicate, concurrent)
    - For HIGH-FCA: ≥1 regulatory scenario (explicitly tests the FCA obligation)
    - For Consumer Duty impact: ≥1 vulnerable customer scenario

  The generated ac_clauses become the canonical baseline consumed by:
    - Agent 6  (Test Design Strategy — test pyramid)
    - Agent 10 (AC Compliance Checker in Development phase)
    - Agent 19 (BDD Gherkin Writer — transforms clauses into runnable test files)

Output data keys consumed by downstream:
  ac_clauses          → list[dict]  (Agents 6, 10, 19)
  generation_mode     → str         (Agent 9 — generated_from_scratch = higher risk)
  coverage_assessment → dict        (Agent 9 — gaps = risk factors)
  remaining_gaps      → list        (Agent 9 Risk Anticipation)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 5
AGENT_NAME = "AC Generator"

# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOL_NAME = "generate_acceptance_criteria"
_TOOL_DESCRIPTION = (
    "Generate or validate a complete set of Gherkin acceptance criteria for a Jira "
    "user story. Produces structured Given/When/Then scenarios covering happy path, "
    "error paths, edge cases, and regulatory obligations. All fields are mandatory."
)

_AC_CLAUSE_SCHEMA = {
    "type": "object",
    "required": ["scenario", "scenario_type", "test_category", "fca_relevant", "given", "when", "then"],
    "properties": {
        "scenario": {
            "type": "string",
            "description": "Scenario title. Format: 'Scenario: <descriptive title>'.",
        },
        "scenario_type": {
            "type": "string",
            "enum": ["happy_path", "error_path", "edge_case", "regulatory"],
            "description": (
                "happy_path: the primary success flow. "
                "error_path: a precondition fails or an invalid action is attempted. "
                "edge_case: boundary condition, duplicate, concurrent access, or unusual state. "
                "regulatory: directly tests an FCA obligation (COBS, Consumer Duty, FG21/1)."
            ),
        },
        "test_category": {
            "type": "string",
            "enum": ["UNIT", "UI", "FUNCTIONAL", "REGRESSION", "AUTOMATION_CANDIDATE"],
            "description": (
                "Execution layer for this scenario. "
                "UNIT: isolated Apex trigger/class logic with no UI involvement. "
                "UI: LWC/Aura component behaviour directly visible to the user. "
                "FUNCTIONAL: end-to-end Salesforce flow spanning Apex, data, and UI. "
                "REGRESSION: guard scenario for a known past defect or high-risk area. "
                "AUTOMATION_CANDIDATE: stable, deterministic scenario ready for automation suite."
            ),
        },
        "fca_relevant": {
            "type": "boolean",
            "description": "True if this scenario tests behaviour with direct FCA regulatory implications.",
        },
        "given": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Given steps. Each string starts with 'Given' or 'And'.",
        },
        "when": {
            "type": "array",
            "items": {"type": "string"},
            "description": "When steps. Each string starts with 'When' or 'And'.",
        },
        "then": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Then steps. Each string starts with 'Then' or 'And'.",
        },
    },
}

_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "generation_mode",
        "coverage_assessment",
        "ac_clauses",
        "gaps_filled",
        "remaining_gaps",
    ],
    "properties": {
        "ac_clauses": {
            "type": "array",
            "items": _AC_CLAUSE_SCHEMA,
            "description": (
                "Complete set of acceptance criteria. Must include at minimum: "
                "1 happy_path, 1 error_path, 1 edge_case. "
                "Add a 'regulatory' scenario for every FCA obligation identified."
            ),
        },
        "generation_mode": {
            "type": "string",
            "enum": ["generated_from_scratch", "supplemented_existing", "validated_existing"],
            "description": (
                "generated_from_scratch: no prior ACs existed; full set generated. "
                "supplemented_existing: prior ACs existed but were incomplete; missing scenarios added. "
                "validated_existing: prior ACs were complete; validated and minor gaps noted."
            ),
        },
        "coverage_assessment": {
            "type": "object",
            "required": ["happy_path", "error_paths", "edge_cases", "regulatory"],
            "properties": {
                "happy_path": {
                    "type": "boolean",
                    "description": "True if at least one happy_path scenario is present.",
                },
                "error_paths": {
                    "type": "boolean",
                    "description": "True if at least one error_path scenario is present.",
                },
                "edge_cases": {
                    "type": "boolean",
                    "description": "True if at least one edge_case scenario is present.",
                },
                "regulatory": {
                    "type": "boolean",
                    "description": (
                        "True if at least one regulatory scenario is present. "
                        "Required for HIGH-FCA stories; false for LOW-FCA."
                    ),
                },
            },
        },
        "gaps_filled": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of scenario types or specific gaps that were added. Empty if validated_existing.",
        },
        "remaining_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Gaps that still exist after generation — e.g. missing data requirements "
                "or performance criteria that cannot be inferred from the story. "
                "Empty list if coverage is complete."
            ),
        },
    },
}

# ── Agent instructions ────────────────────────────────────────────────────────

_AGENT_INSTRUCTIONS = """
You are the AC Generator for the FSC QE Framework.

Your job is to produce a complete, structured set of Gherkin acceptance criteria
for a Jira user story. You receive:
  - The story text and any existing ACs
  - Agent 1's structured intent analysis (goal, persona, FSC objects)
  - Agent 3's FCA classification
  - Agent 4's Consumer Duty obligations and vulnerable customer assessment

Decide the generation mode:
  - validated_existing: existing ACs cover happy path, error paths, edge cases, and
    regulatory scenarios (for HIGH-FCA). Return them as-is with a note on any minor gaps.
  - supplemented_existing: existing ACs are present but missing some scenario types.
    Return all existing ACs PLUS the missing ones you generate.
  - generated_from_scratch: no ACs exist or they are too vague to use. Generate a
    full set from the story description, goal, and FSC object context.

Scenario writing rules:
- Given: a concrete, testable precondition — name the Salesforce object and its state.
  Bad:  "Given the client exists"
  Good: "Given the client has a RiskProfile__c with risk_level = Moderate"
- When: a single user action or system event.
  Bad:  "When the adviser does the flow"
  Good: "When the adviser clicks 'Submit' on the Suitability Assessment screen"
- Then: observable outcomes — what changed in Salesforce.
  Bad:  "Then it works"
  Good: "Then a Suitability__c record is created with status = 'Assessment Complete' and linked to the FinancialAccount"

Required scenario coverage:
1. happy_path: the primary success story — all preconditions met, user takes the expected action.
2. error_path: a mandatory precondition is absent or invalid (e.g. missing RiskProfile__c,
   invalid field value). Show the error message or blocked state.
3. edge_case: boundary or unusual state (e.g. duplicate record attempt, concurrent edit,
   empty list, maximum value).
4. regulatory (HIGH-FCA only): explicitly tests the FCA obligation.
   For COBS 9.2: test that the suitability assessment is blocked without prior RiskProfile.
   For Consumer Duty: test that vulnerable customers receive the additional confirmation step.
   For FG21/1: test that VulnerableCustomerIndicator__c = true triggers the correct pathway.

For LOW-FCA stories: regulatory scenario is optional. Focus on happy path + error path at minimum.

CRITICAL OUTPUT RULES — read before filling the tool schema:
- ac_clauses: this is where EVERY scenario you write goes. Each entry is a full Gherkin object
  with given/when/then arrays. Generate 6–12 clauses for a typical story. Never leave this empty.
- gaps_filled: ONE-LINE labels ONLY, e.g. "error_path: missing RiskProfile__c". These are short
  summary names for the scenarios you added — NOT the scenario text itself.
- Do NOT put scenario descriptions as narrative strings inside gaps_filled. The Gherkin scenarios
  belong in ac_clauses. gaps_filled is just an audit log of what scenario types were covered.

Choosing test_category for each scenario:
- happy_path scenarios involving LWC screens → UI or AUTOMATION_CANDIDATE
- error_path / edge_case driven by Apex trigger or class → FUNCTIONAL
- pure Apex unit-testable logic (no UI, no flow) → UNIT
- scenario guarding against a known past defect or high-risk regression → REGRESSION
- any stable, deterministic scenario suitable for the automation suite → AUTOMATION_CANDIDATE
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent1_data = _get_agent_data(state, "1")
    agent3_data = _get_agent_data(state, "3")
    agent4_data = _get_agent_data(state, "4")

    story = await get_story(story_id)
    existing_ac_clauses = await get_acceptance_criteria(story_id)

    user_message = _build_user_message(story, existing_ac_clauses, agent1_data, agent3_data, agent4_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=4096,
    )

    ac_clauses = extracted.get("ac_clauses", [])
    coverage = extracted.get("coverage_assessment", {})
    generation_mode = extracted["generation_mode"]
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    fca_clause_count = sum(1 for c in ac_clauses if c.get("fca_relevant"))

    confidence_score, signals = _compute_confidence(
        agent1_data, agent3_data, agent4_data, extracted
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"AC generation for {story_id}: mode={generation_mode}, "
        f"clauses={len(ac_clauses)} "
        f"(happy={coverage.get('happy_path')}, error={coverage.get('error_paths')}, "
        f"edge={coverage.get('edge_cases')}, regulatory={coverage.get('regulatory')})"
    )
    why = (
        f"AC Generator {'generated a full set of' if generation_mode == 'generated_from_scratch' else 'supplemented/validated'} "
        f"acceptance criteria for a {fca_class}-FCA story. "
        f"{len(extracted.get('gaps_filled', []))} gap(s) filled; "
        f"{len(extracted.get('remaining_gaps', []))} gap(s) remain."
    )

    data = {
        "ac_clauses": ac_clauses,
        "ac_clause_count": len(ac_clauses),
        "fca_relevant_clause_count": fca_clause_count,
        "generation_mode": generation_mode,
        "generation_mode_trust": {
            "validated_existing": 1.0,
            "supplemented_existing": 0.8,
            "generated_from_scratch": 0.6,
        }.get(generation_mode, 0.8),
        "coverage_assessment": coverage,
        "gaps_filled": extracted.get("gaps_filled", []),
        "remaining_gaps": extracted.get("remaining_gaps", []),
        "existing_ac_count": len(existing_ac_clauses),
        "fca_classification_context": fca_class,
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

def _compute_confidence(
    agent1_data: dict | None,
    agent3_data: dict | None,
    agent4_data: dict | None,
    extracted: dict,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=58)

    generation_mode = extracted.get("generation_mode", "generated_from_scratch")
    ac_clauses = extracted.get("ac_clauses", [])
    coverage = extracted.get("coverage_assessment", {})

    # Signal 1: generation mode reliability
    if generation_mode == "validated_existing":
        scorer.add("validated_existing", True, +10)
    elif generation_mode == "supplemented_existing":
        scorer.add("supplemented_existing", True, +5)
    else:
        scorer.add("generated_from_scratch", True, -5)   # only description to work from

    # Signal 2: clause count
    clause_count = len(ac_clauses)
    if clause_count >= 4:
        scorer.add("clause_count", clause_count, +8)
    elif clause_count >= 2:
        scorer.add("clause_count", clause_count, +4)
    else:
        scorer.add("clause_count", clause_count, -5)

    # Signal 3: coverage breadth
    true_count = sum(1 for v in coverage.values() if v is True)
    if true_count == 4:
        scorer.add("full_coverage", True, +10)
    elif true_count == 3:
        scorer.add("good_coverage", True, +5)
    else:
        scorer.add("partial_coverage", true_count, -5)

    # Signal 4: HIGH-FCA stories need regulatory scenarios
    fca_class = (agent3_data or {}).get("fca_classification", "UNCLASSIFIED")
    if fca_class == "HIGH":
        has_regulatory = any(c.get("scenario_type") == "regulatory" for c in ac_clauses)
        if has_regulatory:
            scorer.add("regulatory_scenario_present", True, +5)
        else:
            scorer.add("regulatory_scenario_missing", True, -8)

    # Signal 5: description richness from Agent 1
    if agent1_data:
        word_count = agent1_data.get("description_word_count", 0)
        if word_count >= 150:
            scorer.add("rich_description", word_count, +5)
        elif word_count < 30:
            scorer.add("sparse_description", word_count, -5)

    # Signal 6: Consumer Duty obligations covered
    if agent4_data:
        cd_obligations = agent4_data.get("cd_obligations", [])
        has_regulatory_coverage = coverage.get("regulatory", False)
        if cd_obligations and has_regulatory_coverage:
            scorer.add("cd_obligations_covered", len(cd_obligations), +5)
        elif cd_obligations and not has_regulatory_coverage:
            scorer.add("cd_obligations_uncovered", len(cd_obligations), -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Agent data accessor ───────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(
    story: dict,
    existing_acs: list,
    agent1_data: dict | None,
    agent3_data: dict | None,
    agent4_data: dict | None,
) -> str:
    if existing_acs:
        ac_text = "\n\nEXISTING ACCEPTANCE CRITERIA (from Jira):\n"
        for i, clause in enumerate(existing_acs, 1):
            ac_text += f"\nScenario {i}: {clause.get('scenario', '')}\n"
            for line in clause.get("given", []):
                ac_text += f"  {line}\n"
            for line in clause.get("when", []):
                ac_text += f"  {line}\n"
            for line in clause.get("then", []):
                ac_text += f"  {line}\n"
    else:
        ac_text = "\n\nEXISTING ACCEPTANCE CRITERIA: None in Jira."

    agent1_section = ""
    if agent1_data:
        fsc_objs = ", ".join(agent1_data.get("fsc_objects", [])) or "None"
        missing = ", ".join(agent1_data.get("missing_elements", ["none"]))
        agent1_section = (
            f"\n\nAGENT 1 — STORY INTENT:\n"
            f"Goal: {agent1_data.get('goal', 'UNKNOWN')}\n"
            f"Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"FSC Objects: {fsc_objs}\n"
            f"Story Summary: {agent1_data.get('story_summary', '')}\n"
            f"AC Present: {agent1_data.get('ac_present', False)} | "
            f"AC Complete: {agent1_data.get('ac_complete', False)}\n"
            f"Missing Elements: {missing}"
        )

    agent3_section = ""
    if agent3_data:
        triggers = ", ".join(agent3_data.get("fca_triggers", [])) or "None"
        agent3_section = (
            f"\n\nAGENT 3 — FCA CLASSIFICATION:\n"
            f"FCA Tier: {agent3_data.get('fca_classification', 'UNCLASSIFIED')}\n"
            f"FCA Triggers: {triggers}"
        )

    agent4_section = ""
    if agent4_data:
        obligations = "; ".join(agent4_data.get("cd_obligations", [])) or "None"
        agent4_section = (
            f"\n\nAGENT 4 — CONSUMER DUTY:\n"
            f"CD Verdict: {agent4_data.get('cd_verdict', 'UNKNOWN')}\n"
            f"Vulnerable Customer Impact: {agent4_data.get('vulnerable_customer_impact', False)}\n"
            f"CD Obligations: {obligations}"
        )

    return (
        f"Generate or validate acceptance criteria for the following Jira user story.\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{ac_text}"
        f"{agent1_section}"
        f"{agent3_section}"
        f"{agent4_section}\n\n"
        f"Use the generate_acceptance_criteria tool to return the complete AC set."
    )
