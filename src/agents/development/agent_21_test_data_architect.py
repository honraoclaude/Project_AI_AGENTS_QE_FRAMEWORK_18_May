"""
Agent 21 — Test Data Architect
Phase       : Development
PACT        : Proactive
Classification: True AI Agent (Claude Sonnet 4.6)
Confidence  : Tier B (base=68)

Runs in Batch 4 (parallel with Agent 20).
Has access to Agents 3, 5, 7, 13, 19.

Purpose:
  Designs the test data strategy for the story based on the metadata scope,
  FCA classification, Gherkin scenarios, and data needs identified during
  refinement (Agent 7). Specifies seed data records, anonymisation
  requirements, and Vulnerable Customer test profiles.

  This is a True AI agent — Sonnet 4.6 handles the full design.
  Structured output via tool use.

Output data keys consumed by downstream:
  test_data_strategy      → dict (Agent 27 CRT Execution setup)
  requires_anonymisation  → bool (FCA data protection compliance)
  vulnerable_profiles     → list (regulatory test coverage)
  data_verdict            → str  (PASS / WARN / INCOMPLETE)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 21
AGENT_NAME = "Test Data Architect"

# ── Sonnet tool ───────────────────────────────────────────────────────────────

_DATA_TOOL_NAME = "design_test_data_strategy"
_DATA_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "seed_records", "requires_anonymisation", "anonymisation_fields",
        "vulnerable_profiles", "data_verdict", "data_setup_notes", "coverage_gaps",
    ],
    "properties": {
        "seed_records": {
            "type": "array",
            "description": "Salesforce seed data records required for testing.",
            "items": {
                "type": "object",
                "required": ["object_name", "record_count", "key_fields", "purpose"],
                "properties": {
                    "object_name": {"type": "string"},
                    "record_count": {"type": "integer"},
                    "key_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "purpose": {"type": "string"},
                },
            },
        },
        "requires_anonymisation": {
            "type": "boolean",
            "description": "True if test data includes PII that must be anonymised.",
        },
        "anonymisation_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fields requiring anonymisation (e.g. Name, NI number, DOB).",
        },
        "vulnerable_profiles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Vulnerable Customer test profiles required (FCA Consumer Duty).",
        },
        "data_verdict": {
            "type": "string",
            "enum": ["PASS", "WARN", "INCOMPLETE"],
            "description": (
                "PASS: Full data strategy designed. "
                "WARN: Strategy partial — some gaps identified. "
                "INCOMPLETE: Cannot design strategy without more information."
            ),
        },
        "data_setup_notes": {
            "type": "string",
            "description": "Developer-facing notes on how to set up the test data.",
        },
        "coverage_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Test scenarios lacking a data strategy.",
        },
    },
}

_DATA_INSTRUCTIONS = """
You are a Salesforce FSC test data architect operating under FCA regulation.
Your task is to design a complete test data strategy for a development story.

Rules:
1. For every Gherkin scenario, specify the seed data records needed.
2. FCA-classified HIGH or MEDIUM stories must include Vulnerable Customer test profiles
   per Consumer Duty requirements.
3. Any record with PII (name, NI number, DOB, account balance) requires anonymisation.
4. Specify the exact Salesforce object, record count, and key field values.
5. If no Gherkin scenarios are available, design data based on the ACs and metadata scope.
6. If data requirements are unclear or missing, note them in coverage_gaps and set WARN.
7. Always consider data isolation — each test must not depend on data from other tests.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data = _get_agent_data(state, "3")
    agent5_data = _get_agent_data(state, "5")
    agent7_data = _get_agent_data(state, "7")
    agent13_data = _get_agent_data(state, "13")
    agent19_data = _get_agent_data(state, "19")

    story = await get_story(story_id)

    fca_class = (agent3_data or {}).get("fca_classification", "LOW")
    data_needs = (agent7_data or {}).get("required_records", [])
    gherkin_scenarios = (agent19_data or {}).get("gherkin_scenarios", [])
    objects_in_scope = (agent13_data or {}).get("detected_objects", [])
    ac_count = (agent5_data or {}).get("ac_count", 0)

    user_message = _build_prompt(
        story_id=story_id,
        story=story,
        fca_class=fca_class,
        data_needs=data_needs,
        gherkin_scenarios=gherkin_scenarios,
        objects_in_scope=objects_in_scope,
        ac_count=ac_count,
    )

    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_DATA_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_DATA_TOOL_NAME,
        tool_description="Design the test data strategy for this story.",
        tool_schema=_DATA_TOOL_SCHEMA,
        max_tokens=2000,
    )

    seed_records = result.get("seed_records", [])
    requires_anon = result.get("requires_anonymisation", False)
    anon_fields = result.get("anonymisation_fields", [])
    vulnerable_profiles = result.get("vulnerable_profiles", [])
    verdict = result.get("data_verdict", "INCOMPLETE")
    setup_notes = result.get("data_setup_notes", "")
    gaps = result.get("coverage_gaps", [])

    # REQ-05 Part 2: reconcile isolation strategy with FCA tier
    # Agent 07 ran before FCA classification was known; if HIGH/MEDIUM FCA was later confirmed
    # and Agent 07 recommended shared_org_data, override to per_test_setup_teardown.
    agent7_isolation = (agent7_data or {}).get("data_isolation_strategy", "per_class_setup")
    isolation_override = False
    isolation_override_reason = ""
    if fca_class in ("HIGH", "MEDIUM") and agent7_isolation == "shared_org_data":
        isolation_override = True
        isolation_override_reason = (
            f"Agent 07 recommended shared_org_data without FCA context; "
            f"overridden to per_test_setup_teardown for {fca_class}-FCA story"
        )

    confidence_score, signals = _compute_confidence(
        agent3_data, agent7_data, agent13_data, agent19_data,
        len(seed_records), verdict, fca_class, vulnerable_profiles,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Test data strategy for {story_id}: {len(seed_records)} seed record type(s), "
        f"anonymisation={'required' if requires_anon else 'not required'}, "
        f"{len(vulnerable_profiles)} vulnerable profile(s) — verdict={verdict}"
    )
    why = setup_notes or (
        f"Designed test data for a {fca_class}-FCA story touching "
        f"{objects_in_scope or ['undetermined']} objects."
    )

    data = {
        "test_data_strategy": {
            "seed_records": seed_records,
            "anonymisation_fields": anon_fields,
            "setup_notes": setup_notes,
        },
        "requires_anonymisation": requires_anon,
        "vulnerable_profiles": vulnerable_profiles,
        "data_verdict": verdict,
        "coverage_gaps": gaps,
        "seed_record_count": len(seed_records),
        "isolation_override": isolation_override,
        "isolation_override_reason": isolation_override_reason,
        "data_design_completeness": _compute_completeness(
            seed_records, fca_class, vulnerable_profiles, verdict, anon_fields
        ),
        "mechanism_signal": _build_mechanism_signal(
            seed_records, fca_class, vulnerable_profiles, verdict, anon_fields
        ),
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


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent3_data: dict | None,
    agent7_data: dict | None,
    agent13_data: dict | None,
    agent19_data: dict | None,
    seed_record_count: int,
    verdict: str,
    fca_class: str,
    vulnerable_profiles: list,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=68)

    if agent3_data:
        scorer.add("fca_classification_available", True, +5)

    if agent7_data:
        scorer.add("data_needs_from_refinement", True, +8)
    else:
        scorer.add("no_data_needs_baseline", 0, -5)

    if agent13_data:
        scorer.add("metadata_scope_available", True, +5)

    if agent19_data:
        scorer.add("gherkin_scenarios_available", True, +7)
    else:
        scorer.add("no_gherkin_scenarios", 0, -5)

    if seed_record_count > 0:
        scorer.add("seed_records_designed", seed_record_count, +5)
    else:
        scorer.add("no_seed_records", 0, -10)

    if fca_class in ("HIGH", "MEDIUM") and not vulnerable_profiles:
        scorer.add("regulated_story_missing_vulnerable_profiles", True, -8)

    if verdict == "INCOMPLETE":
        scorer.add("incomplete_strategy", True, -10)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    story_id: str,
    story: dict,
    fca_class: str,
    data_needs: list,
    gherkin_scenarios: list,
    objects_in_scope: list,
    ac_count: int,
) -> str:
    scenario_titles = [s.get("title", "") for s in gherkin_scenarios[:8]]
    scenario_block = (
        "\n".join(f"  - {t}" for t in scenario_titles)
        if scenario_titles
        else "  (no Gherkin scenarios available)"
    )
    return (
        f"Story ID: {story_id}\n"
        f"Title: {story.get('summary', 'N/A')}\n"
        f"FCA Classification: {fca_class}\n"
        f"Acceptance Criteria count: {ac_count}\n"
        f"Data requirements from refinement: {data_needs or ['not captured']}\n"
        f"Objects in scope: {objects_in_scope or ['not yet determined']}\n"
        f"Gherkin scenarios ({len(gherkin_scenarios)} available):\n"
        f"{scenario_block}\n\n"
        f"Design the test data strategy using the {_DATA_TOOL_NAME} tool."
    )


# ── Mechanism design helpers ──────────────────────────────────────────────────

def _compute_completeness(
    seed_records: list,
    fca_class: str,
    vulnerable_profiles: list,
    verdict: str,
    anon_fields: list,
) -> int:
    score = 100
    if not seed_records:
        score -= 30
    if fca_class in ("HIGH", "MEDIUM") and not vulnerable_profiles:
        score -= 25
    if not anon_fields:
        score -= 15
    if verdict == "INCOMPLETE":
        score -= 20
    return max(0, score)


def _build_mechanism_signal(
    seed_records: list,
    fca_class: str,
    vulnerable_profiles: list,
    verdict: str,
    anon_fields: list,
) -> dict:
    completeness = _compute_completeness(seed_records, fca_class, vulnerable_profiles, verdict, anon_fields)
    return {
        "vulnerable_profile_missing": fca_class in ("HIGH", "MEDIUM") and not vulnerable_profiles,
        "seed_records_missing": not seed_records,
        "downstream_penalty_active": completeness < 70,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
