"""
Agent 9 — Risk Anticipation
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Runs sequentially after Batch 3 (Agents 5 and 6). Has access to all
upstream Refinement agent outputs (1–8).

Purpose:
  Synthesises all prior agent findings into a prioritised risk register.
  Reasons across five risk categories:
    regulatory    : FCA compliance failures (COBS 9.2, Consumer Duty, FG21/1)
    consumer_duty : specific Consumer Duty outcome gaps
    technical     : Apex/Flow bulkification, governor limits, dependency chains
    data          : test data setup order, isolation, sensitive data handling
    quality       : AC coverage gaps, test pyramid gaps, story clarity

  Risk severity:
    CRITICAL : would fail an FCA inspection or cause a production compliance incident.
               Gate G1 does NOT automatically block on CRITICAL risk alone —
               the Compliance Officer sign-off (HIGH-FCA) is the gate control.
               However, CRITICAL risks are flagged in the CO email.
    HIGH     : significant gap requiring Product Owner action before development.
    MEDIUM   : should be addressed in sprint; does not block gate.
    LOW      : informational; fix opportunistically.

  This is the final quality signal before Gate G1 evaluation.
  The Fleet Commander uses overall_risk_level and critical_risk_count in the
  CO sign-off email for HIGH-FCA stories.

Output data keys consumed by downstream:
  risk_register         → list  (Agent 44 FCA Evidence Pack, CO email context)
  overall_risk_level    → str   (Fleet Commander G1 email context)
  critical_risk_count   → int   (Fleet Commander G1 email context)
  recommended_actions   → list  (Jira comment posted by Fleet Commander)
"""

from __future__ import annotations

from src.agents.base import ShapleyAttributor, TierBScorer, build_system, call_with_tool, classify_ta_interaction, get_agent_result, _ta_mult
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_story

AGENT_ID = 9
AGENT_NAME = "Risk Anticipation"

# ── Tool schema ───────────────────────────────────────────────────────────────

_RISK_ITEM_SCHEMA = {
    "type": "object",
    "required": ["risk_id", "category", "description", "severity", "mitigation", "source_agent"],
    "properties": {
        "risk_id": {
            "type": "string",
            "description": "Sequential identifier. Format: 'R-001', 'R-002', ...",
        },
        "category": {
            "type": "string",
            "enum": ["regulatory", "consumer_duty", "technical", "data", "quality"],
        },
        "description": {
            "type": "string",
            "description": "One clear sentence describing the risk.",
        },
        "severity": {
            "type": "string",
            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        },
        "mitigation": {
            "type": "string",
            "description": "One concrete action the team can take to address this risk.",
        },
        "source_agent": {
            "type": "string",
            "description": "The upstream agent that surfaced this risk. e.g. 'agent_3', 'agent_4'.",
        },
    },
}

_TOOL_NAME = "assess_story_risks"
_TOOL_DESCRIPTION = (
    "Synthesise all upstream Refinement agent findings into a prioritised risk register "
    "for a Jira user story. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "critical_risk_count",
        "high_risk_count",
        "overall_risk_level",
        "risk_summary",
        "recommended_actions",
        "risk_register",
    ],
    "properties": {
        "risk_register": {
            "type": "array",
            "items": _RISK_ITEM_SCHEMA,
            "description": (
                "All identified risks, ordered CRITICAL → HIGH → MEDIUM → LOW. "
                "Each upstream agent gap or flag must appear as a risk entry. "
                "Minimum 1 entry — every story has at least one informational risk."
            ),
        },
        "critical_risk_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Count of CRITICAL severity risks.",
        },
        "high_risk_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Count of HIGH severity risks.",
        },
        "overall_risk_level": {
            "type": "string",
            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            "description": "Highest severity present in the risk register.",
        },
        "risk_summary": {
            "type": "string",
            "description": (
                "2–3 sentences summarising the risk posture for this story. "
                "Written to be included in the CO sign-off email for HIGH-FCA stories."
            ),
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Prioritised action list for the Product Owner and QE Lead. "
                "Written to be posted directly to Jira. Minimum 1 entry."
            ),
        },
    },
}

# ── Agent instructions ────────────────────────────────────────────────────────

_AGENT_INSTRUCTIONS = """
You are the Risk Anticipation Agent for the FSC QE Framework.

Your job is to synthesise findings from all prior Refinement agents (1–8) into a
prioritised risk register. You receive a structured summary of what each agent found.

Severity classification:
  CRITICAL : The gap would cause an FCA compliance failure or production incident.
             Examples: missing COBS 9.2 test evidence for a suitability change,
             Consumer Duty confirmation step not implemented for vulnerable customers.
  HIGH     : Significant quality gap requiring Product Owner action before development.
             Examples: no acceptance criteria for a HIGH-FCA story,
             missing error path for a regulated workflow.
  MEDIUM   : Should be addressed in sprint but does not block development.
             Examples: bulkification risk not covered in unit tests,
             test data insertion order not documented.
  LOW      : Informational — fix opportunistically.
             Examples: minor AC wording improvement, factory method not yet named.

Risk categories:
  regulatory    : FCA rule or obligation not addressed (COBS 9.2, Consumer Duty, FG21/1).
  consumer_duty : Specific Consumer Duty outcome gap (PS22/9 Outcome 1–4, FG21/1).
  technical     : Apex bulkification, governor limits, FSC dependency chain risk.
  data          : Test data setup order, isolation strategy, sensitive data masking.
  quality       : AC coverage gap, test pyramid gap, story INVEST quality issue.

Synthesis rules:
1. Every flag from Agent 1 (no_acceptance_criteria, vague_goal) → quality risk.
2. Agent 3 ensemble disagreement → regulatory risk (classification uncertain).
3. Agent 4 AT_RISK or NON_COMPLIANT verdict → consumer_duty risk.
4. Agent 4 vulnerable_customer_impact=True with no regulatory AC scenario (Agent 5) → CRITICAL consumer_duty risk.
5. Agent 5 remaining_gaps → quality risks (one per gap).
6. Agent 5 generated_from_scratch → quality risk (ACs are AI-generated, not PO-authored).
7. Agent 6 risk_areas → technical risks (one per area).
8. Agent 7 risks → data risks (one per risk).
9. Agent 8 dependency_depth ≥ 3 → technical risk (deep dependency chain).

Do not duplicate: if two agents flag the same issue, merge into one risk with both source agents noted.

Use the assess_story_risks tool to return your assessment.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    # Collect all upstream data + confidence scores (game theory: confidence-weighted synthesis)
    agent1_data, agent1_conf = get_agent_result(state, "1")
    agent2_data, agent2_conf = get_agent_result(state, "2")
    agent3_data, agent3_conf = get_agent_result(state, "3")
    agent4_data, agent4_conf = get_agent_result(state, "4")
    agent5_data, agent5_conf = get_agent_result(state, "5")
    agent6_data, agent6_conf = get_agent_result(state, "6")
    agent7_data, agent7_conf = get_agent_result(state, "7")
    agent8_data, agent8_conf = get_agent_result(state, "8")
    # Agent 05B (AC Challenger) — registered as id 54 in the registry
    agent5b_data, agent5b_conf = get_agent_result(state, "54")

    story = await get_story(story_id)

    user_message = _build_user_message(
        story,
        agent1_data, agent1_conf,
        agent2_data, agent2_conf,
        agent3_data, agent3_conf,
        agent4_data, agent4_conf,
        agent5_data, agent5_conf,
        agent6_data, agent6_conf,
        agent7_data, agent7_conf,
        agent8_data, agent8_conf,
        agent5b_data, agent5b_conf,
    )

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=2000,
    )

    confidence_score, signals = _compute_confidence(
        agent3_data, agent3_conf,
        agent4_data,
        agent5_data, agent5b_data,
        agent8_data,
        extracted,
        state,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold
    fca_class = (agent3_data or {}).get("fca_classification", state.get("fca_classification", "UNCLASSIFIED"))

    risk_register = extracted.get("risk_register", [])
    critical = extracted["critical_risk_count"]
    high = extracted["high_risk_count"]

    what = (
        f"Risk register for {story_id}: {len(risk_register)} risk(s) identified — "
        f"CRITICAL={critical}, HIGH={high}, "
        f"overall={extracted['overall_risk_level']}"
    )
    why = (
        f"Risk Anticipation Agent synthesised findings from {_count_available_agents(state)} "
        f"upstream Refinement agents for a {fca_class}-FCA story. "
        f"{len(extracted.get('recommended_actions', []))} recommended action(s) for the Product Owner."
    )

    data = {
        "risk_register": risk_register,
        "risk_count": len(risk_register),
        "critical_risk_count": critical,
        "high_risk_count": high,
        "overall_risk_level": extracted["overall_risk_level"],
        "risk_summary": extracted["risk_summary"],
        "recommended_actions": extracted.get("recommended_actions", []),
        "fca_classification_context": fca_class,
        "upstream_agents_available": _count_available_agents(state),
        "ta_interaction_summary": {
            aid: ("OK" if state["agent_results"].get(aid, {}).get(
                  "confidence", {}).get("final_score", 0) >= 60 else "NOT_OK")
            for aid in ["1", "2", "3", "4", "5", "54", "6", "8"]
        },
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
    agent3_data: dict | None,
    agent3_conf: int,
    agent4_data: dict | None,
    agent5_data: dict | None,
    agent5b_data: dict | None,
    agent8_data: dict | None,
    extracted: dict,
    state: StoryState,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    # Signal 1 (confidence-weighted): Agent 3 ensemble agreement weighted by its confidence.
    # High-confidence FCA classification agreement contributes more than a low-confidence one.
    if agent3_data:
        if agent3_data.get("ensemble_agreement"):
            weighted_bonus = round(8 * (agent3_conf / 100))
            scorer.add("agent3_agreed_weighted", agent3_conf, +weighted_bonus)
        else:
            ta_pos = agent3_data.get("ta_position", "NOT_OK_NOT_OK")
            _TA_PENALTY = {"OK_OK": -3, "OK_NOT_OK": -5, "NOT_OK_OK": -5, "NOT_OK_NOT_OK": -12}
            scorer.add("agent3_disagreed_ta", ta_pos, _TA_PENALTY.get(ta_pos, -5))
    else:
        scorer.add("agent3_unavailable", True, -8)

    # Signal 2: Agent 4 available → CD risks are grounded
    if agent4_data:
        scorer.add("agent4_available", True, +5)
        if agent4_data.get("cd_verdict") in ("AT_RISK", "NON_COMPLIANT"):
            scorer.add("cd_risk_confirmed", agent4_data["cd_verdict"], +3)

    # Signal 3: Agent 8 dependency depth → technical risk grounding
    depth = (agent8_data or {}).get("dependency_depth", 0)
    if depth >= 2:
        scorer.add("dependency_depth_rich", depth, +5)
    elif depth == 0 and agent8_data is not None:
        scorer.add("no_dependencies", 0, -2)

    # Signal 4: Agent 5 remaining gaps increase uncertainty
    remaining_gaps = len((agent5_data or {}).get("remaining_gaps", []))
    if remaining_gaps > 0:
        scorer.add("remaining_ac_gaps", remaining_gaps, -3)

    # Signal 5: Generated-from-scratch ACs → less grounded quality risk assessment
    if (agent5_data or {}).get("generation_mode") == "generated_from_scratch":
        scorer.add("acs_generated_not_authored", True, -5)

    # Signal 6: Rich risk register → thorough analysis
    risk_count = len(extracted.get("risk_register", []))
    if risk_count >= 3:
        scorer.add("rich_risk_register", risk_count, +5)
    elif risk_count == 0:
        scorer.add("empty_risk_register", 0, -10)

    # Signal 7: CRITICAL risks present → high severity clearly detected (confident)
    if extracted.get("critical_risk_count", 0) > 0:
        scorer.add("critical_risks_identified", extracted["critical_risk_count"], +5)

    # Signal 8 (adversarial): Agent 05B challenge — adjust for AC weakness findings
    if agent5b_data is not None:
        critical_weaknesses = agent5b_data.get("critical_weakness_count", 0)
        ac_count = agent5b_data.get("ac_count_challenged", 0)
        survivor_count = agent5b_data.get("survivor_count", ac_count)
        if critical_weaknesses == 0:
            scorer.add("ac_challenge_passed", True, +8)
        elif critical_weaknesses <= 2:
            scorer.add("ac_challenge_minor_issues", critical_weaknesses, 0)
        else:
            scorer.add("ac_challenge_failed", critical_weaknesses, -10)
        scorer.add("ac_survivor_count", survivor_count, 0)  # informational

    # Shapley attribution: fair credit with TA trust multiplier per agent
    attributor = ShapleyAttributor()
    c1  = state["agent_results"].get("1",  {}).get("confidence", {}).get("final_score", 0)
    c2  = state["agent_results"].get("2",  {}).get("confidence", {}).get("final_score", 0)
    c4  = state["agent_results"].get("4",  {}).get("confidence", {}).get("final_score", 0)
    c5  = state["agent_results"].get("5",  {}).get("confidence", {}).get("final_score", 0)
    c5b = state["agent_results"].get("54", {}).get("confidence", {}).get("final_score", 0)
    c6  = state["agent_results"].get("6",  {}).get("confidence", {}).get("final_score", 0)
    c8  = state["agent_results"].get("8",  {}).get("confidence", {}).get("final_score", 0)
    attributor.add_agent("1",  c1,         "1" in state["agent_results"], _ta_mult(c1))
    attributor.add_agent("2",  c2,         "2" in state["agent_results"], _ta_mult(c2))
    attributor.add_agent("3",  agent3_conf, agent3_data is not None,      _ta_mult(agent3_conf))
    attributor.add_agent("4",  c4,         agent4_data is not None,       _ta_mult(c4))
    attributor.add_agent("5",  c5,         agent5_data is not None,       _ta_mult(c5))
    attributor.add_agent("5b", c5b,        agent5b_data is not None,      _ta_mult(c5b))
    attributor.add_agent("6",  c6,         "6" in state["agent_results"], _ta_mult(c6))
    attributor.add_agent("8",  c8,         agent8_data is not None,       _ta_mult(c8))
    scorer.add("shapley_attributions", attributor.compute(), 0)  # informational — no score delta

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_available_agents(state: StoryState) -> int:
    return sum(1 for k in ["1", "2", "3", "4", "5", "6", "7", "8", "54"]
               if k in state["agent_results"])


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(
    story: dict,
    agent1_data: dict | None, agent1_conf: int,
    agent2_data: dict | None, agent2_conf: int,
    agent3_data: dict | None, agent3_conf: int,
    agent4_data: dict | None, agent4_conf: int,
    agent5_data: dict | None, agent5_conf: int,
    agent6_data: dict | None, agent6_conf: int,
    agent7_data: dict | None, agent7_conf: int,
    agent8_data: dict | None, agent8_conf: int,
    agent5b_data: dict | None, agent5b_conf: int,
) -> str:
    lines = [
        f"Synthesise the following upstream Refinement agent findings into a risk register.\n",
        f"STORY ID: {story['story_id']}",
        f"SUMMARY:  {story['summary']}\n",
        "NOTE: Each agent section shows its confidence score in brackets. "
        "Lower-confidence findings carry more uncertainty and should inform risk severity.\n",
    ]

    if agent1_data:
        lines.append(
            f"AGENT 1 — STORY INTENT [confidence: {agent1_conf}]:\n"
            f"  Goal: {agent1_data.get('goal', 'UNKNOWN')}\n"
            f"  Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"  FSC Objects: {', '.join(agent1_data.get('fsc_objects', [])) or 'None'}\n"
            f"  Flags: {', '.join(agent1_data.get('flags', ['none']))}\n"
            f"  Missing Elements: {', '.join(agent1_data.get('missing_elements', ['none']))}"
        )

    if agent2_data:
        lines.append(
            f"AGENT 2 — INVEST QUALITY [confidence: {agent2_conf}]:\n"
            f"  INVEST Score: {agent2_data.get('invest_score', 'N/A')}/100 "
            f"({agent2_data.get('invest_verdict', 'N/A')})\n"
            f"  Blocking Issues: {'; '.join(agent2_data.get('blocking_issues', [])) or 'None'}"
        )

    if agent3_data:
        lines.append(
            f"AGENT 3 — FCA CLASSIFICATION [confidence: {agent3_conf}]:\n"
            f"  Tier: {agent3_data.get('fca_classification', 'UNCLASSIFIED')}\n"
            f"  Ensemble Agreement: {agent3_data.get('ensemble_agreement', 'N/A')}\n"
            f"  Triggers: {', '.join(agent3_data.get('fca_triggers', [])) or 'None'}"
        )

    if agent4_data:
        lines.append(
            f"AGENT 4 — CONSUMER DUTY [confidence: {agent4_conf}]:\n"
            f"  Verdict: {agent4_data.get('cd_verdict', 'N/A')}\n"
            f"  Vulnerable Customer Impact: {agent4_data.get('vulnerable_customer_impact', False)}\n"
            f"  CD Risks: {'; '.join(agent4_data.get('cd_risks', [])) or 'None'}\n"
            f"  CD Obligations: {'; '.join(agent4_data.get('cd_obligations', [])) or 'None'}"
        )

    if agent5_data:
        coverage = agent5_data.get("coverage_assessment", {})
        lines.append(
            f"AGENT 5 — AC GENERATOR [confidence: {agent5_conf}]:\n"
            f"  Mode: {agent5_data.get('generation_mode', 'N/A')}\n"
            f"  Clause Count: {agent5_data.get('ac_clause_count', 0)}\n"
            f"  Coverage: happy={coverage.get('happy_path')}, error={coverage.get('error_paths')}, "
            f"edge={coverage.get('edge_cases')}, regulatory={coverage.get('regulatory')}\n"
            f"  Remaining Gaps: {'; '.join(agent5_data.get('remaining_gaps', [])) or 'None'}"
        )

    if agent5b_data:
        findings_count = len(agent5b_data.get("challenge_findings", []))
        lines.append(
            f"AGENT 5B — AC CHALLENGER [confidence: {agent5b_conf}]:\n"
            f"  Clauses Challenged: {agent5b_data.get('ac_count_challenged', 0)}\n"
            f"  Survivors (no critical/major weakness): {agent5b_data.get('survivor_count', 0)}\n"
            f"  Critical Weaknesses Found: {agent5b_data.get('critical_weakness_count', 0)}\n"
            f"  Total Findings: {findings_count}\n"
            f"  Summary: {agent5b_data.get('challenge_summary', 'N/A')}"
        )

    if agent6_data:
        lines.append(
            f"AGENT 6 — TEST DESIGN [confidence: {agent6_conf}]:\n"
            f"  Coverage Target: {agent6_data.get('coverage_target_pct', 'N/A')}%\n"
            f"  Risk Areas: {'; '.join(agent6_data.get('risk_areas', [])) or 'None'}"
        )

    if agent7_data:
        lines.append(
            f"AGENT 7 — DATA NEEDS [confidence: {agent7_conf}]:\n"
            f"  Data Volume: {agent7_data.get('data_volume', 'N/A')}\n"
            f"  Sensitive Data: {agent7_data.get('sensitive_data_present', False)}\n"
            f"  Data Risks: {'; '.join(agent7_data.get('risks', [])) or 'None'}"
        )

    if agent8_data:
        lines.append(
            f"AGENT 8 — DEPENDENCY MAPPING [confidence: {agent8_conf}]:\n"
            f"  Detected Objects: {', '.join(agent8_data.get('detected_objects', [])) or 'None'}\n"
            f"  Implied Objects: {', '.join(agent8_data.get('implied_objects', [])) or 'None'}\n"
            f"  Dependency Depth: {agent8_data.get('dependency_depth', 0)}"
        )

    lines.append("\nUse the assess_story_risks tool to return the risk register.")
    return "\n\n".join(lines)
