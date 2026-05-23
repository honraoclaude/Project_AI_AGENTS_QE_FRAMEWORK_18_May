"""
Agent 55 — 3 Amigos Facilitator
Phase       : Refinement
PACT        : Collaborative
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=62)

Runs after Agent 09, before Gate G1.
Has access to Agents 01–09, 54.

Purpose:
  Synthesises all Refinement phase outputs (Agents 01–09, 54) into a single
  role-framed facilitation document for a Three Amigos session
  (BA / Developer / Tester / PO). Surfaces open questions, required decisions,
  actor-assigned action items, Definition of Done, regression impact assessment,
  and whether the story is ready to enter Development.

Output data keys consumed by downstream:
  ba_discussion_points          → list[str]   (scope, AC, CD obligation questions)
  developer_discussion_points   → list[str]   (technical complexity, spike, FSC deps)
  tester_discussion_points      → list[str]   (scenario gaps, regulatory, edge cases)
  open_questions                → list[str]   (unresolved before sprint start)
  recommended_decisions         → list[str]   (decisions the 3A session must make)
  story_ready_assessment        → str         (READY / NEEDS_DISCUSSION / BLOCKED)
  facilitator_summary           → str         (2–3 sentences for the facilitator)
  definition_of_done            → list[str]   (specific testable DoD criteria)
  action_items                  → list[dict]  (actor/action/priority — BA/DEV/QA/PO)
  regression_affected_areas     → list[str]   (FSC objects / features that could regress)
  regression_risk_level         → str         (LOW / MEDIUM / HIGH / CRITICAL)
  regression_notes              → str         (2–3 sentence mitigation recommendation)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 55
AGENT_NAME = "3 Amigos Facilitator"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_AMIGOS_TOOL_NAME = "facilitate_3_amigos"
_AMIGOS_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "ba_discussion_points", "developer_discussion_points",
        "tester_discussion_points", "open_questions",
        "recommended_decisions", "story_ready_assessment", "facilitator_summary",
        "definition_of_done", "action_items",
        "regression_affected_areas", "regression_risk_level", "regression_notes",
    ],
    "properties": {
        "ba_discussion_points": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "Open questions for the BA: AC gaps, Consumer Duty obligations, scope ambiguity. "
                "MUST be non-empty. If ac_count=0, list specific AC questions to answer in the session."
            ),
        },
        "developer_discussion_points": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "Technical questions for the Developer: FSC dependency complexity, spike needed, "
                "bulkification, data isolation strategy. MUST be non-empty."
            ),
        },
        "tester_discussion_points": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "Coverage questions for the Tester: missing scenario types, FCA regulatory scenarios, "
                "edge cases, Vulnerable Customer coverage. MUST be non-empty."
            ),
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unresolved questions the team must answer before sprint start.",
        },
        "recommended_decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific decisions the 3 Amigos session must reach (not 'discuss X' but 'decide X').",
        },
        "story_ready_assessment": {
            "type": "string",
            "enum": ["READY", "NEEDS_DISCUSSION", "BLOCKED"],
            "description": (
                "READY: INVEST PASS + no CRITICAL risks + ac_count > 0. "
                "BLOCKED: CRITICAL risks present OR INVEST FAIL. "
                "NEEDS_DISCUSSION: all other cases, including ac_count=0 with clean INVEST."
            ),
        },
        "facilitator_summary": {
            "type": "string",
            "description": "2–3 sentences framing the session agenda and key focus areas for the facilitator.",
        },
        "definition_of_done": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "Specific, testable DoD criteria for this story. Derive from FCA tier, Consumer Duty "
                "obligations, Vulnerable Customer impact, and AC coverage targets. Must include: "
                "Apex coverage threshold, CO sign-off if HIGH-FCA, VC scenario if vc_impact=True, "
                "FCA evidence pack verdict, and AC completeness. MUST be non-empty."
            ),
        },
        "action_items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["actor", "action", "priority"],
                "properties": {
                    "actor": {
                        "type": "string",
                        "enum": ["BA", "DEV", "QA", "PO"],
                        "description": "Role responsible: BA=Business Analyst, DEV=Developer, QA=Tester, PO=Product Owner.",
                    },
                    "action": {
                        "type": "string",
                        "description": "Specific actionable task to complete before or during the sprint.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["MUST", "SHOULD", "COULD"],
                        "description": "MUST=blocks sprint start, SHOULD=recommended, COULD=nice-to-have.",
                    },
                },
            },
            "description": (
                "Actor-assigned action items. Every open question and recommended decision must map "
                "to at least one action item with a named actor. MUST be non-empty."
            ),
        },
        "regression_affected_areas": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "FSC objects, features, or UI pages that could regress due to this change. "
                "MUST be non-empty — at minimum name the primary object touched."
            ),
        },
        "regression_risk_level": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "description": (
                "HIGH/CRITICAL when: destructive changes present, dependency_depth >= 2, "
                "or story touches shared FSC objects (FinancialAccount__c, Suitability__c, FinancialHolding__c)."
            ),
        },
        "regression_notes": {
            "type": "string",
            "description": "2–3 sentences on regression risk and recommended mitigation or regression suite scope.",
        },
    },
}

_AMIGOS_INSTRUCTIONS = """
You are the 3 Amigos Facilitator for a Salesforce FSC Wealth Management story
that has completed the Refinement phase analysis (Agents 01–09).

You receive phase-level outputs: story intent, INVEST quality score, FCA classification,
Consumer Duty obligations, acceptance criteria, test design, data needs, metadata
dependencies, and the risk register.

Generate a facilitation document structured for four roles:
1. BA (Business Analyst): AC gaps, Consumer Duty obligations, scope ambiguity, regulatory scope.
2. Developer: FSC dependencies, technical spikes, bulkification, data isolation strategy.
3. Tester (QA): Missing scenario types, FCA regulatory scenarios, Vulnerable Customer coverage, edge cases.
4. PO (Product Owner): Scope decisions, prioritisation, Definition of Done agreement.

MANDATORY RULES — apply to every story, no exceptions:

Rule 1 — ALWAYS populate all three role discussion lists. Never return empty arrays.
If ac_count = 0, the session's primary purpose IS to define ACs. In that case:
  BA must list: which business outcomes need AC coverage, acceptance thresholds to agree, regulatory scope.
  QA must list: which scenario types are required, what FCA/VC scenarios must be written before testing.
  DEV must list: technical constraints ACs must respect (governor limits, object access, data isolation).

Rule 2 — ALWAYS populate action_items with actor assignments.
Every open question and recommended decision must map to a concrete action item.
Assign the actor who owns it: BA owns AC definition and scope; DEV owns technical spikes and
data isolation; QA owns scenario writing and coverage; PO owns prioritisation and DoD sign-off.
MUST priority = blocks sprint start. Do not return empty action_items.

Rule 3 — ALWAYS produce definition_of_done with specific, testable criteria.
Include DoD items covering:
  - Apex test coverage threshold (HIGH-FCA: ≥90%, MEDIUM: ≥85%, LOW: ≥80%)
  - CO sign-off obtained at G1 (mandatory for HIGH-FCA)
  - Vulnerable Customer scenario present and tagged (if vc_impact=True)
  - All ACs have corresponding Gherkin scenarios
  - FCA evidence pack generated with COMPLETE or PARTIAL verdict
  - No CRITICAL risks in risk register at sprint start
Each DoD item must be measurable, not vague ("tested" is not a DoD item — name the threshold).

Rule 4 — ALWAYS populate regression_affected_areas, regression_risk_level, and regression_notes.
Identify which existing FSC features and objects could regress from this change.
Set regression_risk_level to HIGH or CRITICAL when: destructive metadata changes are present,
dependency_depth >= 2, or the story touches shared FSC objects used across features
(FinancialAccount__c, Suitability__c, FinancialHolding__c, RiskProfile__c).
regression_affected_areas must be non-empty — name a specific regression scope.
regression_notes should name the recommended mitigation suite (e.g. "run Financial Planning module regression suite").

story_ready_assessment rules:
- READY: INVEST verdict = PASS AND no CRITICAL risks AND ac_count > 0.
- BLOCKED: CRITICAL risks present OR INVEST verdict = FAIL.
- NEEDS_DISCUSSION: all other cases, including ac_count = 0 with a passing INVEST.

Output budget — distribute evenly across all 10 required fields. Do not write exhaustive
paragraphs in early fields at the expense of later ones. Target: 3–5 bullet points per
role list, 4–6 items for definition_of_done, 4–6 items for action_items, one concise
object for regression_impact_assessment. Complete all fields before elaborating on any one.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent1_data  = _get_agent_data(state, "1")
    agent2_data  = _get_agent_data(state, "2")
    agent3_data  = _get_agent_data(state, "3")
    agent4_data  = _get_agent_data(state, "4")
    agent5_data  = _get_agent_data(state, "5")
    agent6_data  = _get_agent_data(state, "6")
    agent7_data  = _get_agent_data(state, "7")
    agent8_data  = _get_agent_data(state, "8")
    agent9_data  = _get_agent_data(state, "9")
    agent54_data = _get_agent_data(state, "54")

    amigos_msg = _build_amigos_message(
        story_id, agent1_data, agent2_data, agent3_data, agent4_data,
        agent5_data, agent6_data, agent7_data, agent8_data, agent9_data, agent54_data,
    )
    result_data = await _run_amigos(amigos_msg)

    ba_points    = result_data.get("ba_discussion_points", [])
    dev_points   = result_data.get("developer_discussion_points", [])
    test_points  = result_data.get("tester_discussion_points", [])
    open_qs      = result_data.get("open_questions", [])
    decisions    = result_data.get("recommended_decisions", [])
    assessment   = result_data.get("story_ready_assessment", "NEEDS_DISCUSSION")
    summary      = result_data.get("facilitator_summary", "3 Amigos session facilitation complete.")
    dod          = result_data.get("definition_of_done", [])
    action_items = result_data.get("action_items", [])
    reg_areas    = result_data.get("regression_affected_areas", [])
    reg_risk     = result_data.get("regression_risk_level", "MEDIUM")
    reg_notes    = result_data.get("regression_notes", "")

    confidence_score, signals = _compute_confidence(
        agent2_data, agent9_data, agent3_data, open_qs, assessment,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"3 Amigos facilitation for {story_id}: "
        f"{len(open_qs)} open question(s), {len(action_items)} action item(s) — assessment={assessment}"
    )

    data = {
        "ba_discussion_points": ba_points,
        "developer_discussion_points": dev_points,
        "tester_discussion_points": test_points,
        "open_questions": open_qs,
        "recommended_decisions": decisions,
        "story_ready_assessment": assessment,
        "facilitator_summary": summary,
        "definition_of_done": dod,
        "action_items": action_items,
        "regression_affected_areas": reg_areas,
        "regression_risk_level": reg_risk,
        "regression_notes": reg_notes,
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=summary,
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
    agent2_data: dict | None,
    agent9_data: dict | None,
    agent3_data: dict | None,
    open_questions: list,
    assessment: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    invest_verdict = (agent2_data or {}).get("invest_verdict", "UNKNOWN")
    if invest_verdict == "PASS":
        scorer.add("invest_pass", True, +8)
    elif invest_verdict == "FAIL":
        scorer.add("invest_fail", True, -8)

    critical_risks = (agent9_data or {}).get("critical_risk_count", 0)
    if critical_risks == 0:
        scorer.add("no_critical_risks", True, +5)
    elif critical_risks >= 2:
        scorer.add("multiple_critical_risks", critical_risks, -10)
    else:
        scorer.add("one_critical_risk", critical_risks, -5)

    fca_class = (agent3_data or {}).get("fca_classification", "UNKNOWN")
    if fca_class == "HIGH":
        scorer.add("high_fca_scrutiny", True, -4)

    if len(open_questions) > 5:
        scorer.add("many_open_questions", len(open_questions), -5)

    if assessment == "READY":
        scorer.add("story_ready", True, +5)
    elif assessment == "BLOCKED":
        scorer.add("story_blocked", True, -10)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet call ───────────────────────────────────────────────────────────────

_REQUIRED_LIST_FIELDS = (
    "ba_discussion_points",
    "developer_discussion_points",
    "tester_discussion_points",
    "definition_of_done",
    "action_items",
    "regression_affected_areas",
)


async def _run_amigos(user_message: str) -> dict:
    result = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AMIGOS_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_AMIGOS_TOOL_NAME,
        tool_description="Generate a 3 Amigos facilitation document for this story.",
        tool_schema=_AMIGOS_TOOL_SCHEMA,
        max_tokens=4000,
    )

    empty = [f for f in _REQUIRED_LIST_FIELDS if not result.get(f)]
    if empty:
        field_list = ", ".join(empty)
        retry_msg = (
            f"{user_message}\n\n"
            f"RETRY: Your previous response returned empty arrays for: {field_list}. "
            f"You MUST populate ALL of these fields. Return the complete facilitation "
            f"document with every required field non-empty."
        )
        retry_result = await call_with_tool(
            model=settings.default_model,
            system=build_system(_AMIGOS_INSTRUCTIONS),
            user_message=retry_msg,
            tool_name=_AMIGOS_TOOL_NAME,
            tool_description="Generate a 3 Amigos facilitation document for this story.",
            tool_schema=_AMIGOS_TOOL_SCHEMA,
            max_tokens=4000,
        )
        for field in empty:
            if retry_result.get(field):
                result[field] = retry_result[field]

    return result


def _build_amigos_message(
    story_id: str,
    agent1_data: dict | None,
    agent2_data: dict | None,
    agent3_data: dict | None,
    agent4_data: dict | None,
    agent5_data: dict | None,
    agent6_data: dict | None,
    agent7_data: dict | None,
    agent8_data: dict | None,
    agent9_data: dict | None,
    agent54_data: dict | None,
) -> str:
    goal            = (agent1_data or {}).get("goal", "UNKNOWN")
    invest_verdict  = (agent2_data or {}).get("invest_verdict", "UNKNOWN")
    invest_score    = (agent2_data or {}).get("invest_score", 0)
    blocking_issues = (agent2_data or {}).get("blocking_issues", [])
    fca_class       = (agent3_data or {}).get("fca_classification", "UNKNOWN")
    co_required     = (agent3_data or {}).get("co_signoff_required", False)
    cd_verdict      = (agent4_data or {}).get("cd_verdict", "UNKNOWN")
    cd_obligations  = (agent4_data or {}).get("cd_obligations", [])
    vc_impact       = (agent4_data or {}).get("vulnerable_customer_impact", False)
    ac_count        = (agent5_data or {}).get("ac_clause_count", 0)
    remaining_gaps  = (agent5_data or {}).get("remaining_gaps", [])
    cov_target      = (agent6_data or {}).get("coverage_target_pct", 0)
    test_tools      = (agent6_data or {}).get("test_tools", [])
    data_vol        = (agent7_data or {}).get("data_volume", "UNKNOWN")
    data_isolation  = (agent7_data or {}).get("data_isolation_strategy", "UNKNOWN")
    dep_depth       = (agent8_data or {}).get("dependency_depth", 0)
    has_destructive = (agent8_data or {}).get("has_destructive_changes", False)
    risk_register   = (agent9_data or {}).get("risk_register", [])
    critical_count  = (agent9_data or {}).get("critical_risk_count", 0)
    overall_risk    = (agent9_data or {}).get("overall_risk_level", "UNKNOWN")
    adversarial     = (agent54_data or {}).get("adversarial_verdict", "")

    # Explicit warning when no ACs exist — drives Rule 1 enforcement
    ac_status = (
        "WARNING: 0 ACs written — defining ACs is the PRIMARY output of this session."
        if ac_count == 0
        else f"{ac_count} ACs generated"
    )

    blocking_block    = "\n".join(f"  - {b}" for b in blocking_issues) if blocking_issues else "  none"
    obligations_block = "\n".join(f"  - {o}" for o in cd_obligations) if cd_obligations else "  none"
    gaps_block        = "\n".join(f"  - {g}" for g in remaining_gaps) if remaining_gaps else "  none"
    risks_block       = (
        "\n".join(f"  [{r.get('severity','?')}] {r.get('description','')}" for r in risk_register[:5])
        if risk_register else "  none"
    )

    return (
        f"Story: {story_id}\n"
        f"Goal: {goal}\n\n"
        f"INVEST Quality: verdict={invest_verdict}, score={invest_score}/42\n"
        f"Blocking INVEST issues:\n{blocking_block}\n\n"
        f"FCA Classification: {fca_class}, CO sign-off required: {co_required}\n"
        f"Consumer Duty verdict: {cd_verdict}, Vulnerable Customer impact: {vc_impact}\n"
        f"CD Obligations:\n{obligations_block}\n\n"
        f"Acceptance Criteria: {ac_status}\n"
        f"Remaining AC gaps:\n{gaps_block}\n\n"
        f"Test Design: coverage target={cov_target}%, tools={test_tools}\n"
        f"Data Volume: {data_vol}, Data Isolation Strategy: {data_isolation}\n"
        f"Metadata Dependency Depth: {dep_depth}, Destructive Changes: {has_destructive}\n\n"
        f"Risk Register: overall={overall_risk}, critical count={critical_count}\n"
        f"{risks_block}\n\n"
        f"Adversarial Challenger verdict: {adversarial or 'not run'}\n\n"
        f"Generate the 3 Amigos facilitation document using the {_AMIGOS_TOOL_NAME} tool.\n"
        f"REMINDER: All fields are required and must be non-empty — including "
        f"definition_of_done, action_items (with actor assignments), "
        f"and regression_affected_areas (flat field — NOT nested in regression_impact_assessment)."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
