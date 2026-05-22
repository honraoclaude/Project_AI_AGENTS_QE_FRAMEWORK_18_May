"""
Agent 4 — Consumer Duty Mapper
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (structured signal scoring)

Runs sequentially after Agent 3 (FCA Classifier) — needs fca_classification
and fca_triggers before it can map Consumer Duty obligations.

Purpose:
  Maps the user story to the four FCA Consumer Duty outcomes (PS22/9):
    1. Products and Services — fitness for purpose for target market
    2. Price and Value       — customers receive fair value
    3. Consumer Understanding — communications are clear, not misleading
    4. Consumer Support       — customers can access support and act in interest

  Also applies FG21/1 (Vulnerable Customer Guidance) — flags if the story
  touches VulnerableCustomerIndicator__c or customer vulnerability pathways
  and whether the design adequately protects these customers.

  For LOW-FCA stories, Consumer Duty obligations are typically not applicable.
  The agent produces a short NOT_APPLICABLE verdict quickly in this case.

Output data keys consumed by downstream agents:
  cd_outcomes_affected    → list (Agent 9 Risk Anticipation, Agent 44 FCA Evidence Pack)
  vulnerable_customer_impact → bool (Agent 9, Release gates)
  cd_verdict              → str  (COMPLIANT / AT_RISK / NON_COMPLIANT / NOT_APPLICABLE)
  cd_obligations          → list (Agent 44 FCA Evidence Pack)
  cd_evidence_required    → list (Agent 44 — documents to produce for FCA audit pack)
  cd_risks                → list (Agent 9 Risk Anticipation)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 4
AGENT_NAME = "Consumer Duty Mapper"

# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOL_NAME = "map_consumer_duty"
_TOOL_DESCRIPTION = (
    "Map a Jira user story to FCA Consumer Duty outcomes (PS22/9) and vulnerable "
    "customer obligations (FG21/1). All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "cd_outcomes_affected",
        "vulnerable_customer_impact",
        "vulnerable_customer_rationale",
        "cd_obligations",
        "cd_risks",
        "cd_evidence_required",
        "cd_verdict",
        "cd_rationale",
    ],
    "properties": {
        "cd_outcomes_affected": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "products_and_services",
                    "price_and_value",
                    "consumer_understanding",
                    "consumer_support",
                    "none",
                ],
            },
            "description": (
                "Which of the four Consumer Duty outcomes this story affects. "
                "products_and_services: story changes what products/services clients can access. "
                "price_and_value: story changes fees, AUM calculations, or value delivery. "
                "consumer_understanding: story changes what clients see or are told. "
                "consumer_support: story changes client access to support or remedy processes. "
                "Use ['none'] only for LOW-FCA stories with no customer-facing impact."
            ),
        },
        "vulnerable_customer_impact": {
            "type": "boolean",
            "description": (
                "True if the story touches VulnerableCustomerIndicator__c, adds a Consumer Duty "
                "confirmation step, or changes any flow that handles vulnerable customer pathways."
            ),
        },
        "vulnerable_customer_rationale": {
            "type": "string",
            "description": (
                "One sentence explaining why vulnerable_customer_impact is True or False. "
                "If True, name the specific field or flow involved."
            ),
        },
        "cd_obligations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific Consumer Duty or FG21/1 obligations this story must satisfy. "
                "e.g. ['PS22/9 Outcome 1 — products must be fit for purpose for target market', "
                "'FG21/1 §4.3 — vulnerable customers must not be systematically disadvantaged']. "
                "Empty list if verdict is NOT_APPLICABLE."
            ),
        },
        "cd_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Consumer Duty risks identified in this story's current design. "
                "e.g. 'No error scenario for when Consumer Duty confirmation is bypassed'. "
                "Empty list if verdict is COMPLIANT or NOT_APPLICABLE."
            ),
        },
        "cd_evidence_required": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Evidence items the team must produce for the FCA audit pack. "
                "e.g. ['Screenshot of Consumer Duty confirmation step', "
                "'Test results showing vulnerable customer path is tested', "
                "'Impact assessment on price and value for existing clients']. "
                "Empty list if verdict is NOT_APPLICABLE."
            ),
        },
        "cd_verdict": {
            "type": "string",
            "enum": ["COMPLIANT", "AT_RISK", "NON_COMPLIANT", "NOT_APPLICABLE"],
            "description": (
                "COMPLIANT: story design satisfies all identified Consumer Duty obligations. "
                "AT_RISK: story may have gaps (missing test coverage, unclear ACs) — review recommended. "
                "NON_COMPLIANT: story design violates a specific Consumer Duty obligation. "
                "NOT_APPLICABLE: story is LOW-FCA with no customer-facing Consumer Duty impact."
            ),
        },
        "cd_rationale": {
            "type": "string",
            "description": (
                "Two to three sentences explaining the verdict. "
                "For AT_RISK or NON_COMPLIANT, name the specific obligation and gap."
            ),
        },
        "ui_or_support_touch": {
            "type": "boolean",
            "description": (
                "True if the story touches any client-facing UI, notification template, "
                "Screen Flow, LWC component, or Case/Task/ServiceAppointment objects — "
                "regardless of FCA tier. Triggers consumer_understanding and "
                "consumer_support outcome checks even for LOW-FCA stories."
            ),
        },
    },
}

# ── Agent instructions ────────────────────────────────────────────────────────

_AGENT_INSTRUCTIONS = """
You are the Consumer Duty Mapper for the FSC QE Framework.

Your job is to map a Jira user story to FCA Consumer Duty outcomes (PS22/9) and
vulnerable customer obligations (FG21/1). You receive the story text AND the FCA
classification already determined by Agent 3 (FCA Risk Classifier).

FCA Consumer Duty — four outcomes (PS22/9):
1. Products and Services: products must be fit for purpose for the target market.
   Relevant if story changes what products, accounts, or services clients can access.
   FSC objects: FinancialAccount, IndividualApplication, Suitability__c

2. Price and Value: clients must receive fair value; fees/charges must be proportionate.
   Relevant if story touches AUM calculations, Revenue__c, fee structures, or FinancialHolding.
   FSC objects: Revenue__c, FinancialHolding, AUM roll-ups

3. Consumer Understanding: communications must be clear, fair, and not misleading.
   Relevant if story changes adviser-facing or client-facing UI, disclosures, or notifications.
   FSC objects: Screen Flows, LWC components, notification templates

4. Consumer Support: customers must be able to access support and act in their interests.
   Relevant if story changes complaint handling, escalation paths, or remediation workflows.
   FSC objects: Case, Task, ServiceAppointment

Vulnerable Customer (FG21/1):
- Flag True if story touches VulnerableCustomerIndicator__c or any flow that handles
  customers flagged as vulnerable (physical, mental, financial, or life-event vulnerabilities).
- Key obligation: vulnerable customers must NOT be systematically disadvantaged by
  automated or digital processes. Any mandatory screen-flow step should have a bypass
  or enhanced-support alternative.

UI and Support touch — mandatory check regardless of FCA tier:
Before assigning NOT_APPLICABLE, check whether the story touches any of:
  - Screen Flows, LWC components, notification templates, or Email Templates
  - Case, Task, or ServiceAppointment objects
If ANY of these are present, set ui_or_support_touch=True and evaluate
consumer_understanding (Outcome 3) and consumer_support (Outcome 4) even for
LOW-FCA stories — these obligations apply to client-facing changes regardless of tier.

Verdict guidance:
- COMPLIANT: all CD outcomes are addressed, ACs cover the CD obligations, no gaps identified.
- AT_RISK: story touches a CD outcome but ACs are incomplete or a CD risk exists.
  Missing error/edge-case scenarios for vulnerable customer paths → AT_RISK.
- NON_COMPLIANT: story explicitly removes or disables a Consumer Duty control.
- NOT_APPLICABLE: story has no customer-facing or CD-relevant impact AND
  ui_or_support_touch=False. Do NOT assign NOT_APPLICABLE for LOW-FCA stories
  that touch UI components, notification templates, or support flows.

For HIGH-FCA stories: always check all four outcomes — a suitability story may
affect Products and Services AND Consumer Understanding if the assessment results
are shown to the client.

Use the map_consumer_duty tool to return your assessment.
""".strip()

# ── Vulnerable customer keywords for signal scoring ───────────────────────────

_VULNERABLE_CUSTOMER_KEYWORDS = {
    "vulnerablecustomer", "vulnerablecustomerindicator", "vulnerablecustomerindicator__c",
    "fg21", "fg21/1", "consumer duty", "consumer_duty", "ps22", "ps22/9",
}


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent1_data = _get_agent_data(state, "1")
    agent3_data = _get_agent_data(state, "3")

    story = await get_story(story_id)
    ac_clauses = await get_acceptance_criteria(story_id)

    user_message = _build_user_message(story, ac_clauses, agent1_data, agent3_data)

    extracted = await call_with_tool(
        model=settings.default_model,
        system=build_system(_AGENT_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TOOL_NAME,
        tool_description=_TOOL_DESCRIPTION,
        tool_schema=_TOOL_SCHEMA,
        max_tokens=2048,
    )

    confidence_score, signals = _compute_confidence(agent3_data, extracted)
    escalated = confidence_score < settings.confidence_escalation_threshold

    fca_class = (agent3_data or {}).get("fca_classification", state.get("fca_classification", "UNCLASSIFIED"))
    outcomes = extracted.get("cd_outcomes_affected", [])
    verdict = extracted.get("cd_verdict", "AT_RISK")

    what = (
        f"Consumer Duty mapping for {story_id}: verdict={verdict}, "
        f"outcomes={outcomes}, "
        f"vulnerable_customer={extracted.get('vulnerable_customer_impact', False)}"
    )
    why = (
        f"Consumer Duty Mapper applied FCA PS22/9 and FG21/1 to story classified as "
        f"{fca_class} by Agent 3. "
        f"Found {len(extracted.get('cd_obligations', []))} obligation(s) and "
        f"{len(extracted.get('cd_risks', []))} risk(s)."
    )

    data = {
        "cd_outcomes_affected": outcomes,
        "vulnerable_customer_impact": extracted.get("vulnerable_customer_impact", False),
        "vulnerable_customer_rationale": extracted.get("vulnerable_customer_rationale", ""),
        "ui_or_support_touch": extracted.get("ui_or_support_touch", False),
        "cd_obligations": extracted.get("cd_obligations", []),
        "cd_risks": extracted.get("cd_risks", []),
        "cd_evidence_required": extracted.get("cd_evidence_required", []),
        "cd_verdict": verdict,
        "cd_rationale": extracted.get("cd_rationale", ""),
        "fca_classification_from_agent3": fca_class,
        "agent3_available": agent3_data is not None,
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

def _compute_confidence(agent3_data: dict | None, extracted: dict) -> tuple[int, dict]:
    scorer = TierBScorer(base=55)

    if agent3_data:
        fca_class = agent3_data.get("fca_classification", "UNCLASSIFIED")

        # Signal 1: FCA tier drives how certain the CD mapping is
        if fca_class == "HIGH":
            scorer.add("fca_high", fca_class, +15)   # CD always relevant — confident
        elif fca_class == "MEDIUM":
            scorer.add("fca_medium", fca_class, +8)
        elif fca_class == "LOW":
            scorer.add("fca_low", fca_class, +10)    # NOT_APPLICABLE is an easy confident answer
        else:
            scorer.add("fca_unclassified", fca_class, -10)

        # Signal 2: Agent 3 ensemble agreed — classification is reliable
        if agent3_data.get("ensemble_agreement"):
            scorer.add("agent3_agreed", True, +5)
        else:
            scorer.add("agent3_disagreed", True, -5)

        # Signal 3: Agent 3 confidence — use triggers as proxy
        # (agent3_data is the data dict; triggers are the observable signal)
        triggers = agent3_data.get("fca_triggers", [])
        if len(triggers) >= 2:
            scorer.add("fca_triggers_present", len(triggers), +5)
        elif len(triggers) == 0 and fca_class in ("HIGH", "MEDIUM"):
            scorer.add("no_triggers_but_classified", True, -5)

        # Signal 4: Vulnerable customer explicitly detected in triggers
        triggers_lower = {t.lower() for t in triggers}
        if triggers_lower & _VULNERABLE_CUSTOMER_KEYWORDS:
            scorer.add("vulnerable_customer_in_triggers", True, +8)

    else:
        scorer.add("agent3_unavailable", True, -8)

    # Signal 5: NOT_APPLICABLE verdict is a confident answer only when no UI/support touch
    if extracted.get("cd_verdict") == "NOT_APPLICABLE":
        if extracted.get("ui_or_support_touch", False):
            # LOW-FCA + UI touch should not produce NOT_APPLICABLE — lower confidence
            scorer.add("not_applicable_but_ui_touch", True, -8)
        else:
            scorer.add("not_applicable_verdict", True, +5)

    # Signal 6: Many obligations listed → richer evidence → more confident
    obligations = len(extracted.get("cd_obligations", []))
    if obligations >= 2:
        scorer.add("obligations_present", obligations, +5)

    # Signal 7: Risks identified but no obligations → uncertain mapping
    risks = len(extracted.get("cd_risks", []))
    obligations_count = len(extracted.get("cd_obligations", []))
    if risks > 0 and obligations_count == 0:
        scorer.add("risks_without_obligations", True, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Agent data accessor ───────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(
    story: dict,
    ac_clauses: list,
    agent1_data: dict | None,
    agent3_data: dict | None,
) -> str:
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

    agent3_section = ""
    if agent3_data:
        triggers = ", ".join(agent3_data.get("fca_triggers", [])) or "None"
        obligations = "; ".join(agent3_data.get("regulatory_obligations", [])) or "None"
        agent3_section = (
            f"\n\nAGENT 3 FCA CLASSIFICATION:\n"
            f"FCA Tier: {agent3_data.get('fca_classification', 'UNCLASSIFIED')}\n"
            f"CO Sign-off Required: {agent3_data.get('co_signoff_required', False)}\n"
            f"FCA Triggers Found: {triggers}\n"
            f"Regulatory Obligations: {obligations}\n"
            f"Classification Rationale: {agent3_data.get('classification_rationale', '')}"
        )

    agent1_section = ""
    if agent1_data:
        fsc_objs = ", ".join(agent1_data.get("fsc_objects", [])) or "None identified"
        agent1_section = (
            f"\n\nAGENT 1 STORY INTENT:\n"
            f"Persona: {agent1_data.get('persona', 'UNKNOWN')}\n"
            f"FSC Objects: {fsc_objs}\n"
            f"Story Summary: {agent1_data.get('story_summary', '')}"
        )

    return (
        f"Map the following Jira user story to Consumer Duty outcomes (FCA PS22/9) "
        f"and vulnerable customer obligations (FG21/1).\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{ac_text}"
        f"{agent3_section}"
        f"{agent1_section}\n\n"
        f"Use the map_consumer_duty tool to return your assessment."
    )
