"""
Agent 3 — FCA Risk Classifier
Phase       : Refinement
PACT        : Proactive
Classification: True AI (Sonnet 4.6) — Tier D (Ensemble)
Confidence  : Tier D — two independent Claude calls with different framings.
              Agreement → 78–85 confidence (tier-dependent).
              Disagreement → 38–48 confidence; always auto-escalated to QE Lead.
              "Safer call wins": higher-risk tier always prevails.

FCA Tiers:
  HIGH         : directly modifies Suitability, Appropriateness, RiskProfile,
                 VulnerableCustomer, or Consumer Duty journeys.
                 Requires CO sign-off before Gate G1 opens.
  MEDIUM       : modifies FinancialAccount, Goals, AUM calculations, or the
                 permission model without touching HIGH-FCA objects.
                 85% Apex coverage target applies.
  LOW          : standard platform change — UI labels, non-financial metadata,
                 admin tooling, report adjustments.
  UNCLASSIFIED : insufficient context to classify. Gate G1 blocks.

Output data keys consumed by downstream:
  fca_classification        → str   (Fleet Commander G1; state["fca_classification"])
  co_signoff_required       → bool  (G1 CO interrupt trigger)
  enhanced_testing_required → bool  (Development phase coverage thresholds)
  fca_triggers              → list  (Agent 4 Consumer Duty Mapper, Agent 9 Risk)
  regulatory_obligations    → list  (Agent 4)
  ensemble_agreement        → bool  (audit trace — disagreement signals manual review)
  call_a_classification     → str   (audit trace)
  call_b_classification     → str   (audit trace)
"""

from __future__ import annotations

import asyncio

from src.agents.base import build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState
from src.integrations.jira import get_acceptance_criteria, get_story

AGENT_ID = 3
AGENT_NAME = "FCA Risk Classifier"

# ── Tool schema (shared by both ensemble calls) ───────────────────────────────

_TOOL_NAME = "classify_fca_risk"
_TOOL_DESCRIPTION = (
    "Classify the FCA regulatory risk tier of a Jira user story for an FSC wealth "
    "management platform. All fields are mandatory."
)
_TOOL_SCHEMA = {
    "type": "object",
    "required": [
        "fca_classification",
        "classification_rationale",
        "fca_triggers",
        "regulatory_obligations",
        "co_signoff_required",
        "enhanced_testing_required",
    ],
    "properties": {
        "fca_classification": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW", "UNCLASSIFIED"],
            "description": (
                "HIGH: story directly touches Suitability, Appropriateness, RiskProfile, "
                "VulnerableCustomer, or Consumer Duty journeys. "
                "MEDIUM: touches FinancialAccount, Goals, AUM, FinancialHolding, "
                "AssetsAndLiabilities, Revenue, or permission model. "
                "LOW: cosmetic, administrative, or non-financial metadata change. "
                "UNCLASSIFIED: insufficient context to determine."
            ),
        },
        "classification_rationale": {
            "type": "string",
            "description": (
                "Two to three sentences explaining the classification. "
                "Cite the specific object names, field names, or regulatory references "
                "that drove the decision."
            ),
        },
        "fca_triggers": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of specific FSC objects, field names, or regulatory keywords found "
                "in the story that determined the tier. "
                "e.g. ['Suitability__c', 'COBS 9.2', 'VulnerableCustomerIndicator__c']. "
                "Empty list for LOW or UNCLASSIFIED."
            ),
        },
        "regulatory_obligations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "FCA regulations this story must comply with. "
                "e.g. ['COBS 9.2 Suitability', 'Consumer Duty PS22/9', 'FG21/1 Vulnerable Customers']. "
                "Empty list for LOW tier."
            ),
        },
        "co_signoff_required": {
            "type": "boolean",
            "description": "True only for HIGH-FCA. Compliance Officer must sign off before Gate G1 opens.",
        },
        "enhanced_testing_required": {
            "type": "boolean",
            "description": "True for HIGH and MEDIUM. 85% Apex coverage target applies.",
        },
    },
}

# ── Agent instructions — two framings to reduce anchoring ────────────────────

_INSTRUCTIONS_CAUTIOUS = """
You are a cautious FCA compliance specialist reviewing a Salesforce FSC story.

Your mandate: protect the firm from regulatory risk. When uncertain, classify UP.

Classification rules (apply strictly in order — stop at the first match):
1. HIGH: story text or ACs mention ANY of these objects or concepts:
   Suitability, SuitabilityAssessment, Suitability__c, RiskProfile, RiskProfile__c,
   Appropriateness, Appropriateness__c, VulnerableCustomer, VulnerableCustomerIndicator__c,
   Consumer Duty, COBS 9.2, COBS 10, FG21/1, IndividualApplication (suitability context)
2. MEDIUM: story touches FinancialAccount, FinancialGoal, Goal__c, FinancialHolding,
   AssetsAndLiabilities, AUM, Revenue__c, FinancialAccountTransaction, or modifies
   Permission Sets / Profiles / Roles that control financial object access.
3. LOW: purely cosmetic, administrative, or non-financial metadata.
4. UNCLASSIFIED: description is too sparse to determine (< 15 meaningful words).

If the story is near a tier boundary, classify UP to the higher tier.
""".strip()

_INSTRUCTIONS_EVIDENCE_BASED = """
You are a pragmatic FCA compliance specialist reviewing a Salesforce FSC story.

Your mandate: classify based strictly on explicit evidence in the story text.
Do not infer regulatory impact that is not stated. Do not classify HIGH unless
the story text or ACs explicitly mention HIGH-FCA objects or regulations.

Classification rules (apply strictly in order — stop at the first match):
1. HIGH: story text or ACs EXPLICITLY name: Suitability__c, SuitabilityAssessment,
   RiskProfile__c, Appropriateness__c, VulnerableCustomerIndicator__c, Consumer Duty,
   COBS 9.2, COBS 10, FG21/1. Implicit references (e.g. 'regulatory change' without
   naming the regulation) do NOT qualify for HIGH.
2. MEDIUM: story EXPLICITLY names FinancialAccount, FinancialGoal, Goal__c,
   FinancialHolding, AssetsAndLiabilities, AUM, Revenue__c, or modifies
   Permission Sets / Profiles that control financial record access.
3. LOW: story is a UI change, label update, report adjustment, or other change that
   does not touch any of the above objects.
4. UNCLASSIFIED: description is genuinely too sparse (< 15 meaningful words) or
   contradictory to classify.

Only classify HIGH if the evidence is explicit. Do not up-classify speculatively.
""".strip()

# ── Tier ordering for "safer call wins" logic ─────────────────────────────────

_TIER_ORDER: dict[str, int] = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "UNCLASSIFIED": 0,
}

_AGREEMENT_CONFIDENCE: dict[str, int] = {
    "HIGH": 85,
    "MEDIUM": 80,
    "LOW": 78,
    "UNCLASSIFIED": 55,
}


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    agent1_data = _get_agent_data(state, "1")

    story = await get_story(story_id)
    ac_clauses = await get_acceptance_criteria(story_id)

    user_message = _build_user_message(story, ac_clauses, agent1_data)

    # Two independent calls in parallel — Tier D ensemble
    call_a_result, call_b_result = await asyncio.gather(
        call_with_tool(
            model=settings.default_model,
            system=build_system(_INSTRUCTIONS_CAUTIOUS),
            user_message=user_message,
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
            tool_schema=_TOOL_SCHEMA,
            max_tokens=600,
        ),
        call_with_tool(
            model=settings.default_model,
            system=build_system(_INSTRUCTIONS_EVIDENCE_BASED),
            user_message=user_message,
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
            tool_schema=_TOOL_SCHEMA,
            max_tokens=600,
        ),
    )

    classification, confidence_score, signals = _resolve_ensemble(call_a_result, call_b_result)
    escalated = confidence_score < settings.confidence_escalation_threshold

    # Use the more conservative call's detail fields for the audit trace
    primary = _pick_primary_call(call_a_result, call_b_result, classification)

    co_required = classification == "HIGH"
    enhanced_testing = classification in ("HIGH", "MEDIUM")

    what = (
        f"FCA classification for {story_id}: {classification} "
        f"(call-A={call_a_result['fca_classification']}, "
        f"call-B={call_b_result['fca_classification']}, "
        f"agreement={signals['ensemble_agreement']})"
    )
    why = (
        "FCA Risk Classifier ran two independent classification calls to prevent anchoring. "
        f"{'Both calls agreed.' if signals['ensemble_agreement'] else 'Calls disagreed — safer tier applied, QE Lead escalation triggered.'} "
        f"Triggers: {primary.get('fca_triggers', [])}."
    )

    data = {
        "fca_classification": classification,
        "classification_rationale": primary.get("classification_rationale", ""),
        "fca_triggers": primary.get("fca_triggers", []),
        "regulatory_obligations": primary.get("regulatory_obligations", []),
        "co_signoff_required": co_required,
        "enhanced_testing_required": enhanced_testing,
        "ensemble_agreement": signals["ensemble_agreement"],
        "call_a_classification": call_a_result["fca_classification"],
        "call_b_classification": call_b_result["fca_classification"],
        "tier_gap": signals.get("tier_gap", 0),
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=why,
        data=data,
        confidence=ConfidenceBreakdown(
            tier="D",
            raw_score=confidence_score,
            calibration_multiplier=1.0,
            final_score=confidence_score,
            signals=signals,
            escalated=escalated,
        ),
        model_used=settings.default_model,
    )


# ── Ensemble resolution ───────────────────────────────────────────────────────

def _resolve_ensemble(
    call_a: dict,
    call_b: dict,
) -> tuple[str, int, dict]:
    """
    Resolve two classification calls into a final verdict.
    Safer call always wins on disagreement; confidence reflects certainty.
    """
    class_a = call_a["fca_classification"]
    class_b = call_b["fca_classification"]

    if class_a == class_b:
        confidence = _AGREEMENT_CONFIDENCE[class_a]
        return class_a, confidence, {
            "ensemble_agreement": True,
            "call_a": class_a,
            "call_b": class_b,
            "tier_gap": 0,
        }

    tier_a = _TIER_ORDER[class_a]
    tier_b = _TIER_ORDER[class_b]
    tier_gap = abs(tier_a - tier_b)

    # UNCLASSIFIED loses to any real classification
    if class_a == "UNCLASSIFIED":
        winner = class_b
    elif class_b == "UNCLASSIFIED":
        winner = class_a
    else:
        # Safer (higher regulatory tier) wins
        winner = class_a if tier_a > tier_b else class_b

    # Graduated confidence: adjacent tier gap is more reliable than a 2-tier jump
    confidence = 48 if tier_gap == 1 else (38 if tier_gap == 2 else 30)

    return winner, confidence, {
        "ensemble_agreement": False,
        "call_a": class_a,
        "call_b": class_b,
        "tier_gap": tier_gap,
        "conservative_winner": winner,
    }


# ── Primary call selector ─────────────────────────────────────────────────────

def _pick_primary_call(call_a: dict, call_b: dict, winning_classification: str) -> dict:
    """Return the call whose classification matches the agreed/winning tier."""
    if call_a["fca_classification"] == winning_classification:
        return call_a
    return call_b


# ── Agent data accessor ───────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
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
        flags    = ", ".join(agent1_data.get("flags", ["none"]))
        agent1_section = (
            f"\n\nAGENT 1 PRE-ANALYSIS (Story Intent Agent):\n"
            f"Extracted FSC Objects: {fsc_objs}\n"
            f"Quality Flags: {flags}\n"
            f"Story Summary: {agent1_data.get('story_summary', '')}"
        )

    return (
        f"Classify the FCA regulatory risk tier of the following Jira user story.\n\n"
        f"STORY ID: {story['story_id']}\n"
        f"SUMMARY: {story['summary']}\n"
        f"COMPONENTS: {', '.join(story.get('components', [])) or 'None'}\n\n"
        f"DESCRIPTION:\n{story['description'] or '(empty)'}"
        f"{ac_text}"
        f"{agent1_section}\n\n"
        f"Use the classify_fca_risk tool to return your classification."
    )
