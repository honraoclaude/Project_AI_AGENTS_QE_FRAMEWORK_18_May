"""
Base utilities shared by all True AI agents.

Key patterns:
  - FSC_DOMAIN_CONTEXT: the large shared prompt block with cache_control applied.
    Sent with every agent call. Cached by Anthropic for 5 minutes — ~10x cheaper
    on subsequent calls within the same window.
  - build_messages(): constructs the system+user message list with caching applied.
  - call_with_tool(): calls Claude with a tool definition to force structured JSON output.
    More reliable than asking the model to format JSON in free text.
  - Tier B confidence scoring helpers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic

from src.core.config import settings

if TYPE_CHECKING:
    from src.core.schemas import StoryState

# ── Shared FSC domain context (prompt cached) ─────────────────────────────────
# This block is prepended to every agent call and marked for caching.
# It describes the Salesforce FSC object model, FCA regulatory context,
# and PACT principles so agents don't need to re-derive them.

FSC_DOMAIN_CONTEXT = """
<fsc_domain_context>

## Salesforce Financial Services Cloud (FSC) — Object Model

Key FSC managed package objects agents must recognise:
- FinancialAccount: bank accounts, investment portfolios, pensions. Central to AUM roll-ups.
- FinancialAccountTransaction: individual transactions against a FinancialAccount.
- FinancialGoal (Goal__c): client goals (retirement, education, property). Linked to FinancialAccount.
- Household (Account with RecordType=Household): groups of related clients.
- IndividualApplication: onboarding and suitability assessment records.
- Suitability__c / SuitabilityAssessment: COBS 9.2-mandated risk assessment records.
- RiskProfile__c: client risk tolerance, horizon, capacity for loss.
- Appropriateness__c: COBS 10 non-advised appropriateness checks.
- VulnerableCustomerIndicator__c: flags clients needing enhanced care (Consumer Duty).
- FinancialHolding: holdings within a FinancialAccount (equities, bonds, funds).
- AssetsAndLiabilities: balance sheet items for wealth planning.
- Revenue__c: AUM-based fee calculations.

FSC Platform components:
- Apex triggers and classes: business logic, validation, roll-up calculations.
- Flows (Record-Triggered, Screen): declarative automation — bulkification risks.
- Lightning Web Components (LWC): adviser-facing UI. FLS and locker compliance.
- Permission Sets / Profiles: access control on FSC objects and fields.
- Validation Rules: data integrity on FSC records.

## FCA Regulatory Context (UK)

Applicable regulations for wealth management on FSC:
- COBS 9.2: Suitability — advisers must assess client risk before recommending investments.
- COBS 10: Appropriateness — non-advised services must check client understanding.
- SYSC: Senior Managers and Certification Regime — oversight and governance.
- Consumer Duty (FCA PS22/9): four outcomes — products/services, price/value,
  consumer understanding, consumer support. Applies to ALL customer-facing journeys.
- Vulnerable Customer (FG21/1): enhanced treatment for clients with vulnerabilities.
  Must not be systematically disadvantaged by digital or automated processes.

FCA classification tiers used by this framework:
- HIGH-FCA: story directly modifies Suitability, Appropriateness, Risk Profile,
  Vulnerable Customer indicators, or Consumer Duty journeys. Requires CO sign-off.
- MEDIUM-FCA: story modifies financial data (FinancialAccount, Goals, AUM calculations)
  or permission model without touching the above. Enhanced testing required.
- LOW-FCA: standard platform changes — UI, non-financial metadata, admin tooling.

## Common FSC Personas

- Wealth Adviser: front-line staff recording client interactions, suitability assessments.
- Client / Investor: end customer accessing self-service portals or receiving advice.
- Compliance Officer (CO): responsible for regulatory sign-off on HIGH-FCA changes.
- Operations / Admin: back-office staff managing accounts, transactions, reporting.
- QE Engineer: quality engineering team member building and running test suites.
- Product Owner (PO): story author accountable for business value and acceptance criteria.

## PACT Principles

- Proactive: detect issues before they reach UAT or Production.
- Autonomous: act without human prompts; every decision carries an explainability trace.
- Collaborative: agents work with humans — never replace human regulatory judgement.
- Targeted: 85% Apex coverage on HIGH-FCA components; standard 75% elsewhere.

## Explainability Trace Format

Every agent decision must produce:
  what: one sentence — the action taken
  why: the rule or pattern that triggered it
  data: structured evidence (scores, matched keywords, object lists)
  confidence: 0-100 integer — below 60 the Fleet Commander escalates to QE Lead

</fsc_domain_context>
""".strip()


# ── Anthropic client ──────────────────────────────────────────────────────────

def get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value()
    )


# ── Cached system prompt builder ──────────────────────────────────────────────

def build_system(agent_instructions: str) -> list[dict]:
    """
    Returns the system prompt as a content block list with cache_control
    applied to the shared FSC context block.

    The FSC_DOMAIN_CONTEXT block (~800 tokens) is cached for 5 minutes.
    Subsequent agent calls within that window pay ~10x less for those tokens.
    """
    return [
        {
            "type": "text",
            "text": FSC_DOMAIN_CONTEXT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": agent_instructions,
        },
    ]


# ── Structured output via tool use ────────────────────────────────────────────

async def call_with_tool(
    model: str,
    system: list[dict],
    user_message: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """
    Call Claude with a single tool definition and force it to use that tool.
    Returns the parsed tool input dict — validated against tool_schema by Claude.

    This is more reliable than asking the model to produce JSON in free text,
    especially for structured extraction tasks.
    """
    client = get_client()

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input  # type: ignore[return-value]

    block_types = [b.type for b in response.content]
    raise RuntimeError(
        f"Model did not call tool '{tool_name}' — got {len(response.content)} block(s): {block_types}"
    )


# ── Tier B confidence scoring ─────────────────────────────────────────────────

class TierBScorer:
    """
    Computes a structured confidence score from observable signals.
    Never asks the LLM to self-assess. Signals are measurable inputs.
    """

    def __init__(self, base: int = 60):
        self._score = base
        self._signals: dict[str, Any] = {}

    def add(self, signal_name: str, value: Any, delta: int) -> "TierBScorer":
        self._score += delta
        self._signals[signal_name] = value
        return self

    def cap(self, maximum: int = 92) -> "TierBScorer":
        self._score = min(self._score, maximum)
        return self

    def floor(self, minimum: int = 20) -> "TierBScorer":
        self._score = max(self._score, minimum)
        return self

    def build(self) -> tuple[int, dict[str, Any]]:
        return max(0, min(self._score, 100)), self._signals


# ── Game theory utilities ─────────────────────────────────────────────────────

def get_agent_result(state: StoryState, agent_id: str) -> tuple[dict | None, int]:
    """
    Returns (data, confidence_score) for an upstream agent.
    confidence_score defaults to 0 if the agent has not run.
    Allows downstream agents to weight upstream evidence by quality.
    """
    result = state["agent_results"].get(str(agent_id))
    if not result:
        return None, 0
    return result.get("data"), result.get("confidence", {}).get("final_score", 50)


class ShapleyAttributor:
    """
    Computes Shapley-value marginal contributions for upstream agents.

    For an additive value function (each agent contributes independently),
    the Shapley value equals the agent's individual contribution. We scale
    by confidence (0-100) × data_present to give fair attribution that
    rewards both high-quality evidence and actual data presence.

    The result is normalised to sum to 100.0, making it directly comparable
    across stories and suitable for FCA audit display.
    """

    def __init__(self) -> None:
        self._agents: list[tuple[str, int, bool]] = []

    def add_agent(self, agent_id: str, confidence: int, data_present: bool) -> None:
        if not (0 <= confidence <= 100):
            raise ValueError(f"confidence must be 0-100, got {confidence} for agent {agent_id}")
        self._agents.append((agent_id, confidence, data_present))

    def compute(self) -> dict[str, float]:
        """Returns {agent_id: shapley_value} normalised to sum ≈ 100.0."""
        raw: dict[str, float] = {
            aid: (conf * (1.0 if present else 0.0))
            for aid, conf, present in self._agents
        }
        total = sum(raw.values())
        if total == 0.0:
            equal = round(100.0 / len(self._agents), 2) if self._agents else 0.0
            return {aid: equal for aid, _, _ in self._agents}
        return {aid: round(v / total * 100.0, 2) for aid, v in raw.items()}


_VALID_FCA_TIERS = {"HIGH", "MEDIUM", "LOW", "UNCLASSIFIED"}

def adaptive_threshold(base: int, fca_tier: str, direction: str = "strict") -> int:
    """
    Returns a gate threshold adjusted for FCA regulatory risk tier.

    direction='strict' (default): HIGH-FCA raises the bar (harder to pass),
    creating correct incentives — the riskiest stories face the toughest gates.
    direction='lenient': reserved for future use where LOW-FCA can fast-track.

    Adjustments: HIGH=+5, MEDIUM=0, LOW=-5, UNCLASSIFIED=+10 (unknown = cautious).
    """
    if fca_tier not in _VALID_FCA_TIERS:
        raise ValueError(f"Unknown FCA tier '{fca_tier}'. Expected one of {_VALID_FCA_TIERS}")
    adjustments = {"HIGH": +5, "MEDIUM": 0, "LOW": -5, "UNCLASSIFIED": +10}
    return base + adjustments[fca_tier]
