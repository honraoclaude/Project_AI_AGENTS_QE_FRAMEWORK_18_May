"""
Agent 44 — FCA Evidence Pack
Phase       : Release
PACT        : Targeted
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=65)

Runs sequentially after Gate G9.
Has access to Agents 3, 4, 30, 33, 36.

Purpose:
  Compiles the complete FCA regulatory evidence package required before a
  HIGH or MEDIUM FCA-classified story can be released. Evidence must demonstrate:
  - Consumer Duty PS22/9 obligations addressed
  - COBS 9 suitability assessment tested (where applicable)
  - FCA regulatory scenarios all covered
  - UAT sign-off received for HIGH/MEDIUM stories
  - Test coverage above FCA threshold

  Sonnet 4.6 synthesises evidence from multiple agents into a structured pack.
  Gate G10 depends on evidence_verdict.

Output data keys consumed by downstream:
  evidence_items          → list   (each item: {rule, status, evidence_ref})
  consumer_duty_covered   → bool   (Consumer Duty PS22/9 addressed)
  regulatory_sign_off_ready → bool (all evidence present for CO sign-off)
  evidence_verdict        → str    (COMPLETE / PARTIAL / MISSING)
  evidence_gaps           → list   (rules without evidence)
"""

from __future__ import annotations

from src.agents.base import ShapleyAttributor, TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 44
AGENT_NAME = "FCA Evidence Pack"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_EVIDENCE_TOOL_NAME = "compile_fca_evidence"
_EVIDENCE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["evidence_items", "consumer_duty_covered", "regulatory_sign_off_ready",
                 "evidence_verdict", "evidence_gaps", "narrative"],
    "properties": {
        "evidence_items": {
            "type": "array",
            "description": "Evidence items covering each applicable FCA regulatory requirement.",
            "items": {
                "type": "object",
                "required": ["rule", "status", "evidence_ref"],
                "properties": {
                    "rule":         {"type": "string"},
                    "status":       {"type": "string", "enum": ["COVERED", "PARTIAL", "MISSING"]},
                    "evidence_ref": {"type": "string"},
                },
            },
        },
        "consumer_duty_covered": {
            "type": "boolean",
            "description": "True when Consumer Duty PS22/9 obligations are evidenced.",
        },
        "regulatory_sign_off_ready": {
            "type": "boolean",
            "description": "True when all evidence is present and CO sign-off can be issued.",
        },
        "evidence_verdict": {
            "type": "string",
            "enum": ["COMPLETE", "PARTIAL", "MISSING"],
        },
        "evidence_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of regulatory rules without adequate evidence coverage.",
        },
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences describing the FCA evidence pack status, "
                "which requirements are covered, any gaps, "
                "and whether the story is ready for regulatory sign-off."
            ),
        },
    },
}

_EVIDENCE_INSTRUCTIONS = """
You are compiling the FCA regulatory evidence pack for a Salesforce FSC Wealth Management
feature release under FCA oversight.

You receive the FCA classification, Consumer Duty mapping, FCA regulatory scenario results,
test coverage, and UAT sign-off status.

Compile evidence items for each applicable FCA regulatory requirement:
- Consumer Duty PS22/9: fair outcomes, consumer understanding, vulnerability
- COBS 9: suitability assessment (required for HIGH FCA stories)
- MiFID II Art.25: appropriateness (where applicable)
- FG21/1: Vulnerable Customer treatment

For each rule: determine COVERED (tested and documented), PARTIAL (incomplete testing),
or MISSING (no evidence). If FCA class is LOW and no regulatory scenarios are required,
you may return a minimal evidence pack with evidence_verdict=COMPLETE.

regulatory_sign_off_ready=true only when ALL applicable rules are COVERED and
the UAT coordination is NOT_REQUIRED or SIGNED_OFF or PENDING (not BLOCKED).
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent3_data  = _get_agent_data(state, "3")
    agent4_data  = _get_agent_data(state, "4")
    agent30_data = _get_agent_data(state, "30")
    agent33_data = _get_agent_data(state, "33")
    agent36_data = _get_agent_data(state, "36")

    evidence_msg = _build_evidence_message(
        story_id, agent3_data, agent4_data, agent30_data, agent33_data, agent36_data,
    )
    result_data = await _run_evidence(evidence_msg)

    items     = result_data.get("evidence_items", [])
    cd_cov    = result_data.get("consumer_duty_covered", False)
    sign_rdy  = result_data.get("regulatory_sign_off_ready", False)
    verdict   = result_data.get("evidence_verdict", "MISSING")
    gaps      = result_data.get("evidence_gaps", [])
    narrative = result_data.get("narrative", "FCA Evidence Pack compiled.")

    confidence_score, signals = _compute_confidence(
        agent3_data, agent30_data, agent33_data, agent36_data, verdict, state,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"FCA evidence pack for {story_id}: {len(items)} item(s), "
        f"{len(gaps)} gap(s) — verdict={verdict}"
    )

    data = {
        "evidence_items": items,
        "consumer_duty_covered": cd_cov,
        "regulatory_sign_off_ready": sign_rdy,
        "evidence_verdict": verdict,
        "evidence_gaps": gaps,
        "narrative": narrative,
        "signals": signals,
    }

    return AgentResult(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        what=what,
        why=narrative,
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
    agent30_data: dict | None,
    agent33_data: dict | None,
    agent36_data: dict | None,
    verdict: str,
    state: StoryState,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    if agent3_data:
        scorer.add("fca_classification_known", True, +8)
    else:
        scorer.add("fca_class_unknown", 0, -10)

    if agent30_data:
        scorer.add("fca_scenarios_available", True, +6)
    if agent33_data:
        scorer.add("coverage_data_available", True, +4)

    if verdict == "COMPLETE":
        scorer.add("evidence_complete", True, +5)
    elif verdict == "MISSING":
        scorer.add("evidence_missing", True, -10)

    # Shapley attribution: fair credit for each upstream agent's contribution to evidence pack
    attributor = ShapleyAttributor()
    for aid in ["3", "4", "30", "33", "36"]:
        result = state["agent_results"].get(aid, {})
        conf = result.get("confidence", {}).get("final_score", 0)
        attributor.add_agent(f"agent_{aid}", conf, aid in state["agent_results"])
    scorer.add("shapley_attributions", attributor.compute(), 0)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet evidence call ──────────────────────────────────────────────────────

async def _run_evidence(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_EVIDENCE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_EVIDENCE_TOOL_NAME,
        tool_description="Compile the FCA regulatory evidence pack.",
        tool_schema=_EVIDENCE_TOOL_SCHEMA,
        max_tokens=800,
    )


def _build_evidence_message(
    story_id: str,
    agent3_data: dict | None,
    agent4_data: dict | None,
    agent30_data: dict | None,
    agent33_data: dict | None,
    agent36_data: dict | None,
) -> str:
    fca_class      = (agent3_data or {}).get("fca_classification", "LOW")
    cd_verdict     = (agent4_data or {}).get("consumer_duty_verdict", "PASS")
    cd_obligations = (agent4_data or {}).get("obligations_mapped", [])
    fca_verdict    = (agent30_data or {}).get("fca_scenario_verdict", "PASS")
    fca_count      = (agent30_data or {}).get("fca_scenario_count", 0)
    reg_gaps       = (agent30_data or {}).get("regulatory_gaps", [])
    coverage_pct   = (agent33_data or {}).get("overall_coverage_pct", 0.0)
    cov_verdict    = (agent33_data or {}).get("coverage_verdict", "PASS")
    uat_coord      = (agent36_data or {}).get("uat_coordination_verdict", "NOT_REQUIRED")

    return (
        f"Story: {story_id}\n"
        f"FCA Classification: {fca_class}\n\n"
        f"Consumer Duty PS22/9: verdict={cd_verdict}, obligations={cd_obligations}\n"
        f"FCA Regulatory Scenarios: verdict={fca_verdict}, count={fca_count}, gaps={reg_gaps}\n"
        f"Test Coverage: {coverage_pct:.1f}%, verdict={cov_verdict}\n"
        f"UAT Coordination: {uat_coord}\n\n"
        f"Compile the FCA evidence pack using the {_EVIDENCE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
