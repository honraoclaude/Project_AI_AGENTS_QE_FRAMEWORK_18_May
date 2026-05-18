"""
Agent 35 — Root Cause Analyser
Phase       : Testing
PACT        : Collaborative
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=62)

Runs sequentially after Gate G5.
Has access to Agents 27, 28, 33, 34, 37, 38.

Purpose:
  When defects or test failures are present, performs root cause analysis
  by reasoning across all test outputs. Proposes a structured fix plan
  with owner assignment and estimated resolution effort.

  Only performs deep analysis when defects exist (verdict != PASS).
  When all tests pass, returns a lightweight confirmation trace.

  Sonnet 4.6 synthesises signals across multiple agents to produce
  actionable root cause reasoning.

Output data keys consumed by downstream:
  root_causes         → list   (each: {defect_id, root_cause, fix_action, owner, effort})
  rca_verdict         → str    (RESOLVED_PLAN / NO_ACTION_REQUIRED / INCOMPLETE)
  fix_plan_complete   → bool
  estimated_effort    → str    (story points estimate: LOW / MEDIUM / HIGH)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 35
AGENT_NAME = "Root Cause Analyser"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_RCA_TOOL_NAME = "analyse_root_causes"
_RCA_TOOL_SCHEMA = {
    "type": "object",
    "required": ["root_causes", "rca_verdict", "fix_plan_complete",
                 "estimated_effort", "narrative"],
    "properties": {
        "root_causes": {
            "type": "array",
            "description": "Root cause analysis for each defect or failure.",
            "items": {
                "type": "object",
                "required": ["defect_id", "root_cause", "fix_action", "owner", "effort"],
                "properties": {
                    "defect_id":   {"type": "string"},
                    "root_cause":  {"type": "string"},
                    "fix_action":  {"type": "string"},
                    "owner":       {"type": "string",
                                   "enum": ["Developer", "QE", "DevOps", "Compliance"]},
                    "effort":      {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                },
            },
        },
        "rca_verdict": {
            "type": "string",
            "enum": ["RESOLVED_PLAN", "NO_ACTION_REQUIRED", "INCOMPLETE"],
        },
        "fix_plan_complete": {"type": "boolean"},
        "estimated_effort": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"],
            "description": "Overall effort estimate for all fixes combined.",
        },
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising root causes identified, fix actions proposed, "
                "and overall remediation effort required before the story can be released."
            ),
        },
    },
}

_RCA_INSTRUCTIONS = """
You are a senior QE root cause analyst for a Salesforce FSC Wealth Management platform under FCA regulation.

You receive defect lists and test failure signals from CRT execution, self-heal reviews,
coverage analysis, defect triage, performance tests, and flaky test detection.

For each defect or failure:
1. Identify the most likely root cause (e.g. SOQL query inside loop, stale UI locator,
   missing FCA scenario, financial calculation error, config drift in sandbox).
2. Propose a specific, actionable fix (e.g. "Move SOQL query outside the loop in
   AccountService.updatePortfolio()", "Add @negative scenario for income verification").
3. Assign owner (Developer / QE / DevOps / Compliance) and effort (LOW / MEDIUM / HIGH).

If no defects exist, return rca_verdict=NO_ACTION_REQUIRED with an empty root_causes list.
If defects exist but root cause cannot be determined from available signals, return
rca_verdict=INCOMPLETE for those items.
When all defects have a fix action, return rca_verdict=RESOLVED_PLAN and fix_plan_complete=true.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent27_data = _get_agent_data(state, "27")
    agent28_data = _get_agent_data(state, "28")
    agent33_data = _get_agent_data(state, "33")
    agent34_data = _get_agent_data(state, "34")
    agent37_data = _get_agent_data(state, "37")
    agent38_data = _get_agent_data(state, "38")

    rca_msg = _build_rca_message(
        story_id, agent27_data, agent28_data, agent33_data,
        agent34_data, agent37_data, agent38_data,
    )
    result_data = await _run_rca(rca_msg)

    root_causes  = result_data.get("root_causes", [])
    rca_verdict  = result_data.get("rca_verdict", "NO_ACTION_REQUIRED")
    complete     = result_data.get("fix_plan_complete", True)
    effort       = result_data.get("estimated_effort", "LOW")
    narrative    = result_data.get("narrative", "Root Cause Analyser completed assessment.")

    confidence_score, signals = _compute_confidence(
        agent34_data, agent38_data, len(root_causes), rca_verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Root cause analysis for {story_id}: {len(root_causes)} cause(s) identified — "
        f"verdict={rca_verdict}, effort={effort}"
    )

    data = {
        "root_causes": root_causes,
        "rca_verdict": rca_verdict,
        "fix_plan_complete": complete,
        "estimated_effort": effort,
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
    agent34_data: dict | None,
    agent38_data: dict | None,
    cause_count: int,
    rca_verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=62)

    if agent34_data:
        defect_verdict = agent34_data.get("defect_verdict", "PASS")
        if defect_verdict != "PASS":
            scorer.add("defects_to_analyse", True, +8)
        else:
            scorer.add("no_defects_clean_slate", True, +5)
    else:
        scorer.add("no_defect_triage_data", 0, -10)

    if agent38_data:
        scorer.add("flaky_test_data_available", True, +4)

    if rca_verdict == "RESOLVED_PLAN":
        scorer.add("complete_fix_plan", True, +8)
    elif rca_verdict == "INCOMPLETE":
        scorer.add("incomplete_rca", True, -8)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet RCA call ───────────────────────────────────────────────────────────

async def _run_rca(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_RCA_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_RCA_TOOL_NAME,
        tool_description="Analyse root causes of test failures and produce a fix plan.",
        tool_schema=_RCA_TOOL_SCHEMA,
        max_tokens=1000,
    )


def _build_rca_message(
    story_id: str,
    agent27_data: dict | None,
    agent28_data: dict | None,
    agent33_data: dict | None,
    agent34_data: dict | None,
    agent37_data: dict | None,
    agent38_data: dict | None,
) -> str:
    crt_verdict    = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED")
    crt_fail       = (agent27_data or {}).get("crt_fail_count", 0)
    heal_verdict   = (agent28_data or {}).get("self_heal_verdict", "PASS")
    suspect_heals  = (agent28_data or {}).get("suspect_self_heals", [])
    coverage_pct   = (agent33_data or {}).get("overall_coverage_pct", 0.0)
    cov_verdict    = (agent33_data or {}).get("coverage_verdict", "PASS")
    uncovered      = (agent33_data or {}).get("uncovered_acs", [])
    defects        = (agent34_data or {}).get("defects_found", [])
    def_verdict    = (agent34_data or {}).get("defect_verdict", "PASS")
    perf_verdict   = (agent37_data or {}).get("perf_test_verdict", "SKIPPED")
    perf_concern   = (agent37_data or {}).get("performance_concern", "none")
    flaky_tests    = (agent38_data or {}).get("flaky_tests", [])
    flaky_verdict  = (agent38_data or {}).get("flaky_verdict", "PASS")

    defect_lines = "\n".join(
        f"  - [{d.get('id')}] {d.get('title')} (severity={d.get('severity')}, source={d.get('source')})"
        for d in defects
    ) or "  (none)"

    return (
        f"Story: {story_id}\n\n"
        f"CRT Execution: verdict={crt_verdict}, failed_tests={crt_fail}\n"
        f"Self-Heal Review: verdict={heal_verdict}, suspect_tests={suspect_heals}\n"
        f"Coverage: {coverage_pct:.1f}% overall, verdict={cov_verdict}, uncovered_acs={uncovered}\n"
        f"Performance: verdict={perf_verdict}, concern={perf_concern}\n"
        f"Flaky Tests: verdict={flaky_verdict}, flagged={flaky_tests}\n\n"
        f"Defects to analyse:\n{defect_lines}\n\n"
        f"Overall defect verdict: {def_verdict}\n\n"
        f"Perform root cause analysis using the {_RCA_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
