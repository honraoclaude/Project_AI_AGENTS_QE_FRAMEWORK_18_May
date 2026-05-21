"""
Agent 34 — Defect Triage Agent
Phase       : Testing
PACT        : Proactive
Classification: True AI (Sonnet 4.6)
Confidence  : Tier B (base=60)

Runs in Testing Batch 4 (parallel with Agents 33, 38).
Has access to Agents 27, 28, 30, 31, 37.

Purpose:
  Reviews CRT execution results, self-heal flags, FCA scenario outcomes,
  financial data integrity checks, and performance verdicts to identify,
  classify, and triage defects. Assigns severity (P1–P4) and ownership.

  Sonnet 4.6 reasons across all test outputs to produce a structured
  defect list — the deterministic script cannot fully classify without
  domain reasoning.

Output data keys consumed by downstream:
  defects_found      → list   (each: {id, title, severity, owner, source})
  defect_count       → int
  critical_defects   → list   (P1/P2 defects — triggers Gate G5 FAIL)
  defect_verdict     → str    (PASS / WARN / FAIL)
  triage_complete    → bool
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 34
AGENT_NAME = "Defect Triage Agent"

# ── Sonnet tool ────────────────────────────────────────────────────────────────

_TRIAGE_TOOL_NAME = "triage_defects"
_TRIAGE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["defects_found", "defect_count", "critical_defects",
                 "defect_verdict", "triage_complete", "narrative"],
    "properties": {
        "defects_found": {
            "type": "array",
            "description": "All defects identified across test types.",
            "items": {
                "type": "object",
                "required": ["id", "title", "severity", "owner", "source"],
                "properties": {
                    "id":       {"type": "string"},
                    "title":    {"type": "string"},
                    "severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                    "owner":    {"type": "string",
                                 "enum": ["Developer", "QE", "DevOps", "Compliance"]},
                    "source":   {"type": "string",
                                 "enum": ["CRT", "FCA", "Performance",
                                          "FinancialIntegrity", "SelfHeal"]},
                },
            },
        },
        "defect_count": {"type": "integer", "minimum": 0},
        "critical_defects": {
            "type": "array",
            "description": "P1 and P2 defects — each element is the defect id.",
            "items": {"type": "string"},
        },
        "defect_verdict": {
            "type": "string",
            "enum": ["PASS", "WARN", "FAIL"],
            "description": "PASS=no defects, WARN=P3/P4 only, FAIL=any P1 or P2.",
        },
        "triage_complete": {
            "type": "boolean",
            "description": "True when all defects have been assigned an owner.",
        },
        "narrative": {
            "type": "string",
            "description": (
                "2–3 sentences summarising defects found, their severities, "
                "ownership assignments, and what must be resolved before release."
            ),
        },
    },
}

_TRIAGE_INSTRUCTIONS = """
You are a QE defect triage agent for a Salesforce FSC Wealth Management platform under FCA regulation.
You receive test execution results from CRT, FCA regulatory scenario checks, financial data integrity
checks, performance tests, and CRT self-heal reviews.

For each failure or concern, identify a distinct defect. Assign:
- Severity: P1 (blocker — data corruption, regulatory breach, FCA violation) |
            P2 (critical — functional failure blocking UAT) |
            P3 (major — degraded functionality, workaround exists) |
            P4 (minor — cosmetic, non-blocking)
- Owner:    Developer (code fix needed) | QE (test fix needed) |
            DevOps (infra/config) | Compliance (regulatory gap)
- Source:   CRT | FCA | Performance | FinancialIntegrity | SelfHeal

If all tests passed and no concerns are raised, return an empty defects_found list and verdict=PASS.
If only P3/P4 defects, return verdict=WARN.
If any P1 or P2, return verdict=FAIL.
triage_complete=true when every defect has an owner assigned.

Be precise: one defect per distinct failure. Do not duplicate defects across sources.
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]
    agent27_data = _get_agent_data(state, "27")
    agent28_data = _get_agent_data(state, "28")
    agent30_data = _get_agent_data(state, "30")
    agent31_data = _get_agent_data(state, "31")
    agent37_data = _get_agent_data(state, "37")

    triage_msg = _build_triage_message(
        story_id, agent27_data, agent28_data, agent30_data, agent31_data, agent37_data,
    )
    result_data = await _run_triage(triage_msg)

    defects      = result_data.get("defects_found", [])
    defect_count = result_data.get("defect_count", len(defects))
    critical     = result_data.get("critical_defects", [])
    verdict      = result_data.get("defect_verdict", "PASS")
    complete     = result_data.get("triage_complete", True)
    narrative    = result_data.get("narrative", "Defect Triage Agent completed assessment.")

    # Coalition severity voting — 5 independent test sources each vote on severity
    severity_votes = _build_severity_votes(
        agent27_data, agent28_data, agent30_data, agent31_data, agent37_data,
    )
    final_severity, minimax_escalated = _resolve_severity_votes(severity_votes)
    coalition_dissent = [src for src, sev in severity_votes.items() if sev != final_severity]

    confidence_score, signals = _compute_confidence(
        agent27_data, agent28_data, agent31_data, agent37_data, defect_count,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    what = (
        f"Defect triage for {story_id}: {defect_count} defect(s), "
        f"{len(critical)} critical — verdict={verdict}"
    )

    data = {
        "defects_found": defects,
        "defect_count": defect_count,
        "critical_defects": critical,
        "defect_verdict": verdict,
        "triage_complete": complete,
        "narrative": narrative,
        "severity_votes": severity_votes,
        "coalition_severity": final_severity,
        "minimax_escalated": minimax_escalated,
        "coalition_dissent": coalition_dissent,
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
    agent27_data: dict | None,
    agent28_data: dict | None,
    agent31_data: dict | None,
    agent37_data: dict | None,
    defect_count: int,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=60)

    sources_available = sum(
        1 for d in [agent27_data, agent28_data, agent31_data, agent37_data] if d
    )
    if sources_available >= 3:
        scorer.add("comprehensive_test_sources", sources_available, +10)
    elif sources_available >= 1:
        scorer.add("partial_test_sources", sources_available, +4)
    else:
        scorer.add("no_test_sources", 0, -12)

    crt_verdict = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED")
    if crt_verdict == "PASS":
        scorer.add("crt_execution_passed", True, +5)
    elif crt_verdict == "FAIL":
        scorer.add("crt_execution_failed", True, -5)

    if defect_count == 0:
        scorer.add("no_defects_found", 0, +5)
    elif defect_count >= 3:
        scorer.add("multiple_defects", defect_count, -5)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Sonnet triage call ────────────────────────────────────────────────────────

async def _run_triage(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.default_model,
        system=build_system(_TRIAGE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRIAGE_TOOL_NAME,
        tool_description="Triage defects found across all test types.",
        tool_schema=_TRIAGE_TOOL_SCHEMA,
        max_tokens=800,
    )


def _infer_severity(verdict: str, count: int | bool) -> str:
    if verdict == "FAIL" and count:
        return "P1"
    if verdict == "WARN" and count:
        return "P2"
    if verdict == "FAIL":
        return "P2"
    return "P3"


def _build_severity_votes(
    agent27_data: dict | None,
    agent28_data: dict | None,
    agent30_data: dict | None,
    agent31_data: dict | None,
    agent37_data: dict | None,
) -> dict[str, str]:
    return {
        "crt":          _infer_severity(
            (agent27_data or {}).get("crt_execution_verdict", "PASS"),
            (agent27_data or {}).get("crt_fail_count", 0),
        ),
        "fca_scenario": _infer_severity(
            (agent30_data or {}).get("fca_scenario_verdict", "PASS"),
            len((agent30_data or {}).get("regulatory_gaps", [])),
        ),
        "financial":    _infer_severity(
            (agent31_data or {}).get("integrity_verdict", "PASS"),
            len((agent31_data or {}).get("integrity_violations", [])),
        ),
        "performance":  _infer_severity(
            (agent37_data or {}).get("perf_test_verdict", "PASS"),
            bool((agent37_data or {}).get("performance_concern", "none") not in ("none", "")),
        ),
        "self_heal":    _infer_severity(
            (agent28_data or {}).get("self_heal_verdict", "PASS"),
            len((agent28_data or {}).get("suspect_self_heals", [])),
        ),
    }


def _resolve_severity_votes(votes: dict[str, str]) -> tuple[str, bool]:
    """Returns (final_severity, minimax_escalated). Any P1 → escalate regardless."""
    from collections import Counter
    if "P1" in votes.values():
        majority = "P1"
        escalated = not all(v == "P1" for v in votes.values())
        return majority, escalated
    majority = Counter(votes.values()).most_common(1)[0][0]
    return majority, False


def _build_triage_message(
    story_id: str,
    agent27_data: dict | None,
    agent28_data: dict | None,
    agent30_data: dict | None,
    agent31_data: dict | None,
    agent37_data: dict | None,
) -> str:
    crt_verdict   = (agent27_data or {}).get("crt_execution_verdict", "SKIPPED")
    crt_pass      = (agent27_data or {}).get("crt_pass_count", 0)
    crt_fail      = (agent27_data or {}).get("crt_fail_count", 0)
    heal_verdict  = (agent28_data or {}).get("self_heal_verdict", "PASS")
    suspect_heals = (agent28_data or {}).get("suspect_self_heals", [])
    fca_verdict   = (agent30_data or {}).get("fca_scenario_verdict", "PASS")
    reg_gaps      = (agent30_data or {}).get("regulatory_gaps", [])
    int_verdict   = (agent31_data or {}).get("integrity_verdict", "PASS")
    violations    = (agent31_data or {}).get("integrity_violations", [])
    perf_verdict  = (agent37_data or {}).get("perf_test_verdict", "SKIPPED")
    perf_concern  = (agent37_data or {}).get("performance_concern", "none")

    return (
        f"Story: {story_id}\n\n"
        f"CRT Execution: verdict={crt_verdict}, passed={crt_pass}, failed={crt_fail}\n"
        f"CRT Self-Heal: verdict={heal_verdict}, suspect_tests={suspect_heals}\n"
        f"FCA Scenarios: verdict={fca_verdict}, regulatory_gaps={reg_gaps}\n"
        f"Financial Integrity: verdict={int_verdict}, violations={violations}\n"
        f"Performance: verdict={perf_verdict}, concern={perf_concern}\n\n"
        f"Triage all failures and concerns into defects using the {_TRIAGE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
