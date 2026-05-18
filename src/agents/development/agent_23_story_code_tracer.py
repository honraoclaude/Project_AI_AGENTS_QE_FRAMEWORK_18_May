"""
Agent 23 — Story-to-Code Tracer
Phase       : Development
PACT        : Proactive
Classification: Augmented Script (deterministic + Haiku narrative)
Confidence  : Tier B (base=65)

Runs last in the Development phase (after all other Development agents).
Has access to Agents 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22.

Purpose:
  Produces the definitive story-to-code traceability record for the FCA
  audit ledger. Aggregates signals from all Development phase agents into
  a single structured audit record. This record feeds Gate G4 (go/no-go
  for Development phase completion) and is the primary FCA evidence for
  the Development phase.

  Haiku writes the audit narrative; aggregation is pure Python.

Output data keys consumed by downstream:
  trace_record           → dict (FCA audit ledger — immutable after emission)
  development_verdict    → str  (PASS / PARTIAL / FAIL)
  gate_g4_signals        → dict (Gate G4 input)
  escalation_required    → bool (if any critical finding present)
"""

from __future__ import annotations

from src.agents.base import TierBScorer, build_system, call_with_tool
from src.core.config import settings
from src.core.schemas import AgentResult, ConfidenceBreakdown, StoryState

AGENT_ID = 23
AGENT_NAME = "Story-to-Code Tracer"

# Agents whose FAIL verdict directly impacts gate G4
_CRITICAL_AGENTS = {
    "10": "AC Compliance",
    "12": "Apex Coverage",
    "14": "Code Quality",
    "15": "Apex Security",
}

# Agents whose WARN/FAIL verdict is informational (not gate-blocking)
_ADVISORY_AGENTS = {
    "11": "Branch Tracer",
    "13": "Metadata Dependency",
    "16": "Bulk Quality",
    "17": "SFDX Validator",
    "20": "Performance Risk",
}

# ── Haiku tool ────────────────────────────────────────────────────────────────

_TRACE_TOOL_NAME = "generate_traceability_narrative"
_TRACE_TOOL_SCHEMA = {
    "type": "object",
    "required": ["narrative", "audit_summary"],
    "properties": {
        "narrative": {
            "type": "string",
            "description": (
                "3–4 sentences summarising the Development phase for the FCA audit trail. "
                "State the verdict for each critical check, note any blockers, and confirm "
                "whether the story is ready to proceed to Testing. Be precise and factual."
            ),
        },
        "audit_summary": {
            "type": "string",
            "description": (
                "One sentence audit summary suitable for a compliance report: "
                "e.g. 'Development phase PASSED for FSC-2417 with 90% Apex coverage, "
                "SFDX format valid, no security violations.'"
            ),
        },
    },
}

_TRACE_INSTRUCTIONS = """
You are generating an FCA-grade audit narrative for a Salesforce FSC development story.
You will receive a structured summary of all Development phase agent verdicts. Write a
precise 3–4 sentence narrative that a compliance auditor could rely on — state what
was checked, what passed, what failed, and whether the story is cleared to proceed.
Avoid vague language. Include specific metrics (coverage %, file counts, violation counts).
""".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(state: StoryState) -> AgentResult:
    story_id = state["story_id"]

    # ── Aggregate all Development agent signals ───────────────────────────────
    agent_signals = _collect_agent_signals(state)
    critical_failures, advisory_warnings = _classify_signals(agent_signals)
    dev_verdict = _determine_verdict(critical_failures, advisory_warnings, agent_signals)
    gate_g4 = _build_gate_g4_signals(agent_signals, critical_failures, advisory_warnings)
    escalation_required = len(critical_failures) > 0

    # ── Haiku trace ───────────────────────────────────────────────────────────
    trace_message = _build_trace_message(
        story_id, agent_signals, critical_failures, advisory_warnings, dev_verdict,
    )
    trace = await _generate_trace(trace_message)

    confidence_score, signals = _compute_confidence(
        agent_signals, critical_failures, dev_verdict,
    )
    escalated = confidence_score < settings.confidence_escalation_threshold

    trace_record = {
        "story_id": story_id,
        "development_verdict": dev_verdict,
        "agent_verdicts": agent_signals,
        "critical_failures": critical_failures,
        "advisory_warnings": advisory_warnings,
        "audit_summary": trace.get("audit_summary", ""),
        "narrative": trace.get("narrative", ""),
    }

    what = (
        f"Story-to-code trace for {story_id}: {len(agent_signals)} agent(s) assessed, "
        f"{len(critical_failures)} critical failure(s), {len(advisory_warnings)} advisory warning(s) "
        f"— development_verdict={dev_verdict}"
    )
    why = trace.get(
        "narrative",
        "Story-to-Code Tracer aggregated all Development phase agent verdicts.",
    )

    data = {
        "trace_record": trace_record,
        "development_verdict": dev_verdict,
        "gate_g4_signals": gate_g4,
        "escalation_required": escalation_required,
        "critical_failures": critical_failures,
        "advisory_warnings": advisory_warnings,
        "agents_assessed": len(agent_signals),
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
        model_used=settings.fast_model,
    )


# ── Deterministic aggregation ─────────────────────────────────────────────────

def _collect_agent_signals(state: StoryState) -> dict[str, dict]:
    """Gather verdict + key data from each Development agent."""
    signals: dict[str, dict] = {}
    all_dev_agents = {**_CRITICAL_AGENTS, **_ADVISORY_AGENTS}
    # Also pull from agents 18, 19, 21, 22, 23 (advisory / audit)
    extended = {**all_dev_agents, "18": "Component Attribution", "19": "BDD Gherkin",
                "21": "Test Data Architect", "22": "Sandbox State"}

    for agent_id, agent_name in extended.items():
        result = state["agent_results"].get(agent_id)
        if not result:
            continue
        agent_data = result.get("data", {})
        # Extract the primary verdict field by convention
        verdict = _extract_verdict(agent_id, agent_data)
        signals[agent_id] = {
            "agent_name": agent_name,
            "verdict": verdict,
            "data": agent_data,
        }
    return signals


def _extract_verdict(agent_id: str, data: dict) -> str:
    """Normalise the verdict field name across agents."""
    for key in (
        "coverage_verdict", "quality_verdict", "security_verdict",
        "sfdx_verdict", "component_verdict", "gherkin_verdict",
        "data_verdict", "performance_verdict", "sandbox_verdict",
        "branch_verdict", "ac_compliance_verdict", "verdict",
    ):
        if key in data:
            return data[key]
    return "UNKNOWN"


def _classify_signals(
    agent_signals: dict[str, dict],
) -> tuple[list[str], list[str]]:
    critical_failures: list[str] = []
    advisory_warnings: list[str] = []

    for agent_id, signal in agent_signals.items():
        verdict = signal["verdict"]
        name = signal["agent_name"]
        if agent_id in _CRITICAL_AGENTS and verdict in ("FAIL", "REVIEW_REQUIRED"):
            critical_failures.append(f"{name}: {verdict}")
        elif verdict in ("WARN", "PARTIAL", "REVIEW_REQUIRED", "INCOMPLETE"):
            advisory_warnings.append(f"{name}: {verdict}")

    return critical_failures, advisory_warnings


def _determine_verdict(
    critical_failures: list[str],
    advisory_warnings: list[str],
    agent_signals: dict[str, dict],
) -> str:
    if critical_failures:
        return "FAIL"
    if len(advisory_warnings) >= 3:
        return "PARTIAL"
    # Sandbox must be ready
    sandbox_signal = agent_signals.get("22", {})
    if sandbox_signal.get("verdict") == "BLOCKED":
        return "FAIL"
    return "PASS"


def _build_gate_g4_signals(
    agent_signals: dict[str, dict],
    critical_failures: list[str],
    advisory_warnings: list[str],
) -> dict:
    return {
        "coverage_passed": agent_signals.get("12", {}).get("data", {}).get("coverage_passed", False),
        "security_verdict": agent_signals.get("15", {}).get("verdict", "UNKNOWN"),
        "quality_verdict": agent_signals.get("14", {}).get("verdict", "UNKNOWN"),
        "sfdx_valid": agent_signals.get("17", {}).get("data", {}).get("sfdx_format_valid", True),
        "sandbox_ready": agent_signals.get("22", {}).get("data", {}).get("sandbox_ready", False),
        "critical_failure_count": len(critical_failures),
        "advisory_warning_count": len(advisory_warnings),
    }


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(
    agent_signals: dict[str, dict],
    critical_failures: list[str],
    dev_verdict: str,
) -> tuple[int, dict]:
    scorer = TierBScorer(base=65)

    agent_count = len(agent_signals)
    if agent_count >= 8:
        scorer.add("most_dev_agents_completed", agent_count, +12)
    elif agent_count >= 5:
        scorer.add("some_dev_agents_completed", agent_count, +6)
    else:
        scorer.add("few_dev_agents_completed", agent_count, -8)

    # Critical agents coverage
    critical_present = sum(1 for aid in _CRITICAL_AGENTS if aid in agent_signals)
    if critical_present == len(_CRITICAL_AGENTS):
        scorer.add("all_critical_agents_present", True, +10)
    else:
        scorer.add("some_critical_agents_missing", critical_present, -5)

    if critical_failures:
        penalty = min(len(critical_failures) * 5, 15)
        scorer.add("critical_failures_detected", len(critical_failures), -penalty)

    scorer.cap(92).floor(20)
    return scorer.build()


# ── Haiku trace generation ────────────────────────────────────────────────────

async def _generate_trace(user_message: str) -> dict:
    return await call_with_tool(
        model=settings.fast_model,
        system=build_system(_TRACE_INSTRUCTIONS),
        user_message=user_message,
        tool_name=_TRACE_TOOL_NAME,
        tool_description="Generate an FCA traceability narrative for the Development phase.",
        tool_schema=_TRACE_TOOL_SCHEMA,
        max_tokens=400,
    )


def _build_trace_message(
    story_id: str,
    agent_signals: dict[str, dict],
    critical_failures: list[str],
    advisory_warnings: list[str],
    dev_verdict: str,
) -> str:
    verdict_lines = [
        f"  Agent {aid} ({sig['agent_name']}): {sig['verdict']}"
        for aid, sig in sorted(agent_signals.items())
    ]
    return (
        f"Story: {story_id}\n"
        f"Development Phase Agent Verdicts:\n" + "\n".join(verdict_lines) + "\n"
        f"Critical Failures: {critical_failures or ['none']}\n"
        f"Advisory Warnings: {advisory_warnings or ['none']}\n"
        f"Development Verdict: {dev_verdict}\n\n"
        f"Generate a 3–4 sentence FCA audit narrative using the {_TRACE_TOOL_NAME} tool."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent_data(state: StoryState, agent_id: str) -> dict | None:
    result = state["agent_results"].get(agent_id)
    return result.get("data") if result else None
