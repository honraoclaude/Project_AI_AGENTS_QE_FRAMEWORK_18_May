"""
Tests for Agent 5B — AC Challenger (Adversarial).

Uses mock LLM so tests run without live infrastructure.
Tests: confidence scoring, no-AC fallback, finding severity, survivor count,
required output keys, state key for Agent 5 output, empty challenge_findings path.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_05b_ac_challenger import (
    _build_user_message,
    _FINDING_SCHEMA,
    _TOOL_NAME,
    _TOOL_SCHEMA,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_AC_CLAUSES = [
    {
        "scenario": "Scenario: View consolidated suitability score for a fully assessed client",
        "scenario_type": "happy_path",
        "test_category": "AUTOMATION_CANDIDATE",
        "fca_relevant": True,
        "given": ["Given a client has a complete Suitability__c record"],
        "when": ["When the Wealth Manager opens the suitability-dashboard LWC"],
        "then": ["Then the ConsolidatedSuitabilityScore__c is displayed between 0 and 100"],
    },
    {
        "scenario": "Scenario: COBS 9.2 mandatory warning gate for Retail Client",
        "scenario_type": "regulatory",
        "test_category": "FUNCTIONAL",
        "fca_relevant": True,
        "given": [
            "Given a client is classified as Retail Client",
            "And ConsolidatedSuitabilityScore__c = 44 (below threshold)",
        ],
        "when": ["When the Wealth Manager clicks 'Record Advice'"],
        "then": [
            "Then a mandatory COBS 9.2 warning modal is displayed",
            "And a COBS_9_2_Acknowledgement__c audit record is created",
        ],
    },
    {
        "scenario": "Scenario: Score calculation blocked when RiskProfile__c is absent",
        "scenario_type": "error_path",
        "test_category": "UNIT",
        "fca_relevant": True,
        "given": ["Given a client's FinancialAccount has no linked RiskProfile__c"],
        "when": ["When the SuitabilityScoreCalculator is invoked"],
        "then": [
            "Then SuitabilityScoreCalculator throws SuitabilityCalculationException",
            "And ConsolidatedSuitabilityScore__c is not updated",
        ],
    },
]

MOCK_CHALLENGE_RESULT_NO_FINDINGS = {
    "challenge_summary": "All AC clauses are strong with no critical or major weaknesses.",
    "survivor_count": 3,
    "critical_weakness_count": 0,
    "challenge_findings": [],
}

MOCK_CHALLENGE_RESULT_WITH_FINDINGS = {
    "challenge_summary": "Two clauses have weaknesses that require attention.",
    "survivor_count": 1,
    "critical_weakness_count": 1,
    "challenge_findings": [
        {
            "clause_index": 0,
            "scenario": "Scenario: View consolidated suitability score for a fully assessed client",
            "weakness_type": "ambiguous_given",
            "severity": "CRITICAL",
            "detail": "Given step does not specify the values of objectives_score__c or risk_level — cannot reproduce reliably.",
        },
        {
            "clause_index": 1,
            "scenario": "Scenario: COBS 9.2 mandatory warning gate for Retail Client",
            "weakness_type": "weak_regulatory",
            "severity": "MAJOR",
            "detail": "Regulatory clause does not reference COBS 9.2.4 specifically — auditor cannot map to the exact rule.",
        },
    ],
}


def _state_with_agent5_clauses(clauses: list) -> dict:
    state = initial_story_state("FSC-2417")
    state["fca_classification"] = "HIGH"
    state["agent_results"]["5"] = {
        "data": {
            "ac_clauses": clauses,
            "fca_classification_context": "HIGH",
            "generation_mode": "supplemented_existing",
        },
        "confidence": {"final_score": 85},
    }
    return state


def _state_no_agent5() -> dict:
    state = initial_story_state("FSC-2417")
    state["fca_classification"] = "HIGH"
    return state


# ── Tests: run() — no AC clauses available ────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_early_when_no_ac_clauses():
    """If Agent 5 produced no clauses, Agent 5B skips without calling the LLM."""
    state = _state_no_agent5()
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
    ) as mock_call:
        result = await run(state)

    mock_call.assert_not_called()
    assert result.data["ac_count_challenged"] == 0
    assert result.data["survivor_count"] == 0
    assert result.data["challenge_findings"] == []
    assert result.confidence.final_score == 20
    assert result.confidence.escalated is True


@pytest.mark.asyncio
async def test_run_returns_early_when_agent5_has_empty_clauses():
    state = _state_with_agent5_clauses([])
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
    ) as mock_call:
        result = await run(state)

    mock_call.assert_not_called()
    assert result.data["ac_count_challenged"] == 0


# ── Tests: run() — with AC clauses ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_all_clauses_survive_no_findings():
    """All clauses pass the challenge — survivor_count equals clause count."""
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_NO_FINDINGS,
    ):
        result = await run(state)

    assert result.data["ac_count_challenged"] == 3
    assert result.data["survivor_count"] == 3
    assert result.data["critical_weakness_count"] == 0
    assert result.data["challenge_findings"] == []


@pytest.mark.asyncio
async def test_run_with_critical_findings():
    """Critical and major findings are returned; survivor_count is reduced."""
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_WITH_FINDINGS,
    ):
        result = await run(state)

    assert result.data["critical_weakness_count"] == 1
    assert result.data["survivor_count"] == 1
    assert len(result.data["challenge_findings"]) == 2


# ── Tests: required output keys ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_output_has_all_required_keys():
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_NO_FINDINGS,
    ):
        result = await run(state)

    required_keys = {
        "ac_count_challenged",
        "survivor_count",
        "critical_weakness_count",
        "challenge_findings",
        "challenge_summary",
        "fca_classification_context",
        "signals",
    }
    assert required_keys.issubset(set(result.data.keys()))


@pytest.mark.asyncio
async def test_agent_id_and_name():
    state = _state_no_agent5()
    with patch("src.agents.refinement.agent_05b_ac_challenger.call_with_tool", new_callable=AsyncMock):
        result = await run(state)

    assert result.agent_id == 54
    assert result.agent_name == "AC Challenger"


# ── Tests: finding structure validation ───────────────────────────────────────

@pytest.mark.asyncio
async def test_each_finding_has_required_fields():
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_WITH_FINDINGS,
    ):
        result = await run(state)

    valid_weakness_types = {
        "ambiguous_given", "non_observable_then", "missing_error_path",
        "weak_regulatory", "untestable",
    }
    valid_severities = {"CRITICAL", "MAJOR", "MINOR"}

    for finding in result.data["challenge_findings"]:
        assert "clause_index" in finding
        assert "scenario" in finding
        assert finding.get("weakness_type") in valid_weakness_types
        assert finding.get("severity") in valid_severities
        assert "detail" in finding and len(finding["detail"]) > 0


# ── Tests: _compute_confidence ────────────────────────────────────────────────

def test_confidence_high_survival_no_critical():
    """High survival rate + no critical weaknesses → high confidence."""
    score, signals = _compute_confidence(
        ac_count=8, survivor_count=8, critical_count=0, findings=[]
    )
    assert score >= 75
    assert signals.get("high_survival_rate") is not None or signals.get("rich_ac_set") is not None


def test_confidence_low_survival_critical_issues():
    """Low survival rate + critical weaknesses → lower confidence (still grounded — found real issues)."""
    findings = [
        {"weakness_type": "ambiguous_given", "severity": "CRITICAL"},
        {"weakness_type": "non_observable_then", "severity": "CRITICAL"},
        {"weakness_type": "weak_regulatory", "severity": "MAJOR"},
        {"weakness_type": "missing_error_path", "severity": "MAJOR"},
    ]
    score, signals = _compute_confidence(
        ac_count=4, survivor_count=0, critical_count=2, findings=findings
    )
    # Critical issues found (rewarded), but low survival rate (penalised)
    assert score >= 20  # floor applied
    assert "low_survival_rate" in signals
    assert "critical_weaknesses_found" in signals


def test_confidence_diverse_findings_boost():
    """Multiple weakness types → diverse review → confidence boost."""
    findings = [
        {"weakness_type": "ambiguous_given", "severity": "MAJOR"},
        {"weakness_type": "non_observable_then", "severity": "MINOR"},
        {"weakness_type": "weak_regulatory", "severity": "MAJOR"},
        {"weakness_type": "missing_error_path", "severity": "MINOR"},
    ]
    score, signals = _compute_confidence(
        ac_count=6, survivor_count=3, critical_count=0, findings=findings
    )
    assert "diverse_findings" in signals


def test_confidence_floor_enforced():
    """Confidence never drops below 20 even with worst-case signals."""
    score, _ = _compute_confidence(ac_count=0, survivor_count=0, critical_count=0, findings=[])
    assert score >= 20


def test_confidence_cap_enforced():
    """Confidence never exceeds 92 even with all-positive signals."""
    score, _ = _compute_confidence(ac_count=10, survivor_count=10, critical_count=5, findings=[
        {"weakness_type": "ambiguous_given"},
        {"weakness_type": "non_observable_then"},
        {"weakness_type": "weak_regulatory"},
    ])
    assert score <= 92


def test_confidence_signals_dict_populated():
    """Signals dict is always a non-empty dict."""
    _, signals = _compute_confidence(
        ac_count=3, survivor_count=3, critical_count=0, findings=[]
    )
    assert isinstance(signals, dict)
    assert len(signals) > 0


# ── Tests: untested signal branches ───────────────────────────────────────────

def test_adequate_ac_set_signal_in_signals():
    _, signals = _compute_confidence(ac_count=4, survivor_count=4, critical_count=0, findings=[])
    assert "adequate_ac_set" in signals


def test_moderate_survival_rate_signal_in_signals():
    _, signals = _compute_confidence(ac_count=4, survivor_count=2, critical_count=0, findings=[])
    assert "moderate_survival_rate" in signals


def test_no_acs_to_challenge_signal_in_signals():
    _, signals = _compute_confidence(ac_count=0, survivor_count=0, critical_count=0, findings=[])
    assert "no_acs_to_challenge" in signals


# ── Tests: signal stored values ───────────────────────────────────────────────

def test_rich_ac_set_stores_count():
    _, signals = _compute_confidence(ac_count=10, survivor_count=10, critical_count=0, findings=[])
    assert signals["rich_ac_set"] == 10


def test_high_survival_rate_stores_rounded_rate():
    _, signals = _compute_confidence(ac_count=8, survivor_count=8, critical_count=0, findings=[])
    assert signals["high_survival_rate"] == 1.0


def test_diverse_findings_stores_type_count():
    findings = [
        {"weakness_type": "ambiguous_given"},
        {"weakness_type": "non_observable_then"},
        {"weakness_type": "weak_regulatory"},
    ]
    _, signals = _compute_confidence(ac_count=6, survivor_count=3, critical_count=0, findings=findings)
    assert signals["diverse_findings"] == 3


# ── Tests: integration gaps ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_used_and_tier():
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_NO_FINDINGS,
    ):
        result = await run(state)

    assert result.model_used == "claude-sonnet-4-6"
    assert result.confidence.tier == "B"


@pytest.mark.asyncio
async def test_challenge_summary_is_non_empty_string():
    state = _state_with_agent5_clauses(SAMPLE_AC_CLAUSES)
    with patch(
        "src.agents.refinement.agent_05b_ac_challenger.call_with_tool",
        new_callable=AsyncMock,
        return_value=MOCK_CHALLENGE_RESULT_NO_FINDINGS,
    ):
        result = await run(state)

    assert isinstance(result.data["challenge_summary"], str)
    assert len(result.data["challenge_summary"]) > 0


# ── Tests: prompt content ─────────────────────────────────────────────────────

class TestPromptContent:
    def test_prompt_includes_story_id(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert "FSC-2417" in msg

    def test_prompt_includes_fca_class(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert "HIGH" in msg

    def test_prompt_includes_clause_count(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert str(len(SAMPLE_AC_CLAUSES)) in msg

    def test_prompt_includes_ac_clauses_section(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert "AC CLAUSES TO CHALLENGE:" in msg

    def test_prompt_includes_clause_index(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert "[Clause 0]" in msg

    def test_prompt_includes_scenario_type_and_category(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert "scenario_type" in msg
        assert "test_category" in msg

    def test_prompt_ends_with_tool_instruction(self):
        msg = _build_user_message("FSC-2417", "HIGH", SAMPLE_AC_CLAUSES)
        assert _TOOL_NAME in msg
        assert msg.strip().endswith("challenge_findings.")


# ── Tests: schema contract ────────────────────────────────────────────────────

class TestSchemaContract:
    def test_tool_schema_has_four_required_fields(self):
        expected = {
            "challenge_summary",
            "survivor_count",
            "critical_weakness_count",
            "challenge_findings",
        }
        assert set(_TOOL_SCHEMA["required"]) == expected

    def test_challenge_summary_is_string(self):
        assert _TOOL_SCHEMA["properties"]["challenge_summary"]["type"] == "string"

    def test_survivor_count_has_minimum_zero(self):
        assert _TOOL_SCHEMA["properties"]["survivor_count"]["minimum"] == 0

    def test_critical_weakness_count_has_minimum_zero(self):
        assert _TOOL_SCHEMA["properties"]["critical_weakness_count"]["minimum"] == 0

    def test_challenge_findings_is_array(self):
        assert _TOOL_SCHEMA["properties"]["challenge_findings"]["type"] == "array"

    def test_finding_schema_has_five_required_fields(self):
        expected = {"clause_index", "scenario", "weakness_type", "severity", "detail"}
        assert set(_FINDING_SCHEMA["required"]) == expected

    def test_weakness_type_enum_has_five_values(self):
        schema = _FINDING_SCHEMA["properties"]["weakness_type"]
        assert schema["enum"] == [
            "ambiguous_given",
            "non_observable_then",
            "missing_error_path",
            "weak_regulatory",
            "untestable",
        ]

    def test_severity_enum_has_three_values(self):
        schema = _FINDING_SCHEMA["properties"]["severity"]
        assert schema["enum"] == ["CRITICAL", "MAJOR", "MINOR"]
