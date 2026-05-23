"""Tests for Agent 14 — Code Quality Reviewer (True AI, PMD augmented)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_14_code_quality import (
    _build_user_message,
    _compute_confidence,
    _pmd_baseline_verdict,
    _TOOL_NAME,
    _TOOL_SCHEMA,
    _VIOLATION_ITEM_SCHEMA,
    _worse_verdict,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT3_HIGH = {"fca_classification": "HIGH"}
AGENT3_LOW  = {"fca_classification": "LOW"}

AGENT13_DATA = {
    "detected_objects": ["suitability__c", "riskprofile__c"],
    "changed_files_count": 3,
}

PMD_CLEAN = []

PMD_CRITICAL = [
    {"rule_name": "ApexSOQLInjection", "priority": 1,
     "description": "SOQL injection risk in dynamic query",
     "file_path": "force-app/main/default/classes/SuitabilityService.cls", "line": 42,
     "category": "Security"},
]

PMD_HIGH_VIOLATIONS = [
    {"rule_name": "ApexCRUDViolation", "priority": 2,
     "description": "Apex CRUD violation — no stripInaccessible call",
     "file_path": "force-app/main/default/classes/SuitabilityService.cls", "line": 87,
     "category": "Security"},
    {"rule_name": "ApexCRUDViolation", "priority": 2,
     "description": "Apex CRUD violation — DML without FLS check",
     "file_path": "force-app/main/default/classes/RiskProfileHelper.cls", "line": 23,
     "category": "Security"},
    {"rule_name": "ApexSharingViolations", "priority": 2,
     "description": "Class is missing with sharing keyword",
     "file_path": "force-app/main/default/classes/SuitabilityService.cls", "line": 1,
     "category": "Security"},
]

MOCK_RESULT_PASS = {
    "critical_violations": [],
    "high_violations": [],
    "quality_verdict": "PASS",
    "quality_summary": "No PMD violations detected. Code meets FSC quality standards.",
    "recommended_fixes": ["Maintain consistent with sharing keywords on all Apex classes."],
}

MOCK_RESULT_FAIL = {
    "critical_violations": [
        {"rule_name": "ApexSOQLInjection",
         "description": "SOQL injection risk",
         "file_path": "force-app/.../SuitabilityService.cls", "line": 42},
    ],
    "high_violations": [],
    "quality_verdict": "FAIL",
    "quality_summary": "Critical SOQL injection vulnerability detected. Must be fixed before promotion.",
    "recommended_fixes": [
        "Replace dynamic SOQL with bind variables in SuitabilityService.cls line 42.",
    ],
}

PMD_ALWAYS_FAIL_P2 = [
    {
        "rule_name": "ApexXSSFromURLParam", "priority": 2,
        "description": "XSS risk from URL parameter",
        "file_path": "force-app/main/default/classes/PortalController.cls", "line": 18,
        "category": "Security",
    },
]

PMD_ONE_P2 = [
    {
        "rule_name": "ApexCyclomaticComplexity", "priority": 2,
        "description": "Method complexity too high",
        "file_path": "force-app/main/default/classes/RiskHelper.cls", "line": 55,
        "category": "Design",
    },
]

PMD_P3_ONLY = [
    {
        "rule_name": "VariableNamingConventions", "priority": 3,
        "description": "Variable name does not conform to convention",
        "file_path": "force-app/main/default/classes/SuitabilityService.cls", "line": 12,
        "category": "Code Style",
    },
]

MOCK_RESULT_WARN = {
    "critical_violations": [],
    "high_violations": [
        {"rule_name": "ApexCRUDViolation",
         "description": "CRUD violation", "file_path": "SuitabilityService.cls", "line": 87},
    ],
    "quality_verdict": "WARN",
    "quality_summary": "CRUD violations present. Review required before HIGH-FCA promotion.",
    "recommended_fixes": ["Add stripInaccessible() calls before DML operations."],
}


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_clean_code_scores_high(self):
        score, _ = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert score >= 70

    def test_critical_violations_reduce_confidence(self):
        score_clean, _ = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        score_critical, _ = _compute_confidence(PMD_CRITICAL, MOCK_RESULT_FAIL, AGENT13_DATA)
        assert score_clean > score_critical

    def test_pass_verdict_boosts_confidence(self):
        score_pass, _ = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        score_fail, _ = _compute_confidence(PMD_CRITICAL, MOCK_RESULT_FAIL, AGENT13_DATA)
        assert score_pass > score_fail

    def test_recommended_fixes_boost_confidence(self):
        no_fixes = {**MOCK_RESULT_PASS, "recommended_fixes": []}
        score_with, _ = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        score_without, _ = _compute_confidence(PMD_CLEAN, no_fixes, AGENT13_DATA)
        assert score_with >= score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(PMD_CRITICAL, MOCK_RESULT_FAIL, None)
        assert score >= 20

    def test_metadata_context_available_key_in_signals(self):
        _, signals = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert "metadata_context_available" in signals

    def test_no_pmd_violations_key_in_signals(self):
        _, signals = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert "no_pmd_violations" in signals

    def test_critical_pmd_violations_present_key_in_signals(self):
        _, signals = _compute_confidence(PMD_CRITICAL, MOCK_RESULT_FAIL, AGENT13_DATA)
        assert "critical_pmd_violations_present" in signals

    def test_quality_verdict_pass_key_in_signals(self):
        _, signals = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert "quality_verdict_pass" in signals

    def test_quality_verdict_fail_key_in_signals(self):
        _, signals = _compute_confidence(PMD_CRITICAL, MOCK_RESULT_FAIL, AGENT13_DATA)
        assert "quality_verdict_fail" in signals

    def test_recommended_fixes_present_stores_count(self):
        _, signals = _compute_confidence(PMD_CLEAN, MOCK_RESULT_PASS, AGENT13_DATA)
        assert signals["recommended_fixes_present"] == len(MOCK_RESULT_PASS["recommended_fixes"])


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}
        state["agent_results"]["13"] = {"data": AGENT13_DATA}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.agent_id == 14
        assert result.agent_name == "Code Quality Reviewer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        for key in ["quality_verdict", "critical_violations",
                    "total_violation_count", "quality_summary", "recommended_fixes"]:
            assert key in result.data

    async def test_pass_verdict_propagated(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.data["quality_verdict"] == "PASS"
        assert result.data["total_violation_count"] == 0

    async def test_fail_verdict_with_critical_violations(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CRITICAL
            mock_llm.return_value = MOCK_RESULT_FAIL
            result = await run(state)

        assert result.data["quality_verdict"] == "FAIL"
        assert len(result.data["critical_violations"]) == 1

    async def test_standalone_mode(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.agent_id == 14

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_escalated_when_critical_violations_and_no_metadata(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CRITICAL
            mock_llm.return_value = MOCK_RESULT_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_fca_classification_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.data["fca_classification"] == "HIGH"


# ── Ensemble and TA integration tests ────────────────────────────────────────

@pytest.mark.asyncio
class TestEnsembleAndTA:
    async def test_ensemble_agreement_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert "ensemble_agreement" in result.data

    async def test_ta_position_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert "ta_position" in result.data
        assert "interaction_mode" in result.data
        assert result.data["ta_position"] in ("OK_OK", "OK_NOT_OK", "NOT_OK_OK", "NOT_OK_NOT_OK")

    async def test_call_a_and_call_b_verdicts_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert "call_a_verdict" in result.data
        assert "call_b_verdict" in result.data
        assert result.data["call_a_verdict"] in ("PASS", "WARN", "FAIL")

    async def test_clean_pmd_produces_pass_call_a_verdict(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CLEAN
            mock_llm.return_value = MOCK_RESULT_PASS
            result = await run(state)

        assert result.data["call_a_verdict"] == "PASS"

    async def test_critical_violations_produce_fail_call_a_verdict(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["3"] = {"data": AGENT3_HIGH}

        with (
            patch("src.agents.development.agent_14_code_quality.get_pmd_results",
                  new_callable=AsyncMock) as mock_pmd,
            patch("src.agents.development.agent_14_code_quality.call_with_tool",
                  new_callable=AsyncMock) as mock_llm,
        ):
            mock_pmd.return_value = PMD_CRITICAL
            mock_llm.return_value = MOCK_RESULT_FAIL
            result = await run(state)

        assert result.data["call_a_verdict"] == "FAIL"


# ── _pmd_baseline_verdict unit tests ─────────────────────────────────────────

class TestPmdBaselineVerdict:
    def test_priority_1_gives_fail(self):
        verdict, conf = _pmd_baseline_verdict(PMD_CRITICAL, "HIGH")
        assert verdict == "FAIL"
        assert conf == 80

    def test_always_fail_rule_at_priority2_gives_fail(self):
        # ApexXSSFromURLParam is in _ALWAYS_FAIL_RULES but priority != 1
        verdict, conf = _pmd_baseline_verdict(PMD_ALWAYS_FAIL_P2, "LOW")
        assert verdict == "FAIL"
        assert conf == 85

    def test_crud_violation_high_fca_gives_fail(self):
        verdict, conf = _pmd_baseline_verdict(PMD_HIGH_VIOLATIONS, "HIGH")
        assert verdict == "FAIL"
        assert conf == 80

    def test_crud_violation_medium_fca_gives_fail(self):
        verdict, conf = _pmd_baseline_verdict(PMD_HIGH_VIOLATIONS, "MEDIUM")
        assert verdict == "FAIL"
        assert conf == 80

    def test_crud_violation_low_fca_gives_warn(self):
        # CRUD violations do NOT trigger FAIL on LOW FCA — falls through to p2_count
        verdict, _ = _pmd_baseline_verdict(PMD_HIGH_VIOLATIONS, "LOW")
        assert verdict == "WARN"

    def test_three_p2_violations_gives_warn_75(self):
        verdict, conf = _pmd_baseline_verdict(PMD_HIGH_VIOLATIONS, "LOW")
        assert verdict == "WARN"
        assert conf == 75

    def test_one_p2_violation_gives_warn_70(self):
        verdict, conf = _pmd_baseline_verdict(PMD_ONE_P2, "LOW")
        assert verdict == "WARN"
        assert conf == 70

    def test_clean_gives_pass_72(self):
        verdict, conf = _pmd_baseline_verdict(PMD_CLEAN, "LOW")
        assert verdict == "PASS"
        assert conf == 72

    def test_p3_only_gives_pass_78(self):
        # Non-empty violations list but no p1/p2 — returns PASS with higher confidence
        verdict, conf = _pmd_baseline_verdict(PMD_P3_ONLY, "LOW")
        assert verdict == "PASS"
        assert conf == 78


# ── _worse_verdict unit tests ─────────────────────────────────────────────────

class TestWorseVerdict:
    def test_fail_beats_warn(self):
        assert _worse_verdict("FAIL", "WARN") == "FAIL"

    def test_fail_beats_pass(self):
        assert _worse_verdict("PASS", "FAIL") == "FAIL"

    def test_warn_beats_pass(self):
        assert _worse_verdict("WARN", "PASS") == "WARN"

    def test_equal_verdicts_returns_first(self):
        assert _worse_verdict("WARN", "WARN") == "WARN"


# ── _build_user_message unit tests ────────────────────────────────────────────

class TestBuildUserMessage:
    def test_includes_story_id(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, None)
        assert "FSC-2417" in msg

    def test_includes_fca_class(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, None)
        assert "HIGH" in msg

    def test_violations_listed_when_present(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CRITICAL, None)
        assert "[P1]" in msg
        assert "ApexSOQLInjection" in msg

    def test_no_violations_shows_none_placeholder(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, None)
        assert "(none)" in msg

    def test_detected_objects_from_agent13_shown(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, AGENT13_DATA)
        assert "suitability__c" in msg

    def test_no_agent13_shows_none_for_objects(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, None)
        assert "none" in msg

    def test_violation_format_includes_priority_rule_and_file(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CRITICAL, None)
        assert "ApexSOQLInjection" in msg
        assert "force-app" in msg

    def test_ends_with_tool_name(self):
        msg = _build_user_message("FSC-2417", "HIGH", PMD_CLEAN, None)
        assert _TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_five_required_fields(self):
        assert set(_TOOL_SCHEMA["required"]) == {
            "critical_violations", "high_violations",
            "quality_verdict", "quality_summary", "recommended_fixes",
        }

    def test_quality_verdict_enum_has_three_values(self):
        assert _TOOL_SCHEMA["properties"]["quality_verdict"]["enum"] == ["PASS", "WARN", "FAIL"]

    def test_violation_item_has_four_required_fields(self):
        assert set(_VIOLATION_ITEM_SCHEMA["required"]) == {
            "rule_name", "description", "file_path", "line"
        }

    def test_quality_summary_is_string(self):
        assert _TOOL_SCHEMA["properties"]["quality_summary"]["type"] == "string"

    def test_recommended_fixes_is_array(self):
        assert _TOOL_SCHEMA["properties"]["recommended_fixes"]["type"] == "array"
