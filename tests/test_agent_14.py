"""Tests for Agent 14 — Code Quality Reviewer (True AI, PMD augmented)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_14_code_quality import _compute_confidence, run
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
