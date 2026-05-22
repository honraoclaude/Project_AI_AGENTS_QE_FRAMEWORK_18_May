"""Tests for Agent 18 — Component Attribution Tracer (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_18_component_attribution import (
    _analyse_components,
    _compute_confidence,
    _extract_component_name,
    _infer_component_type,
    _is_regulated,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT11_FOUND = {"branch_found": True, "branch_name": "feature/FSC-2417-suitability-fix"}
AGENT11_NOT_FOUND = {"branch_found": False}

AGENT13_WITH_FILES = {
    "detected_objects": ["suitability__c"],
    "changed_files": [
        {"file_path": "force-app/main/default/classes/SuitabilityService.cls",
         "change_type": "modified", "author_email": "dev1@firm.com"},
        {"file_path": "force-app/main/default/triggers/FinancialAccountTrigger.trigger",
         "change_type": "added", "author_email": "dev2@firm.com"},
    ],
}

AGENT13_MERGE_RISK = {
    "detected_objects": ["financialaccount"],
    "changed_files": [
        {"file_path": "force-app/main/default/classes/FinancialAccountService.cls",
         "change_type": "modified", "author_email": "dev1@firm.com"},
        {"file_path": "force-app/main/default/classes/FinancialAccountService.cls",
         "change_type": "modified", "author_email": "dev2@firm.com"},
    ],
}

AGENT13_EMPTY = {
    "detected_objects": [],
    "changed_files": [],
}

MOCK_TRACE_PASS = {
    "narrative": "Two FSC components changed — SuitabilityService and FinancialAccountTrigger.",
    "attribution_concern": "regulated_components",
}

MOCK_TRACE_WARN = {
    "narrative": "Merge risk detected on FinancialAccountService.",
    "attribution_concern": "merge_risk",
}


# ── Component name extraction tests ──────────────────────────────────────────

class TestComponentNameExtraction:
    def test_extracts_apex_class_name(self):
        name = _extract_component_name("force-app/main/default/classes/SuitabilityService.cls")
        assert name == "SuitabilityService"

    def test_extracts_trigger_name(self):
        name = _extract_component_name("force-app/main/default/triggers/AccountTrigger.trigger")
        assert name == "AccountTrigger"

    def test_extracts_lwc_name(self):
        name = _extract_component_name("force-app/main/default/lwc/suitabilityWidget/suitabilityWidget.js")
        assert name == "suitabilityWidget"

    def test_empty_path_returns_empty(self):
        assert _extract_component_name("") == ""

    def test_unknown_extension_returns_filename(self):
        name = _extract_component_name("force-app/main/default/staticresources/myfile.zip")
        assert name == "myfile.zip"


# ── Component type inference tests ────────────────────────────────────────────

class TestComponentTypeInference:
    def test_cls_inferred_as_apex_class(self):
        assert _infer_component_type("classes/Foo.cls") == "ApexClass"

    def test_trigger_inferred_as_apex_trigger(self):
        assert _infer_component_type("triggers/Foo.trigger") == "ApexTrigger"

    def test_js_in_lwc_path_inferred_as_lwc(self):
        assert _infer_component_type("lwc/foo/foo.js") == "LWC"

    def test_flow_meta_xml_inferred_as_flow(self):
        assert _infer_component_type("flows/MyFlow.flow-meta.xml") == "Flow"

    def test_unknown_extension_returns_unknown(self):
        assert _infer_component_type("someFile.xyz") == "Unknown"


# ── Regulated component detection tests ──────────────────────────────────────

class TestRegulatedDetection:
    def test_suitability_is_regulated(self):
        assert _is_regulated("SuitabilityService") is True

    def test_riskprofile_is_regulated(self):
        assert _is_regulated("RiskProfileCalculator") is True

    def test_vulnerablecustomer_is_regulated(self):
        assert _is_regulated("VulnerableCustomerHandler") is True

    def test_generic_utility_not_regulated(self):
        assert _is_regulated("StringUtils") is False

    def test_financialaccount_is_regulated(self):
        assert _is_regulated("FinancialAccountService") is True

    def test_case_insensitive(self):
        assert _is_regulated("SUITABILITYSERVICE") is True


# ── Deterministic component analysis tests ───────────────────────────────────

class TestComponentAnalysis:
    def test_all_sfdx_files_gives_pass(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev@firm.com"}
        ]
        components, regulated, merge_risk, verdict, author_data_available = _analyse_components(files)
        assert verdict == "PASS"
        assert len(regulated) == 0
        assert len(merge_risk) == 0

    def test_regulated_component_gives_warn(self):
        files = [
            {"file_path": "force-app/main/default/classes/SuitabilityService.cls",
             "author_email": "dev@firm.com"}
        ]
        _, regulated, _, verdict, _ = _analyse_components(files)
        assert len(regulated) >= 1
        assert verdict == "WARN"

    def test_two_authors_on_same_file_gives_merge_risk(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev1@firm.com"},
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev2@firm.com"},
        ]
        _, _, merge_risk, verdict, _ = _analyse_components(files)
        assert len(merge_risk) >= 1
        assert verdict == "WARN"

    def test_regulated_plus_merge_risk_gives_review_required(self):
        files = [
            {"file_path": "force-app/main/default/classes/SuitabilityService.cls",
             "author_email": "dev1@firm.com"},
            {"file_path": "force-app/main/default/classes/SuitabilityService.cls",
             "author_email": "dev2@firm.com"},
        ]
        _, regulated, merge_risk, verdict, _ = _analyse_components(files)
        assert verdict == "REVIEW_REQUIRED"

    def test_empty_files_gives_pass(self):
        components, regulated, merge_risk, verdict, author_data_available = _analyse_components([])
        assert verdict == "PASS"
        assert len(components) == 0


# ── REQ-11: author_data_available tests ──────────────────────────────────────

class TestAuthorDataAvailableREQ11:
    def test_no_author_email_gives_author_data_unavailable(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls", "author_email": ""},
            {"file_path": "force-app/main/default/classes/Other.cls"},
        ]
        _, _, _, _, author_data_available = _analyse_components(files)
        assert author_data_available is False

    def test_with_author_email_gives_author_data_available(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev@firm.com"},
        ]
        _, _, _, _, author_data_available = _analyse_components(files)
        assert author_data_available is True

    def test_no_author_email_merge_risk_not_possible_flag_in_gaps(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls", "author_email": ""},
        ]
        components, _, _, verdict, _ = _analyse_components(files)
        # verdict is PASS (no regulated, no merge risk detected)
        assert verdict == "PASS"

    def test_two_authors_merge_risk_detected_correctly(self):
        files = [
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev1@firm.com"},
            {"file_path": "force-app/main/default/classes/Utils.cls",
             "author_email": "dev2@firm.com"},
        ]
        _, _, merge_risk, verdict, author_data_available = _analyse_components(files)
        assert author_data_available is True
        assert len(merge_risk) >= 1

    @pytest.mark.asyncio
    async def test_author_data_available_in_run_output(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_WITH_FILES}

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "author_data_available" in result.data
        assert result.data["author_data_available"] is True


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_full_context_scores_well(self):
        files = AGENT13_WITH_FILES["changed_files"]
        score, _ = _compute_confidence(
            files, AGENT11_FOUND, AGENT13_WITH_FILES, ["SuitabilityService"], []
        )
        assert score >= 65

    def test_no_files_reduces_confidence(self):
        score_with, _ = _compute_confidence(
            AGENT13_WITH_FILES["changed_files"], AGENT11_FOUND, AGENT13_WITH_FILES, [], []
        )
        score_without, _ = _compute_confidence([], None, None, [], [])
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            AGENT13_WITH_FILES["changed_files"], AGENT11_FOUND, AGENT13_WITH_FILES, [], []
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence([], None, None, [], [])
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_WITH_FILES}

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 18
        assert result.agent_name == "Component Attribution Tracer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["changed_components", "regulated_components",
                    "merge_risk_components", "component_verdict"]:
            assert key in result.data

    async def test_regulated_components_detected(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_WITH_FILES}

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert len(result.data["regulated_components"]) >= 1

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_empty_files_gives_pass_verdict(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_EMPTY}

        with patch("src.agents.development.agent_18_component_attribution.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["component_verdict"] == "PASS"
