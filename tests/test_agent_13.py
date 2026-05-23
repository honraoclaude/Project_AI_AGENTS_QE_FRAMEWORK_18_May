"""
Tests for Agent 13 — Metadata Dependency Mapper (Augmented Script).

The deterministic detection and BFS functions are the primary test targets.
Integration tests mock Copado calls and the Haiku narrative call.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_13_metadata_dependency import (
    _analyse_metadata,
    _build_trace_message,
    _compute_confidence,
    _detect_objects_from_files,
    _run_dependency_bfs,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHANGED_FILES_SUITABILITY = [
    {
        "file_path": "force-app/main/default/classes/SuitabilityService.cls",
        "change_type": "modify",
        "object_type": "ApexClass",
        "object_name": "SuitabilityService",
    },
    {
        "file_path": "force-app/main/default/objects/RiskProfile__c/fields/RiskLevel__c.field-meta.xml",
        "change_type": "modify",
        "object_type": "CustomField",
        "object_name": "RiskProfile__c",
    },
    {
        "file_path": "force-app/main/default/flows/SuitabilityAssessment.flow-meta.xml",
        "change_type": "add",
        "object_type": "Flow",
        "object_name": "SuitabilityAssessment",
    },
]

CHANGED_FILES_FINANCIAL = [
    {
        "file_path": "force-app/main/default/classes/FinancialAccountService.cls",
        "change_type": "modify",
        "object_type": "ApexClass",
        "object_name": "FinancialAccountService",
    },
    {
        "file_path": "force-app/main/default/classes/FinancialHoldingTrigger.cls",
        "change_type": "add",
        "object_type": "ApexClass",
        "object_name": "FinancialHoldingTrigger",
    },
]

CHANGED_FILES_EMPTY = []

CHANGED_FILES_NO_FSC = [
    {
        "file_path": "force-app/main/default/classes/ButtonLabelHelper.cls",
        "change_type": "modify",
        "object_type": "ApexClass",
        "object_name": "ButtonLabelHelper",
    },
]

AGENT8_DATA_SUITABILITY = {
    "detected_objects": ["suitability__c", "riskprofile__c"],
    "implied_objects": ["financialaccount"],
    "dependency_depth": 2,
}

AGENT11_DATA = {
    "branch_name": "feature/FSC-2417-suitability-assessment",
    "branch_found": True,
}

MOCK_TRACE = {
    "narrative": "Suitability__c and RiskProfile__c were detected in changed metadata, matching the Refinement prediction.",
    "dependency_complexity": "medium",
}

CHANGED_FILES_DESTRUCTIVE = [
    {
        "file_path": "force-app/main/default/classes/OldService.cls",
        "change_type": "delete",
        "object_type": "ApexClass",
        "object_name": "OldService",
    },
]

CHANGED_FILES_SINGLE_OBJECT = [
    {
        "file_path": "force-app/main/default/classes/SuitabilityService.cls",
        "change_type": "modify",
        "object_type": "ApexClass",
        "object_name": "SuitabilityService",
    },
]


# ── Object detection tests (no LLM, no network) ───────────────────────────────

class TestObjectDetection:
    def test_detects_suitability_from_class_name(self):
        detected = _detect_objects_from_files(CHANGED_FILES_SUITABILITY)
        assert "suitability__c" in detected

    def test_detects_riskprofile_from_field_path(self):
        detected = _detect_objects_from_files(CHANGED_FILES_SUITABILITY)
        assert "riskprofile__c" in detected

    def test_detects_financial_objects(self):
        detected = _detect_objects_from_files(CHANGED_FILES_FINANCIAL)
        assert "financialaccount" in detected
        assert "financialholding" in detected

    def test_no_fsc_objects_in_non_fsc_files(self):
        detected = _detect_objects_from_files(CHANGED_FILES_NO_FSC)
        assert len(detected) == 0

    def test_empty_files_returns_empty_set(self):
        detected = _detect_objects_from_files(CHANGED_FILES_EMPTY)
        assert len(detected) == 0

    def test_detection_is_case_insensitive(self):
        upper_files = [
            {**f, "file_path": f["file_path"].upper(), "object_name": f["object_name"].upper()}
            for f in CHANGED_FILES_SUITABILITY
        ]
        detected_upper = _detect_objects_from_files(upper_files)
        detected_lower = _detect_objects_from_files(CHANGED_FILES_SUITABILITY)
        assert set(detected_upper) == set(detected_lower)


# ── BFS dependency tests ──────────────────────────────────────────────────────

class TestDependencyBFS:
    def test_suitability_implies_financialaccount(self):
        detected, implied, _, _ = _run_dependency_bfs({"suitability__c"})
        assert "financialaccount" in implied

    def test_depth_nonzero_for_objects_with_parents(self):
        _, _, _, depth = _run_dependency_bfs({"suitability__c"})
        assert depth >= 1

    def test_depth_zero_for_base_objects(self):
        _, _, _, depth = _run_dependency_bfs({"individual"})
        assert depth == 0

    def test_household_and_individual_excluded_from_implied(self):
        _, implied, _, _ = _run_dependency_bfs({"suitability__c"})
        assert "individual" not in implied
        assert "household" not in implied

    def test_empty_detected_gives_empty_results(self):
        detected, implied, graph, depth = _run_dependency_bfs(set())
        assert detected == []
        assert implied == []
        assert depth == 0


# ── Full metadata analysis tests ──────────────────────────────────────────────

class TestMetadataAnalysis:
    def test_scope_matches_refinement_when_no_new_objects(self):
        detected, implied, _, _, scope_delta = _analyse_metadata(
            CHANGED_FILES_SUITABILITY, AGENT8_DATA_SUITABILITY
        )
        # All code objects were predicted in refinement
        assert scope_delta == []

    def test_scope_delta_detected_when_new_objects_appear(self):
        # Agent 8 only predicted suitability; code also touches financialholding
        agent8_narrow = {
            "detected_objects": ["suitability__c"],
            "implied_objects": [],
        }
        _, _, _, _, scope_delta = _analyse_metadata(CHANGED_FILES_FINANCIAL, agent8_narrow)
        assert len(scope_delta) > 0

    def test_no_refinement_baseline_gives_empty_scope_delta(self):
        _, _, _, _, scope_delta = _analyse_metadata(CHANGED_FILES_SUITABILITY, None)
        assert scope_delta == []

    def test_no_changed_files_gives_empty_detection(self):
        detected, implied, _, depth, _ = _analyse_metadata(CHANGED_FILES_EMPTY, None)
        assert detected == []
        assert implied == []
        assert depth == 0


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_rich_files_with_objects_scores_well(self):
        score, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY,
            ["suitability__c", "riskprofile__c"],
            ["financialaccount"],
            2,
            [],
            AGENT8_DATA_SUITABILITY,
        )
        assert score >= 75

    def test_no_changed_files_heavily_penalised(self):
        score_with, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], None
        )
        score_without, _ = _compute_confidence(
            CHANGED_FILES_EMPTY, [], [], 0, [], None
        )
        assert score_with > score_without

    def test_scope_delta_reduces_confidence(self):
        score_clean, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], AGENT8_DATA_SUITABILITY
        )
        score_delta, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, ["revenue__c"], AGENT8_DATA_SUITABILITY
        )
        assert score_clean > score_delta

    def test_scope_matches_refinement_boosts_confidence(self):
        score_match, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], AGENT8_DATA_SUITABILITY
        )
        score_no_baseline, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], None
        )
        assert score_match > score_no_baseline

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(
            CHANGED_FILES_SUITABILITY,
            ["suitability__c", "riskprofile__c"],
            ["financialaccount"],
            2,
            [],
            AGENT8_DATA_SUITABILITY,
        )
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(CHANGED_FILES_EMPTY, [], [], 0, [], None)
        assert score >= 20

    def test_changed_files_present_stores_count(self):
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c", "riskprofile__c"], ["financialaccount"], 2, [], AGENT8_DATA_SUITABILITY
        )
        assert signals["changed_files_present"] == len(CHANGED_FILES_SUITABILITY)

    def test_no_changed_files_key_in_signals(self):
        _, signals = _compute_confidence(CHANGED_FILES_EMPTY, [], [], 0, [], None)
        assert "no_changed_files" in signals

    def test_detected_objects_rich_stores_count(self):
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c", "riskprofile__c"], ["financialaccount"], 2, [], AGENT8_DATA_SUITABILITY
        )
        assert signals["detected_objects_rich"] == 2

    def test_detected_objects_single_key_and_value(self):
        _, signals = _compute_confidence(CHANGED_FILES_SINGLE_OBJECT, ["suitability__c"], [], 1, [], None)
        assert "detected_objects_single" in signals
        assert signals["detected_objects_single"] == 1

    def test_no_fsc_objects_key_in_signals(self):
        _, signals = _compute_confidence(CHANGED_FILES_NO_FSC, [], [], 0, [], None)
        assert "no_fsc_objects_in_changed_files" in signals

    def test_refinement_baseline_available_key_in_signals(self):
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], AGENT8_DATA_SUITABILITY
        )
        assert "refinement_baseline_available" in signals

    def test_scope_matches_refinement_key_in_signals(self):
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], AGENT8_DATA_SUITABILITY
        )
        assert "scope_matches_refinement" in signals

    def test_scope_delta_detected_stores_count(self):
        agent8_narrow = {"detected_objects": ["suitability__c"], "implied_objects": []}
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, ["extra_object__c"], agent8_narrow
        )
        assert "scope_delta_detected" in signals
        assert signals["scope_delta_detected"] == 1

    def test_no_refinement_baseline_key_in_signals(self):
        _, signals = _compute_confidence(CHANGED_FILES_SUITABILITY, ["suitability__c"], [], 1, [], None)
        assert "no_refinement_baseline" in signals

    def test_deep_dependency_chain_stores_depth(self):
        _, signals = _compute_confidence(
            CHANGED_FILES_SUITABILITY, ["suitability__c", "riskprofile__c"], ["financialaccount"], 2, [], AGENT8_DATA_SUITABILITY
        )
        assert "deep_dependency_chain" in signals
        assert signals["deep_dependency_chain"] == 2


# ── Integration tests — full agent run ───────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["8"] = {"data": AGENT8_DATA_SUITABILITY}
        state["agent_results"]["11"] = {"data": AGENT11_DATA}

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": "feature/FSC-2417-suitability"}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.agent_id == 13
        assert result.agent_name == "Metadata Dependency Mapper"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        for key in ["detected_objects", "implied_objects", "dependency_graph",
                    "dependency_depth", "scope_delta_objects", "scope_matches_refinement"]:
            assert key in result.data

    async def test_uses_branch_from_agent_11(self):
        """Agent 13 should use Agent 11's branch name rather than calling Copado again."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["11"] = {"data": AGENT11_DATA}

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": "fallback-branch"}
            mock_haiku.return_value = MOCK_TRACE
            await run(state)

        # get_changed_files called with Agent 11's branch name
        mock_files.assert_called_once_with(
            "FSC-2417", "feature/FSC-2417-suitability-assessment"
        )
        # get_branch_for_story should NOT have been called (Agent 11 provided it)
        mock_branch.assert_not_called()

    async def test_runs_standalone_without_upstream_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_EMPTY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.agent_id == 13
        assert result.data["detected_objects"] == []

    async def test_changed_files_list_in_output_data(self):
        """REQ-10: Agent 13 must emit full changed_files list, not just count."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "changed_files" in result.data, "changed_files list must be in output data"
        assert isinstance(result.data["changed_files"], list)
        assert len(result.data["changed_files"]) == len(CHANGED_FILES_SUITABILITY)

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_has_destructive_changes_detected(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_DESTRUCTIVE
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.data["has_destructive_changes"] is True

    async def test_escalated_when_no_files(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_EMPTY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.development.agent_13_metadata_dependency.get_changed_files",
                  new_callable=AsyncMock) as mock_files,
            patch("src.agents.development.agent_13_metadata_dependency.get_branch_for_story",
                  new_callable=AsyncMock) as mock_branch,
            patch("src.agents.development.agent_13_metadata_dependency.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_files.return_value = CHANGED_FILES_SUITABILITY
            mock_branch.return_value = {"branch_name": ""}
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", [], [], [], 0, [], None)
        assert "FSC-2417" in msg

    def test_detected_objects_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", [], ["suitability__c"], [], 0, [], None)
        assert "suitability__c" in msg

    def test_empty_detected_shows_none(self):
        msg = _build_trace_message("FSC-2417", [], [], ["financialaccount"], 0, [], None)
        assert "FSC objects detected in code: ['none']" in msg

    def test_empty_implied_shows_none(self):
        msg = _build_trace_message("FSC-2417", [], ["suitability__c"], [], 0, [], None)
        assert "Implied parent objects: ['none']" in msg

    def test_empty_scope_delta_shows_none(self):
        msg = _build_trace_message("FSC-2417", [], ["suitability__c"], ["financialaccount"], 1, [], None)
        assert "Scope delta (new objects not in refinement): ['none']" in msg

    def test_refinement_objects_from_agent8_shown(self):
        msg = _build_trace_message("FSC-2417", [], [], [], 0, [], AGENT8_DATA_SUITABILITY)
        assert "suitability__c" in msg

    def test_no_agent8_shows_none_for_refinement(self):
        msg = _build_trace_message("FSC-2417", [], ["suitability__c"], ["financialaccount"], 1, ["extra__c"], None)
        assert "Objects in refinement prediction: ['none']" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", [], [], [], 0, [], None)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "dependency_complexity"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_dependency_complexity_enum_has_three_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["dependency_complexity"]["enum"] == ["low", "medium", "high"]
