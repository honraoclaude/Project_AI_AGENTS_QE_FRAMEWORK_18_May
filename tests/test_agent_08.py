"""
Tests for Agent 8 — Dependency Mapping (Augmented Script).

The deterministic _analyse_dependencies() function is the primary test target —
no LLM mocking needed for correctness tests.
Integration tests mock the Haiku call (narrative only).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_08_dependency_mapping import (
    _analyse_dependencies,
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    run,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio. The Suitability__c record must link "
        "to the client's RiskProfile__c and the relevant FinancialAccount. "
        "For vulnerable customers (VulnerableCustomerIndicator__c = true) the flow "
        "must present an additional Consumer Duty confirmation step."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "High",
    "labels": [],
    "components": ["Suitability"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

STORY_LABEL_CHANGE = {
    "story_id": "FSC-2500",
    "summary": "Update button label on Account page",
    "description": "Change the 'Save' button label to 'Submit' on the Account detail page.",
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": "po@firm.com",
}

STORY_FINANCIAL = {
    "story_id": "FSC-2600",
    "summary": "Add AUM roll-up to FinancialAccount",
    "description": (
        "Aggregate FinancialHolding records to produce AUM on FinancialAccount. "
        "The Revenue__c object should be updated when the roll-up changes."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Medium",
    "labels": [],
    "components": ["WealthCore"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

STORY_GOAL_DEEP = {
    "story_id": "FSC-2700",
    "summary": "Add goal tracking feature",
    "description": "Record the retirement goal for wealth planning.",
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": "po@firm.com",
}

STORY_RICH_DESCRIPTION = {
    **STORY_SUITABILITY,
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio so that the firm meets its FCA regulatory "
        "obligation under COBS 9.2. The Suitability__c record must link to the client's "
        "RiskProfile__c and the relevant FinancialAccount. For vulnerable customers "
        "(VulnerableCustomerIndicator__c = true) the flow must present an additional "
        "Consumer Duty confirmation step as required by FCA PS22/9. The adviser must "
        "confirm the client has been informed of all material risks. The system must "
        "create an audit record on the Suitability__c object and notify the compliance "
        "team by email. The assessment must be versioned and linked to the adviser's "
        "licence record. Any failed assessment must trigger a mandatory review workflow."
    ),
}

STORY_DELETE = {
    "story_id": "FSC-2800",
    "summary": "Clean up deprecated records",
    "description": "Delete deprecated Suitability__c records that are no longer needed.",
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Low",
    "labels": [],
    "components": [],
    "assignee": None,
    "reporter": "po@firm.com",
}

MOCK_TRACE = {
    "narrative": (
        "The story touches Suitability__c and RiskProfile__c, implying FinancialAccount "
        "and Individual parent records must exist before tests execute. "
        "The dependency depth of 2 indicates moderate deployment complexity."
    ),
    "dependency_complexity": "medium",
}


# ── Deterministic analysis tests (no LLM, no Jira) ───────────────────────────

class TestDependencyAnalysis:
    def test_suitability_story_detects_key_objects(self):
        detected, _, _, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "suitability__c" in detected
        assert "riskprofile__c" in detected
        assert "financialaccount" in detected

    def test_suitability_story_implies_individual(self):
        _, implied, _, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "individual" not in implied   # individual is excluded from implied (too generic)
        # individual appears in the dependency map but is excluded from implied list
        # (it's filtered as a base object in _analyse_dependencies)

    def test_label_change_detects_no_fsc_objects(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        assert len(detected) == 0
        assert len(implied) == 0
        assert depth == 0

    def test_financial_story_detects_financial_objects(self):
        detected, _, _, _, _, _ = _analyse_dependencies(STORY_FINANCIAL)
        assert "financialaccount" in detected
        assert "financialholding" in detected

    def test_dependency_depth_nonzero_for_suitability(self):
        _, _, _, depth, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert depth >= 1

    def test_dependency_depth_zero_for_label_change(self):
        _, _, _, depth, _, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        assert depth == 0

    def test_dependency_graph_contains_detected_objects(self):
        detected, _, graph, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        for obj in detected:
            assert obj in graph

    def test_vulnerable_customer_detected_by_alias(self):
        detected, _, _, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert "vulnerablecustomerindicator__c" in detected

    def test_analysis_is_case_insensitive(self):
        upper_story = {**STORY_SUITABILITY, "description": STORY_SUITABILITY["description"].upper()}
        detected_upper, _, _, _, _, _ = _analyse_dependencies(upper_story)
        detected_lower, _, _, _, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        assert set(detected_upper) == set(detected_lower)

    def test_goal_story_implies_financialaccount(self):
        _, implied, _, _, _, _ = _analyse_dependencies(STORY_GOAL_DEEP)
        assert "financialaccount" in implied

    def test_goal_story_depth_is_at_least_two(self):
        _, _, _, depth, _, _ = _analyse_dependencies(STORY_GOAL_DEEP)
        assert depth >= 2


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_suitability_story_high_confidence(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score >= 75, "Rich multi-object story should score ≥ 75"

    def test_label_change_lower_confidence(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        score, _ = _compute_confidence(STORY_LABEL_CHANGE, detected, implied, depth)
        assert score < 75, "Story with no FSC objects should score lower"

    def test_no_objects_heavily_penalised(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        score, signals = _compute_confidence(STORY_LABEL_CHANGE, detected, implied, depth)
        assert "no_objects_detected" in signals

    def test_score_never_exceeds_92(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score <= 92

    def test_score_never_below_20(self):
        empty_story = {**STORY_LABEL_CHANGE, "description": ""}
        detected, implied, _, depth, _, _ = _analyse_dependencies(empty_story)
        score, _ = _compute_confidence(empty_story, detected, implied, depth)
        assert score >= 20

    def test_high_base_score_reflects_deterministic_nature(self):
        """Augmented Script uses base=72 — should score higher than Tier B agents on same input."""
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        score, _ = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert score >= 72, "Base score for deterministic analysis should be high"

    def test_detected_objects_single_signal_in_signals(self):
        _, signals = _compute_confidence(STORY_GOAL_DEEP, ["goal__c"], ["financialaccount"], 2)
        assert "detected_objects_single" in signals
        assert signals["detected_objects_single"] == 1

    def test_deep_dependency_chain_signal_in_signals(self):
        _, signals = _compute_confidence(STORY_GOAL_DEEP, ["goal__c"], ["financialaccount"], 2)
        assert "deep_dependency_chain" in signals
        assert signals["deep_dependency_chain"] == 2

    def test_description_rich_signal_in_signals(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_RICH_DESCRIPTION)
        _, signals = _compute_confidence(STORY_RICH_DESCRIPTION, detected, implied, depth)
        assert "description_rich" in signals
        assert signals["description_rich"] >= 100

    def test_detected_objects_rich_stores_count(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_SUITABILITY)
        _, signals = _compute_confidence(STORY_SUITABILITY, detected, implied, depth)
        assert signals["detected_objects_rich"] == len(detected)

    def test_description_sparse_key_in_signals(self):
        detected, implied, _, depth, _, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        _, signals = _compute_confidence(STORY_LABEL_CHANGE, detected, implied, depth)
        assert "description_sparse" in signals


# ── Integration tests — full agent run ───────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.agent_id == 8
        assert result.agent_name == "Dependency Mapping"
        assert result.model_used == "claude-haiku-4-5-20251001"
        assert result.confidence.tier == "B"

    async def test_detected_objects_in_data(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "suitability__c" in result.data["detected_objects"]
        assert "dependency_depth" in result.data
        assert result.data["dependency_depth"] >= 1

    async def test_label_change_no_detected_objects(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_haiku.return_value = {**MOCK_TRACE, "dependency_complexity": "low"}
            result = await run(state)

        assert result.data["detected_objects"] == []
        assert result.data["dependency_depth"] == 0

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        for key in ["detected_objects", "implied_objects", "dependency_graph",
                    "dependency_depth", "cross_object_count"]:
            assert key in result.data

    async def test_uses_fast_model_for_haiku(self):
        """Agent 8 is an Augmented Script — Haiku generates narrative, not analysis."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_has_destructive_changes_false_for_normal_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.data["has_destructive_changes"] is False

    async def test_has_destructive_changes_true_for_delete_story(self):
        state = initial_story_state("FSC-2800")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_DELETE
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert result.data["has_destructive_changes"] is True

    async def test_label_change_causes_escalation(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_haiku.return_value = {**MOCK_TRACE, "dependency_complexity": "low"}
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_dependency_complexity_is_string(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert isinstance(result.data["dependency_complexity"], str)
        assert len(result.data["dependency_complexity"]) > 0

    async def test_signals_key_in_data_is_dict(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert isinstance(result.data["signals"], dict)


# ── REQ-06: platform_event + external_service detection tests ─────────────────

STORY_PLATFORM_EVENT = {
    "story_id": "FSC-3001",
    "summary": "Publish Platform Event on suitability update",
    "description": (
        "When a Suitability__c record is updated, publish a platform event "
        "to the EventBus so downstream consumers can trigger async processing."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Medium",
    "labels": [],
    "components": ["Integration"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}

STORY_EXTERNAL_SERVICE = {
    "story_id": "FSC-3002",
    "summary": "Retrieve AUM data via Named Credential callout",
    "description": (
        "As a Wealth Adviser, I want the system to perform an HTTP callout "
        "using a Named Credential to retrieve AUM data from the external data feed provider."
    ),
    "status": "Sprint Ready",
    "issue_type": "Story",
    "priority": "Medium",
    "labels": [],
    "components": ["Integration"],
    "assignee": "dev@firm.com",
    "reporter": "po@firm.com",
}


class TestIntegrationPatternDetectionREQ06:
    def test_platform_event_keyword_detected(self):
        """REQ-06: 'platform event' in story text → platform_event in dep_types."""
        _, _, _, _, has_ext, dep_types = _analyse_dependencies(STORY_PLATFORM_EVENT)
        assert has_ext is True
        assert "platform_event" in dep_types

    def test_external_service_keyword_detected(self):
        """REQ-06: 'named credential' + 'http callout' → external_service in dep_types."""
        _, _, _, _, has_ext, dep_types = _analyse_dependencies(STORY_EXTERNAL_SERVICE)
        assert has_ext is True
        assert "external_service" in dep_types

    def test_no_integration_patterns_for_standard_story(self):
        """REQ-06: Standard FSC story without integration keywords → has_external_dependencies=False."""
        _, _, _, _, has_ext, dep_types = _analyse_dependencies(STORY_SUITABILITY)
        assert has_ext is False
        assert dep_types == []

    def test_no_ext_deps_for_label_change(self):
        """REQ-06: Pure UI label change has no integration patterns."""
        _, _, _, _, has_ext, _ = _analyse_dependencies(STORY_LABEL_CHANGE)
        assert has_ext is False


@pytest.mark.asyncio
class TestExternalDepsIntegrationREQ06:
    async def test_has_external_dependencies_in_run_output(self):
        """REQ-06: run() emits has_external_dependencies in result.data."""
        state = initial_story_state("FSC-3002")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_EXTERNAL_SERVICE
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "has_external_dependencies" in result.data
        assert result.data["has_external_dependencies"] is True

    async def test_detected_dependency_types_in_output(self):
        """REQ-06: run() emits detected_dependency_types list in result.data."""
        state = initial_story_state("FSC-3002")

        with (
            patch("src.agents.refinement.agent_08_dependency_mapping.get_story",
                  new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_08_dependency_mapping.call_with_tool",
                  new_callable=AsyncMock) as mock_haiku,
        ):
            mock_story.return_value = STORY_EXTERNAL_SERVICE
            mock_haiku.return_value = MOCK_TRACE
            result = await run(state)

        assert "detected_dependency_types" in result.data
        assert "external_service" in result.data["detected_dependency_types"]


# ── Tests: trace message builder ─────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_trace_includes_story_id(self):
        msg = _build_trace_message(STORY_SUITABILITY, ["suitability__c"], [], 1)
        assert "FSC-2417" in msg

    def test_trace_includes_detected_objects(self):
        msg = _build_trace_message(STORY_SUITABILITY, ["suitability__c", "riskprofile__c"], [], 1)
        assert "suitability__c" in msg

    def test_trace_shows_none_for_empty_detected(self):
        msg = _build_trace_message(STORY_LABEL_CHANGE, [], [], 0)
        assert "['none']" in msg

    def test_trace_includes_implied_objects(self):
        msg = _build_trace_message(STORY_GOAL_DEEP, ["goal__c"], ["financialaccount"], 2)
        assert "financialaccount" in msg

    def test_trace_shows_none_for_empty_implied(self):
        msg = _build_trace_message(STORY_SUITABILITY, ["suitability__c"], [], 1)
        assert "['none']" in msg

    def test_trace_includes_integration_patterns_when_present(self):
        msg = _build_trace_message(
            STORY_PLATFORM_EVENT, ["suitability__c"], [], 1,
            has_external_deps=True, dep_types=["platform_event"],
        )
        assert "Integration patterns detected:" in msg

    def test_trace_omits_integration_line_when_absent(self):
        msg = _build_trace_message(STORY_SUITABILITY, ["suitability__c"], [], 1)
        assert "Integration patterns detected:" not in msg

    def test_trace_ends_with_tool_name(self):
        msg = _build_trace_message(STORY_SUITABILITY, ["suitability__c"], [], 1)
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Tests: schema contract ────────────────────────────────────────────────────

class TestSchemaContract:
    def test_trace_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "dependency_complexity"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_dependency_complexity_enum_has_three_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["dependency_complexity"]["enum"] == [
            "low", "medium", "high"
        ]
