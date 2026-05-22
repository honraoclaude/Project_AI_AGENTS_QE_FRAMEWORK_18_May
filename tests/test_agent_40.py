"""Tests for Agent 40 — Release Composer (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.release.agent_40_release_composer import (
    _compose_release,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT13_RICH = {
    "changed_files_count": 5,
    "detected_objects": ["financialaccount", "suitability__c"],
    "dependency_depth": 2,
    "missing_dependencies": [],
    "has_destructive_changes": False,
}

AGENT13_DESTRUCTIVE = {
    "changed_files_count": 3,
    "detected_objects": ["financialaccount"],
    "dependency_depth": 1,
    "missing_dependencies": [],
    "has_destructive_changes": True,
}

AGENT13_EMPTY = {
    "changed_files_count": 0,
    "detected_objects": [],
    "dependency_depth": 0,
    "missing_dependencies": [],
    "has_destructive_changes": False,
}

AGENT18_APEX = {
    "component_types": {"ApexClass": 3, "ApexTrigger": 1},
}

AGENT18_WITH_OBJECTS = {
    "component_types": {"ApexClass": 2, "CustomObject": 1, "CustomField": 4},
}

AGENT18_PATCH = {
    "component_types": {"CustomLabel": 2, "StaticResource": 1},
}

MOCK_TRACE_COMPOSED = {
    "narrative": "Release FSC-2417-minor-release composed with 4 components. Contains Apex classes and triggers — MINOR release type.",
    "composer_concern": "none",
}

MOCK_TRACE_PARTIAL = {
    "narrative": "Release package could not be fully composed — no components identified in metadata. Manual intervention required.",
    "composer_concern": "no_components",
}


# ── Deterministic composition tests ──────────────────────────────────────────

class TestComposeRelease:
    def test_apex_components_give_minor_release(self):
        _, _, release_type, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert release_type == "MINOR"

    def test_custom_object_gives_major_release(self):
        _, _, release_type, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_WITH_OBJECTS)
        assert release_type == "MAJOR"

    def test_non_apex_non_object_gives_patch(self):
        _, _, release_type, _, _ = _compose_release("FSC-001", None, AGENT13_EMPTY, None, AGENT18_PATCH)
        assert release_type == "PATCH"

    def test_release_name_contains_story_id(self):
        release_name, _, _, _, _ = _compose_release("FSC-2417", None, AGENT13_RICH, None, AGENT18_APEX)
        assert "FSC-2417" in release_name

    def test_release_name_contains_type(self):
        release_name, _, release_type, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert release_type.lower() in release_name

    def test_component_count_from_metadata(self):
        _, count, _, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert count > 0

    def test_no_components_gives_partial(self):
        _, count, _, _, verdict = _compose_release("FSC-001", None, AGENT13_EMPTY, None, None)
        assert verdict == "PARTIAL"

    def test_components_give_composed_verdict(self):
        _, _, _, _, verdict = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert verdict == "COMPOSED"

    def test_components_summary_populated_from_attribution(self):
        _, _, _, summary, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert "ApexClass" in summary

    def test_no_upstream_data_gives_partial(self):
        _, _, _, _, verdict = _compose_release("FSC-001", None, None, None, None)
        assert verdict == "PARTIAL"


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_metadata_available_scores_well(self):
        score, _ = _compute_confidence(AGENT13_RICH, AGENT18_APEX, 4)
        assert score >= 65

    def test_no_metadata_reduces_confidence(self):
        score_with, _ = _compute_confidence(AGENT13_RICH, AGENT18_APEX, 4)
        score_without, _ = _compute_confidence(None, None, 0)
        assert score_with > score_without

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT13_RICH, AGENT18_APEX, 4)
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, 0)
        assert score >= 20


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_RICH}
        state["agent_results"]["18"] = {"data": AGENT18_APEX}

        with patch("src.agents.release.agent_40_release_composer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_COMPOSED
            result = await run(state)

        assert result.agent_id == 40
        assert result.agent_name == "Release Composer"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_40_release_composer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PARTIAL
            result = await run(state)

        for key in ["release_name", "component_count", "release_type",
                    "components_summary", "composer_verdict"]:
            assert key in result.data

    async def test_composed_with_components(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {"data": AGENT13_RICH}
        state["agent_results"]["18"] = {"data": AGENT18_APEX}

        with patch("src.agents.release.agent_40_release_composer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_COMPOSED
            result = await run(state)

        assert result.data["composer_verdict"] == "COMPOSED"
        assert result.data["component_count"] > 0

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.release.agent_40_release_composer.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PARTIAL
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"


# ── REQ-25: new tests ─────────────────────────────────────────────────────────

AGENT17_SFDX_INVALID = {
    "sfdx_format_valid": False,
    "format_violations": ["classes/MyClass.cls: missing -meta.xml"],
}

AGENT8_EXT_DEPS = {
    "has_external_dependencies": True,
}


class TestREQ25SfdxInvalidFailed:
    def test_sfdx_format_invalid_gives_failed_verdict(self):
        _, _, _, _, verdict = _compose_release("FSC-001", None, AGENT13_RICH, AGENT17_SFDX_INVALID, AGENT18_APEX)
        assert verdict == "FAILED"

    def test_sfdx_format_valid_true_gives_composed(self):
        agent17_valid = {"sfdx_format_valid": True, "format_violations": []}
        _, _, _, _, verdict = _compose_release("FSC-001", None, AGENT13_RICH, agent17_valid, AGENT18_APEX)
        assert verdict == "COMPOSED"


class TestREQ25ComponentCountSum:
    def test_component_count_sums_type_values(self):
        agent18 = {"component_types": {"ApexClass": 3, "CustomObject": 1}}
        _, count, _, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, agent18)
        assert count == 4

    def test_mixed_types_summed_correctly(self):
        agent18 = {"component_types": {"ApexClass": 2, "CustomObject": 1, "CustomField": 4}}
        _, count, _, _, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, agent18)
        assert count == 7


class TestREQ25ExternalDepsInSummary:
    def test_external_deps_adds_external_service_to_summary(self):
        _, _, _, summary, _ = _compose_release("FSC-001", AGENT8_EXT_DEPS, AGENT13_RICH, None, AGENT18_APEX)
        assert "ExternalService" in summary

    def test_no_external_deps_no_external_service_key(self):
        _, _, _, summary, _ = _compose_release("FSC-001", None, AGENT13_RICH, None, AGENT18_APEX)
        assert "ExternalService" not in summary
