"""Tests for Agent 17 — SFDX Source-Format Validator (Augmented Script)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.development.agent_17_sfdx_validator import (
    _build_trace_message,
    _compute_confidence,
    _TRACE_TOOL_NAME,
    _TRACE_TOOL_SCHEMA,
    _validate_sfdx_format,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

SFDX_FILES = [
    {"file_path": "force-app/main/default/classes/SuitabilityService.cls", "change_type": "modified"},
    {"file_path": "force-app/main/default/triggers/FinancialAccountTrigger.trigger", "change_type": "added"},
]

MIXED_FILES = [
    {"file_path": "force-app/main/default/classes/SuitabilityService.cls", "change_type": "modified"},
    {"file_path": "src/classes/LegacyApex.cls", "change_type": "modified"},
]

LEGACY_FILES = [
    {"file_path": "src/classes/OldApex.cls", "change_type": "modified"},
    {"file_path": "src/objects/Account.object", "change_type": "modified"},
    {"file_path": "metadata/Suitability__c.object", "change_type": "added"},
]

UNKNOWN_ROOT_FILES = [
    {"file_path": "some/random/path/MyClass.cls", "change_type": "modified"},
]

TWO_INVALID_FILES = [
    {"file_path": "src/classes/OldA.cls", "change_type": "modified"},
    {"file_path": "src/classes/OldB.cls", "change_type": "modified"},
]

FOUR_LEGACY_FILES = [
    {"file_path": "src/classes/A.cls", "change_type": "modified"},
    {"file_path": "src/classes/B.cls", "change_type": "modified"},
    {"file_path": "src/objects/C.object", "change_type": "modified"},
    {"file_path": "metadata/D.object", "change_type": "modified"},
]

MOCK_TRACE_PASS = {
    "narrative": "Both changed files use SFDX source format under force-app/. No migration required.",
    "migration_urgency": "none",
}

MOCK_TRACE_WARN = {
    "narrative": "1 of 2 files is in legacy format. Migrate src/classes/LegacyApex.cls to force-app/.",
    "migration_urgency": "low",
}

MOCK_TRACE_FAIL = {
    "narrative": "3 files are in legacy format. Immediate migration to SFDX format required.",
    "migration_urgency": "high",
}


# ── Deterministic SFDX validation tests ──────────────────────────────────────

class TestSfdxFormatValidation:
    def test_all_sfdx_files_gives_pass(self):
        _, invalid, all_valid, verdict = _validate_sfdx_format(SFDX_FILES)
        assert verdict == "PASS"
        assert all_valid is True
        assert len(invalid) == 0

    def test_empty_files_gives_pass(self):
        valid_count, invalid, all_valid, verdict = _validate_sfdx_format([])
        assert verdict == "PASS"
        assert all_valid is True
        assert valid_count == 0

    def test_legacy_src_classes_flagged(self):
        _, invalid, _, verdict = _validate_sfdx_format(LEGACY_FILES)
        assert verdict == "FAIL"
        assert any("src/classes/" in f for f in invalid)

    def test_legacy_src_objects_flagged(self):
        _, invalid, _, _ = _validate_sfdx_format(LEGACY_FILES)
        assert any("src/objects/" in f for f in invalid)

    def test_legacy_metadata_root_flagged(self):
        _, invalid, _, _ = _validate_sfdx_format(LEGACY_FILES)
        assert any("metadata/" in f for f in invalid)

    def test_mixed_files_gives_warn_for_one_invalid(self):
        _, invalid, all_valid, verdict = _validate_sfdx_format(MIXED_FILES)
        assert verdict == "WARN"
        assert all_valid is False
        assert len(invalid) == 1

    def test_three_invalid_gives_fail(self):
        _, invalid, _, verdict = _validate_sfdx_format(LEGACY_FILES)
        assert verdict == "FAIL"
        assert len(invalid) == 3

    def test_unknown_root_path_flagged_as_invalid(self):
        _, invalid, _, _ = _validate_sfdx_format(UNKNOWN_ROOT_FILES)
        assert len(invalid) == 1

    def test_valid_count_correct_for_mixed(self):
        valid_count, invalid, _, _ = _validate_sfdx_format(MIXED_FILES)
        assert valid_count == len(MIXED_FILES) - len(invalid)

    def test_file_with_empty_path_skipped(self):
        files = [{"file_path": "", "change_type": "modified"}]
        valid_count, invalid, all_valid, verdict = _validate_sfdx_format(files)
        assert len(invalid) == 0
        assert verdict == "PASS"

    def test_case_insensitive_sfdx_root_check(self):
        files = [{"file_path": "Force-App/main/default/classes/Foo.cls"}]
        _, invalid, all_valid, _ = _validate_sfdx_format(files)
        assert all_valid is True
        assert len(invalid) == 0

    def test_src_triggers_flagged_as_legacy(self):
        files = [{"file_path": "src/triggers/AccountTrigger.trigger"}]
        _, invalid, _, verdict = _validate_sfdx_format(files)
        assert len(invalid) == 1
        assert verdict == "WARN"

    def test_unpackaged_root_flagged_as_legacy(self):
        files = [{"file_path": "unpackaged/classes/SomeClass.cls"}]
        _, invalid, _, _ = _validate_sfdx_format(files)
        assert len(invalid) == 1

    def test_two_invalid_gives_warn(self):
        # len(invalid) <= 2 upper boundary — must still be WARN, not FAIL
        _, invalid, all_valid, verdict = _validate_sfdx_format(TWO_INVALID_FILES)
        assert verdict == "WARN"
        assert len(invalid) == 2
        assert all_valid is False


# ── Confidence scoring unit tests ─────────────────────────────────────────────

class TestConfidenceScoring:
    def test_all_sfdx_with_files_scores_high(self):
        score, _ = _compute_confidence(SFDX_FILES, True, [])
        assert score >= 75

    def test_no_files_reduces_confidence(self):
        score_with, _ = _compute_confidence(SFDX_FILES, True, [])
        score_without, _ = _compute_confidence([], True, [])
        assert score_with > score_without

    def test_each_invalid_file_reduces_confidence(self):
        score_pass, _ = _compute_confidence(SFDX_FILES, True, [])
        score_fail, _ = _compute_confidence(LEGACY_FILES, False, ["a", "b", "c"])
        assert score_pass > score_fail

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(SFDX_FILES, True, [])
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence([], False, ["x", "y", "z", "w"])
        assert score >= 20

    def test_signals_dict_not_empty(self):
        _, signals = _compute_confidence(SFDX_FILES, True, [])
        assert len(signals) >= 1

    def test_files_available_for_check_stores_count(self):
        _, signals = _compute_confidence(SFDX_FILES, True, [])
        assert signals["files_available_for_check"] == len(SFDX_FILES)

    def test_no_files_to_check_key_in_signals(self):
        _, signals = _compute_confidence([], True, [])
        assert "no_files_to_check" in signals

    def test_all_files_sfdx_format_key_in_signals(self):
        _, signals = _compute_confidence(SFDX_FILES, True, [])
        assert "all_files_sfdx_format" in signals

    def test_legacy_format_files_found_stores_count(self):
        invalid = ["src/classes/A.cls", "src/classes/B.cls", "metadata/C.object"]
        _, signals = _compute_confidence(LEGACY_FILES, False, invalid)
        assert signals["legacy_format_files_found"] == 3


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.agent_id == 17
        assert result.agent_name == "SFDX Source-Format Validator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        for key in ["sfdx_format_valid", "sfdx_verdict", "invalid_files"]:
            assert key in result.data

    async def test_pass_verdict_when_no_files_available(self):
        """Agent 13 stub returns empty file list — agent defaults to PASS."""
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["sfdx_verdict"] == "PASS"
        assert result.data["sfdx_format_valid"] is True

    async def test_uses_fast_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.model_used == "claude-haiku-4-5-20251001"

    async def test_legacy_files_from_agent13_give_warn_or_fail(self):
        """REQ-10: Agent 17 reads changed_files from Agent 13 — not always returns PASS."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {
            "data": {
                "changed_files": [
                    {"file_path": "src/classes/LegacyApex.cls", "change_type": "modified"},
                ],
                "changed_files_count": 1,
            }
        }

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_WARN
            result = await run(state)

        assert result.data["sfdx_format_valid"] is False
        assert result.data["sfdx_verdict"] in ("WARN", "FAIL")
        assert len(result.data["invalid_files"]) >= 1

    async def test_sfdx_files_from_agent13_give_pass(self):
        """REQ-10: Agent 17 passes validation when Agent 13 changed_files are all SFDX format."""
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {
            "data": {
                "changed_files": SFDX_FILES,
                "changed_files_count": len(SFDX_FILES),
            }
        }

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert result.data["sfdx_format_valid"] is True
        assert result.data["sfdx_verdict"] == "PASS"

    async def test_migration_urgency_in_data(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "migration_urgency" in result.data

    async def test_escalated_when_many_invalid_files(self):
        # 4 invalid → +8 (files) − 16 (penalty capped) = 57 < 60 → escalated
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {
            "data": {"changed_files": FOUR_LEGACY_FILES, "changed_files_count": 4}
        }

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_FAIL
            result = await run(state)

        assert result.confidence.escalated is True

    async def test_what_contains_story_id(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert "FSC-2417" in result.what

    async def test_signals_is_dict(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["signals"], dict)

    async def test_narrative_is_string_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["13"] = {
            "data": {"changed_files": SFDX_FILES, "changed_files_count": 2}
        }

        with patch("src.agents.development.agent_17_sfdx_validator.call_with_tool",
                   new_callable=AsyncMock) as mock_haiku:
            mock_haiku.return_value = MOCK_TRACE_PASS
            result = await run(state)

        assert isinstance(result.data["narrative"], str)


# ── Trace message unit tests ──────────────────────────────────────────────────

class TestBuildTraceMessage:
    def test_includes_story_id(self):
        msg = _build_trace_message("FSC-2417", 2, 2, [], "PASS")
        assert "FSC-2417" in msg

    def test_includes_total_count(self):
        msg = _build_trace_message("FSC-2417", 4, 3, ["src/classes/Old.cls"], "WARN")
        assert "Total files checked: 4" in msg

    def test_includes_verdict(self):
        msg = _build_trace_message("FSC-2417", 2, 2, [], "PASS")
        assert "Verdict: PASS" in msg

    def test_empty_invalid_shows_none(self):
        msg = _build_trace_message("FSC-2417", 2, 2, [], "PASS")
        assert "Invalid files: ['none']" in msg

    def test_invalid_files_shown_when_present(self):
        msg = _build_trace_message("FSC-2417", 2, 1, ["src/classes/Old.cls"], "WARN")
        assert "src/classes/Old.cls" in msg

    def test_valid_count_shown(self):
        msg = _build_trace_message("FSC-2417", 2, 1, ["src/classes/Old.cls"], "WARN")
        assert "Valid SFDX format: 1" in msg

    def test_ends_with_tool_name(self):
        msg = _build_trace_message("FSC-2417", 0, 0, [], "PASS")
        assert _TRACE_TOOL_NAME in msg
        assert msg.strip().endswith("tool.")


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_schema_has_two_required_fields(self):
        assert set(_TRACE_TOOL_SCHEMA["required"]) == {"narrative", "migration_urgency"}

    def test_narrative_is_string(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["narrative"]["type"] == "string"

    def test_migration_urgency_enum_has_three_values(self):
        assert _TRACE_TOOL_SCHEMA["properties"]["migration_urgency"]["enum"] == ["none", "low", "high"]
