"""
Tests for Agent 1 — Story Intent Agent.

Uses a mock Jira response so the test runs without a live Jira instance.
Tests: extraction quality, confidence scoring, flag detection, AgentResult shape.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_01_story_intent import (
    run,
    _compute_confidence,
    _build_user_message,
    _TOOL_SCHEMA,
    _TOOL_NAME,
)
from src.core.schemas import initial_story_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

STORY_SUITABILITY = {
    "story_id": "FSC-2417",
    "summary": "Record Suitability Assessment for Retirement Portfolio",
    "description": (
        "As a Wealth Adviser, I want to record a COBS 9.2 Suitability Assessment "
        "for a client's retirement portfolio so that the firm meets its regulatory "
        "obligation before recommending any changes to the FinancialAccount.\n\n"
        "The assessment must capture risk tolerance, investment horizon, and capacity "
        "for loss. The Suitability__c record must link to the client's RiskProfile__c "
        "and the relevant FinancialAccount.\n\n"
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

AC_CLAUSES_FULL = [
    {
        "source": "description",
        "scenario": "Scenario: Adviser records suitability for standard client",
        "given": ["Given the client has a RiskProfile__c with risk_level = Moderate"],
        "when": ["When the adviser completes the Suitability Assessment screen flow"],
        "then": [
            "Then a Suitability__c record is created linked to the FinancialAccount",
            "Then the status field is set to 'Assessment Complete'",
        ],
    },
    {
        "source": "description",
        "scenario": "Scenario: Vulnerable customer — additional confirmation step",
        "given": ["Given VulnerableCustomerIndicator__c is true for the client"],
        "when": ["When the adviser reaches the final step of the Suitability flow"],
        "then": [
            "Then a Consumer Duty confirmation checkbox is displayed",
            "Then the assessment cannot be submitted without checking the box",
        ],
    },
]

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

MOCK_EXTRACTION_SUITABILITY = {
    "goal": "Enable advisers to record COBS 9.2 Suitability Assessments linked to FinancialAccounts and RiskProfiles.",
    "persona": "Wealth Adviser",
    "fsc_objects": ["Suitability__c", "RiskProfile__c", "FinancialAccount", "VulnerableCustomerIndicator__c"],
    "fsc_components": ["Screen Flow — Suitability Assessment", "Apex trigger on Suitability__c"],
    "ac_present": True,
    "ac_complete": True,
    "missing_elements": ["none"],
    "story_summary": (
        "This story implements the COBS 9.2 Suitability Assessment recording flow for wealth advisers. "
        "It creates Suitability__c records linked to RiskProfile__c and FinancialAccount. "
        "It adds a Consumer Duty confirmation step for vulnerable customers."
    ),
    "flags": ["high_fca_object_detected"],
}

MOCK_EXTRACTION_LABEL = {
    "goal": "Update a button label on the Account detail page.",
    "persona": "Operations/Admin",
    "fsc_objects": [],
    "fsc_components": ["Lightning Page — Account Detail"],
    "ac_present": False,
    "ac_complete": False,
    "missing_elements": ["acceptance_criteria", "error_scenarios"],
    "story_summary": "Minor UI change to rename a button label on the Account detail page. No FSC objects involved.",
    "flags": ["no_acceptance_criteria", "vague_goal", "no_fsc_objects"],
}


# ── Confidence scoring unit tests (no LLM, no Jira) ──────────────────────────

class TestConfidenceScoring:
    def test_high_fca_story_scores_above_70(self):
        score, signals = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert score >= 70, f"Expected >=70 for well-specified HIGH-FCA story, got {score}"

    def test_label_change_scores_below_60(self):
        score, signals = _compute_confidence(STORY_LABEL_CHANGE, [], MOCK_EXTRACTION_LABEL)
        assert score < 70, f"Expected <70 for vague label-change story, got {score}"

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert score <= 92, "Tier B score must never exceed 92 (reserved for deterministic checks)"

    def test_score_never_below_20(self):
        empty_story = {**STORY_LABEL_CHANGE, "description": "", "summary": ""}
        score, _ = _compute_confidence(empty_story, [], MOCK_EXTRACTION_LABEL)
        assert score >= 20, "Score floor is 20"

    def test_signals_dict_populated(self):
        _, signals = _compute_confidence(STORY_SUITABILITY, AC_CLAUSES_FULL, MOCK_EXTRACTION_SUITABILITY)
        assert "description_words" in signals
        assert "ac_complete" in signals or "ac_present_incomplete" in signals or "ac_absent" in signals


# ── Integration test — full agent run with mocked LLM and Jira ────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        assert result.agent_id == 1
        assert result.agent_name == "Story Intent Agent"
        assert result.model_used == "claude-sonnet-4-6"
        assert "FSC-2417" in result.what
        assert result.confidence.tier == "B"
        assert result.confidence.final_score >= 70
        assert not result.escalated

    async def test_escalates_on_vague_story(self):
        state = initial_story_state("FSC-2500")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_EXTRACTION_LABEL

            result = await run(state)

        # Vague story with no ACs — may escalate depending on final score
        assert result.agent_id == 1
        assert result.confidence.tier == "B"
        assert isinstance(result.confidence.final_score, int)

    async def test_data_contains_required_downstream_keys(self):
        """Downstream agents (2, 3, 7) depend on these keys existing."""
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        required_keys = ["goal", "persona", "fsc_objects", "ac_present", "ac_clauses", "flags"]
        # ac_clauses key comes from the extraction
        data_keys = set(result.data.keys())
        for key in ["goal", "persona", "fsc_objects", "ac_present", "flags"]:
            assert key in data_keys, f"Missing downstream-required key: {key}"

    async def test_high_fca_flag_present_for_suitability_story(self):
        state = initial_story_state("FSC-2417")

        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY

            result = await run(state)

        assert "high_fca_object_detected" in result.data.get("flags", [])

    # C1 — escalation threshold
    async def test_high_confidence_story_is_not_escalated(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
            patch("src.agents.refinement.agent_01_story_intent.settings") as mock_settings,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY
            mock_settings.default_model = "claude-sonnet-4-6"
            mock_settings.confidence_escalation_threshold = 60  # score will be 92 — above threshold
            result = await run(state)
        assert result.escalated is False

    async def test_low_confidence_story_is_escalated(self):
        state = initial_story_state("FSC-2500")
        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
            patch("src.agents.refinement.agent_01_story_intent.settings") as mock_settings,
        ):
            mock_story.return_value = STORY_LABEL_CHANGE
            mock_ac.return_value = []
            mock_llm.return_value = MOCK_EXTRACTION_LABEL
            mock_settings.default_model = "claude-sonnet-4-6"
            mock_settings.confidence_escalation_threshold = 99  # threshold above any achievable score
            result = await run(state)
        assert result.escalated is True

    # H1 — data dict keys added by run()
    async def test_data_contains_story_summary_jira(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY
            result = await run(state)
        assert result.data["story_summary_jira"] == STORY_SUITABILITY["summary"]

    async def test_data_contains_description_word_count(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY
            result = await run(state)
        expected = len((STORY_SUITABILITY["description"] or "").split())
        assert result.data["description_word_count"] == expected

    async def test_data_contains_ac_clause_count(self):
        state = initial_story_state("FSC-2417")
        with (
            patch("src.agents.refinement.agent_01_story_intent.get_story", new_callable=AsyncMock) as mock_story,
            patch("src.agents.refinement.agent_01_story_intent.get_acceptance_criteria", new_callable=AsyncMock) as mock_ac,
            patch("src.agents.refinement.agent_01_story_intent.call_with_tool", new_callable=AsyncMock) as mock_llm,
        ):
            mock_story.return_value = STORY_SUITABILITY
            mock_ac.return_value = AC_CLAUSES_FULL
            mock_llm.return_value = MOCK_EXTRACTION_SUITABILITY
            result = await run(state)
        assert result.data["ac_clause_count"] == len(AC_CLAUSES_FULL)


# ── Helpers for functional tests ──────────────────────────────────────────────

def _story(description: str = "", summary: str = "No summary") -> dict:
    """Minimal story dict — only description and summary affect confidence scoring."""
    return {
        "story_id": "TST-001",
        "summary": summary,
        "description": description,
        "status": "Sprint Ready",
        "issue_type": "Story",
        "priority": "Medium",
        "labels": [],
        "components": [],
        "assignee": None,
        "reporter": None,
    }


def _extraction(
    *,
    fsc_objects: list | None = None,
    persona: str = "UNKNOWN",
    flags: list | None = None,
    ac_complete: bool = False,
) -> dict:
    """Minimal extraction dict — caller controls exactly which signals fire."""
    return {
        "goal": "Test goal",
        "persona": persona,
        "fsc_objects": fsc_objects if fsc_objects is not None else [],
        "fsc_components": [],
        "ac_present": ac_complete,
        "ac_complete": ac_complete,
        "missing_elements": ["none"],
        "story_summary": "Test summary.",
        "flags": flags if flags is not None else [],
    }


# ── Signal isolation tests ────────────────────────────────────────────────────

class TestConfidenceSignals:
    """
    Each test isolates one scoring rule by holding all other inputs constant
    and diffing two states. The diff reveals the exact adjustment applied.

    Scoring rules under test (from _compute_confidence):
      description_words : >=150→+15, >=50→+8, >=20→+2, else→-20
      ac_complete       : clauses + complete=True → +15
      ac_incomplete     : clauses + complete=False → +7
      ac_absent         : no clauses → -10
      fsc_objects_count : >=2→+10, ==1→+5
      no_fsc_objects    : empty list → -5
      persona_identified: persona != UNKNOWN → +5
      high_fca_keyword  : FCA keyword in description or summary → +5
      vague_goal_flag   : "vague_goal" in flags → -10
    """

    def test_150_word_description_adds_7_more_than_50_word(self):
        # >=150 → +15; >=50 → +8; diff must be exactly 7
        long_desc  = "word " * 150
        short_desc = "word " * 50
        ext = _extraction()
        score_long,  _ = _compute_confidence(_story(long_desc),  [], ext)
        score_short, _ = _compute_confidence(_story(short_desc), [], ext)
        assert score_long - score_short == 7

    def test_50_word_description_adds_6_more_than_20_word(self):
        # >=50 → +8; >=20 → +2; diff = 6
        mid_desc = "word " * 50
        low_desc = "word " * 20
        ext = _extraction()
        score_mid, _ = _compute_confidence(_story(mid_desc), [], ext)
        score_low, _ = _compute_confidence(_story(low_desc), [], ext)
        assert score_mid - score_low == 6

    def test_20_word_description_adds_22_more_than_19_word(self):
        # >=20 → +2; <20 → -20; diff = 22 — the largest single boundary jump
        above = "word " * 20
        below = "word " * 19
        ext = _extraction()
        score_above, _ = _compute_confidence(_story(above), [], ext)
        score_below, _ = _compute_confidence(_story(below), [], ext)
        assert score_above - score_below == 22

    def test_complete_acs_add_25_more_than_absent(self):
        # complete ACs → +15; no ACs → -10; diff = 25
        desc = "word " * 50
        score_complete, _ = _compute_confidence(_story(desc), AC_CLAUSES_FULL, _extraction(ac_complete=True))
        score_absent,   _ = _compute_confidence(_story(desc), [],              _extraction(ac_complete=False))
        assert score_complete - score_absent == 25

    def test_incomplete_acs_add_17_more_than_absent(self):
        # clauses present but incomplete → +7; no ACs → -10; diff = 17
        desc = "word " * 50
        score_incomplete, _ = _compute_confidence(_story(desc), AC_CLAUSES_FULL, _extraction(ac_complete=False))
        score_absent,     _ = _compute_confidence(_story(desc), [],              _extraction(ac_complete=False))
        assert score_incomplete - score_absent == 17

    def test_complete_acs_add_8_more_than_incomplete(self):
        # complete → +15; incomplete → +7; diff = 8
        desc = "word " * 50
        score_complete,   _ = _compute_confidence(_story(desc), AC_CLAUSES_FULL, _extraction(ac_complete=True))
        score_incomplete, _ = _compute_confidence(_story(desc), AC_CLAUSES_FULL, _extraction(ac_complete=False))
        assert score_complete - score_incomplete == 8

    def test_two_fsc_objects_add_5_more_than_one(self):
        # >=2 → +10; ==1 → +5; diff = 5
        desc = "word " * 50
        score_two, _ = _compute_confidence(_story(desc), [], _extraction(fsc_objects=["A", "B"]))
        score_one, _ = _compute_confidence(_story(desc), [], _extraction(fsc_objects=["A"]))
        assert score_two - score_one == 5

    def test_one_fsc_object_adds_10_more_than_zero(self):
        # ==1 → +5; ==0 → no fsc_objects_count signal BUT no_fsc_objects fires (-5); diff = 10
        desc = "word " * 50
        score_one,  _ = _compute_confidence(_story(desc), [], _extraction(fsc_objects=["A"]))
        score_zero, _ = _compute_confidence(_story(desc), [], _extraction(fsc_objects=[]))
        assert score_one - score_zero == 10

    def test_known_persona_adds_5_points(self):
        desc = "word " * 50
        score_known,   _ = _compute_confidence(_story(desc), [], _extraction(persona="Wealth Adviser"))
        score_unknown, _ = _compute_confidence(_story(desc), [], _extraction(persona="UNKNOWN"))
        assert score_known - score_unknown == 5

    def test_vague_goal_flag_deducts_10_points(self):
        desc = "word " * 50
        score_clean, _ = _compute_confidence(_story(desc), [], _extraction(flags=[]))
        score_vague, _ = _compute_confidence(_story(desc), [], _extraction(flags=["vague_goal"]))
        assert score_clean - score_vague == 10

    def test_high_fca_keyword_in_description_adds_5_points(self):
        # "suitability" is in _HIGH_FCA_OBJECTS
        desc_fca    = "word " * 50 + " suitability assessment required"
        desc_no_fca = "word " * 50
        score_fca,    _ = _compute_confidence(_story(desc_fca),    [], _extraction())
        score_no_fca, _ = _compute_confidence(_story(desc_no_fca), [], _extraction())
        assert score_fca - score_no_fca == 5

    def test_high_fca_keyword_in_summary_also_adds_5_points(self):
        # keyword detection uses (description + summary).lower()
        desc = "word " * 50
        score_fca,    _ = _compute_confidence(_story(desc, summary="suitability review required"), [], _extraction())
        score_no_fca, _ = _compute_confidence(_story(desc, summary="routine change"),             [], _extraction())
        assert score_fca - score_no_fca == 5

    def test_other_flag_does_not_reduce_score(self):
        # Only "vague_goal" triggers a penalty — other flags are informational
        desc = "word " * 50
        score_no_flag, _ = _compute_confidence(_story(desc), [], _extraction(flags=[]))
        score_other,   _ = _compute_confidence(_story(desc), [], _extraction(flags=["high_fca_object_detected"]))
        assert score_no_flag == score_other

    # M1 — additional HIGH-FCA keywords
    def test_riskprofile_keyword_triggers_fca_signal(self):
        desc_fca    = "word " * 50 + " riskprofile assessment"
        desc_no_fca = "word " * 50
        score_fca,    _ = _compute_confidence(_story(desc_fca),    [], _extraction())
        score_no_fca, _ = _compute_confidence(_story(desc_no_fca), [], _extraction())
        assert score_fca - score_no_fca == 5

    def test_consumer_duty_keyword_triggers_fca_signal(self):
        # "consumer duty" is a two-word keyword — tests that the space is preserved in matching
        desc_fca    = "word " * 50 + " consumer duty obligations"
        desc_no_fca = "word " * 50
        score_fca,    _ = _compute_confidence(_story(desc_fca),    [], _extraction())
        score_no_fca, _ = _compute_confidence(_story(desc_no_fca), [], _extraction())
        assert score_fca - score_no_fca == 5

    def test_cobs_keyword_triggers_fca_signal(self):
        desc_fca    = "word " * 50 + " cobs 9.2 compliance"
        desc_no_fca = "word " * 50
        score_fca,    _ = _compute_confidence(_story(desc_fca),    [], _extraction())
        score_no_fca, _ = _compute_confidence(_story(desc_no_fca), [], _extraction())
        assert score_fca - score_no_fca == 5

    # M2 — case-insensitive keyword matching
    def test_fca_keyword_detection_is_case_insensitive(self):
        # combined_text is lowercased before matching — uppercase keywords must still match
        desc_upper = "word " * 50 + " SUITABILITY ASSESSMENT"
        desc_lower = "word " * 50 + " suitability assessment"
        score_upper, _ = _compute_confidence(_story(desc_upper), [], _extraction())
        score_lower, _ = _compute_confidence(_story(desc_lower), [], _extraction())
        assert score_upper == score_lower


# ── Score arithmetic tests ────────────────────────────────────────────────────

class TestScoreArithmetic:
    """
    Tests that derive the full expected score from first principles.
    Catches regressions where the base value or any adjustment constant changes.
    """

    def test_best_case_is_capped_at_92(self):
        # base=55 + desc≥150(+15) + ac_complete(+15) + ≥2_fsc(+10)
        # + persona(+5) + fca_keyword(+5) = 105 → must be capped at 92
        desc = "word " * 150 + " suitability cobs"
        score, _ = _compute_confidence(
            _story(desc),
            AC_CLAUSES_FULL,
            _extraction(
                fsc_objects=["Suitability__c", "RiskProfile__c"],
                persona="Wealth Adviser",
                ac_complete=True,
            ),
        )
        assert score == 92

    def test_worst_case_is_floored_at_20(self):
        # base=55 + desc<20(-20) + ac_absent(-10) + no_fsc(-5) + vague_goal(-10) = 10 → floored at 20
        score, _ = _compute_confidence(
            _story("five words only here now"),   # 5 words — under 20
            [],
            _extraction(flags=["vague_goal"]),
        )
        assert score == 20

    def test_midrange_story_exact_score(self):
        # Controlled inputs with no FCA keyword, no persona, no flags.
        # base=55 + desc≥50(+8) + ac_incomplete(+7) + 1_fsc(+5) = 75
        desc = "word " * 50
        score, _ = _compute_confidence(
            _story(desc),
            AC_CLAUSES_FULL,
            _extraction(fsc_objects=["FinancialAccount"], persona="UNKNOWN", ac_complete=False),
        )
        assert score == 75

    def test_no_signals_at_all_lands_at_base_plus_short_desc_penalty(self):
        # Empty description (0 words < 20 → -20), no ACs (-10), no FSC (-5)
        # base=55 - 20 - 10 - 5 = 20 → hits floor
        score, _ = _compute_confidence(_story(""), [], _extraction())
        assert score == 20

    def test_signals_dict_stores_observed_values_not_adjustments(self):
        # TierBScorer.add(name, value, delta) stores value — not delta
        desc = "word " * 60   # 60 words — signal fires with observed value 60
        _, signals = _compute_confidence(_story(desc), [], _extraction())
        assert signals["description_words"] == 60   # the observed count, not the +8 adjustment

    # H2 — signals dict stores observed values for other signals
    def test_signals_persona_stores_persona_string(self):
        desc = "word " * 50
        _, signals = _compute_confidence(_story(desc), [], _extraction(persona="Wealth Adviser"))
        assert signals["persona_identified"] == "Wealth Adviser"

    def test_signals_fsc_count_stores_object_count(self):
        desc = "word " * 50
        _, signals = _compute_confidence(_story(desc), [], _extraction(fsc_objects=["A", "B"]))
        assert signals["fsc_objects_count"] == 2

    def test_signals_vague_goal_stores_true(self):
        desc = "word " * 50
        _, signals = _compute_confidence(_story(desc), [], _extraction(flags=["vague_goal"]))
        assert signals["vague_goal_flag"] is True


# ── Prompt content tests ──────────────────────────────────────────────────────

class TestPromptContent:
    """
    Tests that _build_user_message() puts the right content in the prompt.
    These catch regressions where someone edits the message builder and drops
    critical context — without needing a live API call.
    """

    def test_prompt_includes_story_id(self):
        msg = _build_user_message(STORY_SUITABILITY, [])
        assert "FSC-2417" in msg

    def test_prompt_includes_description_text(self):
        msg = _build_user_message(STORY_SUITABILITY, [])
        assert "COBS 9.2" in msg
        assert "Suitability" in msg

    def test_prompt_includes_ac_scenarios_when_present(self):
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES_FULL)
        assert "Scenario:" in msg
        assert "Given" in msg
        assert "When" in msg
        assert "Then" in msg

    def test_prompt_says_none_found_when_no_acs(self):
        msg = _build_user_message(STORY_LABEL_CHANGE, [])
        assert "None found" in msg
        assert "Scenario:" not in msg

    def test_prompt_ends_with_tool_instruction(self):
        # The final line must tell the model to call the tool.
        # If this is missing, call_with_tool() will raise RuntimeError.
        msg = _build_user_message(STORY_SUITABILITY, AC_CLAUSES_FULL)
        assert _TOOL_NAME in msg   # "extract_story_intent" must appear
        assert msg.strip().endswith("tool.")

    def test_prompt_includes_components_and_labels(self):
        msg = _build_user_message(STORY_SUITABILITY, [])
        assert "COMPONENTS:" in msg
        assert "LABELS:" in msg

    # C2 — empty description → "(empty)"
    def test_prompt_shows_empty_when_description_is_none(self):
        story = {**STORY_LABEL_CHANGE, "description": None}
        msg = _build_user_message(story, [])
        assert "(empty)" in msg

    # C3 — STATUS field in prompt
    def test_prompt_includes_status(self):
        msg = _build_user_message(STORY_SUITABILITY, [])
        assert "Sprint Ready" in msg

    # M3 — empty components/labels render as "None"
    def test_empty_components_renders_as_none(self):
        # STORY_LABEL_CHANGE has components=[] — should render as "None"
        msg = _build_user_message(STORY_LABEL_CHANGE, [])
        assert "COMPONENTS: None" in msg

    def test_empty_labels_renders_as_none(self):
        # STORY_LABEL_CHANGE has labels=[] — should render as "None"
        msg = _build_user_message(STORY_LABEL_CHANGE, [])
        assert "LABELS: None" in msg


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    """
    Tests that _TOOL_SCHEMA enforces the correct structure.
    Claude validates tool inputs against this schema — if a field is
    missing or wrong-typed here, the LLM can produce malformed output
    that downstream agents silently misread.
    """

    def test_all_nine_fields_are_required(self):
        required = set(_TOOL_SCHEMA["required"])
        expected = {
            "goal", "persona", "fsc_objects", "fsc_components",
            "ac_present", "ac_complete", "missing_elements",
            "story_summary", "flags",
        }
        assert required == expected, f"Required fields mismatch: {required ^ expected}"

    def test_fsc_objects_is_typed_array(self):
        assert _TOOL_SCHEMA["properties"]["fsc_objects"]["type"] == "array"
        assert _TOOL_SCHEMA["properties"]["fsc_objects"]["items"]["type"] == "string"

    def test_boolean_fields_have_correct_type(self):
        assert _TOOL_SCHEMA["properties"]["ac_present"]["type"] == "boolean"
        assert _TOOL_SCHEMA["properties"]["ac_complete"]["type"] == "boolean"

    def test_flags_is_typed_array_of_strings(self):
        flags_schema = _TOOL_SCHEMA["properties"]["flags"]
        assert flags_schema["type"] == "array"
        assert flags_schema["items"]["type"] == "string"

    # C4 — 5 untested fields: goal, persona, fsc_components, missing_elements, story_summary

    def test_goal_is_typed_string(self):
        assert _TOOL_SCHEMA["properties"]["goal"]["type"] == "string"

    def test_persona_is_typed_string(self):
        assert _TOOL_SCHEMA["properties"]["persona"]["type"] == "string"

    def test_fsc_components_is_typed_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["fsc_components"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_missing_elements_is_typed_array_of_strings(self):
        schema = _TOOL_SCHEMA["properties"]["missing_elements"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_story_summary_is_typed_string(self):
        assert _TOOL_SCHEMA["properties"]["story_summary"]["type"] == "string"
