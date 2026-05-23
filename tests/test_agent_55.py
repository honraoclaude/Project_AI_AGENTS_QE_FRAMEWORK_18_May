"""Tests for Agent 55 — 3 Amigos Facilitator (True AI — Sonnet 4.6)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.refinement.agent_55_3_amigos_facilitator import (
    _build_amigos_message,
    _compute_confidence,
    run,
)
from src.core.schemas import initial_story_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT1_DATA = {
    "goal": "Adviser can view suitability scores for FSC clients",
    "persona": "Financial Adviser",
    "fsc_objects": ["FinancialAccount__c", "Suitability__c"],
    "ac_present": True,
    "ac_complete": True,
}

AGENT2_PASS = {
    "invest_verdict": "PASS",
    "invest_score": 36,
    "blocking_issues": [],
    "improvement_suggestions": ["Split epic AC into sub-stories"],
}

AGENT2_FAIL = {
    "invest_verdict": "FAIL",
    "invest_score": 18,
    "blocking_issues": ["Story lacks testability — no measurable outcome defined"],
    "improvement_suggestions": ["Add acceptance criteria with measurable thresholds"],
}

AGENT3_HIGH = {
    "fca_classification": "HIGH",
    "co_signoff_required": True,
    "regulatory_obligations": ["COBS 9", "Consumer Duty PS22/9"],
}

AGENT3_LOW = {
    "fca_classification": "LOW",
    "co_signoff_required": False,
    "regulatory_obligations": [],
}

AGENT4_CLEAN = {
    "cd_verdict": "PASS",
    "cd_obligations": ["fair_outcomes"],
    "vulnerable_customer_impact": False,
}

AGENT4_VC = {
    "cd_verdict": "PASS",
    "cd_obligations": ["fair_outcomes", "consumer_support"],
    "vulnerable_customer_impact": True,
}

AGENT5_DATA = {
    "ac_count": 4,
    "ac_clauses": [
        {"description": "Adviser can view score", "scenario_type": "happy_path"},
        {"description": "FCA check on submit", "scenario_type": "regulatory"},
    ],
    "remaining_gaps": ["Error handling when score unavailable"],
}

AGENT5_NO_ACS = {
    "ac_count": 0,
    "ac_clauses": [],
    "remaining_gaps": [],
}

AGENT7_DATA = {
    "data_volume": "MEDIUM",
    "data_isolation_strategy": "per_class_setup",
}

AGENT8_DEEP = {
    "dependency_depth": 3,
    "has_destructive_changes": True,
}

AGENT9_NO_CRITICAL = {
    "risk_register": [
        {"severity": "HIGH", "description": "Large FSC dependency chain increases deployment risk"},
    ],
    "overall_risk_level": "HIGH",
    "critical_risk_count": 0,
    "high_risk_count": 1,
}

AGENT9_CRITICAL = {
    "risk_register": [
        {"severity": "CRITICAL", "description": "No vulnerable customer test scenario — FG21/1 risk"},
        {"severity": "HIGH", "description": "Data isolation strategy not set for HIGH-FCA"},
    ],
    "overall_risk_level": "CRITICAL",
    "critical_risk_count": 2,
    "high_risk_count": 1,
}

# ── Mock LLM responses (all include new required keys) ────────────────────────

MOCK_AMIGOS_READY = {
    "ba_discussion_points": ["Confirm AC for score refresh timing"],
    "developer_discussion_points": ["Verify FSC FinancialAccount SOQL governor limits"],
    "tester_discussion_points": ["Add FCA COBS 9 suitability scenario"],
    "open_questions": ["What is the refresh SLA for suitability scores?"],
    "recommended_decisions": ["Agree refresh cadence before sprint start"],
    "story_ready_assessment": "READY",
    "facilitator_summary": "Story is well-defined. One open question on score refresh timing. Team to confirm SLA before sprint commitment.",
    "definition_of_done": [
        "All 4 ACs have corresponding Gherkin scenarios",
        "Apex test coverage ≥ 90% (HIGH-FCA threshold)",
        "CO sign-off obtained at G1 before sprint start",
        "FCA evidence pack generated with COMPLETE verdict",
        "No CRITICAL risks in risk register at sprint start",
    ],
    "action_items": [
        {"actor": "BA", "action": "Confirm AC for score refresh timing with PO", "priority": "MUST"},
        {"actor": "DEV", "action": "Verify SOQL governor limits on FinancialAccount queries", "priority": "MUST"},
        {"actor": "QA", "action": "Write FCA COBS 9 suitability scenario in Gherkin", "priority": "MUST"},
        {"actor": "PO", "action": "Define SLA for suitability score refresh cadence", "priority": "SHOULD"},
    ],
    "regression_affected_areas": ["Suitability__c records", "FinancialAccount__c views", "Client 360 page layout"],
    "regression_risk_level": "MEDIUM",
    "regression_notes": "Changes to suitability score display may affect existing Client 360 dashboards. Recommend running regression suite on Financial Planning and Client Profile modules.",
}

MOCK_AMIGOS_NEEDS_DISCUSSION = {
    "ba_discussion_points": ["Scope creep risk — 3 FSC objects not in original brief"],
    "developer_discussion_points": ["Spike required for SOQL optimisation on bulk data"],
    "tester_discussion_points": ["No vulnerable customer scenario despite VC impact flag"],
    "open_questions": [
        "Who owns the suitability calculation service?",
        "Is bulk data testing in scope?",
        "Which COBS rules apply to this change?",
    ],
    "recommended_decisions": [
        "Agree ownership of suitability service",
        "Define bulk data test scope",
    ],
    "story_ready_assessment": "NEEDS_DISCUSSION",
    "facilitator_summary": "Three key questions to resolve. Scope creep and bulk data risks need team alignment before sprint start.",
    "definition_of_done": [
        "All ACs written and agreed by BA, QA, and DEV before sprint start",
        "Apex test coverage ≥ 90% for HIGH-FCA story",
        "Vulnerable Customer scenario present with @fg21_1 tag",
        "CO sign-off at G1 obtained before sprint commitment",
    ],
    "action_items": [
        {"actor": "BA", "action": "Clarify scope — confirm which 3 FSC objects are in scope", "priority": "MUST"},
        {"actor": "DEV", "action": "Spike SOQL optimisation for bulk suitability query", "priority": "SHOULD"},
        {"actor": "QA", "action": "Write Vulnerable Customer scenario before sprint start", "priority": "MUST"},
        {"actor": "PO", "action": "Confirm ownership of suitability calculation service", "priority": "MUST"},
    ],
    "regression_affected_areas": ["Financial Planning module", "Client 360 suitability widget", "Suitability__c trigger chain"],
    "regression_risk_level": "HIGH",
    "regression_notes": "Bulk data changes to Suitability__c may affect the Financial Planning module and existing suitability trigger automations. Run the full Financial Planning regression suite before release.",
}

MOCK_AMIGOS_BLOCKED = {
    "ba_discussion_points": ["Missing acceptance criteria for core flow — must be defined before sprint"],
    "developer_discussion_points": ["Cannot estimate technical complexity without testable ACs"],
    "tester_discussion_points": ["No testable outcome defined — cannot write scenarios"],
    "open_questions": [
        "What does 'view suitability' mean technically?",
        "No INVEST-passing story can be written without this answer",
    ],
    "recommended_decisions": ["Rewrite story with testable ACs before sprint"],
    "story_ready_assessment": "BLOCKED",
    "facilitator_summary": "Story is BLOCKED. INVEST verdict FAIL. Story must be rewritten before entering Development.",
    "definition_of_done": [
        "INVEST verdict = PASS before sprint start",
        "At least 3 testable ACs defined and agreed",
        "Apex test coverage ≥ 90%",
    ],
    "action_items": [
        {"actor": "BA", "action": "Rewrite story with measurable acceptance criteria", "priority": "MUST"},
        {"actor": "PO", "action": "Reprioritise story — cannot enter sprint until ACs defined", "priority": "MUST"},
        {"actor": "QA", "action": "Review rewritten ACs for testability before re-refinement", "priority": "MUST"},
    ],
    "regression_affected_areas": ["Unknown — story scope undefined"],
    "regression_risk_level": "HIGH",
    "regression_notes": "Regression impact cannot be assessed until story scope and ACs are defined. Do not proceed to sprint without a full regression impact review.",
}

MOCK_AMIGOS_NO_ACS = {
    "ba_discussion_points": [
        "What business outcome does this story achieve for the adviser?",
        "What are the acceptance thresholds for the suitability score display?",
        "Which Consumer Duty outcome does this story address?",
    ],
    "developer_discussion_points": [
        "Which FSC objects will be queried — FinancialAccount__c, Suitability__c, or both?",
        "What are the SOQL governor limit constraints for this data volume?",
        "Is per_class_setup sufficient for data isolation given HIGH-FCA classification?",
    ],
    "tester_discussion_points": [
        "What FCA scenario types are required — COBS 9.2, Consumer Duty, or both?",
        "Is a Vulnerable Customer scenario required (vc_impact not yet known)?",
        "What is the minimum Gherkin scenario count for a HIGH-FCA story?",
    ],
    "open_questions": [
        "What are the measurable acceptance thresholds for each AC?",
        "Who owns the suitability calculation service?",
    ],
    "recommended_decisions": [
        "Agree and document at least 3 testable ACs in this session",
        "Confirm Vulnerable Customer applicability before testing begins",
    ],
    "story_ready_assessment": "NEEDS_DISCUSSION",
    "facilitator_summary": "No ACs have been written — this session's primary deliverable is the AC list. All three roles must contribute before the story can enter a sprint.",
    "definition_of_done": [
        "At least 3 testable ACs written and agreed in session",
        "Apex test coverage ≥ 90% (HIGH-FCA)",
        "CO sign-off at G1",
        "FCA evidence pack COMPLETE before production",
    ],
    "action_items": [
        {"actor": "BA", "action": "Write 3 testable ACs with measurable thresholds during session", "priority": "MUST"},
        {"actor": "QA", "action": "Confirm scenario types needed once ACs are defined", "priority": "MUST"},
        {"actor": "DEV", "action": "Identify FSC object access requirements once ACs are defined", "priority": "MUST"},
        {"actor": "PO", "action": "Sign off on agreed ACs before story enters sprint", "priority": "MUST"},
    ],
    "regression_affected_areas": ["Suitability__c", "FinancialAccount__c", "Client 360 page"],
    "regression_risk_level": "HIGH",
    "regression_notes": "Story touches shared FSC objects used across Financial Planning. Full regression assessment required once ACs and scope are confirmed.",
}


# ── Confidence scoring tests ──────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_invest_pass_no_critical_risks_scores_well(self):
        score, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        assert score >= 70

    def test_invest_fail_reduces_score(self):
        score_pass, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        score_fail, _ = _compute_confidence(AGENT2_FAIL, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "NEEDS_DISCUSSION")
        assert score_pass > score_fail

    def test_critical_risks_reduce_score(self):
        score_clean, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        score_critical, _ = _compute_confidence(AGENT2_PASS, AGENT9_CRITICAL, AGENT3_LOW, [], "BLOCKED")
        assert score_clean > score_critical

    def test_high_fca_applies_scrutiny_penalty(self):
        score_low, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        score_high, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_HIGH, [], "READY")
        assert score_low > score_high

    def test_many_open_questions_reduces_score(self):
        many_qs = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        score_few, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        score_many, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, many_qs, "NEEDS_DISCUSSION")
        assert score_few > score_many

    def test_blocked_assessment_reduces_score(self):
        score_ready, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        score_blocked, _ = _compute_confidence(AGENT2_FAIL, AGENT9_CRITICAL, AGENT3_HIGH, [], "BLOCKED")
        assert score_ready > score_blocked

    def test_score_never_exceeds_92(self):
        score, _ = _compute_confidence(AGENT2_PASS, AGENT9_NO_CRITICAL, AGENT3_LOW, [], "READY")
        assert score <= 92

    def test_score_never_below_20(self):
        score, _ = _compute_confidence(None, None, None, [], "BLOCKED")
        assert score >= 20


# ── Prompt building tests ─────────────────────────────────────────────────────

class TestBuildAmigosMessage:
    def _build(self, **kwargs) -> str:
        defaults = dict(
            story_id="FSC-001",
            agent1_data=AGENT1_DATA,
            agent2_data=AGENT2_PASS,
            agent3_data=AGENT3_HIGH,
            agent4_data=AGENT4_CLEAN,
            agent5_data=AGENT5_DATA,
            agent6_data=None,
            agent7_data=AGENT7_DATA,
            agent8_data=None,
            agent9_data=AGENT9_NO_CRITICAL,
            agent54_data=None,
        )
        defaults.update(kwargs)
        return _build_amigos_message(**defaults)

    def test_story_id_in_prompt(self):
        msg = self._build()
        assert "FSC-001" in msg

    def test_invest_verdict_in_prompt(self):
        msg = self._build()
        assert "PASS" in msg

    def test_invest_fail_blocking_issues_in_prompt(self):
        msg = self._build(agent2_data=AGENT2_FAIL)
        assert "testability" in msg

    def test_fca_classification_in_prompt(self):
        msg = self._build()
        assert "HIGH" in msg

    def test_vulnerable_customer_impact_in_prompt(self):
        msg = self._build(agent4_data=AGENT4_VC)
        assert "True" in msg

    def test_remaining_ac_gaps_in_prompt(self):
        msg = self._build()
        assert "Error handling" in msg

    def test_critical_risk_count_in_prompt(self):
        msg = self._build(agent9_data=AGENT9_CRITICAL)
        assert "2" in msg

    def test_no_upstream_data_does_not_crash(self):
        msg = _build_amigos_message("FSC-999", None, None, None, None, None, None, None, None, None, None)
        assert "FSC-999" in msg

    def test_tool_call_instruction_at_end(self):
        msg = self._build()
        assert "facilitate_3_amigos" in msg

    def test_zero_ac_count_produces_warning(self):
        msg = self._build(agent5_data=AGENT5_NO_ACS)
        assert "WARNING" in msg
        assert "0 ACs" in msg or "PRIMARY" in msg

    def test_data_isolation_strategy_in_prompt(self):
        msg = self._build()
        assert "per_class_setup" in msg

    def test_reminder_about_required_fields_at_end(self):
        msg = self._build()
        assert "definition_of_done" in msg
        assert "action_items" in msg
        assert "regression_affected_areas" in msg


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAgentRun:
    async def test_returns_agent_result(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["2"] = {"data": AGENT2_PASS}
        state["agent_results"]["9"] = {"data": AGENT9_NO_CRITICAL}

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_READY
            result = await run(state)

        assert result.agent_id == 55
        assert result.agent_name == "3 Amigos Facilitator"
        assert result.confidence.tier == "B"

    async def test_data_has_required_downstream_keys(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_NEEDS_DISCUSSION
            result = await run(state)

        for key in [
            "ba_discussion_points", "developer_discussion_points",
            "tester_discussion_points", "open_questions",
            "recommended_decisions", "story_ready_assessment",
            "facilitator_summary", "definition_of_done",
            "action_items",
            "regression_affected_areas", "regression_risk_level", "regression_notes",
        ]:
            assert key in result.data, f"Missing key: {key}"

    async def test_ready_assessment_in_data(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["2"] = {"data": AGENT2_PASS}
        state["agent_results"]["9"] = {"data": AGENT9_NO_CRITICAL}

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_READY
            result = await run(state)

        assert result.data["story_ready_assessment"] == "READY"

    async def test_blocked_assessment_propagates(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["2"] = {"data": AGENT2_FAIL}
        state["agent_results"]["9"] = {"data": AGENT9_CRITICAL}

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_BLOCKED
            result = await run(state)

        assert result.data["story_ready_assessment"] == "BLOCKED"

    async def test_open_questions_and_action_items_count_in_what(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_NEEDS_DISCUSSION
            result = await run(state)

        assert "3" in result.what       # 3 open questions
        assert "action item" in result.what

    async def test_facilitator_summary_used_as_why(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_READY
            result = await run(state)

        assert result.why == MOCK_AMIGOS_READY["facilitator_summary"]

    async def test_uses_default_model(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_READY
            result = await run(state)

        assert result.model_used == "claude-sonnet-4-6"

    async def test_no_upstream_data_still_runs(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_NEEDS_DISCUSSION
            result = await run(state)

        assert result.agent_id == 55

    async def test_all_three_role_lists_populated(self):
        state = initial_story_state("FSC-2417")

        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_NEEDS_DISCUSSION
            result = await run(state)

        assert len(result.data["ba_discussion_points"]) > 0
        assert len(result.data["developer_discussion_points"]) > 0
        assert len(result.data["tester_discussion_points"]) > 0


# ── Definition of Done tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDefinitionOfDone:
    async def _run_with(self, mock_data: dict) -> object:
        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = mock_data
            return await run(state)

    async def test_dod_present_in_data(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert "definition_of_done" in result.data

    async def test_dod_is_list(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert isinstance(result.data["definition_of_done"], list)

    async def test_dod_non_empty_when_ready(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert len(result.data["definition_of_done"]) > 0

    async def test_dod_non_empty_when_needs_discussion(self):
        result = await self._run_with(MOCK_AMIGOS_NEEDS_DISCUSSION)
        assert len(result.data["definition_of_done"]) > 0

    async def test_dod_non_empty_when_blocked(self):
        result = await self._run_with(MOCK_AMIGOS_BLOCKED)
        assert len(result.data["definition_of_done"]) > 0

    async def test_dod_non_empty_when_no_acs(self):
        result = await self._run_with(MOCK_AMIGOS_NO_ACS)
        assert len(result.data["definition_of_done"]) > 0

    async def test_dod_items_are_strings(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        for item in result.data["definition_of_done"]:
            assert isinstance(item, str)
            assert len(item) > 0


# ── Action items tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestActionItems:
    async def _run_with(self, mock_data: dict) -> object:
        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = mock_data
            return await run(state)

    async def test_action_items_present_in_data(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert "action_items" in result.data

    async def test_action_items_is_list(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert isinstance(result.data["action_items"], list)

    async def test_action_items_non_empty_when_ready(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert len(result.data["action_items"]) > 0

    async def test_action_items_non_empty_when_no_acs(self):
        result = await self._run_with(MOCK_AMIGOS_NO_ACS)
        assert len(result.data["action_items"]) > 0

    async def test_each_action_item_has_actor(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        for item in result.data["action_items"]:
            assert "actor" in item
            assert item["actor"] in ("BA", "DEV", "QA", "PO")

    async def test_each_action_item_has_action_text(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        for item in result.data["action_items"]:
            assert "action" in item
            assert isinstance(item["action"], str)
            assert len(item["action"]) > 0

    async def test_each_action_item_has_priority(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        for item in result.data["action_items"]:
            assert "priority" in item
            assert item["priority"] in ("MUST", "SHOULD", "COULD")

    async def test_actors_cover_multiple_roles(self):
        result = await self._run_with(MOCK_AMIGOS_NEEDS_DISCUSSION)
        actors = {item["actor"] for item in result.data["action_items"]}
        assert len(actors) >= 2  # at least two distinct roles assigned

    async def test_blocked_story_has_must_priority_items(self):
        result = await self._run_with(MOCK_AMIGOS_BLOCKED)
        must_items = [i for i in result.data["action_items"] if i["priority"] == "MUST"]
        assert len(must_items) > 0

    async def test_action_items_count_in_what(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        expected = str(len(MOCK_AMIGOS_READY["action_items"]))
        assert expected in result.what

    async def test_no_acs_assigns_ba_must_action(self):
        result = await self._run_with(MOCK_AMIGOS_NO_ACS)
        ba_must = [i for i in result.data["action_items"]
                   if i["actor"] == "BA" and i["priority"] == "MUST"]
        assert len(ba_must) > 0


# ── Regression impact assessment tests ───────────────────────────────────────

@pytest.mark.asyncio
class TestRegressionImpactAssessment:
    async def _run_with(self, mock_data: dict) -> object:
        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = mock_data
            return await run(state)

    async def test_regression_flat_fields_present_in_data(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        for key in ("regression_affected_areas", "regression_risk_level", "regression_notes"):
            assert key in result.data, f"Missing flat regression key: {key}"

    async def test_regression_affected_areas_is_list(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert isinstance(result.data["regression_affected_areas"], list)

    async def test_regression_risk_level_is_valid_enum(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        assert result.data["regression_risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    async def test_regression_notes_is_non_empty_string(self):
        result = await self._run_with(MOCK_AMIGOS_READY)
        notes = result.data["regression_notes"]
        assert isinstance(notes, str)
        assert len(notes) > 0

    async def test_high_dependency_depth_produces_high_or_critical_risk(self):
        state = initial_story_state("FSC-2417")
        state["agent_results"]["8"] = {"data": AGENT8_DEEP}
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_sonnet:
            mock_sonnet.return_value = MOCK_AMIGOS_NEEDS_DISCUSSION
            result = await run(state)
        assert result.data["regression_risk_level"] in ("HIGH", "CRITICAL")

    async def test_affected_areas_non_empty(self):
        result = await self._run_with(MOCK_AMIGOS_NEEDS_DISCUSSION)
        assert len(result.data["regression_affected_areas"]) > 0

    async def test_blocked_story_regression_notes_present(self):
        result = await self._run_with(MOCK_AMIGOS_BLOCKED)
        assert len(result.data["regression_notes"]) > 10  # meaningful narrative, not empty string


# ── Retry behaviour tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRetryBehaviour:
    async def test_no_retry_when_all_fields_populated(self):
        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_call:
            mock_call.return_value = MOCK_AMIGOS_READY
            await run(state)

        assert mock_call.call_count == 1

    async def test_retry_triggered_when_field_empty(self):
        first_response = {**MOCK_AMIGOS_READY, "ba_discussion_points": []}
        retry_response = {**MOCK_AMIGOS_READY, "ba_discussion_points": ["BA point from retry"]}

        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [first_response, retry_response]
            result = await run(state)

        assert mock_call.call_count == 2
        assert result.data["ba_discussion_points"] == ["BA point from retry"]
        # Non-empty fields from first call are preserved
        assert result.data["developer_discussion_points"] == MOCK_AMIGOS_READY["developer_discussion_points"]

    async def test_retry_message_names_empty_fields(self):
        first_response = {**MOCK_AMIGOS_READY, "tester_discussion_points": [], "definition_of_done": []}
        retry_response = {
            **MOCK_AMIGOS_READY,
            "tester_discussion_points": ["Tester point"],
            "definition_of_done": ["DoD item"],
        }

        state = initial_story_state("FSC-2417")
        with patch("src.agents.refinement.agent_55_3_amigos_facilitator.call_with_tool",
                   new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [first_response, retry_response]
            await run(state)

        retry_call_kwargs = mock_call.call_args_list[1]
        retry_user_msg = retry_call_kwargs.kwargs.get("user_message", "")
        assert "RETRY" in retry_user_msg
        assert "tester_discussion_points" in retry_user_msg
        assert "definition_of_done" in retry_user_msg


# ── Schema contract tests ─────────────────────────────────────────────────────

class TestSchemaContract:
    def test_required_arrays_have_min_items(self):
        from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA

        required_arrays = (
            "ba_discussion_points",
            "developer_discussion_points",
            "tester_discussion_points",
            "definition_of_done",
            "action_items",
        )
        props = _AMIGOS_TOOL_SCHEMA["properties"]
        for field in required_arrays:
            assert props[field].get("minItems") == 1, (
                f"'{field}' must have minItems=1 to prevent empty array responses"
            )

    def test_regression_affected_areas_has_min_items(self):
        from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA

        props = _AMIGOS_TOOL_SCHEMA["properties"]
        assert props["regression_affected_areas"].get("minItems") == 1, (
            "'regression_affected_areas' must have minItems=1 (flat field, not nested)"
        )

    def test_regression_impact_assessment_not_nested(self):
        from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA

        props = _AMIGOS_TOOL_SCHEMA["properties"]
        assert "regression_impact_assessment" not in props, (
            "regression_impact_assessment must be flattened — "
            "use regression_affected_areas, regression_risk_level, regression_notes instead"
        )

    def test_all_twelve_fields_in_required(self):
        from src.agents.refinement.agent_55_3_amigos_facilitator import _AMIGOS_TOOL_SCHEMA

        expected = {
            "ba_discussion_points", "developer_discussion_points",
            "tester_discussion_points", "open_questions",
            "recommended_decisions", "story_ready_assessment",
            "facilitator_summary", "definition_of_done",
            "action_items",
            "regression_affected_areas", "regression_risk_level", "regression_notes",
        }
        assert set(_AMIGOS_TOOL_SCHEMA["required"]) == expected
