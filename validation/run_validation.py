"""
FSC Agentic QE Framework — Pipeline Validation Runner
======================================================

Runs all 54 agents against a realistic sample FSC Wealth Management story.
Saves each agent's output as JSON, then generates an HTML report.

Usage:
    python -m validation.run_validation                    # default story FSC-2417
    python -m validation.run_validation --story FSC-9001   # custom story id label
    python -m validation.run_validation --skip-report       # JSON only

Requires:
    ANTHROPIC_API_KEY in .env (or environment) to make live Claude API calls.
    Jira and Copado are mocked with realistic sample data — no live credentials needed.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

# ── Bootstrap: load .env and patch DB before any project imports ──────────────
from dotenv import load_dotenv
load_dotenv()

from unittest.mock import MagicMock
patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=MagicMock()).start()
patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=MagicMock()).start()

# Now safe to import project modules
from src.core.schemas import initial_story_state  # noqa: E402

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Agent classification map (for report metadata) ────────────────────────────
AGENT_META: dict[int, dict] = {
    1:  {"name": "Story Intent Agent",          "phase": "Refinement",  "class": "True AI"},
    2:  {"name": "INVEST Quality Agent",         "phase": "Refinement",  "class": "True AI"},
    3:  {"name": "FCA Classifier",               "phase": "Refinement",  "class": "True AI"},
    4:  {"name": "Consumer Duty Mapper",          "phase": "Refinement",  "class": "True AI"},
    5:  {"name": "AC Generator",                 "phase": "Refinement",  "class": "True AI"},
    6:  {"name": "Test Design Strategy",         "phase": "Refinement",  "class": "True AI"},
    54: {"name": "AC Challenger",                "phase": "Refinement",  "class": "True AI"},
    7:  {"name": "Data Need Agent",              "phase": "Refinement",  "class": "Augmented Script"},
    8:  {"name": "Dependency Mapping",           "phase": "Refinement",  "class": "Augmented Script"},
    9:  {"name": "Risk Anticipation",            "phase": "Refinement",  "class": "True AI"},
    10: {"name": "AC Compliance",               "phase": "Development", "class": "True AI"},
    11: {"name": "Branch Tracer",               "phase": "Development", "class": "Augmented Script"},
    12: {"name": "Apex Coverage",               "phase": "Development", "class": "Augmented Script"},
    13: {"name": "Metadata Dependency",         "phase": "Development", "class": "Augmented Script"},
    14: {"name": "Code Quality",                "phase": "Development", "class": "Augmented Script"},
    15: {"name": "Apex Security",               "phase": "Development", "class": "Augmented Script"},
    16: {"name": "Bulk Quality",                "phase": "Development", "class": "Augmented Script"},
    17: {"name": "SFDX Validator",              "phase": "Development", "class": "Augmented Script"},
    18: {"name": "Component Attribution",       "phase": "Development", "class": "Augmented Script"},
    19: {"name": "BDD Gherkin Writer",          "phase": "Development", "class": "True AI"},
    20: {"name": "Performance Risk",            "phase": "Development", "class": "Augmented Script"},
    21: {"name": "Test Data Architect",         "phase": "Development", "class": "True AI"},
    22: {"name": "Sandbox State",               "phase": "Development", "class": "Augmented Script"},
    23: {"name": "Story-to-Code Tracer",        "phase": "Development", "class": "True AI"},
    24: {"name": "Test Strategy Validator",     "phase": "Testing",     "class": "True AI"},
    25: {"name": "Test Env Provisioner",        "phase": "Testing",     "class": "Augmented Script"},
    26: {"name": "CRT Scenario Designer",       "phase": "Testing",     "class": "True AI"},
    27: {"name": "CRT Execution",               "phase": "Testing",     "class": "Augmented Script"},
    28: {"name": "CRT Self-Heal Reviewer",      "phase": "Testing",     "class": "True AI"},
    29: {"name": "UAT Test Case Generator",     "phase": "Testing",     "class": "True AI"},
    30: {"name": "FCA Scenario Agent",          "phase": "Testing",     "class": "True AI"},
    31: {"name": "Financial Data Integrity",    "phase": "Testing",     "class": "Augmented Script"},
    32: {"name": "Regression Risk Assessor",    "phase": "Testing",     "class": "Augmented Script"},
    33: {"name": "Test Coverage Analyser",      "phase": "Testing",     "class": "Augmented Script"},
    34: {"name": "Defect Triage",               "phase": "Testing",     "class": "True AI"},
    35: {"name": "Root Cause Analyser",         "phase": "Testing",     "class": "True AI"},
    36: {"name": "UAT Coordination",            "phase": "Testing",     "class": "Augmented Script"},
    37: {"name": "Performance Test",            "phase": "Testing",     "class": "Augmented Script"},
    38: {"name": "Flaky Test Hunter",           "phase": "Testing",     "class": "Augmented Script"},
    39: {"name": "Release Readiness",           "phase": "Release",     "class": "Augmented Script"},
    40: {"name": "Release Composer",            "phase": "Release",     "class": "Augmented Script"},
    41: {"name": "Change Set Integrity",        "phase": "Release",     "class": "Augmented Script"},
    42: {"name": "Dry Run",                     "phase": "Release",     "class": "Augmented Script"},
    43: {"name": "Smoke on Staging",            "phase": "Release",     "class": "Augmented Script"},
    44: {"name": "FCA Evidence Pack",           "phase": "Release",     "class": "True AI"},
    45: {"name": "Go/No-Go Coordinator",        "phase": "Release",     "class": "Augmented Script"},
    46: {"name": "Production Validation",       "phase": "Release",     "class": "Augmented Script"},
    47: {"name": "Release Notes Writer",        "phase": "Release",     "class": "True AI"},
    48: {"name": "Rollback Readiness",          "phase": "Release",     "class": "Augmented Script"},
    49: {"name": "Post-Release Monitor",        "phase": "Release",     "class": "Augmented Script"},
    50: {"name": "Release Retrospective",       "phase": "Release",     "class": "True AI"},
    51: {"name": "Agent Health Monitor",        "phase": "Monitoring",  "class": "Augmented Script"},
    52: {"name": "Severity Calibration Agent",  "phase": "Monitoring",  "class": "Augmented Script"},
    53: {"name": "Incident Response Agent",     "phase": "Monitoring",  "class": "True AI"},
}

# ── Execution batches (preserves dependency order) ────────────────────────────
EXECUTION_PLAN: list[list[int]] = [
    # Refinement (54=AC Challenger runs after Agent 5, before Agent 9)
    [1, 8], [2, 3, 7], [4], [5], [54, 6], [9],
    # Development
    [10, 11, 13], [12, 14, 15, 16], [17, 18], [19], [20, 21], [22], [23],
    # Testing
    [24, 25, 32], [26, 29, 30], [27], [28, 31, 37], [33, 34, 38], [35], [36],
    # Release
    [39, 47], [40], [41], [42], [43], [44], [45], [46], [48, 49, 50],
    # Monitoring
    [51], [52], [53],
]

# ── Sample FSC story (mocked Jira response) ───────────────────────────────────
SAMPLE_STORY = {
    "story_id": "FSC-2417",
    "summary": "As a Wealth Manager, I want to view a client's consolidated suitability score combining risk profile, investment objectives, and financial circumstances, so that I can make compliant advice recommendations under COBS 9",
    "description": (
        "## Background\n"
        "Our current suitability assessment process requires advisers to manually cross-reference "
        "three separate systems: the MiFID II risk profile tool, the objectives register, and "
        "the client financial circumstances record. This is error-prone and time-consuming, "
        "and increases the risk of non-compliant advice.\n\n"
        "## Goal\n"
        "Build a consolidated suitability dashboard on the FSC Client 360 page that displays "
        "a single suitability score (0–100) computed from the three assessment dimensions. "
        "The score must be recalculated automatically when any underlying assessment changes.\n\n"
        "## Regulatory Context\n"
        "This story directly supports compliance with COBS 9.2 (suitability assessment) and "
        "Consumer Duty PS22/9 (good outcomes for retail clients). The Suitability__c object "
        "in FSC stores the individual dimension scores; the consolidated score will be stored "
        "on FinancialAccount as a custom field ConsolidatedSuitabilityScore__c.\n\n"
        "## Technical Scope\n"
        "- New Apex class: SuitabilityScoreCalculator (triggered by Suitability__c update)\n"
        "- LWC component: suitability-dashboard (embedded on FSC Client 360 page)\n"
        "- Custom field: FinancialAccount.ConsolidatedSuitabilityScore__c (Number, 3,2)\n"
        "- Platform Event: SuitabilityScoreUpdated__e (for real-time dashboard refresh)\n\n"
        "## Out of Scope\n"
        "- Changes to the underlying MiFID II risk assessment questionnaire\n"
        "- Integration with third-party portfolio risk systems"
    ),
    "status": "In Progress",
    "issue_type": "Story",
    "priority": "High",
    "labels": ["FCA-COBS9", "Consumer-Duty", "Suitability", "Sprint-47"],
    "components": ["FSC-ClientPortfolio", "Apex-Services", "LWC-Dashboard"],
    "assignee": "james.chen@wealthfirm.co.uk",
    "reporter": "sarah.patel@wealthfirm.co.uk",
}

SAMPLE_AC = [
    {
        "scenario": "View consolidated suitability score for a fully assessed client",
        "given": [
            "Given a client has a completed MiFID II risk assessment on file",
            "And their investment objectives are recorded in the system",
            "And their financial circumstances were updated within the last 12 months",
        ],
        "when": ["When the Wealth Manager opens the client suitability dashboard"],
        "then": [
            "Then a consolidated suitability score out of 100 is displayed prominently",
            "And a breakdown by dimension (risk 40%, objectives 35%, circumstances 25%) is shown",
            "And the date of the most recent assessment is displayed for each dimension",
        ],
    },
    {
        "scenario": "Automatic score recalculation on assessment update",
        "given": [
            "Given a client has an existing consolidated suitability score of 72",
            "And their risk profile is updated following an annual review",
        ],
        "when": ["When the Suitability__c record is saved with the updated risk classification"],
        "then": [
            "Then the consolidated score is recalculated within 5 seconds",
            "And the updated score is visible on the suitability dashboard without page refresh",
            "And a SuitabilityScoreUpdated__e platform event is fired with the new score",
        ],
    },
    {
        "scenario": "Suitability review alert for stale financial circumstances",
        "given": [
            "Given a client's financial circumstances record was last updated more than 12 months ago",
        ],
        "when": ["When the Wealth Manager views the suitability dashboard"],
        "then": [
            "Then a review required banner is displayed with the date of the last update",
            "And the consolidated score is flagged as 'Requires Review' rather than a numeric value",
            "And the Wealth Manager is prompted to initiate a circumstances review",
        ],
    },
    {
        "scenario": "COBS 9.2 compliance warning for Retail Client advice",
        "given": [
            "Given a client is classified as Retail Client under COBS 3.4",
            "And their consolidated suitability score is below 50",
        ],
        "when": ["When the Wealth Manager prepares to record advice"],
        "then": [
            "Then a mandatory COBS 9.2 suitability warning is displayed",
            "And the adviser must acknowledge the warning before proceeding",
            "And the acknowledgement is recorded against the FinancialAccount for audit purposes",
        ],
    },
]

# ── Monitoring mock data ──────────────────────────────────────────────────────
MOCK_SIGNAL_ROWS = [
    {"agent_id": 1,  "total": 45, "fp": 3, "tp": 38, "fn": 2, "tn": 2},
    {"agent_id": 2,  "total": 45, "fp": 1, "tp": 42, "fn": 1, "tn": 1},
    {"agent_id": 3,  "total": 45, "fp": 2, "tp": 40, "fn": 2, "tn": 1},
    {"agent_id": 5,  "total": 38, "fp": 4, "tp": 30, "fn": 3, "tn": 1},
    {"agent_id": 54, "total": 15, "fp": 1, "tp": 12, "fn": 1, "tn": 1},  # AC Challenger
    {"agent_id": 33, "total": 52, "fp": 0, "tp": 50, "fn": 1, "tn": 1},
    {"agent_id": 44, "total": 30, "fp": 1, "tp": 27, "fn": 1, "tn": 1},
]

# ── Copado mock data (for development agents 11-14) ───────────────────────────
MOCK_BRANCH = {
    "branch_name": "feature/FSC-2417-suitability-dashboard",
    "commit_sha": "a3f8c2e1d94b0571e8c3f2a6b9d4e7f0c1a5b8d2",
    "created_date": "2026-05-12T09:00:00Z",
    "last_commit_date": "2026-05-17T16:42:00Z",
    "author_email": "james.chen@wealthfirm.co.uk",
}
MOCK_CHANGED_FILES = [
    {"file_path": "force-app/main/default/classes/SuitabilityScoreCalculator.cls",
     "change_type": "add", "object_type": "ApexClass", "object_name": "SuitabilityScoreCalculator"},
    {"file_path": "force-app/main/default/classes/SuitabilityScoreCalculator.cls-meta.xml",
     "change_type": "add", "object_type": "ApexClass", "object_name": "SuitabilityScoreCalculator"},
    {"file_path": "force-app/main/default/lwc/suitabilityDashboard/suitabilityDashboard.js",
     "change_type": "add", "object_type": "LightningComponentBundle", "object_name": "suitabilityDashboard"},
    {"file_path": "force-app/main/default/lwc/suitabilityDashboard/suitabilityDashboard.html",
     "change_type": "add", "object_type": "LightningComponentBundle", "object_name": "suitabilityDashboard"},
    {"file_path": "force-app/main/default/objects/FinancialAccount__c/fields/ConsolidatedSuitabilityScore__c.field-meta.xml",
     "change_type": "add", "object_type": "CustomField", "object_name": "ConsolidatedSuitabilityScore__c"},
    {"file_path": "force-app/main/default/triggers/SuitabilityScoreTrigger.trigger",
     "change_type": "add", "object_type": "ApexTrigger", "object_name": "SuitabilityScoreTrigger"},
    {"file_path": "force-app/main/default/platformEventChannels/SuitabilityScoreUpdated__e.evt-meta.xml",
     "change_type": "add", "object_type": "PlatformEvent", "object_name": "SuitabilityScoreUpdated__e"},
]
MOCK_APEX_RESULTS = {
    "test_run_id": "7072v000001OAdxAAG",
    "tests_run": 42,
    "tests_passed": 42,
    "tests_failed": 0,
    "coverage_pct": 87.3,
    "run_date": "2026-05-17T16:55:00Z",
}
MOCK_PMD_RESULTS = [
    {
        "rule_name": "ApexCRUDViolation", "priority": 3,
        "description": "Perform CRUD permission check before querying Suitability__c",
        "file_path": "force-app/main/default/classes/SuitabilityScoreCalculator.cls",
        "line": 47, "category": "Security",
    },
    {
        "rule_name": "AvoidGlobalModifier", "priority": 4,
        "description": "Avoid using global modifier — use public instead",
        "file_path": "force-app/main/default/classes/SuitabilityScoreCalculator.cls",
        "line": 1, "category": "Best Practices",
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_agent(agent_id: int, state: dict, patches: list) -> dict:
    """Import and call an agent's run() function. Returns a serialisable result dict."""
    meta = AGENT_META[agent_id]
    module_path = _module_path(agent_id)

    try:
        module = importlib.import_module(module_path)
        with _apply_patches(agent_id, patches):
            result = await module.run(state)
        return {
            "agent_id": agent_id,
            "agent_name": meta["name"],
            "phase": meta["phase"],
            "classification": meta["class"],
            "status": "ok",
            "elapsed_ms": None,  # set by caller
            **result.model_dump(),
        }
    except Exception as exc:
        return {
            "agent_id": agent_id,
            "agent_name": meta["name"],
            "phase": meta["phase"],
            "classification": meta["class"],
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


async def run_all(story_id: str, output_dir: Path) -> list[dict]:
    """Run all 54 agents in dependency order. Returns list of result dicts."""
    state = initial_story_state(story_id)
    all_results = []
    patches = _build_patches()

    total_agents = sum(len(batch) for batch in EXECUTION_PLAN)
    done = 0

    print(f"\n{'=' * 60}")
    print(f"  FSC QE Framework -- Pipeline Validation")
    print(f"  Story: {story_id}  |  Agents: {total_agents}")
    print(f"{'=' * 60}\n")

    for batch in EXECUTION_PLAN:
        if len(batch) == 1:
            agent_id = batch[0]
            _print_agent_start(agent_id, "sequential")
            t0 = time.monotonic()
            result = await run_agent(agent_id, state, patches)
            elapsed = int((time.monotonic() - t0) * 1000)
            result["elapsed_ms"] = elapsed
            _print_agent_done(result, elapsed)
            _merge_result(state, result)
            all_results.append(result)
            done += 1
        else:
            _print_batch_start(batch)
            t0 = time.monotonic()
            tasks = [run_agent(aid, state, patches) for aid in batch]
            results = await asyncio.gather(*tasks)
            elapsed = int((time.monotonic() - t0) * 1000)
            for result in results:
                result["elapsed_ms"] = elapsed
                _print_agent_done(result, elapsed)
                _merge_result(state, result)
                all_results.append(result)
            done += len(batch)

        _save_results(all_results, output_dir, story_id)
        print(f"  Progress: {done}/{total_agents} agents complete\n")

    return all_results


def _merge_result(state: dict, result: dict) -> None:
    """Push agent output into state so downstream agents can read it."""
    if result.get("status") == "ok" and "data" in result:
        state["agent_results"][str(result["agent_id"])] = {
            "agent_id": result["agent_id"],
            "agent_name": result["agent_name"],
            "what": result.get("what", ""),
            "why": result.get("why", ""),
            "data": result["data"],
            "confidence": result.get("confidence", {}),
            "model_used": result.get("model_used", ""),
        }
        # Propagate FCA classification from Agent 3
        if result["agent_id"] == 3:
            fca = result["data"].get("fca_classification", "")
            if fca:
                state["fca_classification"] = fca


def _save_results(results: list[dict], output_dir: Path, story_id: str) -> None:
    """Save all results to JSON after each batch."""
    story_dir = output_dir / story_id
    story_dir.mkdir(exist_ok=True)
    for r in results:
        aid = r["agent_id"]
        name_slug = r["agent_name"].lower().replace(" ", "_").replace("/", "_")
        filepath = story_dir / f"agent_{aid:02d}_{name_slug}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=str)

    # Also save a pipeline summary
    summary = _build_summary(results)
    with open(story_dir / "_pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _build_summary(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    scores = [r["confidence"]["final_score"] for r in ok if "confidence" in r]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_agents": len(results),
        "passed": len(ok),
        "failed": len(errors),
        "avg_confidence": round(sum(scores) / len(scores), 1) if scores else 0,
        "min_confidence": min(scores) if scores else 0,
        "max_confidence": max(scores) if scores else 0,
        "errors": [{"agent_id": r["agent_id"], "error": r.get("error", "")} for r in errors],
        "total_elapsed_ms": sum(r.get("elapsed_ms") or 0 for r in results),
    }


# ── Patch builders ────────────────────────────────────────────────────────────

def _build_patches() -> list:
    return []  # patches applied per-agent in _apply_patches


from contextlib import contextmanager, ExitStack


# Agents that import get_story / get_acceptance_criteria from src.integrations.jira
_JIRA_AGENTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 21, 29, 30}

# Copado function names needed per agent (at agent module level)
_COPADO_PATCHES: dict[int, dict[str, object]] = {}  # filled lazily below


@contextmanager
def _apply_patches(agent_id: int, _unused):
    """
    Patch external calls at the agent module level.

    Using `from X import Y` in agent modules creates a local binding that is
    unaffected by patching X.Y after import.  We must patch the name inside the
    agent module itself (module_path.func_name) so the mock is seen at call time.
    """
    module_path = _module_path(agent_id)
    patches_to_apply = []

    # ── Jira ──────────────────────────────────────────────────────────────────
    if agent_id in _JIRA_AGENTS:
        # Not every jira agent imports both functions — check hasattr first.
        # patch() defers attribute lookup to __enter__, so try/except on patch()
        # itself doesn't help; we must check the module before constructing the patch.
        mod = importlib.import_module(module_path)
        for func_name, mock_value in [
            ("get_story", AsyncMock(return_value=SAMPLE_STORY)),
            ("get_acceptance_criteria", AsyncMock(return_value=SAMPLE_AC)),
        ]:
            if hasattr(mod, func_name):
                patches_to_apply.append(patch(f"{module_path}.{func_name}", new=mock_value))

    # ── Copado ────────────────────────────────────────────────────────────────
    copado_mocks: dict[str, object] = {
        11: {"get_branch_for_story": AsyncMock(return_value=MOCK_BRANCH)},
        12: {"get_apex_test_results": AsyncMock(return_value=MOCK_APEX_RESULTS)},
        13: {
            "get_branch_for_story": AsyncMock(return_value=MOCK_BRANCH),
            "get_changed_files": AsyncMock(return_value=MOCK_CHANGED_FILES),
        },
        14: {"get_pmd_results": AsyncMock(return_value=MOCK_PMD_RESULTS)},
    }.get(agent_id, {})

    for func_name, mock_value in copado_mocks.items():
        patches_to_apply.append(patch(f"{module_path}.{func_name}", new=mock_value))

    # ── Agent 51 (Health Monitor) ─────────────────────────────────────────────
    if agent_id == 51:
        from src.agents.monitoring.agent_51_health import AGENT_NAMES
        from src.core.schemas import AgentHealthMetric
        from datetime import datetime, timezone
        metrics = [
            AgentHealthMetric(
                agent_id=aid, agent_name=name,
                last_run_at=datetime.now(timezone.utc),
                runs_last_hour=12, errors_last_hour=0,
                avg_latency_ms=850.0, avg_confidence=0.0,
                false_positive_rate_30d=0.04, status="HEALTHY",
            )
            for aid, name in AGENT_NAMES.items()
        ]
        patches_to_apply.append(
            patch("src.agents.monitoring.agent_51_health._collect_metrics",
                  new=AsyncMock(return_value=metrics))
        )

    # ── Agent 52 (Severity Calibration) ───────────────────────────────────────
    if agent_id == 52:
        patches_to_apply.append(
            patch("src.agents.monitoring.agent_52_severity_calibration._fetch_signal_summary",
                  new=AsyncMock(return_value=MOCK_SIGNAL_ROWS))
        )

    with ExitStack() as stack:
        for p in patches_to_apply:
            stack.enter_context(p)
        yield


# ── Console output helpers ────────────────────────────────────────────────────

def _print_agent_start(agent_id: int, mode: str) -> None:
    meta = AGENT_META[agent_id]
    print(f"  -->  Agent {agent_id:02d} | {meta['name']:<35} [{meta['class']}]")


def _print_batch_start(batch: list[int]) -> None:
    names = ", ".join(f"{aid}:{AGENT_META[aid]['name'].split()[0]}" for aid in batch)
    print(f"  ==>  Batch [{names}] (parallel)")


def _print_agent_done(result: dict, elapsed_ms: int) -> None:
    status = result.get("status", "?")
    if status == "ok":
        conf = result.get("confidence", {}).get("final_score", "?")
        icon = "OK"
    else:
        conf = "ERR"
        icon = "FAIL"
    name = result.get("agent_name", f"Agent {result['agent_id']}")
    print(f"         [{icon}] {name:<35} conf={conf}  {elapsed_ms}ms")


# ── Module path helper ────────────────────────────────────────────────────────

def _module_path(agent_id: int) -> str:
    phase = AGENT_META[agent_id]["phase"].lower()
    phase_map = {
        "refinement": "refinement",
        "development": "development",
        "testing": "testing",
        "release": "release",
        "monitoring": "monitoring",
    }
    pkg = phase_map[phase]
    name_slug = AGENT_META[agent_id]["name"].lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    # Manual overrides for agents where slug doesn't match filename exactly
    overrides = {
        1: "agent_01_story_intent",
        2: "agent_02_invest_quality",
        3: "agent_03_fca_classifier",
        4: "agent_04_consumer_duty",
        5: "agent_05_ac_generator",
        6: "agent_06_test_design",
        7: "agent_07_data_need",
        8: "agent_08_dependency_mapping",
        9: "agent_09_risk_anticipation",
        10: "agent_10_ac_compliance",
        11: "agent_11_branch_tracer",
        12: "agent_12_apex_coverage",
        13: "agent_13_metadata_dependency",
        14: "agent_14_code_quality",
        15: "agent_15_apex_security",
        16: "agent_16_bulk_quality",
        17: "agent_17_sfdx_validator",
        18: "agent_18_component_attribution",
        19: "agent_19_bdd_gherkin_writer",
        20: "agent_20_performance_risk",
        21: "agent_21_test_data_architect",
        22: "agent_22_sandbox_state",
        23: "agent_23_story_code_tracer",
        24: "agent_24_test_strategy_validator",
        25: "agent_25_test_env_provisioner",
        26: "agent_26_crt_scenario_designer",
        27: "agent_27_crt_execution",
        28: "agent_28_crt_self_heal_reviewer",
        29: "agent_29_uat_test_case_generator",
        30: "agent_30_fca_scenario_agent",
        31: "agent_31_financial_data_integrity",
        32: "agent_32_regression_risk_assessor",
        33: "agent_33_test_coverage_analyser",
        34: "agent_34_defect_triage",
        35: "agent_35_root_cause_analyser",
        36: "agent_36_uat_coordination",
        37: "agent_37_performance_test",
        38: "agent_38_flaky_test_hunter",
        39: "agent_39_release_readiness",
        40: "agent_40_release_composer",
        41: "agent_41_change_set_integrity",
        42: "agent_42_dry_run",
        43: "agent_43_smoke_on_staging",
        44: "agent_44_fca_evidence_pack",
        45: "agent_45_go_no_go",
        46: "agent_46_production_validation",
        47: "agent_47_release_notes_writer",
        48: "agent_48_rollback_readiness",
        49: "agent_49_post_release_monitor",
        50: "agent_50_retrospective",
        51: "agent_51_health",
        52: "agent_52_severity_calibration",
        53: "agent_53_incident_response",
        54: "agent_05b_ac_challenger",
    }
    return f"src.agents.{pkg}.{overrides[agent_id]}"


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(story_id: str, skip_report: bool) -> None:
    t_start = time.monotonic()
    output_dir = OUTPUT_DIR

    results = await run_all(story_id, output_dir)

    total_ms = int((time.monotonic() - t_start) * 1000)
    ok = sum(1 for r in results if r.get("status") == "ok")
    errors = sum(1 for r in results if r.get("status") == "error")
    scores = [r["confidence"]["final_score"] for r in results if "confidence" in r]

    print(f"\n{'=' * 60}")
    print(f"  Validation Complete")
    print(f"  Agents: {len(results)}  OK: {ok}  Error: {errors}")
    print(f"  Avg confidence: {round(sum(scores)/len(scores), 1) if scores else 0}%")
    print(f"  Total time: {total_ms / 1000:.1f}s")
    print(f"  Outputs: {output_dir / story_id}")
    print(f"{'=' * 60}\n")

    if not skip_report:
        from validation.generate_report import generate_html_report
        report_path = output_dir / f"{story_id}_report.html"
        generate_html_report(output_dir / story_id, report_path, story_id)
        print(f"  Report: {report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", default="FSC-2417")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.story, args.skip_report))
