"""
FSC Agentic QE Framework — Phase-Targeted Validation Runner
============================================================
Runs a single pipeline phase against pre-seeded prior-phase state.
Useful for rapid iteration on a specific phase without waiting for all 55 agents.

Usage:
    python -m validation.run_phase                              # development, FSC-2417
    python -m validation.run_phase --phase development --story FSC-3801
    python -m validation.run_phase --phase testing
    python -m validation.run_phase --skip-report

Requires:
    ANTHROPIC_API_KEY in .env

State seeding:
    Development agents (10–23) need Agent 3 (FCA class) and Agent 5 (AC data)
    from the Refinement phase.  These are pre-seeded from the story's mock data
    so you get realistic inputs without running all 9 refinement agents first.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

# Bootstrap (dotenv + sqlalchemy mock) is handled at import time by run_validation.
# Import it first before any other project module.
from validation.run_validation import (
    STORIES,
    OUTPUT_DIR,
    run_agent,
    _apply_patches,   # noqa: F401 — imported so patching infra is active
    _merge_result,
    _save_results,
    _print_agent_start,
    _print_batch_start,
    _print_agent_done,
)
from src.core.schemas import initial_story_state

# ── Phase execution plans (dependency-ordered, mirrors EXECUTION_PLAN) ────────

PHASE_PLANS: dict[str, list[list[int]]] = {
    "development": [
        [10, 11, 13], [12, 14, 15, 16], [17, 18], [19], [20, 21], [22], [23],
    ],
    "testing": [
        [24, 25, 32], [26, 29, 30], [27], [28, 31, 37], [33, 34, 38], [35], [36],
    ],
    "release": [
        [39, 47], [40], [41], [42], [43], [44], [45], [46], [48, 49, 50],
    ],
    "monitoring": [
        [51], [52], [53],
    ],
}

# ── State seeders ─────────────────────────────────────────────────────────────

def _fca_class_from_story(story: dict) -> str:
    labels = story.get("labels", [])
    if any("COBS" in lbl or lbl.startswith("FCA-") for lbl in labels):
        return "HIGH"
    if "Consumer-Duty" in labels:
        return "MEDIUM"
    return "LOW"


def _seed_refinement(story_id: str, story_data: dict) -> dict:
    """
    Pre-seed state with Agents 3 + 5 outputs for development/testing phase runs.
    FCA class is derived from story labels; ACs come from the mock AC list.
    """
    state = initial_story_state(story_id)
    story = story_data["story"]
    acs = story_data["acs"]
    fca_class = _fca_class_from_story(story)

    state["fca_classification"] = fca_class

    state["agent_results"]["3"] = {
        "agent_id": 3,
        "agent_name": "FCA Risk Classifier",
        "what": f"[SEEDED] FCA classification for {story_id}: {fca_class}",
        "why": "Pre-seeded from story labels — phase-targeted run, not a live API call.",
        "data": {
            "fca_classification": fca_class,
            "fca_triggers": ["COBS 9.2"] if fca_class == "HIGH" else ["Consumer Duty PS22/9"],
            "ensemble_agreement": True,
            "ta_position": "ASSERT",
        },
        "confidence": {"final_score": 85, "tier": "B", "escalated": False, "signals": {}},
        "model_used": "seeded",
    }

    state["agent_results"]["5"] = {
        "agent_id": 5,
        "agent_name": "AC Generator",
        "what": f"[SEEDED] {len(acs)} acceptance criteria for {story_id}",
        "why": "Pre-seeded from story mock ACs — phase-targeted run, not a live API call.",
        "data": {
            "ac_clause_count": len(acs),
            "generation_mode": "validated_existing",
            "ac_clauses": acs,
            "coverage_assessment": {
                "happy_path": True,
                "error_paths": True,
                "edge_cases": True,
                "regulatory": fca_class in ("HIGH", "MEDIUM"),
            },
            "remaining_gaps": [],
        },
        "confidence": {"final_score": 92, "tier": "B", "escalated": False, "signals": {}},
        "model_used": "seeded",
    }

    return state


# Each phase seeds from the phase(s) before it.
# Testing and beyond also need development state, but those agents are seeded
# incrementally as they run — only the refinement baseline needs pre-loading.
_STATE_SEEDERS: dict[str, object] = {
    "development": _seed_refinement,
    "testing":     _seed_refinement,
    "release":     _seed_refinement,
    "monitoring":  _seed_refinement,
}


# ── Phase runner ──────────────────────────────────────────────────────────────

async def run_phase(phase: str, story_id: str, output_dir: Path) -> list[dict]:
    story_data = STORIES.get(story_id, STORIES["FSC-2417"])
    state = _STATE_SEEDERS[phase](story_id, story_data)
    plan = PHASE_PLANS[phase]
    run_id = f"{story_id}_{phase}"

    all_results: list[dict] = []
    total_agents = sum(len(batch) for batch in plan)
    done = 0
    fca_class = state.get("fca_classification", "?")

    print(f"\n{'=' * 60}")
    print(f"  FSC QE Framework -- Phase-Targeted Validation")
    print(f"  Phase: {phase.title()}  |  Story: {story_id}  |  Agents: {total_agents}")
    print(f"  Seeded state: Agent 3 (FCA={fca_class}) + Agent 5 ({len(story_data['acs'])} ACs)")
    print(f"{'=' * 60}\n")

    for batch in plan:
        if len(batch) == 1:
            agent_id = batch[0]
            _print_agent_start(agent_id, "sequential")
            t0 = time.monotonic()
            result = await run_agent(agent_id, state, story_data)
            elapsed = int((time.monotonic() - t0) * 1000)
            result["elapsed_ms"] = elapsed
            _print_agent_done(result, elapsed)
            _merge_result(state, result)
            all_results.append(result)
            done += 1
        else:
            _print_batch_start(batch)
            t0 = time.monotonic()
            tasks = [run_agent(aid, state, story_data) for aid in batch]
            results = await asyncio.gather(*tasks)
            elapsed = int((time.monotonic() - t0) * 1000)
            for result in results:
                result["elapsed_ms"] = elapsed
                _print_agent_done(result, elapsed)
                _merge_result(state, result)
                all_results.append(result)
            done += len(batch)

        _save_results(all_results, output_dir, run_id)
        print(f"  Progress: {done}/{total_agents} agents complete\n")

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(phase: str, story_id: str, skip_report: bool) -> None:
    t_start = time.monotonic()
    run_id = f"{story_id}_{phase}"

    results = await run_phase(phase, story_id, OUTPUT_DIR)

    total_ms = int((time.monotonic() - t_start) * 1000)
    ok = sum(1 for r in results if r.get("status") == "ok")
    errors = sum(1 for r in results if r.get("status") == "error")
    scores = [r["confidence"]["final_score"] for r in results if "confidence" in r]

    print(f"\n{'=' * 60}")
    print(f"  Phase Validation Complete")
    print(f"  Phase: {phase.title()}  |  Story: {story_id}")
    print(f"  Agents: {len(results)}  OK: {ok}  Error: {errors}")
    print(f"  Avg confidence: {round(sum(scores) / len(scores), 1) if scores else 0}%")
    print(f"  Total time: {total_ms / 1000:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR / run_id}")
    print(f"{'=' * 60}\n")

    if not skip_report:
        from validation.generate_report import generate_html_report
        report_path = OUTPUT_DIR / f"{run_id}_report.html"
        # Pass run_id as story_id so the report title clearly shows "FSC-2417_development".
        # No full-pipeline assertions run for phase-only outputs — that is correct behaviour.
        generate_html_report(OUTPUT_DIR / run_id, report_path, run_id)
        print(f"  Report: {report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single FSC QE Framework pipeline phase."
    )
    parser.add_argument(
        "--phase", default="development",
        choices=list(PHASE_PLANS),
        help="Phase to run (default: development)",
    )
    parser.add_argument("--story", default="FSC-2417",
                        help="Story ID (default: FSC-2417)")
    parser.add_argument("--skip-report", action="store_true",
                        help="Skip HTML report generation")
    args = parser.parse_args()
    asyncio.run(main(args.phase, args.story, args.skip_report))
