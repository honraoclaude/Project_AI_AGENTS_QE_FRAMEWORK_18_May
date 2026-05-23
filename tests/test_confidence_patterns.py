"""
Confidence Pattern Tests — Findings 4, 6, 8 (Karpathy Principles 2 & 3)

Finding 4: 53 bespoke _compute_confidence() functions — all should share the same
  TierBScorer guardrails (.cap().floor().build()), verified here at the source level.

Finding 6: Agent 30 ensemble result selection — confirmed correct (call_b on disagree,
  call_a on agree); this test pins the fix so the ternary can't regress.

Finding 8 (REQ-33): _AGENT_BASE_MAP in Agent 52 must stay in sync with each agent file's
  TierBScorer(base=N). This test greps every agent file and asserts the map matches.
"""

import re
from pathlib import Path

import pytest

AGENTS_ROOT = Path(__file__).parent.parent / "src" / "agents"


def _all_agent_files() -> list[Path]:
    return [
        p for p in AGENTS_ROOT.rglob("*.py")
        if p.name not in ("__init__.py", "base.py")
    ]


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Finding 4: TierBScorer guardrail chain ────────────────────────────────────

@pytest.mark.parametrize("agent_file", _all_agent_files(), ids=lambda p: p.name)
def test_tierbscorer_uses_guardrail_chain(agent_file: Path):
    """
    Every agent that creates a TierBScorer must call .cap(), .floor(), and .build()
    in the same file.  This catches an accidental bare .build() that bypasses the
    bounding guardrails introduced to replace 53 hand-rolled confidence functions.
    """
    source = _src(agent_file)
    if "TierBScorer(" not in source:
        pytest.skip("file does not use TierBScorer")

    assert ".cap(" in source, (
        f"{agent_file.name}: TierBScorer used but .cap() missing — "
        "confidence score is not capped; add .cap(92) (or appropriate ceiling)"
    )
    assert ".floor(" in source, (
        f"{agent_file.name}: TierBScorer used but .floor() missing — "
        "confidence score is not floored; add .floor(20) (or appropriate minimum)"
    )
    assert ".build()" in source, (
        f"{agent_file.name}: TierBScorer used but .build() missing — "
        "scorer never finalised"
    )


# ── Finding 6: Agent 30 ensemble result selection ─────────────────────────────

def test_agent30_ensemble_ternary_is_correct():
    """
    Pin Agent 30's ensemble result selection.  The correct logic is:
      cautious (call_b) wins on disagreement; permissive (call_a) used when both agree.

    The wrong version (bug that existed historically) was:
      call_b_result if not ensemble_agreement else call_b_result   ← both branches same

    This test fails if the ternary regresses to always returning call_b.
    """
    agent30 = AGENTS_ROOT / "testing" / "agent_30_fca_scenario_agent.py"
    source = _src(agent30)

    # Correct: disagreement → cautious (call_b); agreement → permissive (call_a)
    assert "call_b_result if not ensemble_agreement else call_a_result" in source, (
        "Agent 30 ensemble ternary is wrong.  Expected: "
        "'call_b_result if not ensemble_agreement else call_a_result'. "
        "The two branches must differ — call_b (cautious) on disagree, "
        "call_a (permissive/minimum) on agree."
    )


# ── Finding 8 (REQ-33): _AGENT_BASE_MAP sync ─────────────────────────────────

def _extract_agent_id(source: str) -> int | None:
    m = re.search(r"^AGENT_ID\s*=\s*(\d+)", source, re.MULTILINE)
    return int(m.group(1)) if m else None


def _extract_tierbscorer_base(source: str) -> int | None:
    m = re.search(r"TierBScorer\s*\(\s*base\s*=\s*(\d+)\s*\)", source)
    return int(m.group(1)) if m else None


def test_agent_base_map_in_sync():
    """
    _AGENT_BASE_MAP in Agent 52 must match the TierBScorer(base=N) value in every
    agent source file that declares both AGENT_ID and a TierBScorer.

    Tier-A agents (confidence=97 hardcoded, no TierBScorer) are automatically excluded
    because they produce no TierBScorer regex match.

    Failure here means an agent's base was changed without updating the map — the
    calibration agent would then compute wrong delta recommendations.
    """
    from src.agents.monitoring.agent_52_severity_calibration import _AGENT_BASE_MAP  # noqa: PLC0415

    mismatches: list[str] = []
    missing_from_map: list[str] = []

    for agent_file in _all_agent_files():
        source = _src(agent_file)
        agent_id = _extract_agent_id(source)
        base = _extract_tierbscorer_base(source)

        if agent_id is None or base is None:
            continue  # no AGENT_ID or no TierBScorer → not a Tier-B pipeline agent

        if agent_id not in _AGENT_BASE_MAP:
            missing_from_map.append(
                f"Agent {agent_id} ({agent_file.name}) uses TierBScorer(base={base}) "
                "but is NOT in _AGENT_BASE_MAP — add it"
            )
        elif _AGENT_BASE_MAP[agent_id] != base:
            mismatches.append(
                f"Agent {agent_id} ({agent_file.name}): "
                f"TierBScorer(base={base}) in source but _AGENT_BASE_MAP has {_AGENT_BASE_MAP[agent_id]}"
            )

    errors = mismatches + missing_from_map
    assert not errors, (
        "_AGENT_BASE_MAP is out of sync with agent source files:\n"
        + "\n".join(f"  {e}" for e in errors)
    )
