"""
FSC Agentic QE Framework — Multi-Story Dashboard Generator
==========================================================

Scans validation/outputs/ for story folders and generates a three-layer
dashboard HTML at validation/outputs/dashboard.html:

  Layer 1 — Executive: headline stats + Chart.js trend/distribution charts
  Layer 2 — Delivery:  sortable story table (PO / release manager view)
  Layer 3 — Technical: expandable per-story accordion (QE / dev detail)

Usage (standalone):
    python -m validation.generate_dashboard

Called automatically by run_validation.py when --dashboard flag is passed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"

PHASE_ORDER = ["Refinement", "Development", "Testing", "Release", "Monitoring"]

VERDICT_COLOUR = {
    "PASS": "green", "GO": "green", "COMPLETE": "green", "HEALTHY": "green",
    "READY": "green", "SIGNED_OFF": "green", "FEASIBLE": "green",
    # Agent 45 coalition verdicts (Wave 5 game theory)
    "UNANIMOUS_GO": "green", "DISSENT_NO_GO": "red",
    "WARN": "amber", "PARTIAL": "amber", "CONDITIONAL": "amber", "PENDING": "amber",
    "FAIL": "red", "NO_GO": "red", "BLOCKED": "red", "MISSING": "red", "CRITICAL": "red",
    "SKIPPED": "gray", "UNKNOWN": "gray", "—": "gray",
    "HIGH": "amber", "MEDIUM": "amber", "LOW": "green", "UNCLASSIFIED": "gray",
}


def _escape(s: object) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _conf_colour(score: float | None) -> str:
    if score is None:
        return "gray"
    if score >= 75:
        return "green"
    if score >= 60:
        return "amber"
    return "red"


def _verdict_colour(v: str | None) -> str:
    return VERDICT_COLOUR.get(str(v or "—").upper(), "gray")


def _load_agent_json(story_dir: Path, agent_id: int) -> dict:
    """Load the first matching agent JSON for a given agent_id. Returns {} if missing."""
    matches = sorted(story_dir.glob(f"agent_{agent_id:02d}_*.json"))
    if not matches:
        return {}
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_story(story_dir: Path) -> dict | None:
    """
    Load all dashboard-relevant data for one story directory.
    Returns None if the directory lacks a pipeline summary.
    """
    summary_path = story_dir / "_pipeline_summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    story_id = story_dir.name

    # Parse generated_at
    generated_at_raw = summary.get("generated_at", "")
    try:
        dt = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
        date_str = dt.strftime("%d %b %Y")
        sort_key = dt.timestamp()
    except Exception:
        date_str = "—"
        sort_key = 0.0

    # Key agent loads
    a2 = _load_agent_json(story_dir, 2).get("data", {})
    a3 = _load_agent_json(story_dir, 3).get("data", {})
    a5 = _load_agent_json(story_dir, 5).get("data", {})
    a9_full = _load_agent_json(story_dir, 9)
    a9 = a9_full.get("data", {})
    a33 = _load_agent_json(story_dir, 33).get("data", {})
    a44 = _load_agent_json(story_dir, 44).get("data", {})
    a45 = _load_agent_json(story_dir, 45).get("data", {})
    a54_full = _load_agent_json(story_dir, 54)
    a54 = a54_full.get("data", {})
    a55 = _load_agent_json(story_dir, 55).get("data", {})

    # Phase confidence breakdown — load all agent files
    phase_scores: dict[str, list[float]] = {p: [] for p in PHASE_ORDER}
    for f in sorted(story_dir.glob("agent_*.json")):
        if f.name.startswith("_"):
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        phase = r.get("phase", "")
        score = r.get("confidence", {}).get("final_score")
        if phase in phase_scores and score is not None:
            phase_scores[phase].append(float(score))

    phase_confidence = {
        p: round(sum(v) / len(v), 1) if v else None
        for p, v in phase_scores.items()
    }

    # Shapley attributions from Agent 09 signals
    shapley = (a9_full.get("confidence", {})
               .get("signals", {})
               .get("shapley_attributions", {}))

    # Recommended actions (top 3 for display)
    recommended_actions: list[str] = a9.get("recommended_actions", [])

    # Duration formatting
    elapsed_ms = summary.get("total_elapsed_ms", 0)
    if elapsed_ms:
        elapsed_str = f"{elapsed_ms / 1000:.0f}s" if elapsed_ms < 120_000 else f"{elapsed_ms / 60_000:.1f}m"
    else:
        elapsed_str = "—"

    # Individual report link
    report_path = story_dir.parent / f"{story_id}_report.html"
    report_exists = report_path.exists()

    return {
        "story_id": story_id,
        "date_str": date_str,
        "sort_key": sort_key,
        "elapsed_str": elapsed_str,
        "total_agents": summary.get("total_agents", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "avg_confidence": summary.get("avg_confidence", 0),
        "min_confidence": summary.get("min_confidence", 0),
        "max_confidence": summary.get("max_confidence", 0),
        # Agent 2 — INVEST
        "invest_score": a2.get("invest_score"),
        "invest_verdict": a2.get("invest_verdict"),
        # Agent 3 — FCA
        "fca_classification": a3.get("fca_classification", "—"),
        # Agent 5 — AC Generator
        "ac_count": a5.get("ac_count"),
        # Agent 9 — Risk
        "overall_risk_level": a9.get("overall_risk_level"),
        "critical_risk_count": a9.get("critical_risk_count", 0),
        "high_risk_count": a9.get("high_risk_count", 0),
        "risk_summary": a9.get("risk_summary"),
        "recommended_actions": recommended_actions,
        "shapley_attributions": shapley,
        # Agent 33 — Coverage
        "overall_coverage_pct": a33.get("overall_coverage_pct"),
        "coverage_verdict": a33.get("coverage_verdict"),
        # Agent 44 — FCA Evidence
        "evidence_verdict": a44.get("evidence_verdict"),
        "regulatory_sign_off_ready": a44.get("regulatory_sign_off_ready"),
        # Agent 45 — Go/No-Go + coalition (Wave 5 game theory)
        "go_decision": a45.get("go_decision"),
        "coordinator_verdict": a45.get("coordinator_verdict"),
        "no_go_reasons": a45.get("no_go_reasons", []),
        "coalition_verdict": a45.get("coalition_verdict"),
        "coalition_dissent": a45.get("coalition_dissent", []),
        # Agent 54 — AC Challenger
        "ac_challenged": a54.get("ac_count_challenged"),
        "survivor_count": a54.get("survivor_count"),
        "critical_weakness_count": a54.get("critical_weakness_count"),
        "challenge_summary": a54.get("challenge_summary"),
        # Agent 55 — 3 Amigos Facilitator
        "story_ready_assessment": a55.get("story_ready_assessment"),
        "open_questions_count": len(a55.get("open_questions", [])),
        # Phase breakdown
        "phase_confidence": phase_confidence,
        "report_exists": report_exists,
        "report_filename": f"{story_id}_report.html",
    }


def scan_stories(output_root: Path) -> list[dict]:
    """Scan for story folders and return list of story dicts, sorted by date."""
    stories = []
    for child in sorted(output_root.iterdir()):
        if child.is_dir():
            data = _load_story(child)
            if data:
                stories.append(data)
    stories.sort(key=lambda s: s["sort_key"])
    return stories


# ── Chip / badge helpers ──────────────────────────────────────────────────────

def _chip(text: str, colour: str) -> str:
    return f'<span class="chip chip-{colour}">{_escape(text)}</span>'



def _mini_bar(pct: float | None, width: int = 60) -> str:
    if pct is None:
        return '<span style="color:#a0aec0;font-size:11px">—</span>'
    col = _conf_colour(pct)
    return (
        f'<div style="display:flex;align-items:center;gap:5px">'
        f'<div style="width:{width}px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden">'
        f'<div style="width:{min(pct,100):.0f}%;height:100%;background:var(--{col})"></div></div>'
        f'<span style="font-size:11px;color:#4a5568">{pct:.0f}%</span></div>'
    )


# ── Layer 2 — Story table row ─────────────────────────────────────────────────

def _render_story_row(s: dict, idx: int) -> str:
    story_id = _escape(s["story_id"])

    fca = s["fca_classification"] or "—"
    fca_chip = _chip(fca, _verdict_colour(fca))

    agents_cell = f'{s["passed"]}/{s["total_agents"]}'
    agents_col = "green" if s["failed"] == 0 else "red"

    invest = f'{s["invest_score"]}' if s.get("invest_score") is not None else "—"
    invest_verdict = s.get("invest_verdict") or "—"
    invest_col = _verdict_colour(invest_verdict)

    risk = s.get("overall_risk_level") or "—"
    risk_col = _verdict_colour(risk)

    cov = s.get("overall_coverage_pct")
    cov_cell = f'{cov:.0f}%' if cov is not None else "—"
    cov_col = "green" if (cov or 0) >= 80 else "red" if (cov or 0) < 60 else "amber"

    evidence = s.get("evidence_verdict") or "—"
    ev_col = _verdict_colour(evidence)

    gng = s.get("coordinator_verdict") or "—"
    gng_col = _verdict_colour(gng)

    coalition = s.get("coalition_verdict") or ""
    coalition_suffix = ""
    if coalition:
        col_col = _verdict_colour(coalition)
        short = "✓ UNANIMOUS" if coalition == "UNANIMOUS_GO" else "✗ DISSENT"
        coalition_suffix = f' <span class="chip chip-{col_col}" style="font-size:9px">{short}</span>'

    report_link = (
        f'<a href="{_escape(s["report_filename"])}" class="report-link">Open →</a>'
        if s["report_exists"]
        else '<span style="color:#a0aec0;font-size:11px">—</span>'
    )

    return f"""      <tr class="story-row" data-idx="{idx}" onclick="toggleAccordion({idx})">
        <td><span class="story-id-btn">{story_id}</span></td>
        <td>{_escape(s["date_str"])}</td>
        <td>{fca_chip}</td>
        <td><span class="chip chip-{agents_col}">{agents_cell}</span></td>
        <td>{_mini_bar(s["avg_confidence"])}</td>
        <td>{invest} <span class="chip chip-{invest_col}" style="font-size:10px">{_escape(invest_verdict)}</span></td>
        <td>{_chip(risk, risk_col)}</td>
        <td><span class="chip chip-{cov_col}">{cov_cell}</span></td>
        <td>{_chip(evidence, ev_col)}</td>
        <td>{_chip(gng, gng_col)}{coalition_suffix}</td>
        <td style="color:#718096;font-size:12px">{_escape(s["elapsed_str"])}</td>
        <td onclick="event.stopPropagation()">{report_link}</td>
      </tr>
      <tr class="accordion-row" id="acc-{idx}" style="display:none">
        <td colspan="12">{_render_accordion_body(s, idx)}</td>
      </tr>"""


# ── Layer 3 — Accordion body ──────────────────────────────────────────────────

def _render_accordion_body(s: dict, idx: int) -> str:
    # Phase confidence mini-bars
    phase_bars = ""
    for phase in PHASE_ORDER:
        conf = s["phase_confidence"].get(phase)
        label_icon = {"Refinement": "🔍", "Development": "⚙️", "Testing": "🧪",
                      "Release": "🚀", "Monitoring": "📡"}.get(phase, "")
        phase_bars += f"""
          <div class="phase-mini">
            <span class="phase-mini-label">{label_icon} {phase}</span>
            {_mini_bar(conf, 90)}
          </div>"""

    # Top risks + coalition dissent
    actions = s.get("recommended_actions", [])[:3]
    risk_items = ""
    for action in actions:
        severity = "critical" if "[BLOCKING" in action else "high" if "[HIGH" in action else "medium"
        short = _escape(action[:200] + ("…" if len(action) > 200 else ""))
        risk_items += f'<div class="mini-risk mini-risk-{severity}">{short}</div>'
    coalition_dissent = s.get("coalition_dissent", [])
    if coalition_dissent:
        dissent_str = _escape(", ".join(coalition_dissent))
        risk_items += f'<div class="mini-risk mini-risk-critical">Coalition dissent: {dissent_str}</div>'
    if not risk_items:
        risk_items = '<div style="color:#a0aec0;font-size:12px">No risk data available</div>'

    # AC Challenger stats
    challenged = s.get("ac_challenged")
    survivors = s.get("survivor_count")
    critical_w = s.get("critical_weakness_count")
    challenge_summary = s.get("challenge_summary") or "—"

    if challenged is not None:
        survival_pct = round((survivors or 0) / challenged * 100) if challenged > 0 else 0
        challenger_html = f"""
          <div class="acc-mini-grid">
            <div class="acc-mini-cell"><div class="acc-mini-val">{challenged}</div><div class="acc-mini-lbl">Clauses Challenged</div></div>
            <div class="acc-mini-cell"><div class="acc-mini-val" style="color:var(--green)">{survivors}</div><div class="acc-mini-lbl">Survived</div></div>
            <div class="acc-mini-cell"><div class="acc-mini-val" style="color:var(--red)">{critical_w}</div><div class="acc-mini-lbl">Critical Issues</div></div>
            <div class="acc-mini-cell"><div class="acc-mini-val">{survival_pct}%</div><div class="acc-mini-lbl">Survival Rate</div></div>
          </div>
          <div class="challenge-summary">{_escape(challenge_summary[:350] + ("…" if len(challenge_summary) > 350 else ""))}</div>"""
    else:
        challenger_html = '<div style="color:#a0aec0;font-size:12px">Agent 54 not run for this story</div>'

    # Shapley mini chart
    shapley = s.get("shapley_attributions", {})
    shapley_html = ""
    if shapley:
        max_val = max(shapley.values()) if shapley else 1
        for agent_label, pct in sorted(shapley.items(), key=lambda x: -x[1]):
            bar_pct = (pct / max_val * 100) if max_val else 0
            is_54 = agent_label in ("5b", "54")
            bar_class = "shapley-bar-highlight" if is_54 else "shapley-bar"
            label = f"Agt {agent_label}" if not is_54 else "Agt 54 (Challenger)"
            shapley_html += f"""
          <div class="shapley-row">
            <span class="shapley-label">{label}</span>
            <div class="shapley-wrap"><div class="{bar_class}" style="width:{bar_pct:.0f}%">{pct:.1f}%</div></div>
          </div>"""
    else:
        shapley_html = '<div style="color:#a0aec0;font-size:12px">No Shapley data</div>'

    # 3 Amigos readiness
    ready_assessment = s.get("story_ready_assessment") or "—"
    open_q_count = s.get("open_questions_count", 0)
    ready_col = _verdict_colour(ready_assessment)
    amigos_html = f"""
          <div class="acc-mini-grid">
            <div class="acc-mini-cell"><div class="acc-mini-val"><span class="chip chip-{ready_col}">{_escape(ready_assessment)}</span></div><div class="acc-mini-lbl">Story Ready Assessment</div></div>
            <div class="acc-mini-cell"><div class="acc-mini-val">{open_q_count}</div><div class="acc-mini-lbl">Open Questions</div></div>
          </div>"""

    return f"""
      <div class="accordion-body">
        <div class="acc-grid">
          <div class="acc-section">
            <div class="acc-section-title">Phase Confidence</div>
            {phase_bars}
          </div>
          <div class="acc-section">
            <div class="acc-section-title">Top Risks (Agent 09)</div>
            {risk_items}
          </div>
          <div class="acc-section">
            <div class="acc-section-title">AC Challenger (Agent 54)</div>
            {challenger_html}
          </div>
          <div class="acc-section">
            <div class="acc-section-title">3 Amigos Facilitator (Agent 55)</div>
            {amigos_html}
          </div>
          <div class="acc-section">
            <div class="acc-section-title">Shapley Attribution (Agent 09)</div>
            {shapley_html}
          </div>
        </div>
      </div>"""


# ── Chart data builders ───────────────────────────────────────────────────────

def _build_chart_data(stories: list[dict]) -> str:
    labels = json.dumps([s["story_id"] for s in stories])
    conf_data = json.dumps([s["avg_confidence"] for s in stories])

    phase_datasets = []
    colours = {
        "Refinement": "rgba(49,130,206,0.8)",
        "Development": "rgba(56,178,172,0.8)",
        "Testing": "rgba(246,173,85,0.8)",
        "Release": "rgba(104,211,145,0.8)",
        "Monitoring": "rgba(159,122,234,0.8)",
    }
    for phase in PHASE_ORDER:
        vals = json.dumps([s["phase_confidence"].get(phase) for s in stories])
        phase_datasets.append(
            f'{{"label":"{phase}","data":{vals},"borderColor":"{colours[phase]}",'
            f'"backgroundColor":"{colours[phase].replace("0.8","0.15")}",'
            f'"tension":0.3,"pointRadius":4}}'
        )

    # FCA tier counts
    fca_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNCLASSIFIED": 0}
    for s in stories:
        tier = (s.get("fca_classification") or "UNCLASSIFIED").upper()
        fca_counts[tier] = fca_counts.get(tier, 0) + 1
    fca_labels = json.dumps(list(fca_counts.keys()))
    fca_vals = json.dumps(list(fca_counts.values()))

    # Evidence verdict counts
    ev_counts: dict[str, int] = {"COMPLETE": 0, "PARTIAL": 0, "MISSING": 0, "—": 0}
    for s in stories:
        ev = (s.get("evidence_verdict") or "—").upper()
        ev_counts[ev] = ev_counts.get(ev, 0) + 1
    ev_labels = json.dumps(list(ev_counts.keys()))
    ev_vals = json.dumps(list(ev_counts.values()))

    return f"""
const chartLabels = {labels};
const confData = {conf_data};
const phaseDatasets = [{",".join(phase_datasets)}];
const fcaLabels = {fca_labels};
const fcaData = {fca_vals};
const evLabels = {ev_labels};
const evData = {ev_vals};
"""


# ── Go/No-Go mini status grid ─────────────────────────────────────────────────

def _render_gng_grid(stories: list[dict]) -> str:
    cells = ""
    for s in stories:
        gng = (s.get("coordinator_verdict") or "—").upper()
        col = _verdict_colour(gng)
        cells += (
            f'<div class="gng-cell gng-{col}" title="{_escape(s["story_id"])}: {_escape(gng)}">'
            f'{_escape(s["story_id"])}'
            f'<div class="gng-verdict">{_escape(gng)}</div></div>'
        )
    return cells


# ── Headline stats ────────────────────────────────────────────────────────────

def _render_headline_stats(stories: list[dict]) -> str:
    if not stories:
        return ""
    total_stories = len(stories)
    all_passed = sum(s["passed"] for s in stories)
    all_agents = sum(s["total_agents"] for s in stories)
    pass_rate = round(all_passed / all_agents * 100, 1) if all_agents else 0
    avg_conf = round(sum(s["avg_confidence"] for s in stories) / total_stories, 1)
    high_fca = sum(1 for s in stories if (s.get("fca_classification") or "").upper() == "HIGH")
    go_count = sum(1 for s in stories if (s.get("coordinator_verdict") or "").upper() == "GO")

    return f"""
      <div class="stat-card"><div class="stat-value">{total_stories}</div><div class="stat-label">Stories Tracked</div></div>
      <div class="stat-card"><div class="stat-value green">{pass_rate:.0f}%</div><div class="stat-label">Agent Pass Rate</div></div>
      <div class="stat-card"><div class="stat-value teal">{avg_conf:.1f}%</div><div class="stat-label">Avg Confidence</div></div>
      <div class="stat-card"><div class="stat-value red">{high_fca}</div><div class="stat-label">HIGH-FCA Stories</div></div>
      <div class="stat-card"><div class="stat-value green">{go_count}</div><div class="stat-label">GO Decisions</div></div>
      <div class="stat-card"><div class="stat-value">{all_agents}</div><div class="stat-label">Total Agent Runs</div></div>"""


# ── Main HTML generator ───────────────────────────────────────────────────────

def generate(output_root: Path | None = None) -> Path:
    if output_root is None:
        output_root = OUTPUT_DIR

    stories = scan_stories(output_root)
    output_path = output_root / "dashboard.html"

    gen_str = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    headline_stats = _render_headline_stats(stories)
    chart_data_js = _build_chart_data(stories) if stories else "const chartLabels=[];const confData=[];const phaseDatasets=[];const fcaLabels=[];const fcaData=[];const evLabels=[];const evData=[];"
    gng_grid = _render_gng_grid(stories)
    table_rows = "\n".join(_render_story_row(s, i) for i, s in enumerate(stories))

    no_data_msg = "" if stories else '<tr><td colspan="12" style="text-align:center;padding:32px;color:#a0aec0">No story runs found in validation/outputs/</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FSC QE Framework — Multi-Story Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--green:#276749;--amber:#b7791f;--red:#c53030;--teal:#0e7490;--gray:#718096}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#edf2f7;color:#2d3748;font-size:14px;line-height:1.5}}
a{{color:#3182ce;text-decoration:none}}

/* Header */
.dash-header{{background:linear-gradient(135deg,#0d1b2a 0%,#1a365d 60%,#2c5282 100%);color:#fff;padding:28px 40px 20px;border-bottom:4px solid #00b4d8}}
.dash-header h1{{font-size:22px;font-weight:700;letter-spacing:.5px}}
.dash-header h2{{font-size:12px;font-weight:400;opacity:.7;margin-top:3px;text-transform:uppercase;letter-spacing:1.2px}}
.header-row{{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px}}
.header-pills{{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;font-size:12px;opacity:.85}}
.pill{{background:rgba(255,255,255,.12);border-radius:4px;padding:3px 10px}}

/* Container */
.container{{max-width:1360px;margin:0 auto;padding:24px 20px 60px}}

/* Section cards */
.card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:22px 26px;margin-bottom:22px}}
.card-title{{font-size:13px;font-weight:700;color:#1a365d;text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #ebf8ff}}
.card-title .badge{{font-size:10px;background:#ebf8ff;color:#2b6cb0;padding:2px 8px;border-radius:4px;margin-left:8px;font-weight:600;vertical-align:middle}}

/* Stat grid */
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:22px}}
.stat-card{{background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center}}
.stat-value{{font-size:28px;font-weight:700;color:#1a365d;line-height:1}}
.stat-value.green{{color:#276749}}.stat-value.red{{color:#c53030}}.stat-value.teal{{color:#0e7490}}.stat-value.amber{{color:#b7791f}}
.stat-label{{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#718096;margin-top:5px}}

/* Charts row */
.charts-row{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:16px;margin-bottom:22px}}
@media(max-width:900px){{.charts-row{{grid-template-columns:1fr}}}}
.chart-card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:18px 20px}}
.chart-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:#4a5568;margin-bottom:12px}}
.chart-wrap{{position:relative;height:180px}}

/* Go/No-Go grid */
.gng-grid{{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}}
.gng-cell{{padding:8px 14px;border-radius:6px;font-size:11px;font-weight:700;cursor:default;text-align:center;min-width:80px}}
.gng-verdict{{font-size:10px;font-weight:400;margin-top:2px;opacity:.85}}
.gng-green{{background:#c6f6d5;color:#22543d}}.gng-red{{background:#fed7d7;color:#742a2a}}
.gng-amber{{background:#fefcbf;color:#744210}}.gng-gray{{background:#e2e8f0;color:#4a5568}}

/* Chips */
.chip{{display:inline-block;border-radius:4px;font-size:11px;font-weight:700;padding:2px 8px;text-transform:uppercase;letter-spacing:.3px;white-space:nowrap}}
.chip-green{{background:#c6f6d5;color:#22543d}}.chip-red{{background:#fed7d7;color:#742a2a}}
.chip-amber{{background:#fefcbf;color:#744210}}.chip-gray{{background:#e2e8f0;color:#4a5568}}
.chip-teal{{background:#e6fffa;color:#234e52}}

/* Story table */
.table-wrap{{overflow-x:auto}}
table.story-table{{width:100%;border-collapse:collapse;font-size:12px}}
.story-table th{{background:#f7fafc;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#718096;padding:9px 12px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0;cursor:pointer;user-select:none;white-space:nowrap}}
.story-table th:hover{{background:#edf2f7;color:#2d3748}}
.story-table th.sort-asc::after{{content:" ▲";font-size:9px}}
.story-table th.sort-desc::after{{content:" ▼";font-size:9px}}
.story-table td{{padding:10px 12px;border-bottom:1px solid #f0f4f8;vertical-align:middle}}
.story-row{{cursor:pointer;transition:background .1s}}
.story-row:hover td{{background:#f7fafc}}
.story-row.open td{{background:#ebf8ff}}
.story-id-btn{{color:#2b6cb0;font-weight:700;font-size:12px}}
.report-link{{color:#2b6cb0;font-weight:600;font-size:11px;padding:3px 10px;background:#ebf8ff;border-radius:4px}}
.report-link:hover{{background:#bee3f8}}
.accordion-row td{{padding:0;border-bottom:1px solid #e2e8f0}}

/* Accordion body */
.accordion-body{{background:#f7fafc;padding:18px 20px;border-top:1px solid #e2e8f0}}
.acc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
.acc-section{{background:#fff;border-radius:8px;padding:14px 16px;border:1px solid #e2e8f0}}
.acc-section-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:#718096;margin-bottom:10px}}
.phase-mini{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.phase-mini-label{{font-size:11px;color:#4a5568;width:100px;flex-shrink:0}}
.mini-risk{{font-size:11px;padding:7px 10px;border-radius:5px;margin-bottom:6px;line-height:1.45}}
.mini-risk-critical{{background:#fff5f5;border-left:3px solid #c53030;color:#4a5568}}
.mini-risk-high{{background:#fffff0;border-left:3px solid #d69e2e;color:#4a5568}}
.mini-risk-medium{{background:#f0fff4;border-left:3px solid #38a169;color:#4a5568}}
.acc-mini-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}}
.acc-mini-cell{{text-align:center;background:#f7fafc;border-radius:6px;padding:8px 4px}}
.acc-mini-val{{font-size:18px;font-weight:700;color:#1a365d;line-height:1}}
.acc-mini-lbl{{font-size:10px;color:#718096;margin-top:3px}}
.challenge-summary{{font-size:11px;color:#4a5568;line-height:1.5;background:#fffff0;border-radius:5px;padding:8px;border:1px solid #fefcbf}}
.shapley-row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.shapley-label{{font-size:11px;color:#4a5568;width:110px;flex-shrink:0;text-align:right}}
.shapley-wrap{{flex:1;background:#edf2f7;border-radius:3px;height:16px}}
.shapley-bar{{height:16px;border-radius:3px;background:#3182ce;display:flex;align-items:center;padding-left:6px;color:#fff;font-size:10px;font-weight:600;min-width:32px}}
.shapley-bar-highlight{{height:16px;border-radius:3px;background:#00b4d8;display:flex;align-items:center;padding-left:6px;color:#fff;font-size:10px;font-weight:600;min-width:32px}}

/* Layer labels */
.layer-label{{display:flex;align-items:center;gap:10px;font-size:12px;color:#718096;margin-bottom:6px}}
.layer-badge{{background:#1a365d;color:#fff;font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:.5px}}

/* Footer */
.footer{{text-align:center;padding:24px;color:#a0aec0;font-size:11px}}
</style>
</head>
<body>

<div class="dash-header">
  <div class="header-row">
    <div>
      <h1>FSC Agentic QE Framework — Multi-Story Dashboard</h1>
      <h2>Cross-Sprint Quality Intelligence · PACT Edition v1 + Game Theory</h2>
    </div>
    <div style="text-align:right">
      <div style="background:rgba(255,255,255,.15);border-radius:6px;padding:8px 18px;font-size:13px;font-weight:700">{len(stories)} Stories Tracked</div>
    </div>
  </div>
  <div class="header-pills">
    <span class="pill">Generated: <strong>{gen_str}</strong></span>
    <span class="pill">Agents: <strong>55 per story</strong></span>
    <span class="pill">Framework: <strong>PACT + Game Theory</strong></span>
  </div>
</div>

<div class="container">

  <!-- ── LAYER 1: EXECUTIVE ──────────────────────────────────────────────── -->
  <div class="layer-label"><span class="layer-badge">LAYER 1</span> Executive Summary — Leadership &amp; FCA</div>

  <div class="stat-grid">{headline_stats}</div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="chart-card">
      <div class="chart-title">Confidence Trend (Avg by Story)</div>
      <div class="chart-wrap"><canvas id="confTrend"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">FCA Risk Tier Distribution</div>
      <div class="chart-wrap"><canvas id="fcaDonut"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Regulatory Evidence Verdict</div>
      <div class="chart-wrap"><canvas id="evBar"></canvas></div>
    </div>
  </div>

  <!-- Go/No-Go Status Grid -->
  <div class="card">
    <div class="card-title">Go / No-Go Decisions <span class="badge">per story</span></div>
    <div class="gng-grid">{gng_grid if gng_grid else '<span style="color:#a0aec0;font-size:12px">No stories yet</span>'}</div>
  </div>

  <!-- ── LAYER 2: DELIVERY ───────────────────────────────────────────────── -->
  <div class="layer-label" style="margin-top:8px"><span class="layer-badge">LAYER 2</span> Delivery View — PO &amp; Release Manager</div>

  <div class="card" style="padding:0">
    <div style="padding:16px 22px 0">
      <div class="card-title" style="border-bottom:none;margin-bottom:0">Story Pipeline Summary
        <span class="badge">click row for detail · click header to sort</span>
      </div>
    </div>
    <div class="table-wrap">
      <table class="story-table" id="storyTable">
        <thead>
          <tr>
            <th onclick="sortTable(0)">Story</th>
            <th onclick="sortTable(1)">Date</th>
            <th onclick="sortTable(2)">FCA Tier</th>
            <th onclick="sortTable(3)">Agents</th>
            <th onclick="sortTable(4)">Avg Conf</th>
            <th onclick="sortTable(5)">INVEST</th>
            <th onclick="sortTable(6)">Risk Level</th>
            <th onclick="sortTable(7)">Coverage</th>
            <th onclick="sortTable(8)">Evidence</th>
            <th onclick="sortTable(9)">Go/No-Go</th>
            <th onclick="sortTable(10)">Duration</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody id="storyTbody">
{table_rows}
{no_data_msg}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── LAYER 3: TECHNICAL (accordion) ─────────────────────────────────── -->
  <div class="layer-label"><span class="layer-badge">LAYER 3</span> Technical Detail — QE &amp; Dev (click a story row above)</div>

</div>

<div class="footer">
  FSC Agentic QE Framework · PACT Edition v1 + Game Theory · Dashboard generated {gen_str}
</div>

<script>
{chart_data_js}

// ── Chart.js charts ──
const commonFont = {{ family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size: 11 }};

// 1. Confidence trend
new Chart(document.getElementById('confTrend'), {{
  type: 'line',
  data: {{
    labels: chartLabels,
    datasets: [
      {{
        label: 'Avg Confidence',
        data: confData,
        borderColor: '#00b4d8',
        backgroundColor: 'rgba(0,180,216,0.1)',
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: '#00b4d8',
        fill: true,
      }},
      {{
        label: 'Target (75%)',
        data: chartLabels.map(() => 75),
        borderColor: '#38a169',
        borderDash: [6, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      }},
      ...phaseDatasets
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: commonFont, boxWidth: 10 }}, position: 'bottom' }} }},
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ font: commonFont }}, grid: {{ color: '#f0f4f8' }} }},
      x: {{ ticks: {{ font: commonFont }} }}
    }}
  }}
}});

// 2. FCA donut
new Chart(document.getElementById('fcaDonut'), {{
  type: 'doughnut',
  data: {{
    labels: fcaLabels,
    datasets: [{{ data: fcaData, backgroundColor: ['#fc8181','#f6e05e','#68d391','#e2e8f0'] }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: commonFont, boxWidth: 10 }}, position: 'bottom' }} }}
  }}
}});

// 3. Evidence bar
new Chart(document.getElementById('evBar'), {{
  type: 'bar',
  data: {{
    labels: evLabels,
    datasets: [{{
      label: 'Stories',
      data: evData,
      backgroundColor: ['#68d391','#f6e05e','#fc8181','#e2e8f0'],
      borderRadius: 4
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, ticks: {{ font: commonFont, stepSize: 1 }}, grid: {{ color: '#f0f4f8' }} }},
      x: {{ ticks: {{ font: commonFont }} }}
    }}
  }}
}});

// ── Accordion ──
function toggleAccordion(idx) {{
  const row = document.getElementById('acc-' + idx);
  const storyRow = document.querySelector('.story-row[data-idx="' + idx + '"]');
  const isOpen = row.style.display !== 'none';
  // Close all
  document.querySelectorAll('.accordion-row').forEach(r => r.style.display = 'none');
  document.querySelectorAll('.story-row').forEach(r => r.classList.remove('open'));
  // Toggle clicked
  if (!isOpen) {{
    row.style.display = 'table-row';
    storyRow.classList.add('open');
  }}
}}

// ── Table sort ──
let sortState = {{ col: -1, dir: 1 }};

function sortTable(col) {{
  const tbody = document.getElementById('storyTbody');
  const rows = [...tbody.querySelectorAll('.story-row')];
  const dir = (sortState.col === col && sortState.dir === 1) ? -1 : 1;
  sortState = {{ col, dir }};

  rows.sort((a, b) => {{
    const aVal = a.cells[col].textContent.trim();
    const bVal = b.cells[col].textContent.trim();
    const aNum = parseFloat(aVal.replace(/[^0-9.]/g, ''));
    const bNum = parseFloat(bVal.replace(/[^0-9.]/g, ''));
    if (!isNaN(aNum) && !isNaN(bNum)) return dir * (aNum - bNum);
    return dir * aVal.localeCompare(bVal);
  }});

  rows.forEach(r => {{
    const accId = r.getAttribute('data-idx');
    const acc = document.getElementById('acc-' + accId);
    tbody.appendChild(r);
    if (acc) tbody.appendChild(acc);
  }});

  // Update header indicators
  document.querySelectorAll('.story-table th').forEach((th, i) => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (i === col) th.classList.add(dir === 1 ? 'sort-asc' : 'sort-desc');
  }});
}}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    size_kb = len(html) / 1024
    print(f"  Dashboard written: {output_path}")
    print(f"  Size: {size_kb:.0f} KB  |  Stories: {len(stories)}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-story QE dashboard")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR,
                        help="Root directory containing story folders")
    args = parser.parse_args()
    generate(args.output_root)
