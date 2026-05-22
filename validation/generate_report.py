"""
FSC Agentic QE Framework — HTML Report Generator
=================================================

Reads JSON output files from the validation runner and produces a
self-contained, polished HTML report suitable for senior management review.

Usage (standalone):
    python -m validation.generate_report --story FSC-2417

Called automatically by run_validation.py after a successful run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"

# ── Verdict → colour mapping ──────────────────────────────────────────────────
VERDICT_COLOUR = {
    # Green
    "PASS": "green", "GO": "green", "COMPLETE": "green", "HEALTHY": "green",
    "READY": "green", "FEASIBLE": "green", "COMPOSED": "green",
    "RESOLVED_PLAN": "green", "MONITORING": "green", "ADJUSTED": "green",
    "SIGNED_OFF": "green", "NOT_REQUIRED": "green", "CONTAINED": "green",
    # Amber
    "WARN": "amber", "PARTIAL": "amber", "CONDITIONAL": "amber",
    "RISKY": "amber", "PENDING": "amber", "NO_CHANGE": "amber",
    "PASS_WITH_CONCERNS": "amber", "MONITORING_ONLY": "amber",
    "INSUFFICIENT_DATA": "amber", "ESCALATING": "amber",
    # Red
    "FAIL": "red", "NO_GO": "red", "BLOCKED": "red", "NOT_FEASIBLE": "red",
    "FAILED": "red", "MISSING": "red", "CRITICAL": "red", "DOWN": "red",
    # Gray
    "SKIPPED": "gray", "UNKNOWN": "gray", "INCOMPLETE": "gray",
    "NO_ACTION_REQUIRED": "gray", "ALERT": "gray",
    # Agent 45 coalition verdicts (Wave 5 game theory)
    "UNANIMOUS_GO": "green", "DISSENT_NO_GO": "red",
    # Error
    "ERROR": "error",
}

PHASE_ORDER = ["Refinement", "Development", "Testing", "Release", "Monitoring"]
PHASE_ICON = {
    "Refinement": "🔍", "Development": "⚙️", "Testing": "🧪",
    "Release": "🚀", "Monitoring": "📡",
}


def _conf_colour(score) -> str:
    if score == "ERR" or score is None:
        return "error"
    score = int(score)
    if score >= 75:
        return "green"
    if score >= 60:
        return "amber"
    return "red"


def _derive_verdict(result: dict) -> str:
    """Extract the primary verdict string from an agent result."""
    if result.get("status") == "error":
        return "ERROR"
    data = result.get("data", {})

    # Agent 05B (AC Challenger) — derive verdict from adversarial challenge outcome
    if result.get("agent_id") == 54:
        critical = data.get("critical_weakness_count", 0)
        ac_count = data.get("ac_count_challenged", 0)
        if ac_count == 0:
            return "SKIPPED"
        if critical == 0:
            return "PASS"
        if critical <= 2:
            return "WARN"
        return "FAIL"

    for key in [
        "retrospective_verdict", "calibration_verdict", "incident_verdict",
        "monitor_verdict", "rollback_verdict", "prod_verdict",
        "coordinator_verdict", "evidence_verdict", "smoke_verdict",
        "dry_run_verdict", "integrity_verdict", "composer_verdict",
        "readiness_verdict", "flaky_verdict", "uat_coordination_verdict",
        "rca_verdict", "defect_verdict", "coverage_verdict",
        "development_verdict", "invest_verdict", "fca_classification",
        "story_ready_assessment",
    ]:
        if key in data:
            return str(data[key])
    return "OK"


def _key_outputs(result: dict) -> list[tuple[str, str]]:
    """Pick the most informative key/value pairs to surface on the card."""
    if result.get("status") == "error":
        return [("Error", result.get("error", "unknown")[:200])]

    data = result.get("data", {})
    items = []
    priority_keys = [
        "goal", "persona", "fsc_objects", "fca_classification",
        # Agents 03 / 14 / 30 — ensemble + TA (Wave 2/3/4)
        "ta_position", "interaction_mode", "ensemble_agreement",
        "invest_score", "invest_verdict", "dimension_scores",
        "ac_count", "ac_clauses",
        # Agent 05 — mechanism design trust signal (Wave 3)
        "generation_mode_trust",
        # Agent 05B (AC Challenger) game theory keys
        "ac_count_challenged", "survivor_count", "critical_weakness_count", "challenge_summary",
        "risk_level", "risk_rating", "risk_factors",
        # Agent 09 — TA interaction summary (Wave 2)
        "ta_interaction_summary",
        "coverage_verdict", "overall_coverage_pct", "gherkin_scenario_count",
        # Agent 19 — Shapley attribution (Wave 3)
        "shapley_attribution", "ac_source_trust",
        # Agent 21 — mechanism design completeness (Wave 3)
        "data_design_completeness",
        "defect_count", "critical_defects", "defect_verdict",
        # Agent 34 — coalition severity voting (Wave 4)
        "severity_votes", "minimax_escalated", "coalition_dissent",
        # Agent 44 — TA-enhanced Shapley evidence summary (Wave 5)
        "ta_evidence_summary",
        "go_decision", "coordinator_verdict", "no_go_reasons",
        # Agent 45 — minimax loss analysis + coalition verdict (Wave 5)
        "coalition_verdict", "minimax_loss_analysis",
        "rollback_verdict", "rollback_risk",
        "health_status", "monitoring_active",
        "agents_adjusted", "calibration_verdict",
        "incident_severity", "incident_verdict", "escalate_to",
        # Agent 55 — 3 Amigos Facilitator
        "story_ready_assessment", "open_questions", "recommended_decisions", "facilitator_summary",
        "narrative", "story_summary",
    ]
    seen = set()
    for k in priority_keys:
        if k in data and k not in seen:
            v = data[k]
            if isinstance(v, list):
                if len(v) == 0:
                    display = "none"
                elif all(isinstance(x, str) for x in v):
                    display = ", ".join(v[:5])
                    if len(v) > 5:
                        display += f" (+{len(v) - 5} more)"
                else:
                    display = json.dumps(v[:3], default=str)
            elif isinstance(v, dict):
                display = ", ".join(f"{dk}:{dv}" for dk, dv in list(v.items())[:4])
            elif isinstance(v, bool):
                display = "Yes" if v else "No"
            elif isinstance(v, float):
                display = f"{v:.1f}"
            else:
                display = str(v)
            if len(display) > 300:
                display = display[:300] + "…"
            items.append((k.replace("_", " ").title(), display))
            seen.add(k)
        if len(items) >= 6:
            break
    return items or [("What", (result.get("what") or "")[:200])]


def _escape(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ── HTML construction ─────────────────────────────────────────────────────────

def _render_agent_card(result: dict) -> str:
    agent_id = result.get("agent_id", 0)
    agent_name = _escape(result.get("agent_name", f"Agent {agent_id}"))
    classification = result.get("classification", "Unknown")
    phase = result.get("phase", "")
    elapsed = result.get("elapsed_ms")
    status = result.get("status", "error")

    conf_data = result.get("confidence", {})
    conf_score = conf_data.get("final_score", "ERR") if status == "ok" else "ERR"
    conf_tier = conf_data.get("tier", "?")
    conf_pct = conf_score if isinstance(conf_score, int) else 0
    escalated = conf_data.get("escalated", False)

    verdict = _derive_verdict(result)
    verdict_col = VERDICT_COLOUR.get(verdict.upper(), "gray")
    conf_col = _conf_colour(conf_score)
    key_outputs = _key_outputs(result)
    what_text = _escape((result.get("what") or "")[:400])
    why_text = _escape((result.get("why") or "")[:400])

    cls_badge_col = {"True AI": "ai", "Augmented Script": "aug", "Workflow Node": "wf"}.get(classification, "aug")
    model = _escape(result.get("model_used", ""))
    elapsed_str = f"{elapsed}ms" if elapsed else "—"

    # Full data JSON for the expandable section
    full_data = _escape(json.dumps(result.get("data", {}), indent=2, default=str))
    signals_data = _escape(json.dumps(conf_data.get("signals", {}), indent=2, default=str))

    escalated_badge = '<span class="badge badge-warn">Escalated</span>' if escalated else ""

    key_rows = ""
    for label, value in key_outputs:
        key_rows += f"""
                <tr>
                  <td class="kv-label">{_escape(label)}</td>
                  <td class="kv-value">{_escape(str(value))}</td>
                </tr>"""

    return f"""
      <div class="agent-card" id="agent-{agent_id}">
        <div class="agent-header" onclick="toggleCard({agent_id})">
          <div class="agent-id-badge">
            <span class="agent-num">{agent_id:02d}</span>
          </div>
          <div class="agent-title-block">
            <span class="agent-name">{agent_name}</span>
            <span class="badge badge-{cls_badge_col}">{classification}</span>
            {escalated_badge}
          </div>
          <div class="agent-metrics">
            <div class="conf-bar-wrap" title="Confidence: {conf_score}%">
              <div class="conf-bar conf-{conf_col}" style="width:{conf_pct}%"></div>
            </div>
            <span class="conf-label conf-text-{conf_col}">{conf_score}%</span>
            <span class="verdict-badge verdict-{verdict_col}">{verdict}</span>
            <span class="elapsed">{elapsed_str}</span>
            <span class="toggle-icon">▾</span>
          </div>
        </div>
        <div class="agent-body" id="body-{agent_id}" style="display:none">
          <div class="agent-body-inner">
            <div class="agent-meta-row">
              <span class="meta-item">Model: <strong>{model}</strong></span>
              <span class="meta-item">Tier: <strong>{conf_tier}</strong></span>
              <span class="meta-item">Time: <strong>{elapsed_str}</strong></span>
            </div>
            <div class="what-why">
              <div class="ww-block">
                <div class="ww-label">WHAT</div>
                <div class="ww-text">{what_text}</div>
              </div>
              <div class="ww-block">
                <div class="ww-label">WHY</div>
                <div class="ww-text">{why_text}</div>
              </div>
            </div>
            <div class="key-outputs">
              <div class="section-label">Key Outputs</div>
              <table class="kv-table">{key_rows}
              </table>
            </div>
            <div class="expand-sections">
              <button class="expand-btn" onclick="toggleJson('data-{agent_id}')">▾ Full Output Data</button>
              <pre class="json-block" id="data-{agent_id}" style="display:none">{full_data}</pre>
              <button class="expand-btn" onclick="toggleJson('sig-{agent_id}')">▾ Confidence Signals</button>
              <pre class="json-block" id="sig-{agent_id}" style="display:none">{signals_data}</pre>
            </div>
          </div>
        </div>
      </div>"""


def generate_html_report(results_dir: Path, output_path: Path, story_id: str) -> None:
    # Load all result files
    results_by_id: dict[int, dict] = {}
    summary = {}
    for f in sorted(results_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if f.name.startswith("_pipeline_summary"):
            summary = data
        elif "agent_id" in data:
            results_by_id[data["agent_id"]] = data

    if not results_by_id:
        print(f"  No agent results found in {results_dir}")
        return

    generated_at = summary.get("generated_at", datetime.now(timezone.utc).isoformat())
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        gen_str = dt.strftime("%d %b %Y  %H:%M UTC")
    except Exception:
        gen_str = generated_at[:19]

    total = summary.get("total_agents", len(results_by_id))
    passed = summary.get("passed", sum(1 for r in results_by_id.values() if r.get("status") == "ok"))
    failed = summary.get("failed", total - passed)
    avg_conf = summary.get("avg_confidence", 0)
    elapsed_s = (summary.get("total_elapsed_ms", 0) / 1000)
    elapsed_str = f"{elapsed_s:.0f}s" if elapsed_s < 120 else f"{elapsed_s/60:.1f}m"

    # Determine overall verdict
    has_error = failed > 0
    overall_verdict = "PIPELINE ERROR" if has_error else "PIPELINE PASS"
    overall_col = "red" if has_error else "green"

    # FCA classification from Agent 3
    fca = (results_by_id.get(3) or {}).get("data", {}).get("fca_classification", "—")

    # Group results by phase
    phases: dict[str, list[dict]] = {p: [] for p in PHASE_ORDER}
    for r in results_by_id.values():
        ph = r.get("phase", "Monitoring")
        if ph in phases:
            phases[ph].append(r)
        else:
            phases["Monitoring"].append(r)
    for ph in phases:
        phases[ph].sort(key=lambda r: r.get("agent_id", 0))

    # Phase summary stats
    phase_stats_html = ""
    for ph in PHASE_ORDER:
        ph_results = phases[ph]
        if not ph_results:
            continue
        ph_ok = sum(1 for r in ph_results if r.get("status") == "ok")
        ph_err = len(ph_results) - ph_ok
        ph_scores = [r["confidence"]["final_score"] for r in ph_results if "confidence" in r]
        ph_avg = round(sum(ph_scores) / len(ph_scores), 0) if ph_scores else 0
        ph_col = "green" if ph_err == 0 else "red"
        icon = PHASE_ICON.get(ph, "")
        phase_stats_html += f"""
          <div class="phase-stat">
            <div class="phase-stat-icon">{icon}</div>
            <div class="phase-stat-name">{ph}</div>
            <div class="phase-stat-count">{len(ph_results)} agents</div>
            <div class="phase-stat-conf">{ph_avg:.0f}% avg</div>
            <div class="phase-stat-status phase-stat-{ph_col}">{ph_ok} ✓  {ph_err} ✗</div>
          </div>"""

    # Phase sections
    phase_sections_html = ""
    for ph in PHASE_ORDER:
        ph_results = phases[ph]
        if not ph_results:
            continue
        icon = PHASE_ICON.get(ph, "")
        cards_html = "".join(_render_agent_card(r) for r in ph_results)
        ph_ok = sum(1 for r in ph_results if r.get("status") == "ok")
        ph_col = "green" if ph_ok == len(ph_results) else "red"
        phase_sections_html += f"""
    <section class="phase-section">
      <div class="phase-header" onclick="togglePhase('{ph}')">
        <span class="phase-icon">{icon}</span>
        <span class="phase-title">{ph}</span>
        <span class="phase-count">{len(ph_results)} agents</span>
        <span class="phase-status-dot phase-dot-{ph_col}">●</span>
        <span class="phase-toggle" id="ptog-{ph}">▾</span>
      </div>
      <div class="phase-body" id="pbody-{ph}">
        {cards_html}
      </div>
    </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FSC QE Framework — Validation Report: {story_id}</title>
<style>
/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #edf2f7;
  color: #2d3748;
  font-size: 14px;
  line-height: 1.5;
}}
a {{ color: #3182ce; text-decoration: none; }}

/* ── Header ── */
.report-header {{
  background: linear-gradient(135deg, #1a365d 0%, #2c5282 60%, #2b6cb0 100%);
  color: white;
  padding: 28px 40px 24px;
  border-bottom: 4px solid #3182ce;
}}
.header-top {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}}
.header-branding {{ flex: 1; }}
.header-branding h1 {{
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.5px;
  opacity: 0.95;
}}
.header-branding h2 {{
  font-size: 13px;
  font-weight: 400;
  opacity: 0.7;
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
}}
.header-verdict {{
  text-align: right;
}}
.verdict-large {{
  display: inline-block;
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.5px;
}}
.verdict-large.green {{ background: #276749; color: #fff; }}
.verdict-large.red   {{ background: #742a2a; color: #fff; }}
.header-meta {{
  margin-top: 16px;
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  font-size: 13px;
  opacity: 0.85;
}}
.meta-pill {{
  background: rgba(255,255,255,0.12);
  border-radius: 4px;
  padding: 3px 10px;
}}
.meta-pill strong {{ opacity: 1; }}

/* ── Container ── */
.container {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px 24px 48px;
}}

/* ── Executive Summary ── */
.exec-summary {{
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  padding: 24px 28px;
  margin-bottom: 24px;
}}
.exec-summary h2 {{
  font-size: 15px;
  font-weight: 600;
  color: #1a365d;
  margin-bottom: 16px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}}
.stat-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
}}
.stat-card {{
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}}
.stat-card .stat-value {{
  font-size: 30px;
  font-weight: 700;
  color: #1a365d;
  line-height: 1.1;
}}
.stat-card .stat-value.green {{ color: #276749; }}
.stat-card .stat-value.red {{ color: #742a2a; }}
.stat-card .stat-value.amber {{ color: #744210; }}
.stat-card .stat-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: #718096;
  margin-top: 4px;
}}

/* ── Phase Summary Row ── */
.phase-overview {{
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  padding: 20px 28px;
  margin-bottom: 24px;
}}
.phase-overview h2 {{
  font-size: 15px;
  font-weight: 600;
  color: #1a365d;
  margin-bottom: 14px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}}
.phase-stat-row {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.phase-stat {{
  flex: 1;
  min-width: 140px;
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px 16px;
  text-align: center;
}}
.phase-stat-icon {{ font-size: 22px; }}
.phase-stat-name {{ font-weight: 600; color: #2d3748; font-size: 13px; margin-top: 4px; }}
.phase-stat-count {{ color: #718096; font-size: 12px; }}
.phase-stat-conf {{ font-size: 18px; font-weight: 700; color: #1a365d; margin: 4px 0; }}
.phase-stat-status {{ font-size: 12px; font-weight: 600; }}
.phase-stat-green {{ color: #276749; }}
.phase-stat-red   {{ color: #c53030; }}

/* ── Phase Sections ── */
.phase-section {{
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  margin-bottom: 16px;
  overflow: hidden;
}}
.phase-header {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  cursor: pointer;
  user-select: none;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}}
.phase-header:hover {{ background: #edf2f7; }}
.phase-icon {{ font-size: 18px; }}
.phase-title {{ font-size: 16px; font-weight: 700; color: #1a365d; flex: 1; }}
.phase-count {{ color: #718096; font-size: 13px; }}
.phase-status-dot {{ font-size: 14px; }}
.phase-dot-green {{ color: #38a169; }}
.phase-dot-red {{ color: #e53e3e; }}
.phase-toggle {{ color: #718096; transition: transform 0.2s; font-size: 16px; }}
.phase-body {{ padding: 12px 16px 16px; }}

/* ── Agent Cards ── */
.agent-card {{
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-bottom: 8px;
  overflow: hidden;
  transition: box-shadow 0.15s;
}}
.agent-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.agent-header {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  background: #fafafa;
}}
.agent-header:hover {{ background: #f0f4f8; }}
.agent-id-badge {{
  background: #1a365d;
  color: white;
  border-radius: 6px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}
.agent-num {{ font-size: 13px; font-weight: 700; }}
.agent-title-block {{
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}}
.agent-name {{ font-weight: 600; color: #2d3748; font-size: 14px; }}
.agent-metrics {{
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}}
.conf-bar-wrap {{
  width: 80px;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}}
.conf-bar {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.4s;
}}
.conf-green {{ background: linear-gradient(90deg, #68d391, #38a169); }}
.conf-amber {{ background: linear-gradient(90deg, #f6e05e, #d69e2e); }}
.conf-red   {{ background: linear-gradient(90deg, #fc8181, #e53e3e); }}
.conf-error {{ background: #e2e8f0; }}
.conf-label {{ font-size: 13px; font-weight: 700; min-width: 35px; text-align: right; }}
.conf-text-green {{ color: #276749; }}
.conf-text-amber {{ color: #744210; }}
.conf-text-red   {{ color: #742a2a; }}
.conf-text-error {{ color: #718096; }}
.elapsed {{ color: #a0aec0; font-size: 11px; min-width: 40px; text-align: right; }}
.toggle-icon {{ color: #a0aec0; font-size: 14px; }}

/* ── Verdict Badges ── */
.badge, .verdict-badge {{
  display: inline-block;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 7px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}}
.badge-ai  {{ background: #e9d8fd; color: #553c9a; }}
.badge-aug {{ background: #c6f6d5; color: #276749; }}
.badge-wf  {{ background: #e2e8f0; color: #4a5568; }}
.badge-warn{{ background: #fefcbf; color: #744210; }}
.verdict-green {{ background: #c6f6d5; color: #276749; }}
.verdict-amber {{ background: #fefcbf; color: #744210; }}
.verdict-red   {{ background: #fed7d7; color: #742a2a; }}
.verdict-gray  {{ background: #e2e8f0; color: #4a5568; }}
.verdict-error {{ background: #fed7d7; color: #742a2a; }}

/* ── Agent Body ── */
.agent-body {{ border-top: 1px solid #e2e8f0; }}
.agent-body-inner {{ padding: 16px 20px; }}
.agent-meta-row {{
  display: flex;
  gap: 20px;
  margin-bottom: 14px;
  font-size: 12px;
  color: #718096;
}}
.meta-item strong {{ color: #4a5568; }}
.what-why {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}}
.ww-block {{
  background: #f7fafc;
  border-left: 3px solid #3182ce;
  padding: 10px 12px;
  border-radius: 0 6px 6px 0;
}}
.ww-label {{
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #3182ce;
  margin-bottom: 5px;
}}
.ww-text {{ font-size: 13px; color: #4a5568; line-height: 1.5; }}
.key-outputs {{ margin-bottom: 12px; }}
.section-label {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: #718096;
  margin-bottom: 8px;
}}
.kv-table {{ width: 100%; border-collapse: collapse; }}
.kv-table tr {{ border-bottom: 1px solid #f0f4f8; }}
.kv-table tr:last-child {{ border-bottom: none; }}
.kv-label {{
  padding: 5px 10px 5px 0;
  font-size: 12px;
  color: #718096;
  font-weight: 500;
  width: 180px;
  vertical-align: top;
  white-space: nowrap;
}}
.kv-value {{
  padding: 5px 0;
  font-size: 12px;
  color: #2d3748;
  word-break: break-word;
}}
.expand-sections {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.expand-btn {{
  background: #edf2f7;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  padding: 5px 12px;
  font-size: 12px;
  color: #4a5568;
  cursor: pointer;
}}
.expand-btn:hover {{ background: #e2e8f0; }}
.json-block {{
  margin-top: 8px;
  background: #1a202c;
  color: #a0aec0;
  border-radius: 6px;
  padding: 14px;
  font-size: 11px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  width: 100%;
}}

/* ── Search ── */
.search-bar {{
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  padding: 14px 24px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.search-bar input {{
  flex: 1;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px 14px;
  font-size: 13px;
  outline: none;
}}
.search-bar input:focus {{ border-color: #3182ce; box-shadow: 0 0 0 2px rgba(49,130,206,0.2); }}
.search-label {{ font-size: 13px; color: #718096; white-space: nowrap; }}

/* ── Footer ── */
.report-footer {{
  text-align: center;
  padding: 28px;
  color: #a0aec0;
  font-size: 12px;
}}

@media (max-width: 768px) {{
  .what-why {{ grid-template-columns: 1fr; }}
  .header-top {{ flex-direction: column; gap: 12px; }}
  .agent-metrics {{ flex-wrap: wrap; gap: 6px; }}
  .conf-bar-wrap {{ display: none; }}
}}
</style>
</head>
<body>

<header class="report-header">
  <div class="header-top">
    <div class="header-branding">
      <h1>FSC Agentic QE Framework</h1>
      <h2>Pipeline Validation Report</h2>
    </div>
    <div class="header-verdict">
      <div class="verdict-large {overall_col}">{overall_verdict}</div>
    </div>
  </div>
  <div class="header-meta">
    <span class="meta-pill">Story: <strong>{_escape(story_id)}</strong></span>
    <span class="meta-pill">Generated: <strong>{gen_str}</strong></span>
    <span class="meta-pill">FCA Classification: <strong>{_escape(str(fca))}</strong></span>
    <span class="meta-pill">Duration: <strong>{elapsed_str}</strong></span>
    <span class="meta-pill">Framework: <strong>PACT Edition v1 + Game Theory</strong></span>
  </div>
</header>

<div class="container">

  <!-- Executive Summary -->
  <section class="exec-summary">
    <h2>Executive Summary</h2>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">{total}</div>
        <div class="stat-label">Agents Run</div>
      </div>
      <div class="stat-card">
        <div class="stat-value green">{passed}</div>
        <div class="stat-label">Succeeded</div>
      </div>
      <div class="stat-card">
        <div class="stat-value {'red' if failed > 0 else 'green'}">{failed}</div>
        <div class="stat-label">Errors</div>
      </div>
      <div class="stat-card">
        <div class="stat-value {'green' if avg_conf >= 75 else 'amber' if avg_conf >= 60 else 'red'}">{avg_conf:.0f}%</div>
        <div class="stat-label">Avg Confidence</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{elapsed_str}</div>
        <div class="stat-label">Total Duration</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{_escape(str(fca))}</div>
        <div class="stat-label">FCA Risk Tier</div>
      </div>
    </div>
  </section>

  <!-- Phase Overview -->
  <section class="phase-overview">
    <h2>Phase Overview</h2>
    <div class="phase-stat-row">
      {phase_stats_html}
    </div>
  </section>

  <!-- Search -->
  <div class="search-bar">
    <span class="search-label">Search agents:</span>
    <input type="text" id="search-input" placeholder="Agent name, verdict, classification…" oninput="filterAgents(this.value)">
  </div>

  <!-- Phase Sections -->
  {phase_sections_html}

</div>

<footer class="report-footer">
  <p>FSC Agentic QE Framework — PACT Edition · Generated {gen_str} · Story {_escape(story_id)}</p>
  <p style="margin-top:4px">All agent outputs are AI-generated and subject to QE Lead review before acting on recommendations.</p>
</footer>

<script>
function toggleCard(id) {{
  const body = document.getElementById('body-' + id);
  const icon = body.closest('.agent-card').querySelector('.toggle-icon');
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    icon.textContent = '▴';
    icon.style.color = '#3182ce';
  }} else {{
    body.style.display = 'none';
    icon.textContent = '▾';
    icon.style.color = '#a0aec0';
  }}
}}

function togglePhase(ph) {{
  const body = document.getElementById('pbody-' + ph);
  const tog = document.getElementById('ptog-' + ph);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    tog.textContent = '▾';
  }} else {{
    body.style.display = 'none';
    tog.textContent = '▸';
  }}
}}

function toggleJson(id) {{
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

function filterAgents(query) {{
  const q = query.toLowerCase().trim();
  document.querySelectorAll('.agent-card').forEach(card => {{
    const text = card.textContent.toLowerCase();
    card.style.display = (!q || text.includes(q)) ? '' : 'none';
  }});
  // Expand phases that have visible cards
  document.querySelectorAll('.phase-body').forEach(pb => {{
    const visible = [...pb.querySelectorAll('.agent-card')].some(c => c.style.display !== 'none');
    if (q && visible) pb.style.display = 'block';
  }});
}}

// Expand first phase (Refinement) by default
window.addEventListener('DOMContentLoaded', () => {{
  const firstBody = document.querySelector('.phase-body');
  if (firstBody) firstBody.style.display = 'block';
  const firstTog = document.querySelector('.phase-toggle');
  if (firstTog) firstTog.textContent = '▾';
}});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report written: {output_path}")
    print(f"  Size: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", default="FSC-2417")
    args = parser.parse_args()
    results_dir = OUTPUT_DIR / args.story
    output_path = OUTPUT_DIR / f"{args.story}_report.html"
    generate_html_report(results_dir, output_path, args.story)
