"""
Jira MCP Server — story and AC retrieval tools for Refinement agents.

Agents use this to pull story text, acceptance criteria, and story metadata.
Write operations are limited to status updates and adding structured comments
(defects, agent trace summaries). Jira is an input/output, not system-of-record.
"""

from jira import JIRA
from mcp.server.fastmcp import FastMCP

from src.core.config import settings

mcp = FastMCP("jira-mcp", description="Jira — story and AC retrieval for FSC QE agents")

_client: JIRA | None = None


def _get_client() -> JIRA:
    global _client
    if _client is None:
        _client = JIRA(
            server=settings.jira_url,
            basic_auth=(settings.jira_username, settings.jira_api_token.get_secret_value()),
        )
    return _client


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_story(story_id: str) -> dict:
    """
    Retrieve a Jira story's full content — summary, description, status,
    labels, components, and all custom fields.
    story_id example: "FSC-2417"
    """
    issue = _get_client().issue(story_id, expand="renderedFields")
    fields = issue.fields

    return {
        "story_id": story_id,
        "summary": fields.summary,
        "description": fields.description or "",
        "status": fields.status.name,
        "issue_type": fields.issuetype.name,
        "priority": fields.priority.name if fields.priority else None,
        "labels": fields.labels,
        "components": [c.name for c in fields.components],
        "sprint": _extract_sprint(fields),
        "assignee": fields.assignee.emailAddress if fields.assignee else None,
        "reporter": fields.reporter.emailAddress if fields.reporter else None,
        "created": str(fields.created),
        "updated": str(fields.updated),
    }


@mcp.tool()
async def get_acceptance_criteria(story_id: str) -> list[dict]:
    """
    Extract acceptance criteria from a Jira story.
    Looks for AC in: dedicated AC custom field, description (Given/When/Then blocks),
    and sub-task descriptions. Returns structured list of AC clauses.
    """
    issue = _get_client().issue(story_id)
    fields = issue.fields
    ac_clauses: list[dict] = []

    # Try dedicated AC custom field first (firm-specific — adjust field name as needed)
    ac_field = getattr(fields, "customfield_10200", None)
    if ac_field:
        ac_clauses.extend(_parse_gherkin_block(ac_field, source="custom_field"))

    # Fall back to parsing description for Given/When/Then
    if not ac_clauses and fields.description:
        ac_clauses.extend(_parse_gherkin_block(fields.description, source="description"))

    return ac_clauses


@mcp.tool()
async def get_stories_in_sprint(sprint_name: str) -> list[dict]:
    """
    Return all stories in a named sprint within the configured project.
    Used by the Fleet Commander webhook listener to trigger the pipeline
    when stories are moved to Sprint Ready status.
    """
    jql = (
        f'project = {settings.jira_project_key} '
        f'AND sprint = "{sprint_name}" '
        f'AND issuetype in (Story, "User Story") '
        f'ORDER BY created ASC'
    )
    issues = _get_client().search_issues(jql, maxResults=100)
    return [
        {
            "story_id": i.key,
            "summary": i.fields.summary,
            "status": i.fields.status.name,
        }
        for i in issues
    ]


@mcp.tool()
async def get_stories_ready_for_sprint() -> list[dict]:
    """
    Return stories that have just been moved to 'Sprint Ready' status
    and have not yet been processed by the Refinement pipeline.
    The Fleet Commander's Jira webhook listener calls this to find new work.
    """
    jql = (
        f'project = {settings.jira_project_key} '
        f'AND status = "Sprint Ready" '
        f'AND issuetype in (Story, "User Story") '
        f'AND labels not in ("qe-pipeline-started") '
        f'ORDER BY updated ASC'
    )
    issues = _get_client().search_issues(jql, maxResults=50)
    return [{"story_id": i.key, "summary": i.fields.summary} for i in issues]


@mcp.tool()
async def add_agent_trace_comment(story_id: str, agent_name: str, trace_summary: str) -> dict:
    """
    Add a structured agent trace comment to a Jira story.
    Used by agents to surface their What/Why/Confidence summary
    directly in Jira for PO and developer visibility.
    """
    body = f"*[QE Agent — {agent_name}]*\n\n{trace_summary}\n\n_Full trace recorded in QDS._"
    _get_client().add_comment(story_id, body)
    return {"story_id": story_id, "commented": True}


@mcp.tool()
async def create_defect(
    story_id: str,
    summary: str,
    description: str,
    severity: str,
    agent_trace: dict,
) -> dict:
    """
    Create a Jira defect linked to a parent story.
    Called by the Defect Triage Agent (Agent 35) with full explainability trace embedded.
    severity: CRITICAL, HIGH, MEDIUM, LOW
    """
    client = _get_client()
    defect = client.create_issue(
        project=settings.jira_project_key,
        summary=summary,
        description=_format_defect_description(description, agent_trace),
        issuetype={"name": "Bug"},
        priority={"name": _severity_to_priority(severity)},
        labels=["qe-agent-raised"],
    )
    client.create_issue_link("is caused by", defect.key, story_id)
    return {"defect_id": defect.key, "severity": severity}


@mcp.tool()
async def add_qe_pipeline_label(story_id: str, label: str) -> dict:
    """
    Add a label to a story to track pipeline progress.
    Used by Fleet Commander to mark stories as pipeline-started,
    g1-passed, etc. so Jira queries can filter processed stories.
    """
    issue = _get_client().issue(story_id)
    current_labels = issue.fields.labels or []
    if label not in current_labels:
        issue.update(fields={"labels": current_labels + [label]})
    return {"story_id": story_id, "label": label, "applied": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_sprint(fields: object) -> str | None:
    sprint_field = getattr(fields, "customfield_10020", None)
    if sprint_field and isinstance(sprint_field, list) and sprint_field:
        sprint = sprint_field[-1]
        return getattr(sprint, "name", str(sprint))
    return None


def _parse_gherkin_block(text: str, source: str) -> list[dict]:
    """Extract Given/When/Then blocks from free text."""
    clauses = []
    current: dict | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("scenario"):
            if current:
                clauses.append(current)
            current = {"source": source, "scenario": stripped, "given": [], "when": [], "then": []}
        elif stripped.lower().startswith("given") and current:
            current["given"].append(stripped)
        elif stripped.lower().startswith("when") and current:
            current["when"].append(stripped)
        elif stripped.lower().startswith("then") and current:
            current["then"].append(stripped)
    if current:
        clauses.append(current)
    return clauses


def _format_defect_description(description: str, trace: dict) -> str:
    return (
        f"{description}\n\n"
        f"*Agent Trace*\n"
        f"* What: {trace.get('what', 'N/A')}\n"
        f"* Why: {trace.get('why', 'N/A')}\n"
        f"* Confidence: {trace.get('final_score', 'N/A')}\n"
        f"* Model: {trace.get('model_used', 'N/A')}\n"
    )


def _severity_to_priority(severity: str) -> str:
    return {"CRITICAL": "Highest", "HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}.get(
        severity.upper(), "Medium"
    )


if __name__ == "__main__":
    mcp.run()
