"""
Async Jira integration — used directly by agents and wrapped by the Jira MCP server.
The jira library is synchronous; all calls are run in a thread pool via asyncio.to_thread.
"""

import asyncio
from functools import lru_cache

from jira import JIRA

from src.core.config import settings


@lru_cache(maxsize=1)
def _get_client() -> JIRA:
    return JIRA(
        server=settings.jira_url,
        basic_auth=(settings.jira_username, settings.jira_api_token.get_secret_value()),
    )


async def get_story(story_id: str) -> dict:
    def _fetch():
        client = _get_client()
        issue = client.issue(story_id)
        fields = issue.fields
        return {
            "story_id": story_id,
            "summary": fields.summary,
            "description": fields.description or "",
            "status": fields.status.name,
            "issue_type": fields.issuetype.name,
            "priority": fields.priority.name if fields.priority else None,
            "labels": list(fields.labels),
            "components": [c.name for c in fields.components],
            "assignee": fields.assignee.emailAddress if fields.assignee else None,
            "reporter": fields.reporter.emailAddress if fields.reporter else None,
        }
    return await asyncio.to_thread(_fetch)


async def get_acceptance_criteria(story_id: str) -> list[dict]:
    def _fetch():
        client = _get_client()
        issue = client.issue(story_id)
        fields = issue.fields
        clauses: list[dict] = []

        ac_field = getattr(fields, "customfield_10200", None)
        source = ac_field or fields.description or ""
        if source:
            clauses.extend(_parse_gherkin(source, "jira"))
        return clauses
    return await asyncio.to_thread(_fetch)


async def add_label(story_id: str, label: str) -> None:
    def _update():
        client = _get_client()
        issue = client.issue(story_id)
        labels = list(issue.fields.labels)
        if label not in labels:
            issue.update(fields={"labels": labels + [label]})
    await asyncio.to_thread(_update)


async def add_comment(story_id: str, body: str) -> None:
    def _post():
        _get_client().add_comment(story_id, body)
    await asyncio.to_thread(_post)


def _parse_gherkin(text: str, source: str) -> list[dict]:
    clauses: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("scenario"):
            if current:
                clauses.append(current)
            current = {"source": source, "scenario": s, "given": [], "when": [], "then": []}
        elif current:
            if s.lower().startswith("given"):
                current["given"].append(s)
            elif s.lower().startswith("when"):
                current["when"].append(s)
            elif s.lower().startswith("then"):
                current["then"].append(s)
    if current:
        clauses.append(current)
    return clauses
