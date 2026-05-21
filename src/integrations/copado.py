"""
Async Copado integration — Development phase agents read branch and artifact data.

If COPADO_URL is not configured (local dev / CI without Copado), all methods
return empty defaults so Development agents degrade gracefully.
"""

from __future__ import annotations

import asyncio

import httpx

from src.core.config import settings

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = (1.0, 2.0, 4.0)  # seconds between attempts


def _is_configured() -> bool:
    return bool(settings.copado_url)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.copado_access_token.get_secret_value()}"}


async def _get(url: str, *, params: dict | None = None, timeout: float = 15.0) -> dict:
    """GET with 3-attempt exponential backoff. Raises on final failure."""
    last_exc: Exception = RuntimeError("unreachable")
    for attempt, delay in enumerate(_RETRY_BACKOFF):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, headers=_headers(), timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                await asyncio.sleep(delay)
    raise last_exc


async def get_branch_for_story(story_id: str) -> dict:
    """
    Returns branch metadata associated with the given Jira story ID.

    Return shape:
      {branch_name, commit_sha, created_date, last_commit_date, author_email}
    """
    if not _is_configured():
        return _empty_branch()

    data = await _get(f"{settings.copado_url}/api/v1/branches", params={"story_id": story_id}, timeout=10.0)
    branch = data.get("branches", [{}])[0]
    return {
        "branch_name": branch.get("name", ""),
        "commit_sha": branch.get("latestCommitSha", ""),
        "created_date": branch.get("createdDate", ""),
        "last_commit_date": branch.get("lastCommitDate", ""),
        "author_email": branch.get("authorEmail", ""),
    }


async def get_changed_files(story_id: str, branch_name: str) -> list[dict]:
    """
    Returns the list of changed metadata files for the given branch.

    Each entry:
      {file_path, change_type ("add"|"modify"|"delete"), object_type, object_name}
    """
    if not _is_configured():
        return []

    data = await _get(f"{settings.copado_url}/api/v1/branches/{branch_name}/changes")
    return [
        {
            "file_path": item.get("filePath", ""),
            "change_type": item.get("changeType", "modify"),
            "object_type": item.get("metadataType", ""),
            "object_name": item.get("componentName", ""),
        }
        for item in data.get("changes", [])
    ]


async def get_apex_test_results(story_id: str) -> dict:
    """
    Returns the latest Apex test run results for the story's branch.
    Used by Agent 12 (Apex Coverage Analyser).

    Return shape:
      {test_run_id, tests_run, tests_passed, tests_failed, coverage_pct, run_date}
    """
    if not _is_configured():
        return _empty_test_results()

    data = await _get(f"{settings.copado_url}/api/v1/test-results", params={"story_id": story_id})
    result = data.get("latestRun", {})
    return {
        "test_run_id": result.get("id", ""),
        "tests_run": result.get("testsRun", 0),
        "tests_passed": result.get("testsPassed", 0),
        "tests_failed": result.get("testsFailed", 0),
        "coverage_pct": result.get("codeCoveragePct", 0),
        "run_date": result.get("runDate", ""),
    }


async def get_pmd_results(story_id: str) -> list[dict]:
    """
    Returns PMD static analysis violations for the story's branch.
    Used by Agent 14 (Code Quality Reviewer).

    Each entry:
      {rule_name, priority (1=critical…5=info), description, file_path, line, category}
    """
    if not _is_configured():
        return []

    data = await _get(f"{settings.copado_url}/api/v1/static-analysis", params={"story_id": story_id}, timeout=20.0)
    return [
        {
            "rule_name": item.get("ruleName", ""),
            "priority": item.get("priority", 3),
            "description": item.get("description", ""),
            "file_path": item.get("filePath", ""),
            "line": item.get("line", 0),
            "category": item.get("category", ""),
        }
        for item in data.get("violations", [])
    ]


def _empty_branch() -> dict:
    return {
        "branch_name": "",
        "commit_sha": "",
        "created_date": "",
        "last_commit_date": "",
        "author_email": "",
    }


def _empty_test_results() -> dict:
    return {
        "test_run_id": "",
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "coverage_pct": 0,
        "run_date": "",
    }
