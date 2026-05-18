# FSC Agentic QE Framework — Production Readiness Plan

**Framework version:** 0.1.0  
**Assessed:** 2026-05-18  
**Scope:** 53 agents | 12 gates | 4 phases | FCA-regulated deployment  
**Validation status:** 53/53 agents pass against sample story FSC-2417 (avg confidence 74.9%)

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| BLOCKER | Cannot deploy without this |
| INTEGRATION | Needed for real data to flow |
| HARDENING | Required before FCA-regulated go-live |
| OPERATIONS | Required for sustained production running |

---

## Tier 1 — Blockers

These four items must be complete before any service can start.

---

### GAP-001 · Fleet Commander entry point missing

**Status:** BLOCKER  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`docker-compose.yml` starts the fleet commander with `python -m src.fleet_commander.main` but `src/fleet_commander/main.py` does not exist. No stories can be processed.

**What must be built**  
`src/fleet_commander/main.py` — an async service that:
- Connects to PostgreSQL via `AsyncPostgresSaver` (LangGraph checkpointer)
- Builds the stateful graph via `build_fleet_commander(checkpointer)`
- Subscribes to the Redis `story_queue` stream
- For each incoming message: calls `graph.ainvoke(initial_story_state(story_id), config={"configurable": {"thread_id": story_id}})`
- Emits a structured log line on start, on each story accepted, and on failure
- Handles `SIGTERM` gracefully (drain in-flight story, then exit)

**Acceptance criteria**
- [ ] Service starts without error when `DATABASE_URL` and `REDIS_URL` are set
- [ ] Posting a message to `story_queue` triggers the refinement phase for that `story_id`
- [ ] Service resumes in-flight graphs after a restart (LangGraph checkpoint round-trip verified)
- [ ] `SIGTERM` allows current phase to complete before shutdown

---

### GAP-002 · Dockerfile missing

**Status:** BLOCKER  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`docker-compose.yml` references a `Dockerfile` (build context `.`) that does not exist. No container image can be built.

**What must be built**  
`Dockerfile` at project root:
- Multi-stage build: `builder` stage installs all deps; `runtime` stage copies only the package
- Base image: `python:3.12-slim`
- Runs as non-root user `appuser` (UID 1000)
- Installs `psycopg[binary]` C extensions in builder stage
- Does NOT include `dev` extras (`pytest`, `ruff`, `mypy`) in the runtime image
- `HEALTHCHECK` instruction on each service's relevant port
- `ENTRYPOINT` / `CMD` left to docker-compose override per service

**Acceptance criteria**
- [ ] `docker build .` completes without error
- [ ] Final image is non-root and contains no dev dependencies
- [ ] Image size is under 800 MB
- [ ] All five docker-compose services start with `docker-compose up`

---

### GAP-003 · Database migrations never run against a live PostgreSQL

**Status:** BLOCKER  
**Effort:** 2 hours (running) + validation  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
The Alembic migration `migrations/versions/001_initial_schema.py` creates all required tables (`decision_events`, `gate_states`, `agent_runs`, `learning_signals`, `pending_approvals`) and configures row-level security. It has never been executed against a real database. Every DB-dependent component (Sign-off service, Agent 51, Agent 52, QDS MCP server) will fail with `relation does not exist`.

**Steps to run**
```bash
# 1. Start the database
docker-compose up -d postgres

# 2. Wait for health check to pass, then run the admin migration
DATABASE_ADMIN_URL=postgresql+psycopg://qe_admin:localdev@localhost:5432/qe_framework \
  alembic upgrade head

# 3. Verify tables exist
psql postgresql://qe_admin:localdev@localhost:5432/qe_framework \
  -c "\dt" -c "\dp decision_events"
```

**Acceptance criteria**
- [ ] `alembic upgrade head` completes with no errors on a clean database
- [ ] `alembic downgrade base` + `alembic upgrade head` completes without errors (idempotency check)
- [ ] All five tables exist with correct columns verified via `\d+ <table>`
- [ ] Row-level security is active on `decision_events` (verify `UPDATE` and `DELETE` are blocked for `qe_agent_writer` role)
- [ ] `scripts/init_roles.sql` grants verified: `qe_agent_writer` can INSERT but not DELETE on `decision_events`

---

### GAP-004 · No Copado webhook receiver to trigger the pipeline

**Status:** BLOCKER  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
The framework is designed to be triggered by Copado pre-promotion hooks. There is no HTTP endpoint that receives the hook payload, maps it to a `story_id`, validates the HMAC signature, and enqueues the story onto the Redis `story_queue`. Without this, the pipeline can only be started by hand.

**What must be built**  
Add a `POST /webhook/copado/pre-promotion` route (in `src/signoff/app.py` or a new `src/webhook/` module):
- Validates the Copado HMAC signature from the `X-Copado-Signature` header
- Extracts `story_id` from the payload (Copado sends the associated Jira ticket key)
- Publishes `{"story_id": story_id, "triggered_at": ...}` to Redis `story_queue`
- Returns `202 Accepted` immediately (async — do not wait for pipeline completion)
- Rejects unsigned or replayed requests with `403`

**Acceptance criteria**
- [ ] Valid signed request with story FSC-2417 returns `202` and the story appears in `story_queue`
- [ ] Request with invalid signature returns `403`
- [ ] Duplicate requests within 5 minutes are idempotent (check for existing in-flight graph)
- [ ] Integration test covers both paths

---

## Tier 2 — Integration Gaps

These items are needed for real data to flow through the framework rather than mocked/empty responses.

---

### GAP-005 · Salesforce integration not implemented

**Status:** INTEGRATION  
**Effort:** 2–3 days  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`pyproject.toml` declares `simple-salesforce>=1.12.0` as a dependency but `src/integrations/salesforce.py` does not exist. Agents that need Salesforce metadata — sandbox state (Agent 22), suitability object schema, FSC managed package version — have no live data source.

**What must be built**  
`src/integrations/salesforce.py`:
- `async get_org_info(domain: str) -> dict` — org name, API version, edition
- `async get_sandbox_status(sandbox_name: str) -> dict` — refresh date, source org, status
- `async describe_object(object_name: str) -> dict` — field list, relationships
- `async run_anonymous_apex(apex: str) -> dict` — execution result + logs (used by Agent 22 sandbox health check)
- Credentials from settings: `SF_USERNAME`, `SF_PASSWORD`, `SF_SECURITY_TOKEN`, `SF_DOMAIN`
- Graceful degradation: if not configured, return empty defaults (same pattern as Copado)

**Acceptance criteria**
- [ ] `get_org_info()` returns the org name and API version from a sandbox
- [ ] `describe_object("FinancialAccount")` returns the FSC field list including custom fields
- [ ] All functions degrade gracefully when `SF_USERNAME` is not set
- [ ] Agent 22 (Sandbox State) uses the live data rather than empty defaults

---

### GAP-006 · Copado MCP server not built

**Status:** INTEGRATION  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
The architecture calls for all external integrations to be exposed as MCP servers so agents consume them as tools. `src/mcp/` has `jira_mcp` and `qds_mcp` but no `copado_mcp`. Agents 11–14 currently bypass the MCP pattern and call `src/integrations/copado.py` directly, which is inconsistent and harder to mock in production.

**What must be built**  
`src/mcp/copado_mcp/server.py` — FastMCP server exposing:
- `get_branch_for_story(story_id)` → branch metadata
- `get_changed_files(story_id, branch_name)` → changed metadata files
- `get_apex_test_results(story_id)` → test run results including coverage %
- `get_pmd_results(story_id)` → PMD violation list

Add to `docker-compose.yml` as a `copado-mcp` service.

**Acceptance criteria**
- [ ] `copado-mcp` service starts and exposes all four tools via MCP protocol
- [ ] Agents 11–14 can be updated to consume via MCP tool calls
- [ ] Server returns empty defaults when `COPADO_URL` is not configured

---

### GAP-007 · Agents do not write audit records to the QDS

**Status:** INTEGRATION  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
Every agent's `run()` function returns an `AgentResult` but never records that result to the `agent_runs` table in the Quality Data Store. Agent 51 (Health Monitor) queries `agent_runs` via `_collect_metrics()` to detect degraded agents. Agent 52 (Severity Calibration) queries `learning_signals` to calibrate confidence bases. Without writes, both agents will always see empty data.

**What must be built**  
In `src/fleet_commander/worker.py`, after `dispatch_agent()` returns, call the QDS MCP `record_agent_run` tool:
```python
await qds.record_agent_run(
    agent_id=result.agent_id,
    story_id=story_id,
    confidence=result.confidence.final_score,
    model_used=result.model_used,
    elapsed_ms=elapsed_ms,
    verdict=...,   # derived from result.data
    false_positive=False,  # updated later by QE Lead feedback
)
```

Also emit a `decision_event` to the audit ledger for every agent verdict (required for FCA 7-year trail).

**Acceptance criteria**
- [ ] After a story runs end-to-end, `agent_runs` contains one row per agent that executed
- [ ] `decision_events` contains at least one row per gate outcome
- [ ] Agent 51 returns real data (not empty metrics) when queried after a story run
- [ ] Agent 52 returns non-empty `signal_rows` when a minimum of 10 runs exist

---

### GAP-008 · Learning signals not collected

**Status:** INTEGRATION  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
The `learning_signals` table exists in the schema but nothing writes to it. The calibration loop (Agent 52) requires QE Lead feedback — when a QE Lead marks an agent verdict as a false positive, that signal must be recorded. Without it, Agent 52 has no data to calibrate from and will always return `INSUFFICIENT_DATA`.

**What must be built**  
Add a `POST /feedback/agent-verdict` endpoint to the sign-off service:
- Accepts `{story_id, agent_id, was_false_positive: bool, qe_lead_email, notes}`
- Validates the QE Lead is in the authorised list (from settings)
- Writes one row to `learning_signals`
- Updates the corresponding `agent_runs` row's `false_positive` flag

**Acceptance criteria**
- [ ] Posting valid feedback creates a `learning_signals` row
- [ ] Posting invalid QE Lead email returns `403`
- [ ] Agent 52 `run_scheduled()` returns `ADJUSTED` or `NO_CHANGE` (not `INSUFFICIENT_DATA`) after 10+ signals exist

---

## Tier 3 — Hardening

Required before FCA-regulated go-live.

---

### GAP-009 · Prompt caching not implemented

**Status:** HARDENING  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
All agents call `call_with_tool()` via `build_system()` without Anthropic `cache_control` headers. The FSC domain context (PACT principles, FCA rule summaries, org schema descriptions) is identical across all agents and is re-sent on every call. At sprint cadence (10+ stories/sprint × 39+ agent calls/story), uncached prompts will significantly increase both cost and latency.

**What must be built**  
In `src/agents/base.py`, update `build_system()` to mark the static domain context block with `cache_control: {"type": "ephemeral"}`. The dynamic, per-story content (story description, AC clauses) must remain outside the cached block.

**Acceptance criteria**
- [ ] Anthropic API response headers show `cache_read_input_tokens > 0` on the second call with the same system prompt
- [ ] End-to-end cost per story run is measurably lower on the second run vs the first

---

### GAP-010 · No rate limiting or retry on Claude API calls

**Status:** HARDENING  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`call_with_tool()` in `src/agents/base.py` makes a single API call with no retry. Concurrent story runs will produce parallel Claude API calls that can hit `429 RateLimitError`, causing silent agent failures with no recovery.

**What must be built**  
Wrap `client.messages.create()` in `call_with_tool()` with:
- Exponential backoff with jitter: start at 1s, cap at 60s, max 4 retries
- Retry only on `429` and `529` (overloaded) status codes
- Raise immediately on `4xx` authentication or schema errors
- Log each retry at `WARNING` level with `story_id`, `agent_id`, and retry count

**Acceptance criteria**
- [ ] A simulated `429` response causes a retry with correct backoff
- [ ] A `401` response raises immediately without retry
- [ ] After 4 failed retries, the exception propagates to the agent's error handler

---

### GAP-011 · FCA audit ledger immutability not enforced at database level

**Status:** HARDENING  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
The `decision_events` table is intended to be immutable (FCA 7-year retention requirement). The migration sets `row-level security` via role grants, but it must be verified that the `qe_agent_writer` role cannot `UPDATE` or `DELETE` rows, and that no application code path issues those statements.

**What must be verified and enforced**
- `qe_agent_writer` role has `INSERT` only on `decision_events` — no `UPDATE`, no `DELETE`
- `qe_admin` role (migration-only) has no `DELETE` trigger defined that could be exploited
- Add a PostgreSQL rule: `CREATE RULE no_update_decision_events AS ON UPDATE TO decision_events DO INSTEAD NOTHING` as belt-and-suspenders
- Document the verification SQL in `scripts/verify_immutability.sql`

**Acceptance criteria**
- [ ] `UPDATE decision_events SET ...` as `qe_agent_writer` returns permission denied
- [ ] `DELETE FROM decision_events` as `qe_agent_writer` returns permission denied
- [ ] `INSERT INTO decision_events ...` as `qe_agent_writer` succeeds
- [ ] `scripts/verify_immutability.sql` can be run as part of post-deployment smoke test

---

### GAP-012 · No CI/CD pipeline defined

**Status:** HARDENING  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
Tests run manually. There is no automated pipeline that runs on every pull request or merge to `main`. In an FCA-regulated environment, deployment without a verifiable CI gate is a compliance risk.

**What must be built**  
`.github/workflows/ci.yml` (GitHub Actions) or equivalent:
- **On pull request:** `ruff check .` → `mypy src/` → `pytest tests/ --cov=src --cov-fail-under=80`
- **On merge to main:** above + `docker build .` (image build validation)
- **On tag `v*`:** above + push image to registry + run `alembic upgrade head` against staging DB
- All secrets injected via GitHub Secrets / Azure Key Vault — never in workflow YAML

**Acceptance criteria**
- [ ] PR pipeline blocks merge if any test fails
- [ ] Coverage below 80% blocks merge
- [ ] Build passes on a clean runner (no cached `site-packages`)
- [ ] Secrets are not visible in pipeline logs

---

### GAP-013 · No structured logging

**Status:** HARDENING  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`structlog` is declared in `pyproject.toml` but is not used anywhere in `src/`. All agents use `print()` statements (in the validation runner) or no logging at all. Production observability for a 53-agent pipeline requires structured JSON logs with consistent fields per call so they can be ingested by Splunk, Datadog, or Azure Monitor.

**What must be built**  
Configure `structlog` in `src/core/logging.py`:
- JSON renderer in production, coloured console renderer in local dev (driven by `LOG_FORMAT` env var)
- Bind `story_id` and `agent_id` to the context at the start of each agent run
- Log fields per agent call: `story_id`, `agent_id`, `agent_name`, `confidence`, `model_used`, `elapsed_ms`, `verdict`, `escalated`
- Log fields per gate: `gate_id`, `story_id`, `verdict`, `blockers`
- Log fields per API call: `model`, `input_tokens`, `output_tokens`, `latency_ms`, `cached`

**Acceptance criteria**
- [ ] Each agent invocation produces exactly one structured log line with all required fields
- [ ] Logs are valid JSON in production mode
- [ ] Log level is configurable via `LOG_LEVEL` environment variable
- [ ] No `print()` statements remain in `src/`

---

### GAP-014 · Production Dockerfile hardening

**Status:** HARDENING  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
Once the Dockerfile is written (GAP-002), it needs production hardening before a regulated deployment: non-root execution, minimal attack surface, health-check endpoint, and graceful signal handling.

**Checklist**
- [ ] Multi-stage build: `builder` (full deps) → `runtime` (package + runtime deps only)
- [ ] `RUN useradd -r -u 1000 appuser && chown -R appuser /app`
- [ ] `USER appuser` before `CMD`
- [ ] No `COPY . .` in runtime stage — only copy the built wheel and `src/`
- [ ] `HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import src.core.config"` (or HTTP health endpoint)
- [ ] `STOPSIGNAL SIGTERM` with graceful drain in `main.py`
- [ ] Image passes `docker scout cves` with no CRITICAL vulnerabilities

---

## Tier 4 — Operations

Required for sustained production running after initial deployment.

---

### GAP-015 · Secrets management — production strategy undefined

**Status:** OPERATIONS  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`.env` files work for local development but must never be used in production. No secrets management strategy has been defined or implemented for the production deployment of `ANTHROPIC_API_KEY`, `COPADO_ACCESS_TOKEN`, `JIRA_API_TOKEN`, `SIGNOFF_HMAC_SECRET`, and database credentials.

**Decision required:** Choose one:
- **Azure Key Vault** — if deploying on Azure (recommended if firm uses Azure)
- **AWS Secrets Manager** — if deploying on AWS
- **Kubernetes Secrets + external-secrets operator** — if deploying on Kubernetes
- **HashiCorp Vault** — if firm has existing Vault infrastructure

**Acceptance criteria**
- [ ] No plaintext secrets appear in any container environment variable visible via `docker inspect`
- [ ] Secret rotation does not require redeployment
- [ ] Access to secrets is audited and logged
- [ ] `SIGNOFF_HMAC_SECRET` rotation invalidates outstanding approval links (by design — links expire at 48h)

---

### GAP-016 · Agent 52 weekly calibration cron not scheduled

**Status:** OPERATIONS  
**Effort:** 2 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`agent_52_severity_calibration.run_scheduled()` exists and is tested, but nothing invokes it on a schedule. The calibration loop — the mechanism by which the framework learns from false positives and adjusts confidence bases — will never run.

**What must be built**  
Choose one scheduling approach:
- **Azure Function Timer Trigger** — `0 0 8 * * 1` (Monday 08:00 UTC)
- **Kubernetes CronJob** — `0 8 * * 1`
- **Celery Beat** — add `CELERYBEAT_SCHEDULE` entry if Celery is adopted

The scheduler must: call `run_scheduled()`, log the calibration result, write the `AgentResult` to the QDS, and send a summary email to QE Lead.

**Acceptance criteria**
- [ ] Calibration runs automatically every Monday at 08:00
- [ ] QE Lead receives a calibration summary email with the `ADJUSTED`/`NO_CHANGE` verdict and per-agent recommendations
- [ ] Missed runs are logged and alerted (not silently skipped)

---

### GAP-017 · Agent 53 incident webhook endpoint not registered

**Status:** OPERATIONS  
**Effort:** 4 hours  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
`agent_53_incident_response.run_incident()` is built and tested as a webhook entry point, but no HTTP route is registered to receive it. Incidents (PagerDuty, Copado failure webhook, monitoring alert) cannot trigger the Incident Response Agent.

**What must be built**  
Add `POST /webhook/incident` to the sign-off service:
```
{
  "story_id": "FSC-2417",
  "incident_type": "apex_exception",
  "severity_hint": "P1",
  "error_details": "...",
  "rollback_feasible": true
}
```
Route calls `run_incident(...)` and returns the `AgentResult` as JSON. Also emit the result to the QDS audit ledger.

**Acceptance criteria**
- [ ] `POST /webhook/incident` with a valid payload returns an `AgentResult` JSON with triage steps
- [ ] `P1` incidents trigger an immediate email to `TECH_LEAD_EMAIL` and `QE_LEAD_EMAIL`
- [ ] The incident result is written to `decision_events` for audit

---

### GAP-018 · No staging environment configuration

**Status:** OPERATIONS  
**Effort:** 1 day  
**Owner:** _______________  
**Target date:** _______________

**Problem**  
There is no staging environment. All changes go directly from local dev to production. For an FCA-regulated deployment, a staging environment that mirrors production data (anonymised) and receives deployments before production is a standard control.

**What must be built**
- `docker-compose.staging.yml` with staging-specific overrides (different DB name, staging Jira project, staging Copado environment)
- Staging Salesforce sandbox (ISV sandbox or partial-copy sandbox with anonymised client data)
- A deployment runbook documenting: staging deploy → smoke test → production deploy sequence
- The Copado webhook in staging points to the staging Fleet Commander endpoint, not production

**Acceptance criteria**
- [ ] `docker-compose -f docker-compose.yml -f docker-compose.staging.yml up` starts a full staging stack
- [ ] A story can run end-to-end in staging without touching production systems
- [ ] The smoke test script (from GAP-011 verification SQL) runs against staging post-deploy

---

## Deployment Sequence

Once all Tier 1 blockers are resolved, the recommended first deployment sequence is:

```
1. docker-compose up -d postgres redis
2. alembic upgrade head                        # GAP-003
3. docker-compose up -d qds-mcp jira-mcp      # MCP servers
4. docker-compose up -d signoff-service        # human-in-the-loop
5. docker-compose up -d fleet-commander        # main pipeline — GAP-001/002
6. POST /webhook/copado/pre-promotion with a test story  # GAP-004
7. Monitor agent_runs and decision_events tables
8. Review HTML report at validation/outputs/<story_id>_report.html
```

---

## Open Questions Requiring Business Decision

| # | Question | Impact |
|---|---|---|
| Q1 | Which cloud provider — Azure, AWS, or Kubernetes on-prem? | Drives secrets management (GAP-015) and cron scheduling (GAP-016) |
| Q2 | Which FCA audit ledger technology — Azure Immutable Blob, AWS QLDB, or PostgreSQL WAL archival? | Drives GAP-011 enforcement strategy |
| Q3 | Is the Compliance Officer sign-off email-based (current DD-001 design) or Salesforce Experience Cloud page? | Drives sign-off service UI effort |
| Q4 | Which stories are in Wave 1 scope? All 53 agents run on every story — should lower-risk stories skip certain gates? | Drives cost model and gate configuration |
| Q5 | What is the go-live date? | Determines which tiers must be complete vs. deferred |

---

## Summary

| Tier | Gaps | Total effort |
|------|------|-------------|
| 1 — Blockers | GAP-001 to GAP-004 | ~3 days |
| 2 — Integration | GAP-005 to GAP-008 | ~5–6 days |
| 3 — Hardening | GAP-009 to GAP-014 | ~4 days |
| 4 — Operations | GAP-015 to GAP-018 | ~3 days |
| **Total** | **18 gaps** | **~15–16 engineering days** |

**Realistic timeline to production-ready:** 3–4 weeks with one focused engineer, assuming business decisions on Q1–Q5 are made in Week 1.
