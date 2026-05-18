"""
pytest configuration — patches SQLAlchemy engine creation at import time.

database.py creates a real async engine at module level.  In tests, psycopg is
not installed and no database is running.  Patching create_async_engine and
async_sessionmaker before any test module is imported lets us import agent_51
and agent_52 (which reference database.py at the top level) without a live DB.

Individual test files patch _collect_metrics / _fetch_signal_summary with
AsyncMock, so the DB is never actually called during a test run.
"""

from unittest.mock import MagicMock, patch

# These patches must start at module level — conftest.py is imported by pytest
# before it collects and imports test files, so the patches are already active
# when `from src.agents.monitoring.agent_51_health import ...` runs.
patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=MagicMock()).start()
patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=MagicMock()).start()
