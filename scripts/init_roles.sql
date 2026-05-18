-- Runs once on first postgres container start.
-- Creates the two application roles with least-privilege access.

CREATE ROLE qe_agent_writer WITH LOGIN PASSWORD 'writer_localdev';
CREATE ROLE qe_audit_reader WITH LOGIN PASSWORD 'reader_localdev';

-- Grants applied after Alembic creates the tables (see migrations).
-- See migrations/versions/001_initial_schema.py for GRANT statements.
