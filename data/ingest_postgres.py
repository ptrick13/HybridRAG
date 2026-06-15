"""Ingest Software Development Analytics data into PostgreSQL.

Creates all tables (idempotent) and loads tickets, sprint_metrics, deployments,
test_coverage, and incidents from JSON sample files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tools.postgres_client import execute_ddl, execute_insert

logger = logging.getLogger(__name__)

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS tickets (
        id                  UUID PRIMARY KEY,
        title               TEXT NOT NULL,
        type                VARCHAR(20),
        priority            VARCHAR(5),
        status              VARCHAR(20),
        component_id        VARCHAR(20),
        assignee_id         VARCHAR(20),
        team_id             VARCHAR(20),
        story_points        INTEGER,
        created_at          TIMESTAMPTZ,
        resolved_at         TIMESTAMPTZ,
        reopened_count      INTEGER DEFAULT 0,
        sprint_id           VARCHAR(20),
        description         TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sprint_metrics (
        id                  TEXT PRIMARY KEY,
        sprint_id           VARCHAR(20),
        team_id             VARCHAR(20),
        start_date          DATE,
        end_date            DATE,
        planned_points      INTEGER,
        completed_points    INTEGER,
        velocity            DOUBLE PRECISION,
        bug_count           INTEGER,
        feature_count       INTEGER,
        carried_over_count  INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS deployments (
        id                  UUID PRIMARY KEY,
        component_id        VARCHAR(20),
        version             VARCHAR(50),
        status              VARCHAR(20),
        environment         VARCHAR(20),
        deployed_by         VARCHAR(20),
        deployed_at         TIMESTAMPTZ,
        duration_seconds    INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS test_coverage (
        id                  UUID PRIMARY KEY,
        component_id        VARCHAR(20),
        measured_at         DATE,
        line_coverage       DOUBLE PRECISION,
        branch_coverage     DOUBLE PRECISION,
        test_count          INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS incidents (
        id                          UUID PRIMARY KEY,
        title                       TEXT,
        severity                    VARCHAR(5),
        component_id                VARCHAR(20),
        root_cause_component_id     VARCHAR(20),
        started_at                  TIMESTAMPTZ,
        resolved_at                 TIMESTAMPTZ,
        duration_minutes            INTEGER,
        affected_users              INTEGER,
        status                      VARCHAR(20)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tickets_component ON tickets(component_id)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_type ON tickets(type)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_sprint_metrics_team ON sprint_metrics(team_id)",
    "CREATE INDEX IF NOT EXISTS idx_deployments_component ON deployments(component_id)",
    "CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status)",
    "CREATE INDEX IF NOT EXISTS idx_test_coverage_component ON test_coverage(component_id)",
    "CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)",
    "CREATE INDEX IF NOT EXISTS idx_incidents_started ON incidents(started_at)",
]


def create_schema() -> None:
    """Create all tables and indexes (idempotent)."""
    for ddl in _DDL_STATEMENTS:
        execute_ddl(ddl.strip())
    logger.info("PostgreSQL schema created.")


def load_tickets(tickets: list[dict[str, Any]]) -> None:
    sql = """
        INSERT INTO tickets
            (id, title, type, priority, status, component_id, assignee_id,
             team_id, story_points, created_at, resolved_at, reopened_count,
             sprint_id, description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    params = [
        (
            t["id"],
            t["title"],
            t["type"],
            t["priority"],
            t["status"],
            t["component_id"],
            t["assignee_id"],
            t["team_id"],
            t.get("story_points"),
            t.get("created_at"),
            t.get("resolved_at"),
            t.get("reopened_count", 0),
            t.get("sprint_id"),
            t.get("description"),
        )
        for t in tickets
    ]
    execute_insert(sql, params)
    logger.info("Loaded %d tickets.", len(tickets))


def load_sprint_metrics(metrics: list[dict[str, Any]]) -> None:
    sql = """
        INSERT INTO sprint_metrics
            (id, sprint_id, team_id, start_date, end_date, planned_points,
             completed_points, velocity, bug_count, feature_count, carried_over_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    params = [
        (
            m["id"],
            m["sprint_id"],
            m["team_id"],
            m["start_date"],
            m["end_date"],
            m["planned_points"],
            m["completed_points"],
            m["velocity"],
            m["bug_count"],
            m["feature_count"],
            m["carried_over_count"],
        )
        for m in metrics
    ]
    execute_insert(sql, params)
    logger.info("Loaded %d sprint metrics.", len(metrics))


def load_deployments(deployments: list[dict[str, Any]]) -> None:
    sql = """
        INSERT INTO deployments
            (id, component_id, version, status, environment, deployed_by,
             deployed_at, duration_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    params = [
        (
            d["id"],
            d["component_id"],
            d["version"],
            d["status"],
            d["environment"],
            d["deployed_by"],
            d["deployed_at"],
            d.get("duration_seconds"),
        )
        for d in deployments
    ]
    execute_insert(sql, params)
    logger.info("Loaded %d deployments.", len(deployments))


def load_test_coverage(records: list[dict[str, Any]]) -> None:
    sql = """
        INSERT INTO test_coverage
            (id, component_id, measured_at, line_coverage, branch_coverage, test_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    params = [
        (
            r["id"],
            r["component_id"],
            r["measured_at"],
            r["line_coverage"],
            r["branch_coverage"],
            r["test_count"],
        )
        for r in records
    ]
    execute_insert(sql, params)
    logger.info("Loaded %d test coverage records.", len(records))


def load_incidents(incidents: list[dict[str, Any]]) -> None:
    sql = """
        INSERT INTO incidents
            (id, title, severity, component_id, root_cause_component_id,
             started_at, resolved_at, duration_minutes, affected_users, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    params = [
        (
            i["id"],
            i["title"],
            i["severity"],
            i["component_id"],
            i.get("root_cause_component_id"),
            i["started_at"],
            i["resolved_at"],
            i["duration_minutes"],
            i["affected_users"],
            i["status"],
        )
        for i in incidents
    ]
    execute_insert(sql, params)
    logger.info("Loaded %d incidents.", len(incidents))


def ingest_from_files(sample_dir: Path) -> None:
    """Run the full PostgreSQL ingestion pipeline from JSON files."""
    create_schema()
    load_tickets(json.loads((sample_dir / "tickets.json").read_text()))
    load_sprint_metrics(json.loads((sample_dir / "sprint_metrics.json").read_text()))
    load_deployments(json.loads((sample_dir / "deployments.json").read_text()))
    load_test_coverage(json.loads((sample_dir / "test_coverage.json").read_text()))
    load_incidents(json.loads((sample_dir / "incidents.json").read_text()))
    logger.info("PostgreSQL ingestion complete.")
