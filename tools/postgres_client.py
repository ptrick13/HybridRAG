"""PostgreSQL structured query tool.

Exposes ``query_postgres``, which executes a read-only SQL query and returns
results as a formatted string with column headers.

All queries are strictly read-only. LIMIT 50 is enforced at the agent
system-prompt level. A regex guard also blocks DML/DDL statements at the
tool boundary, complementing the agent-level prompt constraint.
"""

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool

from config.settings import settings

logger = logging.getLogger(__name__)

_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _get_pool() -> "psycopg2.pool.ThreadedConnectionPool":
    """Initialise and return the shared connection pool (1–5 connections)."""
    return psycopg2.pool.ThreadedConnectionPool(1, 5, settings.postgres_dsn)


@contextmanager
def _connection() -> "Generator[psycopg2.connection, None, None]":
    """Check out a pooled connection; roll back and return it on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.rollback()  # no-op after commit; clears idle transactions from read-only sessions
        pool.putconn(conn)


def query_postgres(sql: str) -> str:
    """Execute a read-only SQL query and return a formatted result string.

    Results are formatted as a table with column headers separated by ``|``
    and rows separated by newlines.  The SQL Agent includes this raw string
    in its retrieval result; the Answer Agent synthesises it into a citation.

    Args:
        sql: A read-only SELECT statement. The agent system prompt enforces
             LIMIT 50. DDL or DML statements are rejected before execution.

    Returns:
        A multi-line string with column headers on the first line and one
        result row per subsequent line, or ``"No results found."`` when the
        query returns an empty set.

    Raises:
        ValueError: If the SQL contains a non-SELECT statement.
        psycopg2.Error: If the SQL is malformed or the connection fails. The
            SQL Agent catches this and retries with a corrected query.
    """
    if _WRITE_PATTERN.search(sql):
        logger.warning("PostgreSQL write query rejected: %s", sql[:200])
        raise ValueError("Write operations are not permitted via query_postgres.")

    with _connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql)
            rows: list[dict[str, Any]] = cursor.fetchall()

    if not rows:
        return "No results found."

    columns = list(rows[0].keys())
    header = " | ".join(columns)
    separator = "-" * len(header)
    data_lines = [" | ".join(str(row[col]) for col in columns) for row in rows]

    result = "\n".join([header, separator, *data_lines])
    logger.debug("PostgreSQL returned %d rows.", len(rows))
    return result


def execute_ddl(sql: str) -> None:
    """Execute a DDL statement (CREATE TABLE, etc.) for schema setup.

    Used only by the data ingestion pipeline, never by agents.

    Args:
        sql: A DDL statement to execute.
    """
    with _connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def execute_insert(sql: str, params: tuple | list[tuple]) -> None:
    """Execute an INSERT or batch INSERT for data ingestion.

    Used only by the data ingestion pipeline.

    Args:
        sql: An INSERT statement (optionally parameterised).
        params: Either a single tuple for ``execute`` or a list of tuples
                for ``executemany``.
    """
    with _connection() as conn:
        with conn.cursor() as cursor:
            if isinstance(params, list):
                cursor.executemany(sql, params)
            else:
                cursor.execute(sql, params)
        conn.commit()
