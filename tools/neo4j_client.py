"""Neo4j graph query tool.

Exposes ``query_neo4j``, which executes a read-only Cypher query and returns
raw results as a list of record dicts.

All queries are strictly read-only. LIMIT 25 is enforced at the agent
system-prompt level to prevent context-window overflow. Property names in Neo4j
are case-sensitive — the graph schema section of the Graph Agent's system
prompt makes this explicit.
"""

import logging
import re
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Path, Relationship

from config.settings import settings

logger = logging.getLogger(__name__)


def _to_python(value: Any) -> Any:
    """Convert Neo4j graph objects to JSON-serializable Python types."""
    if isinstance(value, Node):
        return {"_labels": list(value.labels), **{k: _to_python(v) for k, v in value.items()}}
    if isinstance(value, Relationship):
        return {"_type": value.type, **{k: _to_python(v) for k, v in value.items()}}
    if isinstance(value, Path):
        return {
            "nodes": [_to_python(n) for n in value.nodes],
            "relationships": [_to_python(r) for r in value.relationships],
        }
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s*\{)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Return the cached Neo4j driver. Shared by query and ingestion code."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def query_neo4j(cypher: str) -> list[dict[str, Any]]:
    """Execute a read-only Cypher query and return raw results.

    Results are returned as a list of record dicts without interpretation
    or summarisation — the Answer Agent performs contextual compression
    during synthesis.

    Args:
        cypher: A Cypher query string. Should be read-only (MATCH / RETURN).
                The agent system prompt enforces LIMIT 25.

    Returns:
        A list of dicts, one per result row, mapping field names to values.

    Raises:
        Neo4jError: If the query is malformed or the connection fails. The
            Graph Agent catches this and retries with a corrected query.
    """
    if _WRITE_PATTERN.search(cypher):
        logger.warning("Neo4j write query rejected: %s", cypher[:200])
        raise ValueError("Write operations are not permitted via query_neo4j.")

    driver = get_neo4j_driver()

    def _run_query(tx: ManagedTransaction) -> list[dict[str, Any]]:
        result = tx.run(cypher)
        return [{k: _to_python(v) for k, v in record.items()} for record in result]

    with driver.session() as session:
        records = session.execute_read(_run_query)

    logger.debug("Neo4j returned %d records.", len(records))
    return records


def close_driver() -> None:
    """Close the cached Neo4j driver. Call on application shutdown."""
    try:
        driver = get_neo4j_driver()
        driver.close()
    except Exception:
        pass
    finally:
        get_neo4j_driver.cache_clear()
