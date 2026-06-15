"""Ingest Software Development Analytics data into Neo4j.

Builds the knowledge graph from shared_ids.json and graph_relationships.json.
Idempotent — uses MERGE so re-running is safe.

Graph schema:
  Nodes: Team, Developer, Component, Repository, Epic
  Relationships:
    (Developer)-[:MEMBER_OF {since}]->(Team)
    (Developer)-[:OWNS]->(Component)
    (Team)-[:RESPONSIBLE_FOR]->(Component)
    (Component)-[:DEPENDS_ON {type}]->(Component)
    (Component)-[:HOSTED_IN]->(Repository)
    (Epic)-[:AFFECTS]->(Component)
    (Developer)-[:CONTRIBUTED_TO {commits}]->(Component)
"""

import json
import logging
from pathlib import Path
from typing import Any

from tools.neo4j_client import close_driver, get_neo4j_driver

logger = logging.getLogger(__name__)


def _check_rel_summary(summary, rel_name: str, input_count: int) -> None:
    """Log actual relationship creation count and warn on potential MATCH failures."""
    created = summary.counters.relationships_created
    if input_count > 0 and created == 0:
        logger.warning(
            "%s: 0 relationships created from %d input rows — "
            "MATCH may have found no nodes (expected only on re-runs, not first ingestion).",
            rel_name,
            input_count,
        )
    logger.info("Loaded %d %s relationships (%d new).", input_count, rel_name, created)


def _create_constraints(driver) -> None:
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Developer) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Epic) REQUIRE e.id IS UNIQUE",
    ]
    with driver.session() as session:
        for c in constraints:
            session.run(c)
    logger.info("Neo4j constraints created.")


def _load_teams(driver, teams: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MERGE (t:Team {id: row.id})
        SET t.name = row.name, t.department = row.department
    """
    with driver.session() as session:
        session.run(cypher, rows=teams)
    logger.info("Loaded %d Team nodes.", len(teams))


def _load_developers(driver, devs: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MERGE (d:Developer {id: row.id})
        SET d.name = row.name, d.seniority = row.seniority
    """
    with driver.session() as session:
        session.run(cypher, rows=devs)
    logger.info("Loaded %d Developer nodes.", len(devs))


def _load_components(driver, comps: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MERGE (c:Component {id: row.id})
        SET c.name = row.name, c.type = row.type
    """
    with driver.session() as session:
        session.run(cypher, rows=comps)
    logger.info("Loaded %d Component nodes.", len(comps))


def _load_repositories(driver, repos: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MERGE (r:Repository {id: row.id})
        SET r.name = row.name, r.url = row.url, r.primary_language = row.language
    """
    with driver.session() as session:
        session.run(cypher, rows=repos)
    logger.info("Loaded %d Repository nodes.", len(repos))


def _load_epics(driver, epics: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MERGE (e:Epic {id: row.id})
        SET e.title = row.title, e.status = row.status
    """
    with driver.session() as session:
        session.run(cypher, rows=epics)
    logger.info("Loaded %d Epic nodes.", len(epics))


def _load_member_of(driver, rels: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (d:Developer {id: row.developer_id})
        MATCH (t:Team {id: row.team_id})
        MERGE (d)-[r:MEMBER_OF]->(t)
        SET r.since = row.since
    """
    with driver.session() as session:
        result = session.run(cypher, rows=rels)
        summary = result.consume()
    _check_rel_summary(summary, "MEMBER_OF", len(rels))


def _load_owns(driver, comps: list[dict[str, Any]]) -> None:
    """OWNS: component.owner_id → (Developer)-[:OWNS]->(Component)."""
    cypher = """
        UNWIND $rows AS row
        MATCH (d:Developer {id: row.owner_id})
        MATCH (c:Component {id: row.comp_id})
        MERGE (d)-[:OWNS]->(c)
    """
    rows = [{"owner_id": c["owner_id"], "comp_id": c["id"]} for c in comps]
    with driver.session() as session:
        result = session.run(cypher, rows=rows)
        summary = result.consume()
    _check_rel_summary(summary, "OWNS", len(rows))


def _load_responsible_for(driver, comps: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (t:Team {id: row.team_id})
        MATCH (c:Component {id: row.comp_id})
        MERGE (t)-[:RESPONSIBLE_FOR]->(c)
    """
    rows = [{"team_id": c["team_id"], "comp_id": c["id"]} for c in comps]
    with driver.session() as session:
        result = session.run(cypher, rows=rows)
        summary = result.consume()
    _check_rel_summary(summary, "RESPONSIBLE_FOR", len(rows))


def _load_depends_on(driver, rels: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (a:Component {id: row.from_id})
        MATCH (b:Component {id: row.to_id})
        MERGE (a)-[r:DEPENDS_ON]->(b)
        SET r.type = row.type
    """
    with driver.session() as session:
        result = session.run(cypher, rows=rels)
        summary = result.consume()
    _check_rel_summary(summary, "DEPENDS_ON", len(rels))


def _load_hosted_in(driver, comps: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (c:Component {id: row.comp_id})
        MATCH (r:Repository {id: row.repo_id})
        MERGE (c)-[:HOSTED_IN]->(r)
    """
    rows = [{"comp_id": c["id"], "repo_id": c["repo_id"]} for c in comps]
    with driver.session() as session:
        result = session.run(cypher, rows=rows)
        summary = result.consume()
    _check_rel_summary(summary, "HOSTED_IN", len(rows))


def _load_affects(driver, epics: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (e:Epic {id: row.epic_id})
        MATCH (c:Component {id: row.comp_id})
        MERGE (e)-[:AFFECTS]->(c)
    """
    rows = [
        {"epic_id": e["id"], "comp_id": c_id} for e in epics for c_id in e.get("component_ids", [])
    ]
    with driver.session() as session:
        result = session.run(cypher, rows=rows)
        summary = result.consume()
    _check_rel_summary(summary, "AFFECTS", len(rows))


def _load_contributed_to(driver, rels: list[dict[str, Any]]) -> None:
    cypher = """
        UNWIND $rows AS row
        MATCH (d:Developer {id: row.developer_id})
        MATCH (c:Component {id: row.component_id})
        MERGE (d)-[r:CONTRIBUTED_TO]->(c)
        SET r.commits = row.commits
    """
    with driver.session() as session:
        result = session.run(cypher, rows=rels)
        summary = result.consume()
    _check_rel_summary(summary, "CONTRIBUTED_TO", len(rels))


def ingest_from_files(sample_dir: Path) -> None:
    """Run the full Neo4j ingestion pipeline from JSON files."""
    shared = json.loads((sample_dir / "shared_ids.json").read_text())
    rels = json.loads((sample_dir / "graph_relationships.json").read_text())

    driver = get_neo4j_driver()
    _create_constraints(driver)
    _load_teams(driver, shared["teams"])
    _load_developers(driver, shared["developers"])
    _load_components(driver, shared["components"])
    _load_repositories(driver, shared["repositories"])
    _load_epics(driver, shared["epics"])
    _load_member_of(driver, rels["member_of"])
    _load_owns(driver, shared["components"])
    _load_responsible_for(driver, shared["components"])
    _load_depends_on(driver, rels["depends_on"])
    _load_hosted_in(driver, shared["components"])
    _load_affects(driver, shared["epics"])
    _load_contributed_to(driver, rels["contributed_to"])
    close_driver()
    logger.info("Neo4j ingestion complete.")
