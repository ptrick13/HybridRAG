"""Tools package — thin wrappers around the three data stores.

Agent-facing entry points:

- ``qdrant_client.search_documents``  — hybrid vector search (dense + BM25 + RRF)
- ``neo4j_client.query_neo4j``        — read-only Cypher graph queries
- ``postgres_client.query_postgres``  — read-only SQL relational queries

Ingestion-only helpers (not used by agents):

- ``postgres_client.execute_ddl``     — DDL for schema setup
- ``postgres_client.execute_insert``  — batch inserts for data ingestion
- ``qdrant_client.ensure_collection_exists`` / ``index_document`` — collection setup
"""
