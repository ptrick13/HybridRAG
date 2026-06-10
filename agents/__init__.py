"""Agents package.

Each agent is implemented as a module exposing one async entry point:

- ``orchestrator.route_query``  — query routing and decomposition
- ``vector_agent.retrieve``     — semantic search via Qdrant
- ``graph_agent.retrieve``      — structural queries via Neo4j
- ``sql_agent.retrieve``        — structured queries via PostgreSQL
- ``judge_agent.evaluate``      — quality evaluation and closed-loop control (V2 only)
- ``answer_agent.synthesise``   — response synthesis with source citations
"""
