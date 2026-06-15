"""MCP server — exposes the hybrid RAG system as an MCP-compatible tool.

Framework: FastMCP
Port: 8001 (configurable via MCP_PORT in .env)

Exposes a single tool:

    query_hybrid_rag(query, variant="v1") -> str

MCP-compatible clients discover this tool automatically via MCP Discovery.
Only the workflow entry point is exposed — internal agent communication is
not visible to external systems.

Usage:
    python -m integrations.mcp_server
    # or
    uvicorn integrations.mcp_server:app --port 8001
"""

import logging

from fastmcp import FastMCP

from config.settings import settings
from workflows import v1_workflow, v2_workflow

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="HybridRAG",
    instructions=(
        "A hybrid multi-agent RAG system for Software Development Analytics data. "
        "Queries are routed across three specialised retrieval agents: "
        "semantic search (Qdrant), graph queries (Neo4j), and structured "
        "SQL queries (PostgreSQL). "
        "Variant 'v1' uses a single-pass retrieval pipeline. "
        "Variant 'v2' adds a Judge Agent for closed-loop quality control "
        "with up to 3 retrieval iterations and automatic query rewriting."
    ),
)


@mcp.tool()
async def query_hybrid_rag(query: str, variant: str = "v1") -> str:
    """Answer a question about Software Development Analytics using the hybrid RAG system.

    The query is automatically routed to one or more specialised retrieval
    agents (Vector / Graph / SQL) based on its content. The Answer Agent
    synthesises a cited response from the consolidated results.

    Args:
        query: A natural language question about engineering teams, components,
               incidents, or architecture decisions. Examples:
               - "Which teams own components that depend on the auth-gateway?"
               - "Find postmortems caused by external dependency failures."
               - "Which components have the most P1 incidents in Q1 2024?"
               - "What architectural decisions mention circuit breaker patterns?"
        variant: Orchestration variant.
                 - ``"v1"`` (default): single-pass, lower latency (~10s).
                 - ``"v2"``: closed-loop with Judge Agent, higher quality (~15s).

    Returns:
        A cited answer string. Each factual claim references its data source.
    """
    if variant not in ("v1", "v2"):
        return f"Unknown variant '{variant}'. Choose 'v1' or 'v2'."

    try:
        if variant == "v1":
            result = await v1_workflow.run(query)
        else:
            result = await v2_workflow.run(query)

        logger.info(
            "MCP query completed | variant=%s latency=%.2fs",
            variant,
            result.latency_seconds,
        )
        return result.answer

    except Exception as exc:
        logger.exception("MCP query failed: %s", exc)
        return f"An error occurred while processing your query: {exc}"


if __name__ == "__main__":
    mcp.run(transport="sse", port=settings.mcp_port)
