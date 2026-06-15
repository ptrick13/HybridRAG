"""A2A server — Google Agent2Agent protocol endpoint.

Framework: FastAPI
Port: 8002 (configurable via A2A_PORT in .env)

Implements the Agent2Agent (A2A) protocol:
    GET  /.well-known/agent.json  →  Agent Card (available capabilities)
    POST /tasks/send              →  Synchronous query, returns A2A Artifact

Only the workflow entry point is exposed. Internal agent communication is
not visible to external A2A clients.

Usage:
    uvicorn integrations.a2a_server:app --port 8002
"""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings
from workflows import v1_workflow, v2_workflow

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HybridRAG A2A Server",
    description="Google Agent2Agent protocol endpoint for the hybrid multi-agent RAG system.",
    version="1.0.0",
)


# ── A2A Protocol Models ───────────────────────────────────────────────────────


class A2ATaskInput(BaseModel):
    """Input payload for an A2A task."""

    query: str = Field(
        ..., description="Natural language query about Software Development Analytics data."
    )
    variant: str = Field(default="v1", description="Workflow variant: 'v1' or 'v2'.")


class A2ATask(BaseModel):
    """Incoming A2A task request body following the Agent2Agent protocol."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Client-assigned task ID."
    )
    input: A2ATaskInput


class A2AArtifact(BaseModel):
    """A2A Artifact returned in the task response."""

    type: str = "text"
    content: str


class A2ATaskResult(BaseModel):
    """A2A task result following the Agent2Agent protocol."""

    id: str
    status: str
    artifacts: list[A2AArtifact]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Agent Card ────────────────────────────────────────────────────────────────

_AGENT_CARD = {
    "name": "HybridRAG",
    "description": (
        "A hybrid multi-agent RAG system for Software Development Analytics data. "
        "Routes queries across semantic search (Qdrant), graph queries (Neo4j), "
        "and structured SQL queries (PostgreSQL). "
        "Supports two orchestration variants: V1 (single-pass) and V2 (closed-loop with Judge Agent)."
    ),
    "url": settings.a2a_base_url,
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "skills": [
        {
            "id": "query_hybrid_rag",
            "name": "Query Hybrid RAG",
            "description": (
                "Answer questions about Software Development Analytics data. Queries are automatically "
                "routed to the appropriate retrieval agents. Every answer includes "
                "source citations."
            ),
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Which teams own components that depend on the auth-gateway?",
                "Find postmortems caused by external dependency failures.",
                "Which components have the most P1 incidents in Q1 2024?",
                "What architectural decisions mention circuit breaker patterns?",
            ],
        }
    ],
    "parameters": {
        "variant": {
            "type": "string",
            "enum": ["v1", "v2"],
            "default": "v1",
            "description": "Orchestration variant. v1: single-pass. v2: closed-loop with quality control.",
        }
    },
}


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/.well-known/agent.json", summary="Agent Card")
async def agent_card() -> dict[str, Any]:
    """Return the Agent Card describing this agent's capabilities.

    MCP-compatible and A2A-compatible clients use this endpoint for discovery.
    """
    return _AGENT_CARD


@app.post("/tasks/send", response_model=A2ATaskResult, summary="Send a task")
async def send_task(task: A2ATask) -> A2ATaskResult:
    """Execute a hybrid RAG query and return the result as an A2A Artifact.

    The query is routed to the appropriate retrieval agents and the synthesised
    answer is returned synchronously.

    Args:
        task: A2A task with a ``query`` and optional ``variant`` parameter.

    Returns:
        An ``A2ATaskResult`` with a single text artifact containing the answer.
    """
    if task.input.variant not in ("v1", "v2"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{task.input.variant}'. Choose 'v1' or 'v2'.",
        )

    try:
        if task.input.variant == "v1":
            result = await v1_workflow.run(task.input.query)
        else:
            result = await v2_workflow.run(task.input.query)

        logger.info(
            "A2A task %s completed | variant=%s latency=%.2fs",
            task.id,
            task.input.variant,
            result.latency_seconds,
        )

        return A2ATaskResult(
            id=task.id,
            status="completed",
            artifacts=[A2AArtifact(content=result.answer)],
            metadata={
                "variant": result.variant,
                "activated_agents": result.activated_agents,
                "iterations": result.iterations,
                "latency_seconds": round(result.latency_seconds, 3),
            },
        )

    except Exception as exc:
        logger.exception("A2A task %s failed: %s", task.id, exc)
        return A2ATaskResult(
            id=task.id,
            status="failed",
            artifacts=[A2AArtifact(content=f"An error occurred: {exc}")],
        )


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
