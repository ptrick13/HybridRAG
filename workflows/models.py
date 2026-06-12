"""Shared Pydantic models used across both workflow variants."""

from typing import Any

from pydantic import BaseModel, Field


class WorkflowResult(BaseModel):
    """Top-level result object returned by both V1 and V2 ``run()`` functions."""

    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="Synthesised answer with source citations.")
    variant: str = Field(..., description="Workflow variant: 'v1' or 'v2'.")
    activated_agents: list[str] = Field(
        default_factory=list,
        description="Which retrieval agents were activated in the final iteration.",
    )
    retrieval_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw results from each retrieval agent.",
    )
    iterations: int = Field(
        default=1,
        description="Number of retrieval iterations executed (always 1 for V1).",
    )
    judge_decision: dict[str, Any] | None = Field(
        default=None,
        description="Final Judge Agent decision (V2 only).",
    )
    latency_seconds: float = Field(
        default=0.0,
        description="Total wall-clock time from query receipt to answer generation.",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Estimated total cost in USD — LLM tokens plus embedding tokens.",
    )
    embedding_tokens: int = Field(
        default=0,
        description="Total embedding tokens consumed (dense retrieval calls).",
    )
    agent_latencies: dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent wall-clock time in seconds, summed across all iterations.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for observability (token counts, model, etc.).",
    )
