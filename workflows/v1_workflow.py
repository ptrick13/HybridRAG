"""Variant 1 — Hierarchical workflow without closed-loop (LangGraph implementation).

Graph topology:

    START → orchestrator → [Send(single_retrieval) × N] → answer → END

The Orchestrator decides which retrieval agents to activate. All retrieval agents
run concurrently via LangGraph Send(). Their results accumulate in
``WorkflowState.retrieval_results`` and are passed directly to the Answer Agent.
No quality evaluation occurs between retrieval and synthesis.
"""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.usage import (
    compute_cost,
    get_embedding_token_count,
    get_llm_input_token_count,
    get_llm_output_token_count,
    init_tracking,
)
from config.settings import settings
from workflows.models import WorkflowResult
from workflows.nodes import answer_node, orchestrator_node, single_retrieval_node
from workflows.registry import AGENT_REGISTRY
from workflows.state import DEFAULT_WORKFLOW_STATE, WorkflowState

logger = logging.getLogger(__name__)


def _dispatch(state: WorkflowState):
    """After orchestrator: fan out one Send() per subtask, or skip to answer."""
    routing = state.get("routing")
    if not routing or not routing.subtasks:
        logger.info("[V1] Orchestrator returned no subtasks — out-of-scope query.")
        return "answer"
    sends = [
        Send("single_retrieval", {"_subtask_agent": st.agent, "_subtask_query": st.query})
        for st in routing.subtasks
        if st.agent in AGENT_REGISTRY
    ]
    return sends if sends else "answer"


def _build_graph() -> StateGraph:
    g = StateGraph(WorkflowState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("single_retrieval", single_retrieval_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", _dispatch)
    g.add_edge("single_retrieval", "answer")
    g.add_edge("answer", END)

    return g.compile()


_graph = _build_graph()


async def run(query: str) -> WorkflowResult:
    """Execute Variant 1: hierarchical retrieval without closed-loop evaluation.

    Stages run sequentially via the LangGraph graph:
    Orchestrator → parallel retrieval (Send) → Answer Agent.

    Args:
        query: The user's natural language query.

    Returns:
        A ``WorkflowResult`` containing the answer, activated agents, raw
        retrieval results, timing information, per-agent latencies, and cost.
    """
    init_tracking()
    start_time = time.perf_counter()
    logger.info("[V1] Starting workflow for query: %s", query[:120])

    initial: WorkflowState = {
        **DEFAULT_WORKFLOW_STATE,
        "original_query": query,
        "current_query": query,
        "retrieval_results": [],
        "previous_decisions": [],
    }
    final = await _graph.ainvoke(initial)

    elapsed = time.perf_counter() - start_time
    routing = final.get("routing")
    activated = [st.agent for st in routing.subtasks] if routing else []
    cost = compute_cost(settings.openai_model)

    logger.info("[V1] Completed in %.2fs | agents: %s | cost: $%.4f", elapsed, activated, cost)

    return WorkflowResult(
        query=query,
        answer=final["answer"],
        variant="v1",
        activated_agents=activated,
        retrieval_results=final["retrieval_results"],
        iterations=1,
        judge_decision=None,
        latency_seconds=elapsed,
        cost_usd=cost,
        embedding_tokens=get_embedding_token_count(),
        agent_latencies=final["agent_latencies"],
        metadata={
            "routing_reasoning": routing.reasoning if routing else "",
            "prompt_tokens": get_llm_input_token_count(),
            "completion_tokens": get_llm_output_token_count(),
        },
    )


async def run_streaming(query: str) -> AsyncGenerator[dict[str, Any], None]:
    """V1 workflow that yields SSE-style events for live demos.

    Runs the same LangGraph graph as :func:`run` but via ``astream_events``,
    emitting one structured event per agent lifecycle transition so the web UI
    can update its agent-status panel in real time.  The answer is delivered as
    a single ``result`` event (V1 does not stream tokens).

    SSE event schema mirrors the V2 streaming contract:

    .. code-block:: text

        agent_start   {"event": "agent_start",  "agent": <name>}
        routing       {"event": "routing",       "agents": [...], "reasoning": "..."}
        agent_done    {"event": "agent_done",    "agent": <name>, "latency": <s>}
        result        {"event": "result",        "data": <WorkflowResult.model_dump()>}

    Args:
        query: The user's natural language query.

    Yields:
        Event dicts in chronological order.
    """
    init_tracking()
    start_time = time.perf_counter()
    logger.info("[V1 streaming] Starting for query: %s", query[:120])

    initial: WorkflowState = {
        **DEFAULT_WORKFLOW_STATE,
        "original_query": query,
        "current_query": query,
        "retrieval_results": [],
        "previous_decisions": [],
    }

    _STREAMING_NODES = frozenset({"orchestrator", "single_retrieval", "answer"})

    accumulated_results: list[dict[str, Any]] = []
    all_latencies: dict[str, float] = {}
    full_answer = ""
    routing_obj = None

    async for event in _graph.astream_events(initial, version="v2"):
        kind: str = event["event"]
        name: str = event.get("name", "")
        data: dict = event.get("data", {}) or {}

        if kind == "on_chain_start" and name in _STREAMING_NODES:
            if name == "single_retrieval":
                input_state = data.get("input") or {}
                agent_label = (
                    input_state.get("_subtask_agent", "retrieval")
                    if isinstance(input_state, dict)
                    else "retrieval"
                )
                yield {"event": "agent_start", "agent": agent_label}
            else:
                yield {"event": "agent_start", "agent": name}

        elif kind == "on_chain_end" and name in _STREAMING_NODES:
            output: dict = data.get("output") or {}

            if name == "orchestrator":
                routing = output.get("routing")
                latency = output.get("agent_latencies", {}).get("orchestrator", 0.0)
                all_latencies["orchestrator"] = latency
                routing_obj = routing
                if routing:
                    yield {
                        "event": "routing",
                        "agents": [st.agent for st in routing.subtasks],
                        "reasoning": routing.reasoning,
                    }
                yield {"event": "agent_done", "agent": "orchestrator", "latency": latency}

            elif name == "single_retrieval":
                new_results: list = output.get("retrieval_results", [])
                node_latencies: dict = output.get("agent_latencies", {})
                accumulated_results.extend(new_results)
                for agent_name, latency in node_latencies.items():
                    all_latencies[agent_name] = all_latencies.get(agent_name, 0.0) + latency
                    yield {"event": "agent_done", "agent": agent_name, "latency": latency}

            elif name == "answer":
                latency = output.get("agent_latencies", {}).get("answer", 0.0)
                all_latencies["answer"] = latency
                full_answer = output.get("answer", "")
                yield {"event": "agent_done", "agent": "answer", "latency": latency}

    elapsed = time.perf_counter() - start_time
    activated = [st.agent for st in routing_obj.subtasks] if routing_obj else []
    cost = compute_cost(settings.openai_model)

    logger.info(
        "[V1 streaming] Completed in %.2fs | agents: %s | cost: $%.4f",
        elapsed, activated, cost,
    )

    result = WorkflowResult(
        query=query,
        answer=full_answer or "Unable to generate an answer from the retrieved context.",
        variant="v1",
        activated_agents=activated,
        retrieval_results=accumulated_results,
        iterations=1,
        judge_decision=None,
        latency_seconds=elapsed,
        cost_usd=cost,
        embedding_tokens=get_embedding_token_count(),
        agent_latencies=all_latencies,
        metadata={
            "routing_reasoning": routing_obj.reasoning if routing_obj else "",
            "prompt_tokens": get_llm_input_token_count(),
            "completion_tokens": get_llm_output_token_count(),
        },
    )
    yield {"event": "result", "data": result.model_dump()}
