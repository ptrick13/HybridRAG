"""Variant 2 — Hierarchical workflow with closed-loop quality control (LangGraph implementation).

Two graphs are compiled:

``_graph`` (full, used by ``run()``):

    START → orchestrator → [Send(single_retrieval) × N] → judge
                ↑                                              │ REJECT
                └──────────────────────────────────────────────┘
                                                               │ ACCEPT / MAX_ITERATIONS_REACHED
                                                           answer → END

``_retrieval_graph`` (retrieval-only, used by ``run_streaming()``):

    START → orchestrator → [Send(single_retrieval) × N] → judge
                ↑                                              │ REJECT
                └──────────────────────────────────────────────┘
                                                               │ ACCEPT / MAX_ITERATIONS_REACHED
                                                           END

``run_streaming()`` streams LangGraph events via ``.astream_events()`` for the
retrieval loop, then pumps the answer Token-by-Token directly via the OpenAI
streaming API — preserving the original progressive-render behaviour.
"""

import logging
import time
from typing import Any, AsyncGenerator, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import answer_agent
from agents.judge_agent import JudgeDecision
from agents.usage import (
    compute_cost,
    get_embedding_token_count,
    get_llm_input_token_count,
    get_llm_output_token_count,
    init_tracking,
)
from config.settings import settings
from workflows.registry import AGENT_REGISTRY
from workflows.models import WorkflowResult
from workflows.nodes import answer_node, judge_node, orchestrator_node, single_retrieval_node
from workflows.state import DEFAULT_WORKFLOW_STATE, WorkflowState

logger = logging.getLogger(__name__)

# Nodes observed in run_streaming() (answer_node is NOT in the retrieval graph).
_STREAMING_NODES = frozenset({"orchestrator", "single_retrieval", "judge"})


# ── Shared routing helpers ────────────────────────────────────────────────────

def _sends_for_routing(state: WorkflowState, fallback: str):
    """Return Send() list for each subtask, or ``fallback`` node name if none."""
    routing = state.get("routing")
    if not routing or not routing.subtasks:
        logger.info("[V2] Orchestrator returned no subtasks — out-of-scope query.")
        return fallback
    sends = [
        Send("single_retrieval", {"_subtask_agent": st.agent, "_subtask_query": st.query})
        for st in routing.subtasks
        if st.agent in AGENT_REGISTRY
    ]
    return sends if sends else fallback


# ── Full graph (run) ──────────────────────────────────────────────────────────

def _dispatch(state: WorkflowState):
    return _sends_for_routing(state, fallback="answer")


def _judge_routing(state: WorkflowState) -> str:
    decision: Optional[JudgeDecision] = state.get("final_decision")
    if decision is None or decision.decision in ("ACCEPT", "MAX_ITERATIONS_REACHED"):
        return "answer"
    if state.get("iteration", 0) >= settings.max_retrieval_iterations:
        return "answer"
    return "orchestrator"


def _build_graph() -> StateGraph:
    g = StateGraph(WorkflowState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("single_retrieval", single_retrieval_node)
    g.add_node("judge", judge_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", _dispatch)
    g.add_edge("single_retrieval", "judge")
    g.add_conditional_edges("judge", _judge_routing)
    g.add_edge("answer", END)

    return g.compile()


_graph = _build_graph()


# ── Retrieval-only graph (run_streaming) ──────────────────────────────────────


def _dispatch_retrieval(state: WorkflowState):
    """Like _dispatch but routes the no-subtasks case directly to END."""
    return _sends_for_routing(state, fallback=END)


def _judge_routing_retrieval(state: WorkflowState) -> str:
    decision: Optional[JudgeDecision] = state.get("final_decision")
    if decision is None or decision.decision in ("ACCEPT", "MAX_ITERATIONS_REACHED"):
        return END
    if state.get("iteration", 0) >= settings.max_retrieval_iterations:
        return END
    return "orchestrator"


def _build_retrieval_graph() -> StateGraph:
    """Retrieval loop without answer_node — used by run_streaming()."""
    g = StateGraph(WorkflowState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("single_retrieval", single_retrieval_node)
    g.add_node("judge", judge_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", _dispatch_retrieval)
    g.add_edge("single_retrieval", "judge")
    g.add_conditional_edges("judge", _judge_routing_retrieval)

    return g.compile()


_retrieval_graph = _build_retrieval_graph()


# ── run() ─────────────────────────────────────────────────────────────────────

async def run(query: str) -> WorkflowResult:
    """Execute Variant 2: hierarchical retrieval with closed-loop evaluation.

    Iterates retrieval ↔ Judge until acceptance or MAX_RETRIEVAL_ITERATIONS.
    Results accumulate across all iterations so the Answer Agent receives the
    full evidence base. The Judge's conflict flags and gap documentation are
    forwarded to the Answer Agent for transparent disclosure.

    Args:
        query: The user's natural language query.

    Returns:
        A ``WorkflowResult`` containing the answer, iteration count, Judge
        decision details, timing information, per-agent latencies, and cost.
    """
    init_tracking()
    start_time = time.perf_counter()
    logger.info("[V2] Starting workflow for query: %s", query[:120])

    initial: WorkflowState = {
        **DEFAULT_WORKFLOW_STATE,
        "original_query": query,
        "current_query": query,
        "retrieval_results": [],
        "previous_decisions": [],
    }
    final = await _graph.ainvoke(initial)

    elapsed = time.perf_counter() - start_time
    retrieval_results = final["retrieval_results"]
    activated = list(dict.fromkeys(r["source"] for r in retrieval_results))
    final_decision: Optional[JudgeDecision] = final.get("final_decision")
    iterations = final.get("iteration", 0)
    cost = compute_cost(settings.openai_model)

    logger.info(
        "[V2] Completed in %.2fs | iterations: %d | agents: %s | cost: $%.4f",
        elapsed, iterations, activated, cost,
    )

    return WorkflowResult(
        query=query,
        answer=final["answer"],
        variant="v2",
        activated_agents=activated,
        retrieval_results=retrieval_results,
        iterations=iterations,
        judge_decision=final_decision.model_dump() if final_decision else None,
        latency_seconds=elapsed,
        cost_usd=cost,
        embedding_tokens=get_embedding_token_count(),
        agent_latencies=final["agent_latencies"],
        metadata={
            "judge_decision": final_decision.decision if final_decision else None,
            "criteria_scores": (
                final_decision.criteria_scores.model_dump() if final_decision else None
            ),
            "prompt_tokens": get_llm_input_token_count(),
            "completion_tokens": get_llm_output_token_count(),
        },
    )


# ── run_streaming() ───────────────────────────────────────────────────────────

async def run_streaming(query: str) -> AsyncGenerator[dict[str, Any], None]:
    """V2 workflow that yields SSE-style events for live demos.

    Architecture:
    1. ``_retrieval_graph.astream_events()`` drives the closed retrieval loop and
       emits structured events for each step (iteration_start, routing,
       judge_decision, …).
    2. After the loop the answer is streamed token-by-token via the OpenAI
       streaming API — preserving the same progressive-render behaviour as the
       original implementation.

    SSE event schema:

    .. code-block:: text

        iteration_start  {"event": "iteration_start", "iteration": <n>}
        agent_start      {"event": "agent_start",     "agent": <name>}
        routing          {"event": "routing",          "agents": [...], "reasoning": "..."}
        agent_done       {"event": "agent_done",       "agent": <name>, "latency": <s>}
        judge_decision   {"event": "judge_decision",   "decision": "...", "gaps": [...],
                                                        "conflicts": [...],
                                                        "criteria_scores": {...},
                                                        "latency": <s>}
        answer_token     {"event": "answer_token",     "token": "..."}   (one per token)
        result           {"event": "result",           "data": <WorkflowResult.model_dump()>}

    Args:
        query: The user's natural language query.

    Yields:
        Event dicts in chronological order.
    """
    init_tracking()
    start_time = time.perf_counter()
    logger.info("[V2 streaming] Starting for query: %s", query[:120])

    initial: WorkflowState = {
        **DEFAULT_WORKFLOW_STATE,
        "original_query": query,
        "current_query": query,
        "retrieval_results": [],
        "previous_decisions": [],
    }

    # Accumulated from LangGraph events — mirrors WorkflowState reducers manually.
    accumulated_results: list[dict[str, Any]] = []
    final_decision: Optional[JudgeDecision] = None
    all_latencies: dict[str, float] = {}
    final_iteration = 0
    current_iteration_number = 0

    # ── Phase 1: retrieval loop via LangGraph astream_events ─────────────────
    async for event in _retrieval_graph.astream_events(initial, version="v2"):
        kind: str = event["event"]
        name: str = event.get("name", "")
        data: dict = event.get("data", {}) or {}

        if kind == "on_chain_start" and name in _STREAMING_NODES:
            if name == "orchestrator":
                current_iteration_number += 1
                yield {"event": "iteration_start", "iteration": current_iteration_number}
                yield {"event": "agent_start", "agent": "orchestrator"}
            elif name == "single_retrieval":
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
                all_latencies["orchestrator"] = all_latencies.get("orchestrator", 0.0) + latency
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

            elif name == "judge":
                fd: Optional[JudgeDecision] = output.get("final_decision")
                judge_latency = output.get("agent_latencies", {}).get("judge", 0.0)
                all_latencies["judge"] = all_latencies.get("judge", 0.0) + judge_latency
                final_iteration = output.get("iteration", final_iteration)
                if fd:
                    final_decision = fd
                    yield {
                        "event": "judge_decision",
                        "decision": fd.decision,
                        "gaps": fd.gaps,
                        "conflicts": fd.conflicts,
                        "criteria_scores": fd.criteria_scores.model_dump(),
                        "latency": judge_latency,
                    }
                yield {"event": "agent_done", "agent": "judge", "latency": judge_latency}

    # ── Phase 2: answer — streamed token-by-token via answer_agent ───────────
    yield {"event": "agent_start", "agent": "answer"}
    t0 = time.perf_counter()

    full_answer = ""
    async for token in answer_agent.synthesise_streaming(query, accumulated_results, final_decision):
        full_answer += token
        yield {"event": "answer_token", "token": token}

    answer_latency = time.perf_counter() - t0
    all_latencies["answer"] = answer_latency
    yield {"event": "agent_done", "agent": "answer", "latency": answer_latency}

    # ── Final result event ────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    activated_all = list(dict.fromkeys(r["source"] for r in accumulated_results))
    cost = compute_cost(settings.openai_model)

    logger.info(
        "[V2 streaming] Completed in %.2fs | iterations: %d | agents: %s | cost: $%.4f",
        elapsed, final_iteration, activated_all, cost,
    )

    result = WorkflowResult(
        query=query,
        answer=full_answer or "Unable to generate an answer from the retrieved context.",
        variant="v2",
        activated_agents=activated_all,
        retrieval_results=accumulated_results,
        iterations=final_iteration,
        judge_decision=final_decision.model_dump() if final_decision else None,
        latency_seconds=elapsed,
        cost_usd=cost,
        embedding_tokens=get_embedding_token_count(),
        agent_latencies=all_latencies,
        metadata={
            "judge_decision": final_decision.decision if final_decision else None,
            "criteria_scores": (
                final_decision.criteria_scores.model_dump() if final_decision else None
            ),
            "prompt_tokens": get_llm_input_token_count(),
            "completion_tokens": get_llm_output_token_count(),
        },
    )
    yield {"event": "result", "data": result.model_dump()}
