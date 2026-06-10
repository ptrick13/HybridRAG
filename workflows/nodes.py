"""Shared LangGraph node functions used by both V1 and V2 workflows.

All agent function calls are unchanged — only the orchestration wrapper changes.
Each node returns a partial state update dict; LangGraph merges it into the
full WorkflowState using the reducers declared in state.py.
"""

import logging
import time
from typing import Any

from agents import answer_agent, judge_agent, orchestrator
from agents.judge_agent import JudgeDecision
from config.settings import settings
from workflows.registry import AGENT_REGISTRY
from workflows.state import SingleRetrievalState, WorkflowState

logger = logging.getLogger(__name__)


def _build_enriched_query(
    original_query: str,
    decision: JudgeDecision,
    iteration: int,
) -> str:
    """Enrich the next-cycle query with the Judge's gap context."""
    gaps_section = ""
    if decision.gaps:
        gaps_section = " [JUDGE GAPS: " + "; ".join(decision.gaps) + "]"
    base = decision.reformulated_query or original_query
    return f"{base}{gaps_section}"


async def orchestrator_node(state: WorkflowState) -> dict[str, Any]:
    """Call the Orchestrator Agent and record routing + latency."""
    t0 = time.perf_counter()
    routing = await orchestrator.route_query(state["current_query"])
    latency = time.perf_counter() - t0
    logger.info(
        "[orchestrator] routed to %s | latency=%.2fs",
        [st.agent for st in routing.subtasks],
        latency,
    )
    return {
        "routing": routing,
        "agent_latencies": {"orchestrator": latency},
    }


async def single_retrieval_node(state: SingleRetrievalState) -> dict[str, Any]:
    """Execute one retrieval agent subtask — invoked in parallel via Send().

    Input comes entirely from the Send() payload; results are merged back into
    WorkflowState via the ``operator.add`` and ``_merge_latencies`` reducers.
    """
    agent_name: str = state["_subtask_agent"]
    query: str = state["_subtask_query"]
    agent_fn = AGENT_REGISTRY.get(agent_name)
    if agent_fn is None:
        raise ValueError(f"Unknown agent {agent_name!r}. Available: {list(AGENT_REGISTRY)}")

    t0 = time.perf_counter()
    result = await agent_fn(query)
    latency = time.perf_counter() - t0

    logger.info("[single_retrieval] agent=%s latency=%.2fs", agent_name, latency)
    return {
        "retrieval_results": [result],
        "agent_latencies": {agent_name: latency},
    }


async def judge_node(state: WorkflowState) -> dict[str, Any]:
    """Evaluate accumulated retrieval results and decide accept / reject (V2).

    On REJECT: enriches ``current_query`` for the next orchestrator call.
    Always increments ``iteration`` and appends the decision to
    ``previous_decisions`` for anti-loop detection in subsequent calls.
    """
    iteration = state.get("iteration", 0)
    previous_decisions = state.get("previous_decisions") or []

    t0 = time.perf_counter()
    decision = await judge_agent.evaluate(
        original_query=state["original_query"],
        retrieval_results=state["retrieval_results"],
        iteration=iteration,
        max_iterations=settings.max_retrieval_iterations,
        previous_decisions=previous_decisions,
    )
    latency = time.perf_counter() - t0

    logger.info(
        "[judge] decision=%s iteration=%d latency=%.2fs gaps=%d",
        decision.decision,
        iteration + 1,
        latency,
        len(decision.gaps),
    )

    updates: dict[str, Any] = {
        "final_decision": decision,
        "previous_decisions": previous_decisions + [decision],
        "iteration": iteration + 1,
        "agent_latencies": {"judge": latency},
    }

    if decision.decision == "REJECT":
        updates["current_query"] = _build_enriched_query(
            state["original_query"], decision, iteration
        )

    return updates


async def answer_node(state: WorkflowState) -> dict[str, Any]:
    """Synthesise a cited answer from all accumulated retrieval results."""
    t0 = time.perf_counter()
    answer = await answer_agent.synthesise(
        original_query=state["original_query"],
        retrieval_results=state["retrieval_results"],
        judge_decision=state.get("final_decision"),
    )
    latency = time.perf_counter() - t0

    logger.info("[answer] synthesised %d chars in %.2fs", len(answer or ""), latency)
    return {
        "answer": answer,
        "agent_latencies": {"answer": latency},
    }
