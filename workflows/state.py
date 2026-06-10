"""LangGraph state definitions shared by V1 and V2 workflows."""

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from agents.judge_agent import JudgeDecision
from agents.orchestrator import RoutingDecision


def _merge_latencies(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Merge two per-agent latency dicts by summing values for matching keys."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0.0) + v
    return result


class WorkflowState(TypedDict):
    """Shared graph state for V1 and V2 LangGraph workflows.

    Fields with ``Annotated`` reducers accumulate across parallel Send() branches
    and loop iterations. All other fields use last-write-wins semantics.
    """

    # ── Query fields ──────────────────────────────────────────────────────────
    original_query: str
    current_query: str  # may be enriched by judge_node on V2 rejection

    # ── Orchestrator output ───────────────────────────────────────────────────
    routing: Optional[RoutingDecision]

    # ── Retrieval results — accumulate across parallel branches & iterations ──
    retrieval_results: Annotated[list[dict[str, Any]], operator.add]

    # ── Timing — summed across all nodes and iterations ───────────────────────
    agent_latencies: Annotated[dict[str, float], _merge_latencies]

    # ── Final answer ──────────────────────────────────────────────────────────
    answer: str

    # ── V2-only fields (unused by V1 graph) ───────────────────────────────────
    iteration: int                          # completed iteration count (0-indexed increment)
    final_decision: Optional[JudgeDecision]
    previous_decisions: list[JudgeDecision]


class SingleRetrievalState(TypedDict):
    """Minimal state passed to single_retrieval_node via LangGraph Send().

    Return values (``retrieval_results``, ``agent_latencies``) are merged back
    into the parent ``WorkflowState`` using the declared reducers.
    """

    _subtask_agent: str
    _subtask_query: str


# retrieval_results and previous_decisions are mutable objects shared by all
# shallow copies of this dict.  Always override them with fresh lists when
# building the initial state for a workflow run.
DEFAULT_WORKFLOW_STATE: WorkflowState = {
    "original_query": "",
    "current_query": "",
    "routing": None,
    "retrieval_results": [],
    "agent_latencies": {},
    "answer": "",
    "iteration": 0,
    "final_decision": None,
    "previous_decisions": [],
}
