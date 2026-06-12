import pytest
from unittest.mock import patch

from langgraph.types import Send

from agents.judge_agent import CriteriaScores, JudgeDecision
from agents.orchestrator import RoutingDecision, SubTask
from workflows.v1_workflow import _dispatch
from workflows.v2_workflow import _judge_routing

_MOCK_REGISTRY = {"vector": None, "graph": None, "sql": None}


def make_state(routing=None, final_decision=None, iteration=0):
    return {
        "original_query": "test",
        "current_query": "test",
        "routing": routing,
        "retrieval_results": [],
        "agent_latencies": {},
        "answer": "",
        "iteration": iteration,
        "final_decision": final_decision,
        "previous_decisions": [],
    }


def make_routing(*agent_names):
    subtasks = [SubTask(agent=name, query=f"query for {name}") for name in agent_names]
    return RoutingDecision(subtasks=subtasks, reasoning="test routing")


def make_judge_decision(decision: str):
    return JudgeDecision(
        decision=decision,
        criteria_scores=CriteriaScores(completeness=4, relevance=4, consistency=4, specificity=4),
        gaps=[],
        conflicts=[],
        reformulated_query=None if decision != "REJECT" else "retry query",
        reasoning="test decision",
    )


# V1 _dispatch

def test_dispatch_two_subtasks_returns_two_sends():
    state = make_state(routing=make_routing("vector", "sql"))
    with patch("workflows.v1_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, Send) for s in result)


def test_dispatch_send_target_is_single_retrieval():
    state = make_state(routing=make_routing("vector", "graph"))
    with patch("workflows.v1_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        sends = _dispatch(state)
    assert all(s.node == "single_retrieval" for s in sends)


def test_dispatch_empty_subtasks_returns_answer():
    routing = RoutingDecision(subtasks=[], reasoning="out of scope")
    state = make_state(routing=routing)
    with patch("workflows.v1_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch(state)
    assert result == "answer"


# V2 _judge_routing

def test_judge_routing_accept():
    state = make_state(final_decision=make_judge_decision("ACCEPT"), iteration=1)
    with patch("workflows.v2_workflow.settings") as mock_settings:
        mock_settings.max_retrieval_iterations = 3
        result = _judge_routing(state)
    assert result == "answer"


def test_judge_routing_reject_within_limit():
    state = make_state(final_decision=make_judge_decision("REJECT"), iteration=1)
    with patch("workflows.v2_workflow.settings") as mock_settings:
        mock_settings.max_retrieval_iterations = 3
        result = _judge_routing(state)
    assert result == "orchestrator"


def test_judge_routing_max_iterations_reached():
    state = make_state(final_decision=make_judge_decision("MAX_ITERATIONS_REACHED"), iteration=3)
    with patch("workflows.v2_workflow.settings") as mock_settings:
        mock_settings.max_retrieval_iterations = 3
        result = _judge_routing(state)
    assert result == "answer"


def test_judge_routing_reject_at_iteration_limit():
    state = make_state(final_decision=make_judge_decision("REJECT"), iteration=3)
    with patch("workflows.v2_workflow.settings") as mock_settings:
        mock_settings.max_retrieval_iterations = 3
        result = _judge_routing(state)
    assert result == "answer"
