from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph import END

from agents.judge_agent import CriteriaScores, JudgeDecision
from agents.orchestrator import RoutingDecision, SubTask
from workflows.models import WorkflowResult
from workflows.v2_workflow import (
    _dispatch,
    _dispatch_retrieval,
    _judge_routing_retrieval,
    _sends_for_routing,
    run,
    run_streaming,
)

_MOCK_REGISTRY = {"vector": None, "graph": None, "sql": None}


def make_routing(*agents):
    return RoutingDecision(
        subtasks=[SubTask(agent=a, query=f"query for {a}") for a in agents],
        reasoning="test",
    )


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


def make_judge_decision(decision: str):
    return JudgeDecision(
        decision=decision,
        criteria_scores=CriteriaScores(completeness=4, relevance=4, consistency=4, specificity=4),
        gaps=[],
        conflicts=[],
        reformulated_query=None if decision != "REJECT" else "retry",
        reasoning="test",
    )


@pytest.fixture(autouse=True)
def mock_usage():
    with (
        patch("workflows.v2_workflow.init_tracking"),
        patch("workflows.v2_workflow.compute_cost", return_value=0.001),
        patch("workflows.v2_workflow.get_embedding_token_count", return_value=0),
        patch("workflows.v2_workflow.get_llm_input_token_count", return_value=0),
        patch("workflows.v2_workflow.get_llm_output_token_count", return_value=0),
    ):
        yield


# _dispatch (full graph)


def test_dispatch_empty_subtasks_returns_answer():
    routing = RoutingDecision(subtasks=[], reasoning="out of scope")
    state = make_state(routing=routing)
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch(state)
    assert result == "answer"


def test_dispatch_with_agents_returns_sends():
    from langgraph.types import Send

    state = make_state(routing=make_routing("vector", "graph"))
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


# _sends_for_routing


def test_sends_for_routing_no_routing_returns_fallback():
    state = make_state(routing=None)
    result = _sends_for_routing(state, "answer")
    assert result == "answer"


def test_sends_for_routing_empty_subtasks_returns_fallback():
    routing = RoutingDecision(subtasks=[], reasoning="out of scope")
    state = make_state(routing=routing)
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _sends_for_routing(state, "answer")
    assert result == "answer"


def test_sends_for_routing_returns_sends_for_known_agents():
    from langgraph.types import Send

    state = make_state(routing=make_routing("vector", "sql"))
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _sends_for_routing(state, "answer")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, Send) for s in result)


def test_sends_for_routing_unknown_agents_excluded():
    state = make_state(routing=make_routing("vector", "unknown_agent"))
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _sends_for_routing(state, "answer")
    assert isinstance(result, list)
    assert len(result) == 1


# _dispatch_retrieval


def test_dispatch_retrieval_empty_subtasks_returns_END():
    routing = RoutingDecision(subtasks=[], reasoning="out of scope")
    state = make_state(routing=routing)
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch_retrieval(state)
    assert result == END


def test_dispatch_retrieval_with_agents_returns_sends():
    from langgraph.types import Send

    state = make_state(routing=make_routing("graph"))
    with patch("workflows.v2_workflow.AGENT_REGISTRY", _MOCK_REGISTRY):
        result = _dispatch_retrieval(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


# _judge_routing_retrieval


def test_judge_routing_retrieval_none_decision_returns_END():
    state = make_state(final_decision=None)
    with patch("workflows.v2_workflow.settings") as s:
        s.max_retrieval_iterations = 3
        result = _judge_routing_retrieval(state)
    assert result == END


def test_judge_routing_retrieval_accept_returns_END():
    state = make_state(final_decision=make_judge_decision("ACCEPT"), iteration=1)
    with patch("workflows.v2_workflow.settings") as s:
        s.max_retrieval_iterations = 3
        result = _judge_routing_retrieval(state)
    assert result == END


def test_judge_routing_retrieval_reject_within_limit_returns_orchestrator():
    state = make_state(final_decision=make_judge_decision("REJECT"), iteration=1)
    with patch("workflows.v2_workflow.settings") as s:
        s.max_retrieval_iterations = 3
        result = _judge_routing_retrieval(state)
    assert result == "orchestrator"


def test_judge_routing_retrieval_reject_at_limit_returns_END():
    state = make_state(final_decision=make_judge_decision("REJECT"), iteration=3)
    with patch("workflows.v2_workflow.settings") as s:
        s.max_retrieval_iterations = 3
        result = _judge_routing_retrieval(state)
    assert result == END


def test_judge_routing_retrieval_max_iterations_reached_returns_END():
    state = make_state(final_decision=make_judge_decision("MAX_ITERATIONS_REACHED"), iteration=3)
    with patch("workflows.v2_workflow.settings") as s:
        s.max_retrieval_iterations = 3
        result = _judge_routing_retrieval(state)
    assert result == END


# run()


async def test_v2_run_returns_workflow_result():
    decision = make_judge_decision("ACCEPT")
    final_state = {
        "answer": "V2 answer with citations.",
        "retrieval_results": [{"source": "vector"}, {"source": "sql"}],
        "final_decision": decision,
        "iteration": 1,
        "agent_latencies": {"orchestrator": 0.1, "answer": 0.3},
    }
    with patch("workflows.v2_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        result = await run("test query")
    assert isinstance(result, WorkflowResult)
    assert result.variant == "v2"


async def test_v2_run_answer_matches_state():
    final_state = {
        "answer": "Specific V2 answer.",
        "retrieval_results": [],
        "final_decision": None,
        "iteration": 0,
        "agent_latencies": {},
    }
    with patch("workflows.v2_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        result = await run("test query")
    assert result.answer == "Specific V2 answer."


async def test_v2_run_activated_agents_deduplicated():
    final_state = {
        "answer": "Answer.",
        "retrieval_results": [
            {"source": "vector"},
            {"source": "vector"},
            {"source": "sql"},
        ],
        "final_decision": None,
        "iteration": 2,
        "agent_latencies": {},
    }
    with patch("workflows.v2_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        result = await run("test query")
    assert result.activated_agents == ["vector", "sql"]


async def test_v2_run_judge_decision_serialized():
    decision = make_judge_decision("ACCEPT")
    final_state = {
        "answer": "Answer.",
        "retrieval_results": [],
        "final_decision": decision,
        "iteration": 1,
        "agent_latencies": {},
    }
    with patch("workflows.v2_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        result = await run("test query")
    assert result.judge_decision is not None
    assert result.judge_decision["decision"] == "ACCEPT"


async def test_v2_run_no_judge_decision_is_none():
    final_state = {
        "answer": "Answer.",
        "retrieval_results": [],
        "final_decision": None,
        "iteration": 0,
        "agent_latencies": {},
    }
    with patch("workflows.v2_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        result = await run("test query")
    assert result.judge_decision is None


# run_streaming()


async def test_v2_run_streaming_yields_result_event():
    decision = make_judge_decision("ACCEPT")
    result_item = {"source": "vector"}

    async def fake_astream_events(initial, version):
        yield {"event": "on_chain_start", "name": "orchestrator", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "orchestrator",
            "data": {
                "output": {
                    "routing": make_routing("vector"),
                    "agent_latencies": {"orchestrator": 0.1},
                }
            },
        }
        yield {
            "event": "on_chain_start",
            "name": "single_retrieval",
            "data": {"input": {"_subtask_agent": "vector"}},
        }
        yield {
            "event": "on_chain_end",
            "name": "single_retrieval",
            "data": {
                "output": {
                    "retrieval_results": [result_item],
                    "agent_latencies": {"vector": 0.2},
                }
            },
        }
        yield {"event": "on_chain_start", "name": "judge", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "judge",
            "data": {
                "output": {
                    "final_decision": decision,
                    "agent_latencies": {"judge": 0.1},
                    "iteration": 1,
                }
            },
        }

    async def fake_synthesise_streaming(query, results, fd):
        yield "Streamed"
        yield " answer."

    with (
        patch("workflows.v2_workflow._retrieval_graph") as mock_graph,
        patch(
            "workflows.v2_workflow.answer_agent.synthesise_streaming", new=fake_synthesise_streaming
        ),
    ):
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    event_types = [e["event"] for e in events]
    assert "result" in event_types
    assert "answer_token" in event_types
    assert "judge_decision" in event_types

    result_event = next(e for e in events if e["event"] == "result")
    assert result_event["data"]["variant"] == "v2"


async def test_v2_run_streaming_yields_iteration_start():
    async def fake_astream_events(initial, version):
        yield {"event": "on_chain_start", "name": "orchestrator", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "orchestrator",
            "data": {
                "output": {"routing": make_routing(), "agent_latencies": {"orchestrator": 0.05}}
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "judge",
            "data": {
                "output": {
                    "final_decision": make_judge_decision("ACCEPT"),
                    "agent_latencies": {"judge": 0.1},
                    "iteration": 1,
                }
            },
        }

    async def fake_synthesise_streaming(query, results, fd):
        yield "Done."

    with (
        patch("workflows.v2_workflow._retrieval_graph") as mock_graph,
        patch(
            "workflows.v2_workflow.answer_agent.synthesise_streaming", new=fake_synthesise_streaming
        ),
    ):
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    iteration_events = [e for e in events if e["event"] == "iteration_start"]
    assert len(iteration_events) == 1
    assert iteration_events[0]["iteration"] == 1


async def test_v2_run_streaming_accumulates_retrieval_results():
    item_a = {"source": "vector"}
    item_b = {"source": "sql"}

    async def fake_astream_events(initial, version):
        yield {
            "event": "on_chain_end",
            "name": "single_retrieval",
            "data": {"output": {"retrieval_results": [item_a], "agent_latencies": {"vector": 0.1}}},
        }
        yield {
            "event": "on_chain_end",
            "name": "single_retrieval",
            "data": {"output": {"retrieval_results": [item_b], "agent_latencies": {"sql": 0.1}}},
        }
        yield {
            "event": "on_chain_end",
            "name": "judge",
            "data": {
                "output": {
                    "final_decision": make_judge_decision("ACCEPT"),
                    "agent_latencies": {"judge": 0.05},
                    "iteration": 1,
                }
            },
        }

    async def fake_synthesise_streaming(query, results, fd):
        yield "ok"

    with (
        patch("workflows.v2_workflow._retrieval_graph") as mock_graph,
        patch(
            "workflows.v2_workflow.answer_agent.synthesise_streaming", new=fake_synthesise_streaming
        ),
    ):
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    result_event = next(e for e in events if e["event"] == "result")
    assert result_event["data"]["retrieval_results"] == [item_a, item_b]


async def test_v2_run_streaming_judge_start_emits_agent_start():
    async def fake_astream_events(initial, version):
        yield {"event": "on_chain_start", "name": "judge", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "judge",
            "data": {
                "output": {
                    "final_decision": make_judge_decision("ACCEPT"),
                    "agent_latencies": {"judge": 0.1},
                    "iteration": 1,
                }
            },
        }

    async def fake_synthesise_streaming(query, results, fd):
        yield "done"

    with (
        patch("workflows.v2_workflow._retrieval_graph") as mock_graph,
        patch(
            "workflows.v2_workflow.answer_agent.synthesise_streaming", new=fake_synthesise_streaming
        ),
    ):
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    agent_starts = [e for e in events if e["event"] == "agent_start"]
    assert any(e.get("agent") == "judge" for e in agent_starts)
