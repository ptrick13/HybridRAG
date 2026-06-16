from unittest.mock import AsyncMock, patch

import pytest

from agents.orchestrator import RoutingDecision, SubTask
from workflows.models import WorkflowResult
from workflows.v1_workflow import run, run_streaming


def make_routing(*agents):
    return RoutingDecision(
        subtasks=[SubTask(agent=a, query=f"query for {a}") for a in agents],
        reasoning="test reasoning",
    )


def make_final_state(routing=None, answer="Test answer", results=None):
    return {
        "answer": answer,
        "routing": routing if routing is not None else make_routing("vector"),
        "retrieval_results": results
        if results is not None
        else [{"source": "vector", "results": []}],
        "agent_latencies": {"orchestrator": 0.1, "vector": 0.2, "answer": 0.3},
    }


@pytest.fixture(autouse=True)
def mock_usage():
    with (
        patch("workflows.v1_workflow.init_tracking"),
        patch("workflows.v1_workflow.compute_cost", return_value=0.001),
        patch("workflows.v1_workflow.get_embedding_token_count", return_value=100),
        patch("workflows.v1_workflow.get_llm_input_token_count", return_value=200),
        patch("workflows.v1_workflow.get_llm_output_token_count", return_value=50),
    ):
        yield


async def test_run_returns_workflow_result():
    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=make_final_state())
        result = await run("test query")
    assert isinstance(result, WorkflowResult)
    assert result.variant == "v1"


async def test_run_answer_matches_state():
    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=make_final_state(answer="The real answer."))
        result = await run("test query")
    assert result.answer == "The real answer."


async def test_run_activated_agents_from_routing():
    routing = make_routing("vector", "sql")
    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=make_final_state(routing=routing))
        result = await run("test query")
    assert result.activated_agents == ["vector", "sql"]


async def test_run_no_routing_gives_empty_agents():
    state = make_final_state(answer="No agents used.")
    state["routing"] = None
    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state)
        result = await run("test query")
    assert result.activated_agents == []


async def test_run_cost_and_tokens_populated():
    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=make_final_state())
        result = await run("test query")
    assert result.cost_usd == 0.001
    assert result.embedding_tokens == 100
    assert result.metadata["prompt_tokens"] == 200
    assert result.metadata["completion_tokens"] == 50


async def test_run_streaming_yields_routing_event():
    routing = make_routing("vector")

    async def fake_astream_events(initial, version):
        yield {"event": "on_chain_start", "name": "orchestrator", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "orchestrator",
            "data": {"output": {"routing": routing, "agent_latencies": {"orchestrator": 0.1}}},
        }
        yield {
            "event": "on_chain_start",
            "name": "single_retrieval",
            "data": {"input": {"_subtask_agent": "vector"}},
        }
        yield {
            "event": "on_chain_end",
            "name": "single_retrieval",
            "data": {"output": {"retrieval_results": [], "agent_latencies": {"vector": 0.2}}},
        }
        yield {"event": "on_chain_start", "name": "answer", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "answer",
            "data": {"output": {"answer": "Test answer", "agent_latencies": {"answer": 0.3}}},
        }

    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    event_types = [e["event"] for e in events]
    assert "routing" in event_types
    assert "result" in event_types


async def test_run_streaming_result_event_has_v1_variant():
    async def fake_astream_events(initial, version):
        yield {
            "event": "on_chain_end",
            "name": "answer",
            "data": {"output": {"answer": "SQL result", "agent_latencies": {"answer": 0.1}}},
        }

    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    result_events = [e for e in events if e["event"] == "result"]
    assert len(result_events) == 1
    assert result_events[0]["data"]["variant"] == "v1"


async def test_run_streaming_agent_start_uses_subtask_agent_label():
    async def fake_astream_events(initial, version):
        yield {
            "event": "on_chain_start",
            "name": "single_retrieval",
            "data": {"input": {"_subtask_agent": "graph"}},
        }
        yield {
            "event": "on_chain_end",
            "name": "answer",
            "data": {"output": {"answer": "done", "agent_latencies": {"answer": 0.1}}},
        }

    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    agent_starts = [e for e in events if e["event"] == "agent_start"]
    assert any(e.get("agent") == "graph" for e in agent_starts)


async def test_run_streaming_accumulates_retrieval_results():
    routing = make_routing("vector")
    result_item = {"source": "vector", "results": [{"id": "1"}]}

    async def fake_astream_events(initial, version):
        yield {
            "event": "on_chain_end",
            "name": "single_retrieval",
            "data": {
                "output": {"retrieval_results": [result_item], "agent_latencies": {"vector": 0.1}}
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "orchestrator",
            "data": {"output": {"routing": routing, "agent_latencies": {"orchestrator": 0.05}}},
        }
        yield {
            "event": "on_chain_end",
            "name": "answer",
            "data": {"output": {"answer": "done", "agent_latencies": {"answer": 0.1}}},
        }

    with patch("workflows.v1_workflow._graph") as mock_graph:
        mock_graph.astream_events = fake_astream_events
        events = [e async for e in run_streaming("test query")]

    result_event = next(e for e in events if e["event"] == "result")
    assert result_event["data"]["retrieval_results"] == [result_item]
