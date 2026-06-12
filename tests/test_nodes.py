from unittest.mock import AsyncMock, patch

import pytest

from workflows.nodes import judge_node, orchestrator_node, single_retrieval_node


@pytest.fixture
def base_state():
    return {
        "original_query": "test query",
        "current_query": "test current query",
        "routing": None,
        "retrieval_results": [],
        "agent_latencies": {},
        "answer": "",
        "iteration": 0,
        "final_decision": None,
        "previous_decisions": [],
    }


async def test_orchestrator_node_returns_routing(base_state, sample_routing):
    with patch("agents.orchestrator.route_query", new=AsyncMock(return_value=sample_routing)):
        result = await orchestrator_node(base_state)
    assert result["routing"] is sample_routing


async def test_orchestrator_node_returns_latency(base_state, sample_routing):
    with patch("agents.orchestrator.route_query", new=AsyncMock(return_value=sample_routing)):
        result = await orchestrator_node(base_state)
    assert "orchestrator" in result["agent_latencies"]
    assert result["agent_latencies"]["orchestrator"] >= 0.0


async def test_orchestrator_node_uses_current_query(base_state, sample_routing):
    mock_route = AsyncMock(return_value=sample_routing)
    with patch("agents.orchestrator.route_query", new=mock_route):
        await orchestrator_node(base_state)
    mock_route.assert_called_once_with("test current query")


async def test_retrieval_node_calls_correct_agent():
    mock_result = {"source": "vector", "results": []}
    mock_agent = AsyncMock(return_value=mock_result)
    state = {"_subtask_agent": "vector", "_subtask_query": "search tickets"}
    with patch("workflows.nodes.AGENT_REGISTRY", {"vector": mock_agent}):
        await single_retrieval_node(state)
    mock_agent.assert_called_once_with("search tickets")


async def test_retrieval_node_returns_list():
    mock_result = {"source": "vector", "results": []}
    mock_agent = AsyncMock(return_value=mock_result)
    state = {"_subtask_agent": "vector", "_subtask_query": "search tickets"}
    with patch("workflows.nodes.AGENT_REGISTRY", {"vector": mock_agent}):
        result = await single_retrieval_node(state)
    assert isinstance(result["retrieval_results"], list)
    assert result["retrieval_results"] == [mock_result]


async def test_retrieval_node_propagates_exception():
    mock_agent = AsyncMock(side_effect=RuntimeError("DB connection failed"))
    state = {"_subtask_agent": "vector", "_subtask_query": "search tickets"}
    with patch("workflows.nodes.AGENT_REGISTRY", {"vector": mock_agent}):
        with pytest.raises(RuntimeError, match="DB connection failed"):
            await single_retrieval_node(state)


async def test_judge_node_accept_no_query_update(base_state, accept_decision):
    with patch("agents.judge_agent.evaluate", new=AsyncMock(return_value=accept_decision)):
        with patch("workflows.nodes.settings") as mock_settings:
            mock_settings.max_retrieval_iterations = 3
            result = await judge_node(base_state)
    assert "current_query" not in result


async def test_judge_node_reject_updates_current_query(base_state, reject_decision):
    with patch("agents.judge_agent.evaluate", new=AsyncMock(return_value=reject_decision)):
        with patch("workflows.nodes.settings") as mock_settings:
            mock_settings.max_retrieval_iterations = 3
            result = await judge_node(base_state)
    assert "current_query" in result
    assert "JUDGE GAPS" in result["current_query"]


async def test_judge_node_appends_to_previous_decisions(base_state, accept_decision):
    with patch("agents.judge_agent.evaluate", new=AsyncMock(return_value=accept_decision)):
        with patch("workflows.nodes.settings") as mock_settings:
            mock_settings.max_retrieval_iterations = 3
            result = await judge_node(base_state)
    assert result["previous_decisions"] == [accept_decision]


async def test_judge_node_increments_iteration(base_state, accept_decision):
    base_state["iteration"] = 1
    with patch("agents.judge_agent.evaluate", new=AsyncMock(return_value=accept_decision)):
        with patch("workflows.nodes.settings") as mock_settings:
            mock_settings.max_retrieval_iterations = 3
            result = await judge_node(base_state)
    assert result["iteration"] == 2
