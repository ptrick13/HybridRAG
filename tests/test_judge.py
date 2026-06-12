import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.judge_agent import evaluate

_ACCEPT_JSON = json.dumps({
    "decision": "ACCEPT",
    "criteria_scores": {"completeness": 5, "relevance": 5, "consistency": 5, "specificity": 4},
    "gaps": [],
    "conflicts": [],
    "reformulated_query": None,
    "reasoning": "All criteria met.",
})

_REJECT_JSON = json.dumps({
    "decision": "REJECT",
    "criteria_scores": {"completeness": 2, "relevance": 4, "consistency": 4, "specificity": 3},
    "gaps": ["Expert data missing from Graph Agent"],
    "conflicts": [],
    "reformulated_query": "Find top contributors with commit counts",
    "reasoning": "Graph data missing. Retry with broader traversal.",
})

_REJECT_WITH_CONFLICTS_JSON = json.dumps({
    "decision": "REJECT",
    "criteria_scores": {"completeness": 3, "relevance": 4, "consistency": 2, "specificity": 3},
    "gaps": [],
    "conflicts": ["Vector Agent states 8 incidents; SQL Agent reports 47 total incidents"],
    "reformulated_query": "Clarify incident count scope",
    "reasoning": "Conflicting incident counts detected.",
})

_MAX_ITER_JSON = json.dumps({
    "decision": "MAX_ITERATIONS_REACHED",
    "criteria_scores": {"completeness": 3, "relevance": 4, "consistency": 4, "specificity": 3},
    "gaps": ["Contributor details still incomplete"],
    "conflicts": [],
    "reformulated_query": None,
    "reasoning": "Max iterations reached. Documenting remaining gaps.",
})


@pytest.fixture
def mock_client(make_response):
    client = AsyncMock()
    with patch("agents.judge_agent.get_async_client", return_value=client):
        yield client, make_response


async def test_accept_decision_parsed(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    result = await evaluate("some query", sample_retrieval_results, 0, 3)
    assert result.decision == "ACCEPT"


async def test_accept_reformulated_query_is_none(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    result = await evaluate("some query", sample_retrieval_results, 0, 3)
    assert result.reformulated_query is None


async def test_accept_record_llm_call(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(
        _ACCEPT_JSON, prompt_tokens=80, completion_tokens=40
    )
    with patch("agents.judge_agent.record_llm_call") as mock_record:
        await evaluate("some query", sample_retrieval_results, 0, 3)
    mock_record.assert_called_once_with("judge", 80, 40)


async def test_reject_decision_parsed(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_REJECT_JSON)
    result = await evaluate("some query", sample_retrieval_results, 0, 3)
    assert result.decision == "REJECT"
    assert len(result.gaps) >= 1


async def test_reject_reformulated_query_populated(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_REJECT_JSON)
    result = await evaluate("some query", sample_retrieval_results, 0, 3)
    assert result.reformulated_query is not None
    assert len(result.reformulated_query) > 0


async def test_reject_conflicts_populated(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_REJECT_WITH_CONFLICTS_JSON)
    result = await evaluate("some query", sample_retrieval_results, 0, 3)
    assert len(result.conflicts) == 1
    assert "Vector Agent" in result.conflicts[0]


async def test_max_iterations_decision_parsed(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_MAX_ITER_JSON)
    result = await evaluate("some query", sample_retrieval_results, 2, 3)
    assert result.decision == "MAX_ITERATIONS_REACHED"


async def test_max_iterations_prompt_contains_final_notice(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_MAX_ITER_JSON)
    await evaluate("some query", sample_retrieval_results, iteration=2, max_iterations=3)
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "FINAL allowed iteration" in user_content


async def test_not_final_iteration_no_final_notice(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    await evaluate("some query", sample_retrieval_results, iteration=0, max_iterations=3)
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "FINAL allowed iteration" not in user_content


async def test_previous_decisions_none_no_history_section(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    await evaluate("some query", sample_retrieval_results, 0, 3, previous_decisions=None)
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Previous Iteration Decisions" not in user_content


async def test_previous_decisions_empty_no_history_section(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    await evaluate("some query", sample_retrieval_results, 0, 3, previous_decisions=[])
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Previous Iteration Decisions" not in user_content


async def test_previous_decisions_serialized_in_prompt(mock_client, sample_retrieval_results, accept_decision):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_ACCEPT_JSON)
    await evaluate("some query", sample_retrieval_results, 1, 3, previous_decisions=[accept_decision])
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Previous Iteration Decisions" in user_content
    assert "ACCEPT" in user_content
