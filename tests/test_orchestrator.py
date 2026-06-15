import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agents.orchestrator import route_query

_SINGLE_VECTOR = json.dumps(
    {
        "subtasks": [{"agent": "vector", "query": "database timeout ticket search"}],
        "reasoning": "Pure semantic question about ticket descriptions.",
    }
)

_MULTI_AGENT = json.dumps(
    {
        "subtasks": [
            {"agent": "graph", "query": "Who owns the auth-gateway component?"},
            {"agent": "sql", "query": "Sprint velocity for auth-gateway team last 10 sprints"},
        ],
        "reasoning": "Ownership requires Graph; sprint velocity requires SQL.",
    }
)

_EMPTY_SUBTASKS = json.dumps(
    {
        "subtasks": [],
        "reasoning": "Weather data is not available in Software Development Analytics.",
    }
)


@pytest.fixture
def mock_client(make_response):
    client = AsyncMock()
    with patch("agents.orchestrator.get_async_client", return_value=client):
        yield client, make_response


async def test_single_subtask_parsed(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_SINGLE_VECTOR)
    result = await route_query("Find tickets about database timeouts")
    assert len(result.subtasks) == 1
    assert result.subtasks[0].agent == "vector"


async def test_multiple_subtasks_parsed(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_MULTI_AGENT)
    result = await route_query("Who owns auth-gateway and what is their sprint velocity?")
    assert len(result.subtasks) == 2
    assert {st.agent for st in result.subtasks} == {"graph", "sql"}


async def test_empty_subtasks_parsed(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_EMPTY_SUBTASKS)
    result = await route_query("What is today's weather in Berlin?")
    assert result.subtasks == []
    assert result.reasoning != ""


async def test_reasoning_preserved(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_SINGLE_VECTOR)
    result = await route_query("Find tickets about database timeouts")
    assert result.reasoning == "Pure semantic question about ticket descriptions."


async def test_missing_required_field_raises(mock_client):
    client, make_response = mock_client
    bad_json = json.dumps({"subtasks": [{"agent": "vector"}], "reasoning": "test"})
    client.chat.completions.create.return_value = make_response(bad_json)
    with pytest.raises(ValidationError):
        await route_query("some query")


async def test_invalid_json_raises(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response("not valid json {{")
    with pytest.raises(json.JSONDecodeError):
        await route_query("some query")


async def test_empty_content_raises(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response("")
    with pytest.raises(ValueError, match="LLM returned empty content"):
        await route_query("some query")


async def test_record_llm_call_invoked_once(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(
        _SINGLE_VECTOR, prompt_tokens=50, completion_tokens=20
    )
    with patch("agents.orchestrator.record_llm_call") as mock_record:
        await route_query("some query")
    mock_record.assert_called_once_with("orchestrator", 50, 20)


async def test_query_passed_to_llm(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_SINGLE_VECTOR)
    query = "Find tickets about connection pool exhaustion"
    await route_query(query)
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert user_msg["content"] == query


async def test_response_format_schema_used(mock_client):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(_SINGLE_VECTOR)
    await route_query("some query")
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
