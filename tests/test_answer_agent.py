from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.answer_agent import synthesise, synthesise_streaming
from agents.judge_agent import CriteriaScores, JudgeDecision


def make_judge_decision(decision: str, gaps=None, conflicts=None):
    return JudgeDecision(
        decision=decision,
        criteria_scores=CriteriaScores(completeness=4, relevance=4, consistency=4, specificity=4),
        gaps=gaps or [],
        conflicts=conflicts or [],
        reformulated_query=None,
        reasoning="test",
    )


@pytest.fixture
def mock_client(make_response):
    client = AsyncMock()
    with patch("agents.answer_agent.get_async_client", return_value=client):
        yield client, make_response


async def test_synthesise_returns_answer(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response("The answer is 42.")
    result = await synthesise("What is the answer?", sample_retrieval_results)
    assert result == "The answer is 42."


async def test_synthesise_records_llm_call(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response(
        "Test answer", prompt_tokens=100, completion_tokens=50
    )
    with patch("agents.answer_agent.record_llm_call") as mock_record:
        await synthesise("query", sample_retrieval_results)
    mock_record.assert_called_once_with("answer", 100, 50)


async def test_synthesise_with_judge_decision(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    client.chat.completions.create.return_value = make_response("Cited answer.")
    decision = make_judge_decision("ACCEPT", conflicts=["Source A vs Source B"])
    result = await synthesise("query", sample_retrieval_results, judge_decision=decision)
    assert result == "Cited answer."


async def test_synthesise_null_content_returns_fallback(mock_client, sample_retrieval_results):
    client, make_response = mock_client
    response = make_response("placeholder")
    response.choices[0].message.content = None
    client.chat.completions.create.return_value = response
    result = await synthesise("query", sample_retrieval_results)
    assert "unable" in result.lower()


async def test_synthesise_streaming_yields_tokens(mock_client, sample_retrieval_results):
    client, _ = mock_client

    async def fake_stream():
        for token in ["The", " answer", " is", " 42."]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            chunk.usage = None
            yield chunk
        final = MagicMock()
        final.choices = []
        final.usage = MagicMock()
        final.usage.prompt_tokens = 100
        final.usage.completion_tokens = 20
        yield final

    client.chat.completions.create.return_value = fake_stream()

    tokens = []
    async for token in synthesise_streaming("What is the answer?", sample_retrieval_results):
        tokens.append(token)

    assert tokens == ["The", " answer", " is", " 42."]


async def test_synthesise_streaming_records_usage(mock_client, sample_retrieval_results):
    client, _ = mock_client

    async def fake_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "response"
        chunk.usage = None
        yield chunk
        final = MagicMock()
        final.choices = []
        final.usage = MagicMock()
        final.usage.prompt_tokens = 50
        final.usage.completion_tokens = 10
        yield final

    client.chat.completions.create.return_value = fake_stream()

    with patch("agents.answer_agent.record_llm_call") as mock_record:
        async for _ in synthesise_streaming("query", sample_retrieval_results):
            pass

    mock_record.assert_called_once_with("answer", 50, 10)
