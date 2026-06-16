from openai import AsyncOpenAI

from agents.client import get_async_client


def test_get_async_client_returns_async_openai_instance():
    get_async_client.cache_clear()
    client = get_async_client()
    assert isinstance(client, AsyncOpenAI)
    get_async_client.cache_clear()


def test_get_async_client_is_cached():
    get_async_client.cache_clear()
    client_a = get_async_client()
    client_b = get_async_client()
    assert client_a is client_b
    get_async_client.cache_clear()
