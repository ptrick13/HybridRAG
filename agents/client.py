"""Shared OpenAI client factory.

All agents import ``get_async_client()`` from this module to ensure a single
consistent client configuration across the system. Both standard OpenAI and
Azure OpenAI are supported via the ``OPENAI_BASE_URL`` setting.
"""

from functools import lru_cache

from openai import AsyncOpenAI

from config.settings import settings


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client configured from settings.

    The client is created once and reused across all agent calls, which
    keeps connection overhead low and respects rate-limit back-off state.

    Returns:
        A configured ``AsyncOpenAI`` instance. When ``OPENAI_BASE_URL`` is
        set to an Azure OpenAI endpoint, the same client is used for Azure.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
