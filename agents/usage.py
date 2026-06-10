"""Token usage tracking and cost computation for LLM and embedding calls.

Each LLM-calling agent records its token counts here via ``record_llm_call``.
Embedding calls are tracked separately via ``record_embedding_tokens``.
The workflow layer calls ``init_tracking`` at the start of each run and
``compute_cost`` at the end to get the total USD cost for the run.

Uses a ``ContextVar`` so that concurrent workflow runs (in tests or eval)
do not pollute each other's counters.
"""

from contextvars import ContextVar
from dataclasses import dataclass

# (price_input_per_token, price_output_per_token) in USD
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4-turbo": (10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-4": (30.00 / 1_000_000, 60.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}
_DEFAULT_PRICING: tuple[float, float] = (2.50 / 1_000_000, 10.00 / 1_000_000)

# Price per token in USD for embedding models
_EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-large": 0.13 / 1_000_000,
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-ada-002": 0.10 / 1_000_000,
}
_DEFAULT_EMBEDDING_PRICING: float = 0.13 / 1_000_000


@dataclass
class LLMCall:
    agent: str
    input_tokens: int
    output_tokens: int


@dataclass
class EmbeddingCall:
    model: str
    tokens: int


_CALL_LOG: ContextVar[list[LLMCall] | None] = ContextVar("_llm_call_log", default=None)
_EMBEDDING_LOG: ContextVar[list[EmbeddingCall] | None] = ContextVar("_embedding_call_log", default=None)


def init_tracking() -> None:
    """Reset both call logs for the current async context (call once per workflow run)."""
    _CALL_LOG.set([])
    _EMBEDDING_LOG.set([])


def record_llm_call(agent: str, input_tokens: int, output_tokens: int) -> None:
    """Append one LLM call record to the current context's log.

    Safe to call from concurrently running coroutines because all share the
    same list reference (the ContextVar holds the object, not the value).
    """
    log = _CALL_LOG.get()
    if log is not None:
        log.append(LLMCall(agent=agent, input_tokens=input_tokens, output_tokens=output_tokens))


def record_embedding_tokens(model: str, tokens: int) -> None:
    """Append one embedding call record to the current context's log.

    Args:
        model: The embedding model name (e.g. ``"text-embedding-3-large"``).
        tokens: Total tokens consumed by the embedding request.
    """
    log = _EMBEDDING_LOG.get()
    if log is not None:
        log.append(EmbeddingCall(model=model, tokens=tokens))


def get_llm_token_count() -> int:
    """Return total LLM tokens (input + output) consumed in the current context."""
    return sum(c.input_tokens + c.output_tokens for c in (_CALL_LOG.get() or []))


def get_llm_input_token_count() -> int:
    """Return total LLM input (prompt) tokens consumed in the current context."""
    return sum(c.input_tokens for c in (_CALL_LOG.get() or []))


def get_llm_output_token_count() -> int:
    """Return total LLM output (completion) tokens consumed in the current context."""
    return sum(c.output_tokens for c in (_CALL_LOG.get() or []))


def get_embedding_token_count() -> int:
    """Return total embedding tokens consumed in the current context."""
    return sum(e.tokens for e in (_EMBEDDING_LOG.get() or []))


def model_price(model: str) -> tuple[float, float]:
    """Return (input_price_per_token, output_price_per_token) in USD for *model*."""
    return _PRICING.get(model, _DEFAULT_PRICING)


def embedding_price(model: str) -> float:
    """Return price per token in USD for *model*."""
    return _EMBEDDING_PRICING.get(model, _DEFAULT_EMBEDDING_PRICING)


def compute_cost(model: str) -> float:
    """Return total USD cost for all LLM and embedding calls in the current context.

    Args:
        model: The chat completion model name (e.g. ``"gpt-4o"``). Falls back
            to gpt-4o pricing when the model is not in the pricing table.
            Embedding costs are tracked per-model independently.
    """
    price_in, price_out = _PRICING.get(model, _DEFAULT_PRICING)
    calls = _CALL_LOG.get() or []
    llm_cost = sum(c.input_tokens * price_in + c.output_tokens * price_out for c in calls)

    embedding_cost = sum(
        e.tokens * _EMBEDDING_PRICING.get(e.model, _DEFAULT_EMBEDDING_PRICING)
        for e in (_EMBEDDING_LOG.get() or [])
    )
    return llm_cost + embedding_cost
