import asyncio

from agents.usage import (
    compute_cost,
    get_embedding_token_count,
    get_llm_input_token_count,
    get_llm_output_token_count,
    get_llm_token_count,
    init_tracking,
    record_embedding_tokens,
    record_llm_call,
)

_GPT4O_IN = 2.50 / 1_000_000
_GPT4O_OUT = 10.00 / 1_000_000
_GPT4O_MINI_IN = 0.15 / 1_000_000
_GPT4O_MINI_OUT = 0.60 / 1_000_000
_EMBED_3_LARGE = 0.13 / 1_000_000


def test_init_tracking_resets():
    record_llm_call("agent", 100, 50)
    record_embedding_tokens("text-embedding-3-large", 200)
    init_tracking()
    assert get_llm_token_count() == 0
    assert get_embedding_token_count() == 0


def test_record_llm_accumulates():
    record_llm_call("orchestrator", 100, 50)
    record_llm_call("judge", 200, 80)
    assert get_llm_input_token_count() == 300
    assert get_llm_output_token_count() == 130


def test_input_output_tracked_separately():
    record_llm_call("orchestrator", 100, 50)
    assert get_llm_input_token_count() == 100
    assert get_llm_output_token_count() == 50
    assert get_llm_token_count() == 150


def test_embedding_tracked_separately():
    record_llm_call("orchestrator", 100, 50)
    record_embedding_tokens("text-embedding-3-large", 300)
    assert get_embedding_token_count() == 300
    assert get_llm_token_count() == 150


def test_multiple_agents_sum():
    record_llm_call("orchestrator", 100, 20)
    record_llm_call("judge", 150, 30)
    record_llm_call("answer", 200, 50)
    assert get_llm_input_token_count() == 450
    assert get_llm_output_token_count() == 100


def test_cost_gpt4o():
    record_llm_call("orchestrator", 1_000, 500)
    cost = compute_cost("gpt-4o")
    expected = 1_000 * _GPT4O_IN + 500 * _GPT4O_OUT
    assert abs(cost - expected) < 1e-12


def test_cost_gpt4o_mini_cheaper():
    record_llm_call("agent", 1_000, 500)
    cost_big = compute_cost("gpt-4o")

    init_tracking()
    record_llm_call("agent", 1_000, 500)
    cost_mini = compute_cost("gpt-4o-mini")

    assert cost_mini < cost_big


def test_embedding_cost_adds_to_total():
    record_llm_call("orchestrator", 1_000, 0)
    record_embedding_tokens("text-embedding-3-large", 1_000)
    cost = compute_cost("gpt-4o")
    expected = 1_000 * _GPT4O_IN + 1_000 * _EMBED_3_LARGE
    assert abs(cost - expected) < 1e-12


def test_unknown_model_falls_back_to_default_pricing():
    record_llm_call("agent", 1_000, 500)
    cost_unknown = compute_cost("gpt-999-mystery")

    init_tracking()
    record_llm_call("agent", 1_000, 500)
    cost_gpt4o = compute_cost("gpt-4o")

    assert abs(cost_unknown - cost_gpt4o) < 1e-12


def test_zero_tokens_zero_cost():
    cost = compute_cost("gpt-4o")
    assert cost == 0.0


async def test_contextvars_isolated_across_tasks():
    async def run_task(agent_name: str, tokens: int) -> int:
        init_tracking()
        record_llm_call(agent_name, tokens, 0)
        return get_llm_input_token_count()

    count_a, count_b = await asyncio.gather(
        asyncio.create_task(run_task("a", 100)),
        asyncio.create_task(run_task("b", 200)),
    )
    assert count_a == 100
    assert count_b == 200


async def test_contextvars_no_cross_contamination():
    async def run_task(tokens: int) -> int:
        init_tracking()
        record_llm_call("agent", tokens, 0)
        await asyncio.sleep(0)
        return get_llm_input_token_count()

    results = await asyncio.gather(
        asyncio.create_task(run_task(100)),
        asyncio.create_task(run_task(999)),
    )
    assert sorted(results) == [100, 999]
