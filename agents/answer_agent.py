"""Answer Agent — final response synthesis with mandatory source citations.

Synthesises consolidated retrieval results from all activated agents into a
coherent, well-structured answer. Every factual claim must be backed by a
source citation from the retrieval results — the agent may NOT add information
from its general knowledge.

Citation format by source type:
- Vector Agent: [document title | chunk ID]
- Graph Agent:  [Neo4j | <entity name> via <relationship path>]
- SQL Agent:    [PostgreSQL | <table> WHERE <filter criteria>]
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from agents.client import get_async_client
from agents.judge_agent import JudgeDecision
from agents.usage import record_llm_call
from config.settings import settings

logger = logging.getLogger(__name__)

# Limits per-chunk text preview in prompts; full content is in Qdrant payload.
_MAX_CHUNK_PREVIEW_CHARS = 800

_SYSTEM_PROMPT = """\
You are the Answer Agent in a hybrid RAG system. Your job is to synthesise
retrieval results from multiple data sources into a clear, accurate, and
well-cited answer.

## Strict Rules

1. **Citations are mandatory.** Every factual claim must be backed by a
   source citation from the retrieval results. Use the format below.
   Do NOT include any claim that is not supported by the provided results.

2. **No hallucination.** You MUST NOT add information from your general
   knowledge. If the retrieval results do not contain an answer, say so
   explicitly.

3. **Conflict handling.** If conflicts between sources are passed to you:
   - Report BOTH conflicting claims side by side
   - Explicitly flag the conflict: "Source A states X, while Source B states Y"
   - Recommend manual verification
   - Do NOT favour either source

4. **Gap acknowledgement.** If remaining gaps are passed (from max-iteration
   Judge): explicitly state what could not be answered due to retrieval limits.

## Citation Format

- **Vector Agent results** (Qdrant documents):
  - Tickets: `[Ticket | id: <ticket_id> | title: "<title>"]`
  - Architecture docs: `[ArchDoc | id: <doc_id> | title: "<title>"]`
  - Postmortems: `[Postmortem | id: <postmortem_id> | title: "<title>"]`

- **Graph Agent results** (Neo4j queries):
  `[Neo4j | <entity> via <relationship_path>]`
  Example: `[Neo4j | User: "JohnDoe" via :EXPERT_IN → Tag: "python"]`

- **SQL Agent results** (PostgreSQL queries):
  `[PostgreSQL | table: <table> | filter: <WHERE clause>]`
  Example: `[PostgreSQL | table: questions | filter: score > 100 AND tag = 'python']`

## Structure
- Begin with a direct answer to the query
- Support each claim with inline citations
- If multiple sources agree on a fact, cite all of them
- End with a "Sources" section listing unique citations

## Output Quality

- **Language.** Respond in the same language as the query. Technical identifiers
  and source citations are exempt — keep them exactly as returned by the retrieval
  agents.

- **Identifiers.** Technical names returned by retrieval agents (component ids,
  team ids, ticket ids) must be reproduced verbatim, wrapped in backticks. Do not
  paraphrase, abbreviate, or reformat them.

- **Completeness.** When retrieval results contain an enumerable list of distinct
  named entities, output all of them. Do not truncate silently. For large numeric
  aggregations, a summary is acceptable. Only add a note about query limits if an
  agent explicitly reported hitting one.
"""


def _format_results_for_prompt(
    retrieval_results: list[dict[str, Any]],
    judge_decision: JudgeDecision | None,
) -> str:
    """Build the user-facing prompt section from retrieval results and judge metadata.

    Args:
        retrieval_results: Consolidated results from all activated agents.
        judge_decision: Judge decision (V2 only) carrying conflict and gap info.

    Returns:
        A formatted string describing all retrieval results for the LLM prompt.
    """
    sections: list[str] = []

    for result in retrieval_results:
        source = result.get("source", "unknown")
        if source == "vector":
            docs = result.get("results", [])
            doc_lines = []
            for doc in docs:
                payload = doc.get("payload", {})
                title = payload.get("title", "Untitled")
                # Prefer the human-readable typed ID for citations; fall back to
                # source_id (chunk-level) then the UUID5 Qdrant point ID.
                doc_ref = (
                    payload.get("ticket_id")
                    or payload.get("doc_id")
                    or payload.get("pm_id")
                    or payload.get("source_id")
                    or doc.get("id", "?")
                )
                text = payload.get("text", "")[:_MAX_CHUNK_PREVIEW_CHARS]
                doc_lines.append(f"  - ID: {doc_ref!r} | Title: {title!r}\n    Text: {text}")
            sections.append("=== VECTOR AGENT RESULTS ===\n" + "\n".join(doc_lines))

        elif source == "graph":
            cypher = result.get("cypher", "")
            records = result.get("results", [])
            sections.append(
                f"=== GRAPH AGENT RESULTS ===\n"
                f"Executed Cypher: {cypher}\n"
                f"Records ({len(records)}):\n" + json.dumps(records, indent=2, default=str)
            )

        elif source == "sql":
            sql = result.get("sql", "")
            results_str = result.get("results", "No results.")
            sections.append(
                f"=== SQL AGENT RESULTS ===\nExecuted SQL: {sql}\nResults:\n{results_str}"
            )

    if judge_decision:
        if judge_decision.conflicts:
            sections.append(
                "=== JUDGE AGENT: CONFLICTS DETECTED ===\n"
                + "\n".join(f"- {c}" for c in judge_decision.conflicts)
            )
        if judge_decision.decision == "MAX_ITERATIONS_REACHED" and judge_decision.gaps:
            sections.append(
                "=== JUDGE AGENT: UNRESOLVED GAPS (max iterations reached) ===\n"
                + "\n".join(f"- {g}" for g in judge_decision.gaps)
            )

    return "\n\n".join(sections)


async def synthesise(
    original_query: str,
    retrieval_results: list[dict[str, Any]],
    judge_decision: JudgeDecision | None = None,
) -> str:
    """Synthesise a cited answer from consolidated retrieval results.

    Args:
        original_query: The user's original query.
        retrieval_results: Consolidated results from all activated retrieval agents.
        judge_decision: Optional ``JudgeDecision`` (V2 only) providing conflict
            flags and gap documentation to pass through to the answer.

    Returns:
        A well-structured answer string with inline citations and a sources section.
    """
    results_section = _format_results_for_prompt(retrieval_results, judge_decision)

    user_content = f"""
User Query: {original_query}

Retrieved Context:
{results_section}

Please synthesise a cited answer to the user's query using ONLY the information
above. Follow all citation and conflict-handling rules from your system prompt.
"""

    client = get_async_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    if response.usage:
        record_llm_call("answer", response.usage.prompt_tokens, response.usage.completion_tokens)
    logger.info("Answer Agent synthesised response (%d chars).", len(answer or ""))
    return answer or "I was unable to generate an answer from the retrieved context."


async def synthesise_streaming(
    original_query: str,
    retrieval_results: list[dict[str, Any]],
    judge_decision: JudgeDecision | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a cited answer token-by-token from consolidated retrieval results.

    Reuses the same prompt-building logic as :func:`synthesise` but calls the
    OpenAI streaming API (``stream=True``) and yields each token as it arrives.

    Args:
        original_query: The user's original query.
        retrieval_results: Consolidated results from all activated retrieval agents.
        judge_decision: Optional ``JudgeDecision`` (V2 only) providing conflict
            flags and gap documentation to pass through to the answer.

    Yields:
        Individual text tokens as they stream from the model.
    """
    results_section = _format_results_for_prompt(retrieval_results, judge_decision)

    user_content = f"""
User Query: {original_query}

Retrieved Context:
{results_section}

Please synthesise a cited answer to the user's query using ONLY the information
above. Follow all citation and conflict-handling rules from your system prompt.
"""

    client = get_async_client()
    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage:
            record_llm_call("answer", chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
    logger.info("Answer Agent finished streaming response.")
