"""Vector Agent — semantic search over Qdrant.

Handles queries where content similarity matters: conceptual questions, "how
to" queries, finding related discussions. Uses hybrid retrieval (dense +
sparse BM25 with RRF fusion) via the ``search_documents`` tool.

Unlike the Graph and SQL agents, the Vector Agent does not generate a formal
query language — it optimises the natural language query for hybrid retrieval.
"""

import json
import logging
from typing import Any

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings
from tools.qdrant_client import search_documents

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Vector Agent in a hybrid RAG system. Your sole responsibility is
to retrieve relevant Software Development Analytics documents using semantic hybrid search.

## Data Source
Qdrant vector database containing chunked engineering documents across three collections:

| collection     | contents                                                        |
|----------------|-----------------------------------------------------------------|
| tickets        | Bug/feature/tech-debt/incident ticket titles and descriptions   |
| arch_docs      | Architecture decision records (ADRs), design docs, and RFCs     |
| postmortems    | Incident postmortem reports                                     |

## Your Task
1. Analyse the incoming sub-task query.
2. Select the correct collection based on the query subject:
   - Use **tickets** for queries about bugs, features, tech debt, or incident tickets.
   - Use **arch_docs** for queries about architectural decisions, design rationale, or RFCs.
   - Use **postmortems** for queries about incident postmortems, root causes, or outage reports.
3. Optimise the query for hybrid retrieval: use specific technical terms,
   include synonyms for key concepts, and keep the query focused.
   Do NOT translate it into a formal query language — just refine the natural
   language query.
4. Call the search_documents tool with the optimised query and chosen collection.
5. Return the raw retrieval results exactly as received — no interpretation,
   no summarisation.

## Important
- Return ONLY the raw results from the tool call.
- Do not add any explanation, analysis, or personal knowledge.
- If the tool returns empty results, report that clearly.
"""

_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Retrieve Software Development Analytics documents using hybrid semantic search "
            "(dense embedding + BM25 sparse, fused with RRF)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The optimised natural language search query. "
                        "Use specific technical terms and synonyms."
                    ),
                },
                "collection": {
                    "type": "string",
                    "enum": ["tickets", "arch_docs", "postmortems"],
                    "description": (
                        "Which collection to search. "
                        "'tickets' for bug/feature/tech-debt/incident tickets. "
                        "'arch_docs' for ADRs, design docs, and RFCs. "
                        "'postmortems' for incident postmortem reports."
                    ),
                },
            },
            "required": ["query", "collection"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def retrieve(subtask_query: str) -> dict[str, Any]:
    """Execute semantic retrieval for the given sub-task.

    Runs one LLM call that optimises the query and calls ``search_documents``,
    then returns the raw results for consolidation by the workflow layer.

    Args:
        subtask_query: The focused retrieval sub-task from the Orchestrator.

    Returns:
        A dict with ``source="vector"``, the ``query`` used, and ``results``
        containing the list of retrieved document dicts.
    """
    client = get_async_client()
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": subtask_query},
    ]

    # First LLM call — agent decides what to search for
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=[_TOOL_DEFINITION],
        tool_choice={"type": "function", "function": {"name": "search_documents"}},
        temperature=0,
    )

    if response.usage:
        record_llm_call("vector", response.usage.prompt_tokens, response.usage.completion_tokens)
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        logger.warning("Vector Agent: LLM returned no tool call; returning empty results.")
        return {"source": "vector", "query": subtask_query, "collection": "tickets", "results": []}
    tool_call = tool_calls[0]
    tool_args = json.loads(tool_call.function.arguments)
    optimised_query = tool_args["query"]
    collection = tool_args.get("collection", "tickets")

    logger.debug("Vector Agent searching '%s': %s", collection, optimised_query)

    try:
        results = await search_documents(optimised_query, collection=collection)
    except Exception as exc:
        logger.warning("Vector Agent search failed: %s", exc, exc_info=True)
        return {"source": "vector", "query": optimised_query, "collection": collection, "results": []}

    logger.info("Vector Agent retrieved %d results from '%s'.", len(results), collection)

    return {
        "source": "vector",
        "query": optimised_query,
        "collection": collection,
        "results": results,
    }
