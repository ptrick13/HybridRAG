"""Graph Agent — structural queries over the Neo4j knowledge graph.

Handles queries where relationships between entities are decisive: tag
co-occurrence, user expertise, and structural path queries. Uses the
``query_neo4j`` tool to execute Cypher queries.

Key behaviours:
- Generates read-only Cypher with LIMIT 25 on every query
- Property names are case-sensitive in Neo4j — the system prompt makes this explicit
- Self-corrects: if a query fails or returns empty, the agent retries with a
  refined Cypher query (up to MAX_SELF_CORRECTION_ATTEMPTS times)
- Returns raw query results without interpretation
"""

import asyncio
import json
import logging
from typing import Any

from neo4j.exceptions import Neo4jError

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings
from tools.neo4j_client import query_neo4j

logger = logging.getLogger(__name__)

MAX_SELF_CORRECTION_ATTEMPTS = 3

_SYSTEM_PROMPT = """\
You are the Graph Agent in a hybrid RAG system. Your sole responsibility is
to retrieve structural information from the Software Development Analytics Neo4j knowledge graph
by writing and executing read-only Cypher queries.

## Graph Schema
IMPORTANT: Neo4j property names are CASE-SENSITIVE. Use exactly the casing shown below.

Nodes:
  - Developer  {id: string, name: string, seniority: string}
  - Team       {id: string, name: string, department: string}
  - Component  {id: string, name: string, type: string}
  - Repository {id: string, name: string, url: string, primary_language: string}
  - Epic       {id: string, title: string, status: string}

Relationships:
  - (Developer)-[:MEMBER_OF {since: string}]->(Team)
  - (Developer)-[:OWNS]->(Component)
  - (Developer)-[:CONTRIBUTED_TO {commits: integer}]->(Component)
  - (Team)-[:RESPONSIBLE_FOR]->(Component)
  - (Component)-[:DEPENDS_ON {type: string}]->(Component)   ← key: supports multi-hop
  - (Component)-[:HOSTED_IN]->(Repository)
  - (Epic)-[:AFFECTS]->(Component)

## Rules
1. ALL queries must be read-only (MATCH / RETURN / WITH / WHERE / ORDER BY / LIMIT only).
2. EVERY query MUST end with LIMIT 25 to prevent context-window overflow.
3. Use exact property name casing from the schema above (e.g., "seniority" not "Seniority").
4. If a query fails or returns unexpected empty results, refine the Cypher and retry.
5. Return raw query results — no interpretation, no summarisation.
6. Always include the Cypher query you executed before presenting results. Structure
   your response as: the query in a ```cypher``` code block first, then the raw
   results below it. Omitting the query is not acceptable — it must appear in every
   response for traceability.

## Self-Correction
If the tool call returns an error or empty result:
- Check for typos in property names (case sensitivity is a common source of errors)
- Broaden the WHERE clause (e.g., use toLower() for string comparisons)
- Verify the relationship direction in the schema
- Retry with the corrected query

## Important
- Return ONLY the raw results from the tool call.
- Do not add explanation or use general LLM knowledge.
"""

_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_neo4j",
        "description": (
            "Execute a read-only Cypher query against the Software Development Analytics "
            "Neo4j knowledge graph. LIMIT 25 is required on every query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": (
                        "A read-only Cypher query (MATCH/RETURN). "
                        "Must include LIMIT 25."
                    ),
                }
            },
            "required": ["cypher"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def retrieve(subtask_query: str) -> dict[str, Any]:
    """Execute a graph query for the given sub-task with self-correction.

    The agent generates Cypher, executes it, and retries up to
    ``MAX_SELF_CORRECTION_ATTEMPTS`` times if the query fails or returns
    an empty result set.

    Args:
        subtask_query: The focused graph retrieval sub-task from the Orchestrator.

    Returns:
        A dict with ``source="graph"``, the executed ``cypher``, and ``results``
        containing the list of record dicts.
    """
    client = get_async_client()
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": subtask_query},
    ]

    last_cypher = ""
    last_results: list[dict] = []
    loop = asyncio.get_running_loop()

    for attempt in range(MAX_SELF_CORRECTION_ATTEMPTS):
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=[_TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": "query_neo4j"}},
            temperature=0,
        )

        if response.usage:
            record_llm_call("graph", response.usage.prompt_tokens, response.usage.completion_tokens)
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            logger.warning("Graph Agent: no tool call on attempt %d.", attempt + 1)
            messages.append({"role": "assistant", "content": assistant_message.content or ""})
            messages.append({"role": "user", "content": "You must call the query_neo4j tool. Please try again."})
            continue
        tool_call = assistant_message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        last_cypher = tool_args["cypher"]

        logger.debug("Graph Agent Cypher (attempt %d): %s", attempt + 1, last_cypher)

        try:
            last_results = await loop.run_in_executor(None, query_neo4j, last_cypher)
            tool_result_content = json.dumps(last_results)
            success = True
        except Neo4jError as exc:
            tool_result_content = f"ERROR: {exc}"
            success = False
            logger.warning("Graph Agent Cypher failed (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            tool_result_content = f"ERROR: {exc}"
            success = False
            logger.warning("Graph Agent unexpected error (attempt %d): %s", attempt + 1, exc, exc_info=True)

        # Feed the tool result back to the LLM for potential self-correction
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": "query_neo4j", "arguments": tool_call.function.arguments},
            }
        ]})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result_content,
        })

        if success and last_results:
            break

        if attempt < MAX_SELF_CORRECTION_ATTEMPTS - 1:
            correction_prompt = (
                "The previous query returned empty results or an error. "
                "Analyse the issue, correct the Cypher query, and try again."
            )
            messages.append({"role": "user", "content": correction_prompt})

    logger.info("Graph Agent retrieved %d records.", len(last_results))

    return {
        "source": "graph",
        "cypher": last_cypher,
        "results": last_results,
    }
