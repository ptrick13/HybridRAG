"""SQL Agent — structured queries over PostgreSQL.

Handles queries requiring precise filtering, aggregation, statistics, or
access to structured Software Development Analytics metrics. Uses the ``query_postgres``
tool to execute SQL queries.

Key behaviours:
- Generates read-only SELECT statements with LIMIT 50
- Self-corrects on SQL errors or unexpected empty results
- Returns results as a formatted string with column headers
"""

import asyncio
import json
import logging
from typing import Any

import psycopg2

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings
from tools.postgres_client import query_postgres

logger = logging.getLogger(__name__)

MAX_SELF_CORRECTION_ATTEMPTS = 3

_SYSTEM_PROMPT = """\
You are the SQL Agent in a hybrid RAG system. Your sole responsibility is
to retrieve structured information from the Software Development Analytics PostgreSQL database
by writing and executing read-only SQL queries.

## Database Schema

CREATE TABLE tickets (
    id UUID PRIMARY KEY, title TEXT NOT NULL, type VARCHAR(20), priority VARCHAR(5),
    status VARCHAR(20), component_id VARCHAR(20), assignee_id VARCHAR(20),
    team_id VARCHAR(20), story_points INTEGER, created_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ, reopened_count INTEGER DEFAULT 0, sprint_id VARCHAR(20),
    description TEXT
);

CREATE TABLE sprint_metrics (
    id TEXT PRIMARY KEY, sprint_id VARCHAR(20), team_id VARCHAR(20),
    start_date DATE, end_date DATE,
    planned_points INTEGER, completed_points INTEGER,
    velocity DOUBLE PRECISION,
    bug_count INTEGER, feature_count INTEGER, carried_over_count INTEGER
);

CREATE TABLE deployments (
    id UUID PRIMARY KEY, component_id VARCHAR(20), version VARCHAR(50),
    status VARCHAR(20), environment VARCHAR(20), deployed_by VARCHAR(20),
    deployed_at TIMESTAMPTZ, duration_seconds INTEGER
);

CREATE TABLE test_coverage (
    id UUID PRIMARY KEY, component_id VARCHAR(20), measured_at DATE,
    line_coverage DOUBLE PRECISION, branch_coverage DOUBLE PRECISION, test_count INTEGER
);

CREATE TABLE incidents (
    id UUID PRIMARY KEY, title TEXT, severity VARCHAR(5),
    component_id VARCHAR(20), root_cause_component_id VARCHAR(20),
    started_at TIMESTAMPTZ, resolved_at TIMESTAMPTZ,
    duration_minutes INTEGER, affected_users INTEGER, status VARCHAR(20)
);

## Rules
1. ALL queries must be read-only SELECT statements — no INSERT, UPDATE, DELETE, or DDL.
2. EVERY query MUST include LIMIT 50 to prevent context-window overflow.
3. Use parameterised-style queries if filtering by user input, but since this
   is an agent environment, inline the values directly.
4. If a query fails or returns empty unexpectedly, refine and retry.
5. Return results as a formatted string with column headers (the tool handles formatting).

## Self-Correction
If the tool call returns an error or "No results found.":
- Check table and column names for typos
- Broaden date ranges or score thresholds
- Try ILIKE for case-insensitive text matching
- Verify JOIN conditions
- Retry with the corrected query

## Important
- Return ONLY the raw results from the tool call.
- Do not add explanation or use general LLM knowledge.
"""

_TOOL_DEFINITION: Any = {
    "type": "function",
    "function": {
        "name": "query_postgres",
        "description": (
            "Execute a read-only SQL SELECT query against the Software Development Analytics "
            "PostgreSQL database. Must include LIMIT 50."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": ("A read-only SELECT statement. Must include LIMIT 50."),
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_SQL_TOOL_CHOICE: Any = {"type": "function", "function": {"name": "query_postgres"}}


async def retrieve(subtask_query: str) -> dict[str, Any]:
    """Execute a SQL query for the given sub-task with self-correction.

    The agent generates SQL, executes it, and retries up to
    ``MAX_SELF_CORRECTION_ATTEMPTS`` times if the query fails or returns
    no data.

    Args:
        subtask_query: The focused SQL retrieval sub-task from the Orchestrator.

    Returns:
        A dict with ``source="sql"``, the executed ``sql`` statement, and
        ``results`` as the formatted string returned by ``query_postgres``.
    """
    client = get_async_client()
    messages: Any = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": subtask_query},
    ]

    last_sql = ""
    last_results = "No results found."
    loop = asyncio.get_running_loop()

    for attempt in range(MAX_SELF_CORRECTION_ATTEMPTS):
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=[_TOOL_DEFINITION],
            tool_choice=_SQL_TOOL_CHOICE,
            temperature=0.0,
        )

        if response.usage:
            record_llm_call("sql", response.usage.prompt_tokens, response.usage.completion_tokens)
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            logger.warning("SQL Agent: no tool call on attempt %d.", attempt + 1)
            messages.append({"role": "assistant", "content": assistant_message.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "You must call the query_postgres tool. Please try again.",
                }
            )
            continue
        tool_call: Any = assistant_message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        last_sql = tool_args["sql"]

        logger.debug("SQL Agent query (attempt %d): %s", attempt + 1, last_sql)

        try:
            last_results = await loop.run_in_executor(None, query_postgres, last_sql)
            tool_result_content = last_results
            success = True
        except (psycopg2.Error, ValueError) as exc:
            tool_result_content = f"ERROR: {exc}"
            success = False
            logger.warning("SQL Agent query failed (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            tool_result_content = f"ERROR: {exc}"
            success = False
            logger.warning(
                "SQL Agent unexpected error (attempt %d): %s", attempt + 1, exc, exc_info=True
            )

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": "query_postgres",
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_content,
            }
        )

        if success and last_results != "No results found.":
            break

        if attempt < MAX_SELF_CORRECTION_ATTEMPTS - 1:
            correction_prompt = (
                "The previous SQL query returned no results or an error. "
                "Analyse the issue, correct the SQL query, and try again."
            )
            messages.append({"role": "user", "content": correction_prompt})

    logger.info("SQL Agent completed retrieval.")

    return {
        "source": "sql",
        "sql": last_sql,
        "results": last_results,
    }
