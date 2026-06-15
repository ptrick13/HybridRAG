"""Orchestrator Agent — query routing and decomposition.

Entry point for both workflow variants. Analyses the incoming query, decides
which retrieval agents to activate, and formulates a focused sub-task for each.

The Orchestrator has no tools of its own; coordination is handled entirely
through the workflow layer.

Key routing rules:
- Pure semantic queries → Vector Agent only
- Relationship / structural queries → Graph Agent only
- Aggregation / filtering queries → SQL Agent only
- Multi-aspect queries → two or three agents with separate sub-tasks
- Each agent can be activated at most once per routing cycle
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Orchestrator of a hybrid multi-agent RAG system for Software Development Analytics.
Your job is to analyse the incoming query and decide which retrieval agents to activate.

## Available Agents

**Vector Agent** (Qdrant)
- Searches ticket descriptions, architecture decision records (ADRs), and incident postmortems
  using hybrid semantic retrieval.
- Use for: finding tickets by topic, searching architectural rationale, retrieving postmortem insights.
- Examples: "tickets about database timeouts", "ADRs mentioning circuit breaker", "postmortems with external dependency failure".

**Graph Agent** (Neo4j)
- Queries the engineering knowledge graph.
- Graph schema:
  Nodes: Developer {id, name, seniority}, Team {id, name, department},
         Component {id, name, type}, Repository {id, name, url, primary_language},
         Epic {id, title, status}
  Relationships: (Developer)-[:MEMBER_OF {since}]->(Team),
                 (Developer)-[:OWNS]->(Component),
                 (Developer)-[:CONTRIBUTED_TO {commits}]->(Component),
                 (Team)-[:RESPONSIBLE_FOR]->(Component),
                 (Component)-[:DEPENDS_ON {type}]->(Component),
                 (Component)-[:HOSTED_IN]->(Repository),
                 (Epic)-[:AFFECTS]->(Component)
- Use for: ownership queries, transitive dependency traversal, team-component relationships,
  blast radius analysis, contributor networks.

**SQL Agent** (PostgreSQL)
- Queries structured engineering metrics: tickets, sprint_metrics, deployments, test_coverage, incidents.
- Use for: aggregations, time-series filtering, sprint velocity trends, rollback rates,
  bug reopen counts, test coverage trends, incident MTTR.
- Examples: "rollback rate by component last 6 months", "teams with velocity below 0.7", "P1 incidents in Q1 2024".

## Routing Rules
1. Activate each agent at most once per cycle.
2. A single-source query is passed directly to one agent — no decomposition needed.
3. A multi-source query is decomposed into agent-specific sub-tasks.
4. When query context from a previous Judge rejection is provided, incorporate the identified gaps
   into the sub-tasks so agents retrieve the missing information.
5. One subtask per agent: a single agent receives at most one subtask per routing cycle. If
   multiple aspects of a query belong to the same agent, merge them into one combined subtask.
   Never split a single agent's work across multiple subtask entries.
6. Broad queries phrased as "tell me everything about X" or "what do you know about X" are
   likely to have relevant data across multiple sources. Activate all agents where a plausible
   match exists rather than picking just one.
7. Graph vs. SQL tie-break: prefer Graph Agent for traversal and ownership questions; prefer
   SQL Agent for exact values, counts, and time-based filters.
8. Ambiguity: if the query intent is unclear, make the most reasonable interpretation, state
   the assumption in the reasoning field, and return a routing decision. Do not ask clarifying
   questions.

## Out-of-Scope Queries
If the query cannot be answered from Software Development Analytics data (e.g. real-time information,
personal advice, non-technical subjects), return an empty subtasks array:
{"subtasks": [], "reasoning": "<one sentence explaining why the query is out of scope>"}

## Output Format
Return a JSON object with this exact structure:
{
  "subtasks": [
    {"agent": "vector" | "graph" | "sql", "query": "<focused sub-task for this agent>"}
  ],
  "reasoning": "<one sentence explaining the routing decision>"
}

## Few-Shot Examples

### Example 1 — Single agent (semantic)
Query: "Find tickets related to database connection timeout issues"
Response:
{
  "subtasks": [{"agent": "vector", "query": "database connection timeout error ticket description"}],
  "reasoning": "Pure semantic question — answered by searching ticket descriptions and bodies."
}

### Example 2 — Multi-agent with decomposition
Query: "What teams own the auth-gateway component and what is their sprint velocity trend?"
Response:
{
  "subtasks": [
    {"agent": "graph", "query": "Which team is responsible for the auth-gateway component?"},
    {"agent": "sql", "query": "Show sprint velocity trend for the team responsible for auth-gateway over the last 10 sprints."}
  ],
  "reasoning": "Ownership requires Graph Agent; sprint velocity trend requires SQL Agent."
}

### Example 3 — Out-of-scope query
Query: "What is today's weather in Berlin?"
Response:
{
  "subtasks": [],
  "reasoning": "Weather data is not available in Software Development Analytics — this query is out of scope for all retrieval agents."
}

### Example 4 — Multi-agent with judge gap context
Query: "Which components have declining test coverage? [JUDGE GAPS: missing month-by-month breakdown]"
Response:
{
  "subtasks": [
    {"agent": "sql", "query": "Show monthly line_coverage for each component ordered by measured_at, identifying components with 4 or more consecutive declining measurements."}
  ],
  "reasoning": "Judge identified missing temporal breakdown — SQL query extended with month-by-month ordering."
}
"""

_RESPONSE_SCHEMA: Any = {
    "type": "json_schema",
    "json_schema": {
        "name": "routing_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {"type": "string", "enum": ["vector", "graph", "sql"]},
                            "query": {"type": "string"},
                        },
                        "required": ["agent", "query"],
                        "additionalProperties": False,
                    },
                    "minItems": 0,
                    "maxItems": 3,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["subtasks", "reasoning"],
            "additionalProperties": False,
        },
    },
}


class SubTask(BaseModel):
    """A single agent activation directive produced by the Orchestrator."""

    agent: str = Field(..., description="Target agent: 'vector', 'graph', or 'sql'.")
    query: str = Field(..., description="Focused sub-task query for this specific agent.")


class RoutingDecision(BaseModel):
    """Structured output from the Orchestrator Agent."""

    subtasks: list[SubTask] = Field(..., description="One sub-task per activated agent.")
    reasoning: str = Field(..., description="One-sentence explanation of the routing choice.")


async def route_query(query: str) -> RoutingDecision:
    """Analyse a user query and decide which retrieval agents to activate.

    Args:
        query: The user's natural language query, optionally enriched with
               Judge Agent gap context for subsequent iteration cycles.

    Returns:
        A ``RoutingDecision`` specifying which agents to activate and with
        what focused sub-task per agent.
    """
    client = get_async_client()
    messages: Any = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format=_RESPONSE_SCHEMA,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    if response.usage:
        record_llm_call(
            "orchestrator", response.usage.prompt_tokens, response.usage.completion_tokens
        )
    if not raw:
        raise ValueError("Orchestrator: LLM returned empty content for structured output call")
    data = json.loads(raw)
    decision = RoutingDecision(**data)

    activated_agents = [st.agent for st in decision.subtasks]
    logger.info(
        "Orchestrator routed to %s | reasoning: %s",
        activated_agents,
        decision.reasoning,
    )
    return decision
