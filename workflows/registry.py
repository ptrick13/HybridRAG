from collections.abc import Callable

from agents import graph_agent, sql_agent, vector_agent

AGENT_REGISTRY: dict[str, Callable] = {
    "vector": vector_agent.retrieve,
    "graph": graph_agent.retrieve,
    "sql": sql_agent.retrieve,
}
