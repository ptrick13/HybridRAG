import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-does-not-matter")

from unittest.mock import MagicMock

import pytest

from agents.judge_agent import CriteriaScores, JudgeDecision
from agents.orchestrator import RoutingDecision, SubTask
from agents.usage import init_tracking


@pytest.fixture(autouse=True)
def reset_tracking():
    init_tracking()


@pytest.fixture
def make_response():
    def _make(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        return response

    return _make


@pytest.fixture
def sample_routing():
    return RoutingDecision(
        subtasks=[
            SubTask(agent="vector", query="search semantic tickets"),
            SubTask(agent="sql", query="aggregate deployment metrics"),
        ],
        reasoning="Multi-source query requires semantic and structured retrieval.",
    )


@pytest.fixture
def sample_retrieval_results():
    return [
        {
            "source": "vector",
            "results": [
                {
                    "id": "uuid-1",
                    "payload": {
                        "ticket_id": "TKT-001",
                        "title": "Database timeout issue",
                        "text": "Connection pool exhausted during peak load.",
                    },
                }
            ],
        },
        {
            "source": "graph",
            "cypher": "MATCH (t:Team)-[:RESPONSIBLE_FOR]->(c:Component) RETURN t, c",
            "results": [{"team": "backend", "component": "auth-gateway"}],
        },
        {
            "source": "sql",
            "sql": "SELECT component_id, COUNT(*) FROM deployments WHERE status='rollback'",
            "results": "component_id | count\nauth-gateway | 12\napi-proxy | 3",
        },
    ]


@pytest.fixture
def accept_decision():
    return JudgeDecision(
        decision="ACCEPT",
        criteria_scores=CriteriaScores(completeness=5, relevance=5, consistency=5, specificity=4),
        gaps=[],
        conflicts=[],
        reformulated_query=None,
        reasoning="All criteria met with high scores. No gaps or conflicts detected.",
    )


@pytest.fixture
def reject_decision():
    return JudgeDecision(
        decision="REJECT",
        criteria_scores=CriteriaScores(completeness=2, relevance=4, consistency=4, specificity=3),
        gaps=["Expert user data is missing from Graph Agent results"],
        conflicts=[],
        reformulated_query="Find top contributors to auth-gateway with commit counts",
        reasoning="Graph data missing for expert analysis. Retry with broader traversal.",
    )
