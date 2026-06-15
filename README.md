# Hybrid Multi-Agent RAG Architecture

![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.4-green)
![License](https://img.shields.io/badge/license-MIT-blue)
[![CI](https://github.com/ptrick13/HybridRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ptrick13/HybridRAG/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ptrick13/HybridRAG/graph/badge.svg)](https://codecov.io/gh/ptrick13/HybridRAG)

A **hybrid multi-agent Retrieval-Augmented Generation (RAG) system** that routes queries across three specialised retrieval agents connected to three heterogeneous data stores — a vector database, a knowledge graph, and a relational database.

---

## Architecture Overview

```mermaid
---
config:
  layout: elk
---
flowchart TD
    Q([User Query]) --> O["Orchestrator"]

    subgraph PAR[" "]
        V["Vector Agent\n(Qdrant)"]
        G["Graph Agent\n(Neo4j)"]
        S["SQL Agent\n(Postgres)"]
    end

    O --> V & G & S
    V & G & S --> J["Judge Agent\n(Variant 2 only)"]:::v2only

    J -.->|"ACCEPT"| A["Answer Agent"]
    J -.->|"REJECT"| O

    A --> E([Result])

    classDef v2only stroke-dasharray: 5 5
```

*Variant 1 skips the Judge and the retry loop — retrieval results flow directly to the Answer Agent.*

### Two Workflow Variants

| Feature | V1 (without closed-loop) | V2 (with closed-loop) |
|---|---|---|
| Stages | Orchestrator → Retrieval → Answer | Orchestrator → Retrieval ↔ Judge → Answer |
| Quality control | None | Judge Agent (4 criteria) |
| Max iterations | 1 | 3 |
| Query rewriting | No | Yes (on rejection) |

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM** | GPT-4o via OpenAI API (Azure OpenAI also supported) |
| **Embeddings** | text-embedding-3-large (3072 dims) |
| **Vector DB** | Qdrant — hybrid dense + BM25 sparse with RRF fusion |
| **Graph DB** | Neo4j — Cypher queries |
| **Relational DB** | PostgreSQL — SQL queries |
| **Workflow** | LangGraph — parallel retrieval via `Send()` fan-out |
| **MCP Server** | FastMCP on port 8001 |
| **A2A Server** | FastAPI on port 8002 (Google Agent2Agent protocol) |
| **Configuration** | pydantic-settings — all parameters from `.env` |

---

## Project Structure

```
hybridrag/
├── agents/
│   ├── client.py          # Shared AsyncOpenAI client
│   ├── orchestrator.py    # Query routing + decomposition
│   ├── vector_agent.py    # Semantic search (Qdrant)
│   ├── graph_agent.py     # Graph queries (Neo4j) with self-correction
│   ├── sql_agent.py       # SQL queries (PostgreSQL) with self-correction
│   ├── judge_agent.py     # Quality evaluation (V2 only)
│   ├── answer_agent.py    # Response synthesis with citations
│   └── usage.py           # Token usage tracking + cost computation
├── config/
│   └── settings.py        # Centralised config via pydantic-settings
├── tools/
│   ├── qdrant_client.py   # Hybrid dense + BM25 retrieval with RRF
│   ├── neo4j_client.py    # Cypher query execution
│   └── postgres_client.py # SQL query execution
├── workflows/
│   ├── models.py          # Shared Pydantic data models
│   ├── state.py           # LangGraph WorkflowState TypedDict + reducers
│   ├── nodes.py           # Shared node functions (orchestrator, retrieval, judge, answer)
│   ├── registry.py        # Agent name → callable mapping
│   ├── v1_workflow.py     # V1: START → orchestrator → Send() × N → answer → END
│   └── v2_workflow.py     # V2: closed-loop — judge routes back to orchestrator on REJECT
├── integrations/
│   ├── mcp_server.py      # FastMCP endpoint (port 8001)
│   └── a2a_server.py      # FastAPI A2A endpoint (port 8002)
├── evaluation/
│   ├── runner.py          # Evaluation runner + CLI
│   ├── metrics.py         # LLM-as-Judge scoring
│   └── test_queries.json  # 17 test queries across 6 categories
├── data/
│   ├── generate_sample.py # Generates synthetic sample data
│   ├── load_data.py       # Data loading orchestrator (CLI)
│   ├── ingest_postgres.py # PostgreSQL ingestion
│   ├── ingest_neo4j.py    # Neo4j graph ingestion
│   ├── ingest_qdrant.py   # Qdrant embedding + indexing
│   └── sample/            # Generated sample dataset (JSON)
├── scripts/
│   ├── web_ui.py          # FastAPI web UI (port 8000)
│   └── static/
│       └── index.html     # Single-page frontend
├── docker-compose.yml     # PostgreSQL + Neo4j + Qdrant + app
├── requirements.txt
└── .env.example
```

---

## Demo

Interface showing the Closed-Loop variant answering a graph query, with per-agent status, latency breakdown, token count, and cost:

<img src="demo/ui.png">

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (or Azure OpenAI credentials)

### 1. Clone and configure

```bash
git clone https://github.com/ptrick13/HybridRAG.git
cd HybridRAG
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Start everything

```bash
docker-compose up --build
# Builds the app image and starts PostgreSQL, Neo4j, Qdrant, and the web UI.
# The app waits for all databases to be healthy before starting.
# Open http://localhost:8000
```

### 3. Generate and load sample data

```bash
docker-compose exec app python -m data.generate_sample
docker-compose exec app python -m data.load_data
```

---

### Alternative: manual setup (without Docker for the app)

<details>
<summary>Expand manual setup instructions</summary>

**Prerequisites:** Python 3.11+, Docker and Docker Compose

```bash
# Install dependencies
pip install -r requirements.txt

# Start only the databases
docker-compose up -d postgres neo4j qdrant
# Wait until all services are healthy — Neo4j may take ~60s on first start
docker-compose ps

# Generate and load sample data
python -m data.generate_sample
python -m data.load_data

# Start the web UI
uvicorn scripts.web_ui:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

</details>

---

## Usage

### MCP Server

```bash
python -m integrations.mcp_server
# Starts at http://localhost:8001
```

Tool: `query_hybrid_rag(query: str, variant: str = "v1") -> str`

### A2A Server

```bash
uvicorn integrations.a2a_server:app --port 8002
```

- `GET /.well-known/agent.json` — Agent Card
- `POST /tasks/send` — Execute a query

### Evaluation

```bash
# All 17 queries against both variants
python -m evaluation.runner

# Filter by category or variant
python -m evaluation.runner --category SEM --variant v1
```

Results are saved to `evaluation/results/` and include faithfulness, relevancy, completeness, citation accuracy, latency, and cost per query.

Sample run: [evaluation/results/RESULTS.md](evaluation/results/RESULTS.md)

---

## Testing

No running databases, no API keys required — all LLM and database calls are replaced with mocks so the suite runs in seconds.

```bash
# Lint + Format
ruff check .
ruff format --check .

# Type check
mypy agents/ workflows/ --ignore-missing-imports

# Tests
pytest -v
```

| Test file | What is tested |
|---|---|
| `tests/test_orchestrator.py` | `route_query()` — JSON parsing, multi-agent routing, error handling, token tracking |
| `tests/test_judge.py` | `evaluate()` — ACCEPT / REJECT / MAX_ITERATIONS decisions, prompt construction, history serialization |
| `tests/test_nodes.py` | `orchestrator_node`, `single_retrieval_node`, `judge_node` — latency recording, state mutations |
| `tests/test_routing.py` | V1 `_dispatch()` fan-out via `Send()` and V2 `_judge_routing()` edge conditions |
| `tests/test_answer_format.py` | `_format_results_for_prompt()` — per-source headers, conflict and gap sections |
| `tests/test_state.py` | `_merge_latencies()` reducer — accumulation, immutability, disjoint merge |
| `tests/test_usage.py` | Token accumulation, per-model cost computation, `asyncio` context-var isolation |

---

## Example Queries

### SEM — Semantic (Vector Agent)
```
Which postmortems had an external dependency failure as root cause?
What architectural decisions mention circuit breaker patterns?
Find tickets related to database connection timeout issues
```

### REL — Relational (Graph Agent)
```
Which teams own components that depend on the either-gateway?
Who are the top contributors to database-type components by commit count?
Which components have the deepest transitive dependency chains?
```

### STR — Structured (SQL Agent)
```
Which components have a deployment rollback rate above 20% in 2024?
Show sprint velocity trends for the backend department over the last 10 sprints.
Which components show 4 or more consecutive months of declining test coverage?
```

### MIX — Multi-source
```
What teams own components with the highest bug reopen rates, and what do their ADRs say about quality?
Which components have the most P1 incidents, and who owns them in the knowledge graph?
Find postmortems mentioning cascading failures and show the rollback rate for the involved components.
```

### CON — Conflict detection
```
Show components flagged in postmortems and with a high rollback rate — do the two lists agree?
```

### EDGE — Edge cases
```
What is today's weather in Berlin?              # out of scope
Find postmortems for service XYZ123Foobar.      # empty result
```

---

## Data Sources

Synthetic dataset representing a fictional software organisation: 80 developers, 12 teams, 70 components. Generated by `data/generate_sample.py` — no external downloads required.

### Neo4j — Knowledge Graph

Nodes: `Team`, `Developer`, `Component`, `Repository`, `Epic`

| Relationship | Direction | Properties |
|---|---|---|
| `MEMBER_OF` | Developer → Team | `since` |
| `OWNS` | Developer → Component | — |
| `RESPONSIBLE_FOR` | Team → Component | — |
| `DEPENDS_ON` | Component → Component | `type` |
| `HOSTED_IN` | Component → Repository | — |
| `AFFECTS` | Epic → Component | — |
| `CONTRIBUTED_TO` | Developer → Component | `commits` |

### PostgreSQL — Structured Metrics

| Table | Description |
|---|---|
| `tickets` | Bug, feature, and incident tickets with priority and status |
| `sprint_metrics` | Per-sprint velocity, planned/completed points, bug counts |
| `deployments` | Deployment history with status (success/failed/rollback) |
| `test_coverage` | Monthly line and branch coverage per component |
| `incidents` | P1–P3 incidents with root cause component and MTTR |

### Qdrant — Semantic Search

Three collections using hybrid retrieval (dense + BM25 sparse, RRF fusion):

| Collection | Content |
|---|---|
| `dev_tickets` | Ticket descriptions |
| `dev_arch_docs` | ADRs, design docs, RFCs |
| `dev_postmortems` | Postmortem narratives |

Embeddings: `text-embedding-3-large` (3072 dims), chunked at 512 tokens with 64-token overlap.

---

## Context

An independent project exploring hybrid multi-agent RAG over heterogeneous data sources, combining a vector database, a knowledge graph, and a relational database. Everything runs on a synthetic dataset generated within this repository, which keeps the whole system fully reproducible without any external data.

---

## License

MIT © Patrick Vorreiter