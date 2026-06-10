"""Judge Agent — quality control for Variant 2 (closed-loop).

Evaluates consolidated retrieval results against the original query on four
criteria and decides whether to accept or reject the results.

On rejection: formulates a revised query enriched with full evaluation context
so the Orchestrator understands exactly what information is still missing.

On acceptance: passes any detected conflicts to the Answer Agent for
transparent disclosure.

Maximum 3 retrieval iterations are enforced by the workflow layer; when reached
the Judge documents remaining gaps so the Answer Agent can acknowledge them.
"""

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Judge Agent in a hybrid RAG system with a closed-loop quality
control mechanism. Your job is to evaluate consolidated retrieval results
against the original query and decide whether they are sufficient for a
well-founded answer.

## Evaluation Criteria

1. **Completeness**: Do the results cover ALL aspects of the query? Are any
   sub-questions left unanswered?

2. **Relevance**: Are the results actually relevant to what was asked? Or do
   they discuss a related but different topic?

3. **Consistency**: Do results from different sources (Vector/Graph/SQL)
   contradict each other? If so, name the conflicting agents and describe
   the contradiction explicitly.

4. **Specificity**: Are the results specific enough to give a precise answer,
   or are they too vague / generic?

## Decision Logic

- **ACCEPT**: All four criteria are sufficiently met. Pass results to Answer Agent.
- **REJECT**: One or more criteria are not met. Formulate a revised query that
  targets the specific gaps identified. Include: which aspects are missing,
  which agents should cover them, and why the current results are insufficient.
- **MAX_ITERATIONS_REACHED**: Use this decision ONLY when instructed in the
  prompt. Document remaining gaps so the Answer Agent can acknowledge them.

A structurally valid graph or SQL query that returns zero rows is not a retrieval
failure — it means the requested data does not exist in that source. This should
result in an ACCEPT decision with a note that no matching records were found, not
a REJECT. Only reject if a required agent was never called at all, or if the query
itself was clearly wrong (wrong entity type, wrong table).

## Anti-Loop Rules

These rules take precedence over the standard evaluation criteria when the
Previous Iteration Decisions history shows a retrieval loop:

1. **Persistent gaps**: If a gap you would identify already appears verbatim or
   near-verbatim in two or more previous REJECT decisions, that gap cannot be
   resolved through additional retrieval. Use ACCEPT and document the gap.

2. **Agent exhaustion**: If all three retrieval agents (vector, graph, sql) have
   already been activated in prior iterations without improving the criteria
   scores, further iterations will not help. Use ACCEPT.

3. **Stagnant scores**: If the criteria scores across two consecutive previous
   iterations have not improved despite query reformulation, the retrieval space
   is exhausted. Use ACCEPT.

When triggering an anti-loop ACCEPT: set decision="ACCEPT", list the
unresolvable gaps in the "gaps" field, set reformulated_query to null, and
explain the anti-loop trigger in "reasoning".

## Conflict Detection
If Vector, Graph, and SQL results contain contradictory facts:
- Name the conflicting sources explicitly (e.g., "Vector Agent states X, but SQL Agent shows Y")
- Describe the contradiction precisely
- Do NOT resolve or favour either source — flag it for the Answer Agent

## Output Format
Return a JSON object with this exact structure:
{
  "decision": "ACCEPT" | "REJECT" | "MAX_ITERATIONS_REACHED",
  "criteria_scores": {
    "completeness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5,
    "specificity": 1-5
  },
  "gaps": ["<gap 1>", "<gap 2>"],
  "conflicts": ["<conflict description if any>"],
  "reformulated_query": "<revised query for next retrieval cycle, or null if ACCEPT>",
  "reasoning": "<two-sentence explanation of the decision>"
}

## Few-Shot Examples

### Example 1 — ACCEPT
Original query: "What Python libraries are most used for data processing?"
Results: Vector returned 5 relevant SO answers about pandas, numpy, polars.
         SQL showed top 10 tags by question count including pandas and numpy.
Response:
{
  "decision": "ACCEPT",
  "criteria_scores": {"completeness": 5, "relevance": 5, "consistency": 5, "specificity": 4},
  "gaps": [],
  "conflicts": [],
  "reformulated_query": null,
  "reasoning": "Both Vector and SQL results consistently cover popular Python data processing libraries with sufficient specificity. No additional retrieval needed."
}

### Example 2 — REJECT (gap identified)
Original query: "Find high-voted questions about RAG and list top experts."
Results: Vector returned relevant RAG questions but Graph returned no expert data.
Response:
{
  "decision": "REJECT",
  "criteria_scores": {"completeness": 2, "relevance": 4, "consistency": 4, "specificity": 3},
  "gaps": ["Expert user data for RAG/LangChain tags is missing — Graph Agent returned empty results"],
  "conflicts": [],
  "reformulated_query": "Find top experts in RAG systems [JUDGE GAPS: Graph Agent returned empty — try broader tag names: 'langchain', 'llm', 'vector-database', 'retrieval-augmented-generation']",
  "reasoning": "Vector results are relevant but incomplete: the user asked for both questions AND expert listings, and expert data is entirely missing. Retry with Graph Agent using broader tag variants."
}

### Example 3 — Conflict detected
Original query: "How many questions about pandas exist?"
Vector: found 8 highly-voted pandas questions. SQL: reports 1,247 total pandas questions.
Response:
{
  "decision": "ACCEPT",
  "criteria_scores": {"completeness": 4, "relevance": 5, "consistency": 3, "specificity": 4},
  "gaps": [],
  "conflicts": ["Vector Agent retrieved 8 documents (a semantic sample); SQL Agent reports 1,247 total pandas questions — these are not contradictory but represent different scopes (semantic sample vs. full count). Answer Agent should clarify both figures."],
  "reformulated_query": null,
  "reasoning": "The apparent conflict is a scope difference, not a factual contradiction. Both results are accepted; the Answer Agent must clarify the distinction."
}
"""

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["ACCEPT", "REJECT", "MAX_ITERATIONS_REACHED"]},
                "criteria_scores": {
                    "type": "object",
                    "properties": {
                        "completeness": {"type": "integer"},
                        "relevance": {"type": "integer"},
                        "consistency": {"type": "integer"},
                        "specificity": {"type": "integer"},
                    },
                    "required": ["completeness", "relevance", "consistency", "specificity"],
                    "additionalProperties": False,
                },
                "gaps": {"type": "array", "items": {"type": "string"}},
                "conflicts": {"type": "array", "items": {"type": "string"}},
                "reformulated_query": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["decision", "criteria_scores", "gaps", "conflicts", "reformulated_query", "reasoning"],
            "additionalProperties": False,
        },
    },
}


class CriteriaScores(BaseModel):
    """Per-criterion quality scores from the Judge Agent (1–5 scale)."""

    completeness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    consistency: int = Field(..., ge=1, le=5)
    specificity: int = Field(..., ge=1, le=5)


class JudgeDecision(BaseModel):
    """Structured output from the Judge Agent."""

    decision: str = Field(..., description="ACCEPT, REJECT, or MAX_ITERATIONS_REACHED.")
    criteria_scores: CriteriaScores
    gaps: list[str] = Field(default_factory=list, description="Identified information gaps.")
    conflicts: list[str] = Field(default_factory=list, description="Detected cross-source conflicts.")
    reformulated_query: Optional[str] = Field(
        default=None,
        description="Revised query for the next retrieval cycle (None on ACCEPT).",
    )
    reasoning: str = Field(..., description="Two-sentence explanation of the decision.")


async def evaluate(
    original_query: str,
    retrieval_results: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    previous_decisions: Optional[list["JudgeDecision"]] = None,
) -> "JudgeDecision":
    """Evaluate consolidated retrieval results and decide accept or reject.

    Args:
        original_query: The user's original query (unmodified across iterations).
        retrieval_results: Consolidated results from all activated agents
            (accumulated across all iterations so far).
        iteration: Current iteration number (0-indexed).
        max_iterations: Maximum allowed iterations. When ``iteration + 1 ==
            max_iterations``, the judge is instructed to document gaps and accept.
        previous_decisions: Ordered list of all prior ``JudgeDecision`` objects
            from earlier iterations, used for anti-loop detection.

    Returns:
        A ``JudgeDecision`` with the accept/reject verdict, criteria scores,
        gaps, conflicts, and an optional reformulated query for the next cycle.
    """
    results_summary = json.dumps(retrieval_results, indent=2, default=str)
    is_last_iteration = (iteration + 1) >= max_iterations

    history_section = ""
    if previous_decisions:
        history = [
            {
                "iteration": i + 1,
                "decision": pd.decision,
                "gaps": pd.gaps,
                "criteria_scores": pd.criteria_scores.model_dump(),
            }
            for i, pd in enumerate(previous_decisions)
        ]
        history_section = (
            "\n\nPrevious Iteration Decisions (for anti-loop detection):\n"
            + json.dumps(history, indent=2)
        )

    user_content = f"""
Original Query: {original_query}

Consolidated Retrieval Results (all iterations):
{results_summary}

Current Iteration: {iteration + 1} / {max_iterations}
{('IMPORTANT: This is the FINAL allowed iteration. You MUST use decision="MAX_ITERATIONS_REACHED", document all remaining gaps, and set reformulated_query to null.' if is_last_iteration else '')}
{history_section}

Please evaluate the results and return your structured decision.
"""

    client = get_async_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_RESPONSE_SCHEMA,
        temperature=0,
    )

    raw = response.choices[0].message.content
    if response.usage:
        record_llm_call("judge", response.usage.prompt_tokens, response.usage.completion_tokens)
    if not raw:
        raise ValueError("Judge: LLM returned empty content for structured output call")
    data = json.loads(raw)
    decision = JudgeDecision(**data)

    logger.info(
        "Judge decision: %s | completeness=%d relevance=%d | gaps=%d",
        decision.decision,
        decision.criteria_scores.completeness,
        decision.criteria_scores.relevance,
        len(decision.gaps),
    )
    return decision
