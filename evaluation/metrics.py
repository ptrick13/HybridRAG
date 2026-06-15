"""LLM-as-a-Judge evaluation metrics.

Scores answers on four dimensions using an independent LLM call, each rated
on a 1–5 integer scale.

Metrics:
- **Faithfulness** (1–5):    Are all factual claims supported by the retrieved context?
- **Relevancy** (1–5):       Does the answer address the original question?
- **Completeness** (1–5):    Are all aspects of the question covered?
- **Citation Accuracy** (1–5): Are citations correctly formatted and verifiable?
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.client import get_async_client
from agents.usage import record_llm_call
from config.settings import settings

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator of RAG system outputs. Score the given answer
on four dimensions using a 1–5 integer scale.

## Scoring Rubrics

**Faithfulness** — Are all factual claims in the answer directly supported by
the retrieved context? 5 = fully grounded; 1 = mostly hallucinated.

**Relevancy** — Does the answer directly address the original question?
5 = fully addresses all aspects; 1 = off-topic.

**Completeness** — Are all aspects of the question covered in the answer?
5 = fully covered; 1 = significant aspects ignored.

**Citation Accuracy** — Are the inline citations present, correctly formatted,
and traceable to the retrieved context? 5 = all claims cited correctly;
1 = no citations or all incorrect.

Return a JSON object:
{
  "faithfulness": 1-5,
  "relevancy": 1-5,
  "completeness": 1-5,
  "citation_accuracy": 1-5,
  "reasoning": "<two sentences explaining the scores>"
}
"""

_SCORE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evaluation_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "faithfulness": {"type": "integer"},
                "relevancy": {"type": "integer"},
                "completeness": {"type": "integer"},
                "citation_accuracy": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": [
                "faithfulness",
                "relevancy",
                "completeness",
                "citation_accuracy",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}


class EvaluationScores(BaseModel):
    """LLM-as-a-Judge scores for a single query–answer pair."""

    faithfulness: int = Field(..., ge=1, le=5)
    relevancy: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    citation_accuracy: int = Field(..., ge=1, le=5)
    reasoning: str

    @property
    def average(self) -> float:
        """Mean of all four dimension scores."""
        return (
            self.faithfulness + self.relevancy + self.completeness + self.citation_accuracy
        ) / 4.0


async def score_answer(
    query: str,
    answer: str,
    retrieval_results: list[dict[str, Any]],
) -> EvaluationScores:
    """Score a single answer using LLM-as-a-Judge.

    Args:
        query: The original user question.
        answer: The synthesised answer to evaluate.
        retrieval_results: Raw retrieval results used as the reference context.

    Returns:
        An ``EvaluationScores`` instance with per-dimension integer scores.
    """
    # Truncate at record boundaries so the JSON sent to the judge is never malformed.
    _CONTEXT_BUDGET = 3000
    context_parts: list[str] = []
    budget = _CONTEXT_BUDGET
    for rec in retrieval_results:
        serialised = json.dumps(rec, default=str)
        if len(serialised) + 1 > budget:
            break
        context_parts.append(serialised)
        budget -= len(serialised) + 1
    context_summary = "\n".join(context_parts)

    user_content = f"""
Original Question: {query}

Retrieved Context (truncated):
{context_summary}

Answer to Evaluate:
{answer}

Please score this answer on all four dimensions.
"""

    client = get_async_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_SCORE_SCHEMA,
        temperature=0,
    )

    raw = response.choices[0].message.content
    if response.usage:
        record_llm_call(
            "eval_judge", response.usage.prompt_tokens, response.usage.completion_tokens
        )
    if not raw:
        raise ValueError("Eval judge: LLM returned empty content for structured output call")
    data = json.loads(raw)
    scores = EvaluationScores(**data)

    logger.debug(
        "Scores | F=%d R=%d C=%d CA=%d | avg=%.2f",
        scores.faithfulness,
        scores.relevancy,
        scores.completeness,
        scores.citation_accuracy,
        scores.average,
    )
    return scores
