"""Evaluation runner — executes test queries and produces a results report.

Runs each query in ``test_queries.json`` against both V1 and V2 workflows,
scores each answer with LLM-as-a-Judge, and writes a JSON report.

Usage:
    # Run all queries against both variants
    python -m evaluation.runner

    # Run only a specific category
    python -m evaluation.runner --category SEM

    # Run only V1
    python -m evaluation.runner --variant v1

    # Limit to first N queries per category
    python -m evaluation.runner --limit 3
"""

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from agents.usage import compute_cost, embedding_price, model_price
from config.settings import settings
from evaluation.metrics import score_answer
from workflows import v1_workflow, v2_workflow

logger = logging.getLogger(__name__)

_TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
_RESULTS_DIR = Path("evaluation/results")

_WORKFLOW_MAP = {
    "v1": v1_workflow.run,
    "v2": v2_workflow.run,
}


async def _run_single(
    query_item: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Execute one query against one workflow variant and score the result.

    Args:
        query_item: A single item from ``test_queries.json`` with ``id``,
                    ``category``, and ``query`` fields.
        variant: Workflow variant identifier: ``"v1"`` or ``"v2"``.

    Returns:
        A result dict containing the query, answer, scores, latency, and metadata.
    """
    query = query_item["query"]
    workflow_fn = _WORKFLOW_MAP[variant]

    try:
        result = await workflow_fn(query)
        scores = await score_answer(
            query=query,
            answer=result.answer,
            retrieval_results=result.retrieval_results,
        )
        # compute_cost() now includes workflow tokens + eval-judge tokens; subtract
        # the already-captured workflow cost to isolate the evaluator's share.
        eval_cost_usd = max(0.0, compute_cost(settings.openai_model) - result.cost_usd)
        return {
            "id": query_item["id"],
            "category": query_item["category"],
            "query": query,
            "variant": variant,
            "answer": result.answer,
            "activated_agents": result.activated_agents,
            "iterations": result.iterations,
            "latency_seconds": round(result.latency_seconds, 3),
            "cost_usd": round(result.cost_usd, 6),
            "eval_cost_usd": round(eval_cost_usd, 6),
            "cost_breakdown": {
                "llm_input_usd": round(
                    result.metadata.get("prompt_tokens", 0) * model_price(settings.openai_model)[0], 6
                ),
                "llm_output_usd": round(
                    result.metadata.get("completion_tokens", 0) * model_price(settings.openai_model)[1], 6
                ),
                "embedding_usd": round(
                    result.embedding_tokens * embedding_price(settings.embedding_model), 6
                ),
            },
            "agent_latencies": {k: round(v, 3) for k, v in result.agent_latencies.items()},
            "scores": {
                "faithfulness": scores.faithfulness,
                "relevancy": scores.relevancy,
                "completeness": scores.completeness,
                "citation_accuracy": scores.citation_accuracy,
                "average": round(scores.average, 2),
            },
            "judge_decision": result.judge_decision,
            "manual_faithfulness": "",
            "manual_relevancy": "",
            "manual_completeness": "",
            "manual_citation_accuracy": "",
            "manual_notes": "",
            "error": None,
        }
    except Exception as exc:
        logger.exception("Query %s / %s failed: %s", query_item["id"], variant, exc)
        return {
            "id": query_item["id"],
            "category": query_item["category"],
            "query": query,
            "variant": variant,
            "answer": "",
            "activated_agents": [],
            "iterations": 0,
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
            "eval_cost_usd": 0.0,
            "cost_breakdown": {"llm_input_usd": 0.0, "llm_output_usd": 0.0, "embedding_usd": 0.0},
            "agent_latencies": {},
            "scores": None,
            "judge_decision": None,
            "manual_faithfulness": "",
            "manual_relevancy": "",
            "manual_completeness": "",
            "manual_citation_accuracy": "",
            "manual_notes": "",
            "error": str(exc),
        }


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics from individual query results.

    Args:
        results: List of result dicts from ``_run_single``.

    Returns:
        A summary dict with per-variant and per-category averages.
    """
    by_variant: dict[str, list] = defaultdict(list)
    by_category: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    cost_by_variant: dict[str, list[float]] = defaultdict(list)
    eval_cost_by_variant: dict[str, list[float]] = defaultdict(list)
    embedding_usd_by_variant: dict[str, list[float]] = defaultdict(list)
    latency_by_variant: dict[str, list[float]] = defaultdict(list)

    for r in results:
        variant = r["variant"]
        cost_by_variant[variant].append(r.get("cost_usd", 0.0))
        eval_cost_by_variant[variant].append(r.get("eval_cost_usd", 0.0))
        breakdown = r.get("cost_breakdown", {})
        embedding_usd_by_variant[variant].append(breakdown.get("embedding_usd", 0.0))
        latency_by_variant[variant].append(r.get("latency_seconds", 0.0))
        if r["scores"] is None:
            continue
        category = r["category"]
        scores = r["scores"]

        by_variant[variant].append(scores)
        by_category[category][variant].append(scores)

    def _avg_scores(score_list: list[dict]) -> dict:
        if not score_list:
            return {}
        keys = ["faithfulness", "relevancy", "completeness", "citation_accuracy", "average"]
        return {k: round(sum(s[k] for s in score_list) / len(score_list), 2) for k in keys}

    by_variant_summary = {}
    for v, s in by_variant.items():
        entry = _avg_scores(s)
        costs = cost_by_variant.get(v, [])
        eval_costs = eval_cost_by_variant.get(v, [])
        embedding_usds = embedding_usd_by_variant.get(v, [])
        latencies = latency_by_variant.get(v, [])
        entry["avg_cost_usd"] = round(sum(costs) / len(costs), 6) if costs else 0.0
        entry["total_embedding_usd"] = round(sum(embedding_usds), 6)
        entry["total_cost_usd"] = round(sum(costs), 6)
        entry["total_eval_cost_usd"] = round(sum(eval_costs), 6)
        entry["avg_latency_seconds"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        by_variant_summary[v] = entry

    summary = {
        "by_variant": by_variant_summary,
        "by_category": {
            cat: {v: _avg_scores(s) for v, s in variant_map.items()}
            for cat, variant_map in by_category.items()
        },
    }
    return summary


async def run_evaluation(
    variants: list[str],
    category_filter: Optional[str],
    limit: Optional[int],
) -> dict[str, Any]:
    """Run the full evaluation suite and return a structured report.

    Args:
        variants: Which workflow variants to evaluate (e.g., ``["v1", "v2"]``).
        category_filter: If set, only evaluate queries in this category.
        limit: If set, cap the number of queries per category.

    Returns:
        An evaluation report dict with per-query results and aggregate summary.
    """
    with open(_TEST_QUERIES_PATH) as f:
        all_queries: list[dict] = json.load(f)

    if category_filter:
        all_queries = [q for q in all_queries if q["category"] == category_filter]

    if limit:
        # Apply limit per category to keep evaluation balanced
        limited: list[dict] = []
        sorted_queries = sorted(all_queries, key=lambda q: q["category"])
        for _, group in groupby(sorted_queries, key=lambda q: q["category"]):
            limited.extend(list(group)[:limit])
        all_queries = limited

    logger.info(
        "Running %d queries × %d variants = %d total calls",
        len(all_queries),
        len(variants),
        len(all_queries) * len(variants),
    )

    all_results: list[dict] = []
    total_tasks = len(all_queries) * len(variants)

    with tqdm(total=total_tasks, desc="Evaluating") as pbar:
        for query_item in all_queries:
            for variant in variants:
                result = await _run_single(query_item, variant)
                all_results.append(result)
                status = "OK" if result["error"] is None else f"ERR: {result['error'][:40]}"
                pbar.set_postfix(id=result["id"], variant=variant, status=status)
                pbar.update(1)

    summary = _compute_summary(all_results)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configuration": {
            "variants": variants,
            "category_filter": category_filter,
            "query_count": len(all_queries),
        },
        "summary": summary,
        "results": all_results,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Evaluation report written to %s", report_path)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    summary = report["summary"]
    for variant, scores in summary["by_variant"].items():
        print(f"\nVariant {variant.upper()}:")
        for metric, value in scores.items():
            if metric in ("avg_cost_usd",):
                print(f"  {metric:<20} ${value:.6f}")
            else:
                print(f"  {metric:<20} {value:.3f}")

    print("\nBy Category:")
    for category, variant_map in summary["by_category"].items():
        print(f"\n  {category}:")
        for variant, scores in variant_map.items():
            avg = scores.get("average", 0)
            print(f"    {variant}: avg={avg:.2f}")

    print("=" * 70)


def main() -> None:
    """CLI entry point for the evaluation runner."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="HybridRAG evaluation runner")
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "both"],
        default="both",
        help="Workflow variant(s) to evaluate.",
    )
    parser.add_argument(
        "--category",
        choices=["SEM", "REL", "STR", "MIX", "CON", "EDGE"],
        default=None,
        help="Evaluate only this query category.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max queries per category (for quick smoke tests).",
    )
    args = parser.parse_args()

    variants = ["v1", "v2"] if args.variant == "both" else [args.variant]
    report = asyncio.run(
        run_evaluation(
            variants=variants,
            category_filter=args.category,
            limit=args.limit,
        )
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
