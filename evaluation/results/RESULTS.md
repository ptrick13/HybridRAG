# Evaluation Results

**Run:** 17 queries · both variants (V1 and V2)

## Overall Scores (LLM-as-a-Judge, 1–5)

| Metric              | V1     | V2     |
|---------------------|--------|--------|
| Faithfulness        | 4.65   | 4.65   |
| Relevancy           | 5.00   | 5.00   |
| Completeness        | 4.71   | 4.76   |
| Citation Accuracy   | 4.53   | 4.65   |
| **Average**         | **4.72** | **4.76** |

## Cost & Latency

| Metric                  | V1        | V2        |
|-------------------------|-----------|-----------|
| Avg cost / query        | $0.000870 | $0.002357 |
| Total workflow cost     | $0.014785 | $0.040064 |
| Total eval (judge) cost | $0.002822 | $0.003720 |
| Avg latency             | 10.4 s    | 20.6 s    |

V2 costs ~2.7× more and takes ~2× longer due to its iterative judge loop.

## By Category (average score)

| Category | V1   | V2   | Notes                                      |
|----------|------|------|--------------------------------------------|
| SEM      | 4.94 | 5.00 | Pure semantic retrieval — both near-perfect |
| REL      | 5.00 | 5.00 | Graph traversal — perfect across the board |
| STR      | 4.50 | 4.75 | V2 judge catches schema drift in STR-02    |
| MIX      | 4.83 | 4.42 | V1 wins — V2 judge penalises cross-source faithfulness |
| CON      | 4.25 | 4.75 | V2 iterates to resolve cross-source conflicts |
| EDGE     | 4.50 | 4.50 | Both handle out-of-scope gracefully         |

---
Full results: [`sample_eval.json`](sample_eval.json)