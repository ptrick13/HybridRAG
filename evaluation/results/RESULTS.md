# Evaluation Results

**Run:** 17 queries · both variants (V1 and V2) · illustrative example on a small synthetic dataset — not a benchmark and not intended as a generalizable claim.

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

## By Category (average score)

| Category | V1   | V2   |
|----------|------|------|
| SEM      | 4.94 | 5.00 |
| REL      | 5.00 | 5.00 |
| STR      | 4.50 | 4.75 |
| MIX      | 4.83 | 4.42 |
| CON      | 4.25 | 4.75 |
| EDGE     | 4.50 | 4.50 |

---
Full results: [`sample_eval.json`](sample_eval.json)
