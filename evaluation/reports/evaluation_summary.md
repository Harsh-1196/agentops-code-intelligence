# AgentOps Benchmark Evaluation Summary Report

Generated: `2026-11-16T20:24:00.967399+00:00` | Total Cases: `3` | Duration: `41.01s`

---

## 1. Trajectory Exact-Match (TEM)

| Metric | Result | Description |
|---|---|---|
| **Trajectory Exact Match Rate** | **100.0%** | Exact matching trajectories / total evaluated |
| Exact Sequence Matches | 3 | Perfect alignment with canonical sequence |
| Sequence Mismatches | 0 | Missing or unexpected agent/tool actions |

---

## 2. Finding-Level Classification Metrics

| Metric | Macro Average | Micro Average |
|---|---|---|
| **Precision** | **33.3%** | 33.3% |
| **Recall** | **33.3%** | 36.4% |
| **F1 Score** | **33.3%** | 34.8% |
| Counts | TP: 4 | FP: 8, FN: 7 |

---

## 3. LLM-as-a-Judge Evaluation (0–5 Rubric)

| Dimension | Average Score (0–5) | Evaluation Scope |
|---|---|---|
| **Hallucination Score** | **5.00 / 5.00** | Grounding against source context; flags unsupported claims |
| **Response Match Score** | **4.90 / 5.00** | Completeness against expected reference answer |
| Judge LLM Tokens | 1,966 tokens | Judge inference token accounting |
| Judge LLM Cost | $0.000223 USD | Cost calculated via Gemini pricing table |

*Note: LLM-as-a-Judge scores represent automated model-based evaluations using Google AI Studio Gemini API.*
