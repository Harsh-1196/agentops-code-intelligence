# AgentOps Benchmark Evaluation Summary Report

Generated: `2026-11-16T19:44:38.947090+00:00` | Total Cases: `32` | Duration: `0.56s`

---

## 1. Trajectory Exact-Match (TEM)

| Metric | Result | Description |
|---|---|---|
| **Trajectory Exact Match Rate** | **96.9%** | Exact matching trajectories / total evaluated |
| Exact Sequence Matches | 31 | Perfect alignment with canonical sequence |
| Sequence Mismatches | 1 | Missing or unexpected agent/tool actions |

---

## 2. Finding-Level Classification Metrics

| Metric | Macro Average | Micro Average |
|---|---|---|
| **Precision** | **3.1%** | 3.1% |
| **Recall** | **3.1%** | 3.6% |
| **F1 Score** | **3.1%** | 3.4% |
| Counts | TP: 4 | FP: 124, FN: 107 |

---

## 3. LLM-as-a-Judge Evaluation (0–5 Rubric)

| Dimension | Average Score (0–5) | Evaluation Scope |
|---|---|---|
| **Hallucination Score** | **5.00 / 5.00** | Grounding against source context; flags unsupported claims |
| **Response Match Score** | **4.90 / 5.00** | Completeness against expected reference answer |
| Judge LLM Tokens | 20,194 tokens | Judge inference token accounting |
| Judge LLM Cost | $0.002321 USD | Cost calculated via Gemini pricing table |

*Note: LLM-as-a-Judge scores represent automated model-based evaluations using Google AI Studio Gemini API.*
