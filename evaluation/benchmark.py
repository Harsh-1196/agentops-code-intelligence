"""Benchmark harness orchestrating multi-agent evaluations across golden datasets."""

import asyncio
import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from agentops.config import Settings, get_settings
from agentops.orchestrator import ADKOrchestrator
from evaluation.evaluators.trajectory_evaluator import TrajectoryExactMatchEvaluator, TrajectoryBatchResult
from evaluation.evaluators.classification_metrics import ClassificationMetricsEvaluator, BatchClassificationMetrics
from evaluation.evaluators.hallucination_evaluator import HallucinationEvaluator, HallucinationEvaluationResult
from evaluation.evaluators.response_match_evaluator import ResponseMatchEvaluator, ResponseMatchEvaluationResult

logger = logging.getLogger("agentops.benchmark")


class BenchmarkHarness:
    """Executes evaluation benchmarks and generates summary reports and CSV/JSON artifacts."""

    def __init__(self, settings: Optional[Settings] = None, dataset_path: Optional[str] = None):
        self.settings = settings or get_settings()
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(__file__), "datasets", "golden_dataset.json")
        self.orchestrator = ADKOrchestrator(settings=self.settings)

        self.trajectory_evaluator = TrajectoryExactMatchEvaluator()
        self.classification_evaluator = ClassificationMetricsEvaluator()
        self.hallucination_evaluator = HallucinationEvaluator(settings=self.settings)
        self.response_match_evaluator = ResponseMatchEvaluator(settings=self.settings)

        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_dataset(self) -> Dict[str, Any]:
        """Loads golden benchmark dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run_trajectory_evaluation(self, cases: List[Dict[str, Any]]) -> TrajectoryBatchResult:
        """Runs trajectory exact-match evaluation."""
        pairs = []
        for case in cases:
            # Run workflow or generate canonical execution
            state = await self.orchestrator.run(
                task=case["task"],
                max_iterations=self.settings.max_agent_iterations,
            )
            actual_events = [s.canonical_event for s in state.trajectory]
            pairs.append({
                "case_id": case["id"],
                "actual": actual_events,
                "expected": case["expected_trajectory"],
            })

        batch_result = self.trajectory_evaluator.evaluate_batch(pairs)

        # Write trajectory_results.csv
        csv_path = os.path.join(self.reports_dir, "trajectory_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["case_id", "is_exact_match", "alignment_ratio", "missing_steps", "unexpected_steps"])
            for r in batch_result.results:
                writer.writerow([
                    r.case_id,
                    r.is_exact_match,
                    r.alignment_ratio,
                    ";".join(r.missing_steps),
                    ";".join(r.unexpected_steps),
                ])
        logger.info(f"Saved trajectory results to {csv_path}")
        return batch_result

    async def run_metrics_evaluation(self, cases: List[Dict[str, Any]]) -> BatchClassificationMetrics:
        """Runs finding-level Precision, Recall, and F1 evaluation."""
        items = []
        for case in cases:
            state = await self.orchestrator.run(
                task=case["task"],
                max_iterations=self.settings.max_agent_iterations,
            )
            predicted_facts = []
            if state.research_findings:
                predicted_facts = state.research_findings.identified_facts

            items.append({
                "case_id": case["id"],
                "predicted_findings": predicted_facts,
                "reference_facts": case["expected_facts"],
            })

        batch_metrics = self.classification_evaluator.evaluate_batch(items)

        # Write precision_recall_results.csv
        csv_path = os.path.join(self.reports_dir, "precision_recall_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["case_id", "precision", "recall", "f1", "tp", "fp", "fn"])
            for c in batch_metrics.case_results:
                writer.writerow([
                    c.case_id,
                    c.precision,
                    c.recall,
                    c.f1,
                    c.true_positives,
                    c.false_positives,
                    c.false_negatives,
                ])
        logger.info(f"Saved classification metrics to {csv_path}")
        return batch_metrics

    async def run_judge_evaluation(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs independent LLM-as-a-Judge for Hallucination and Response Match."""
        hallucination_results = []
        response_match_results = []

        total_tokens = 0
        total_cost = 0.0

        for case in cases:
            state = await self.orchestrator.run(
                task=case["task"],
                max_iterations=self.settings.max_agent_iterations,
            )
            candidate_resp = state.final_response or ""

            # 1. Hallucination Judge
            h_res = await self.hallucination_evaluator.evaluate(
                candidate_response=candidate_resp,
                reference_context=case["reference_context"],
                case_id=case["id"],
            )
            hallucination_results.append(h_res.model_dump())
            total_tokens += h_res.total_tokens
            if h_res.cost_usd:
                total_cost += h_res.cost_usd

            # 2. Response Match Judge
            r_res = await self.response_match_evaluator.evaluate(
                candidate_response=candidate_resp,
                expected_response=case["expected_response"],
                task=case["task"],
                case_id=case["id"],
            )
            response_match_results.append(r_res.model_dump())
            total_tokens += r_res.total_tokens
            if r_res.cost_usd:
                total_cost += r_res.cost_usd

        # Calculate averages
        avg_h_score = (
            sum(r["hallucination_score"] for r in hallucination_results) / len(hallucination_results)
            if hallucination_results else 0.0
        )
        avg_r_score = (
            sum(r["response_match_score"] for r in response_match_results) / len(response_match_results)
            if response_match_results else 0.0
        )

        judge_data = {
            "evaluated_cases": len(cases),
            "average_hallucination_score": round(avg_h_score, 2),
            "average_response_match_score": round(avg_r_score, 2),
            "total_judge_tokens": total_tokens,
            "total_judge_cost_usd": round(total_cost, 6),
            "hallucination_evaluations": hallucination_results,
            "response_match_evaluations": response_match_results,
        }

        # Write judge_results.json
        json_path = os.path.join(self.reports_dir, "judge_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(judge_data, f, indent=2)
        logger.info(f"Saved judge evaluations to {json_path}")
        return judge_data

    async def run_full_benchmark(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Runs all evaluation modes and generates comprehensive summary reports."""
        dataset = self.load_dataset()
        cases = dataset.get("cases", [])
        if limit:
            cases = cases[:limit]

        start_time = time.perf_counter()

        logger.info(f"Running full benchmark on {len(cases)} cases...")
        trajectory_res = await self.run_trajectory_evaluation(cases)
        metrics_res = await self.run_metrics_evaluation(cases)
        judge_res = await self.run_judge_evaluation(cases)

        total_elapsed = round(time.perf_counter() - start_time, 2)

        summary = {
            "benchmark_run_at": datetime.now(timezone.utc).isoformat(),
            "total_evaluated_cases": len(cases),
            "dataset_categories": dataset.get("metadata", {}).get("categories", []),
            "elapsed_seconds": total_elapsed,
            "trajectory_exact_match": {
                "rate": trajectory_res.trajectory_exact_match,
                "exact_matches": trajectory_res.exact_matches,
                "mismatches": trajectory_res.mismatches,
            },
            "classification_metrics": {
                "macro_precision": metrics_res.macro_precision,
                "macro_recall": metrics_res.macro_recall,
                "macro_f1": metrics_res.macro_f1,
                "micro_precision": metrics_res.micro_precision,
                "micro_recall": metrics_res.micro_recall,
                "micro_f1": metrics_res.micro_f1,
                "total_tp": metrics_res.total_true_positives,
                "total_fp": metrics_res.total_false_positives,
                "total_fn": metrics_res.total_false_negatives,
            },
            "llm_as_a_judge": {
                "average_hallucination_score": judge_res["average_hallucination_score"],
                "average_response_match_score": judge_res["average_response_match_score"],
                "total_judge_tokens": judge_res["total_judge_tokens"],
                "total_judge_cost_usd": judge_res["total_judge_cost_usd"],
            }
        }

        # Write evaluation_summary.json
        summary_json_path = os.path.join(self.reports_dir, "evaluation_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write evaluation_summary.md
        summary_md_path = os.path.join(self.reports_dir, "evaluation_summary.md")
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown_summary(summary))

        logger.info(f"Full benchmark complete. Reports written to {self.reports_dir}")
        return summary

    def _generate_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Formats benchmark results into clean Markdown table."""
        tem = summary["trajectory_exact_match"]
        cls = summary["classification_metrics"]
        jdg = summary["llm_as_a_judge"]

        return f"""# AgentOps Benchmark Evaluation Summary Report

Generated: `{summary['benchmark_run_at']}` | Total Cases: `{summary['total_evaluated_cases']}` | Duration: `{summary['elapsed_seconds']}s`

---

## 1. Trajectory Exact-Match (TEM)

| Metric | Result | Description |
|---|---|---|
| **Trajectory Exact Match Rate** | **{tem['rate'] * 100:.1f}%** | Exact matching trajectories / total evaluated |
| Exact Sequence Matches | {tem['exact_matches']} | Perfect alignment with canonical sequence |
| Sequence Mismatches | {tem['mismatches']} | Missing or unexpected agent/tool actions |

---

## 2. Finding-Level Classification Metrics

| Metric | Macro Average | Micro Average |
|---|---|---|
| **Precision** | **{cls['macro_precision'] * 100:.1f}%** | {cls['micro_precision'] * 100:.1f}% |
| **Recall** | **{cls['macro_recall'] * 100:.1f}%** | {cls['micro_recall'] * 100:.1f}% |
| **F1 Score** | **{cls['macro_f1'] * 100:.1f}%** | {cls['micro_f1'] * 100:.1f}% |
| Counts | TP: {cls['total_tp']} | FP: {cls['total_fp']}, FN: {cls['total_fn']} |

---

## 3. LLM-as-a-Judge Evaluation (0–5 Rubric)

| Dimension | Average Score (0–5) | Evaluation Scope |
|---|---|---|
| **Hallucination Score** | **{jdg['average_hallucination_score']:.2f} / 5.00** | Grounding against source context; flags unsupported claims |
| **Response Match Score** | **{jdg['average_response_match_score']:.2f} / 5.00** | Completeness against expected reference answer |
| Judge LLM Tokens | {jdg['total_judge_tokens']:,} tokens | Judge inference token accounting |
| Judge LLM Cost | ${jdg['total_judge_cost_usd']:.6f} USD | Cost calculated via Gemini pricing table |

*Note: LLM-as-a-Judge scores represent automated model-based evaluations using Google AI Studio Gemini API.*
"""
