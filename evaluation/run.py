"""CLI runner for AgentOps evaluation harness.

Usage:
    python -m evaluation.run --mode trajectory
    python -m evaluation.run --mode metrics
    python -m evaluation.run --mode judge
    python -m evaluation.run --mode full
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

from agentops.config import Settings
from evaluation.benchmark import BenchmarkHarness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation.run")


def parse_args():
    parser = argparse.ArgumentParser(description="AgentOps Evaluation Runner")
    parser.add_argument(
        "--mode",
        choices=["trajectory", "metrics", "judge", "full"],
        default="full",
        help="Evaluation mode to execute: trajectory, metrics, judge, or full."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom golden benchmark dataset JSON file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate (default: all cases in dataset)."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum agent revision loop iterations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name to use for evaluation."
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info(f"Starting evaluation runner in mode: '{args.mode}'")

    settings_kwargs = {"max_agent_iterations": args.max_iterations}
    if args.model:
        settings_kwargs["gemini_model"] = args.model

    settings = Settings(**settings_kwargs)
    harness = BenchmarkHarness(settings=settings, dataset_path=args.dataset)

    dataset = harness.load_dataset()
    cases = dataset.get("cases", [])
    if args.limit:
        cases = cases[:args.limit]

    logger.info(f"Loaded {len(cases)} evaluation cases from dataset.")

    if args.mode == "trajectory":
        res = await harness.run_trajectory_evaluation(cases)
        print("\n" + "=" * 50)
        print("TRAJECTORY EXACT-MATCH RESULTS")
        print("=" * 50)
        print(f"Total Evaluated: {res.total_evaluated}")
        print(f"Exact Matches:   {res.exact_matches}")
        print(f"Mismatches:      {res.mismatches}")
        print(f"TEM Rate:        {res.trajectory_exact_match * 100:.2f}%")
        print(f"Detailed CSV:    evaluation/reports/trajectory_results.csv")

    elif args.mode == "metrics":
        res = await harness.run_metrics_evaluation(cases)
        print("\n" + "=" * 50)
        print("CLASSIFICATION METRICS RESULTS")
        print("=" * 50)
        print(f"Macro Precision: {res.macro_precision * 100:.2f}%")
        print(f"Macro Recall:    {res.macro_recall * 100:.2f}%")
        print(f"Macro F1:        {res.macro_f1 * 100:.2f}%")
        print(f"Micro Precision: {res.micro_precision * 100:.2f}%")
        print(f"Micro Recall:    {res.micro_recall * 100:.2f}%")
        print(f"Micro F1:        {res.micro_f1 * 100:.2f}%")
        print(f"Detailed CSV:    evaluation/reports/precision_recall_results.csv")

    elif args.mode == "judge":
        res = await harness.run_judge_evaluation(cases)
        print("\n" + "=" * 50)
        print("LLM-AS-A-JUDGE RESULTS")
        print("=" * 50)
        print(f"Evaluated Cases:       {res['evaluated_cases']}")
        print(f"Avg Hallucination:     {res['average_hallucination_score']:.2f} / 5.00")
        print(f"Avg Response Match:    {res['average_response_match_score']:.2f} / 5.00")
        print(f"Total Judge Tokens:    {res['total_judge_tokens']:,}")
        print(f"Total Judge Cost:      ${res['total_judge_cost_usd']:.6f} USD")
        print(f"Detailed JSON:         evaluation/reports/judge_results.json")

    elif args.mode == "full":
        summary = await harness.run_full_benchmark(limit=args.limit)
        print("\n" + "=" * 50)
        print("FULL BENCHMARK EVALUATION COMPLETE")
        print("=" * 50)
        print(f"Cases Evaluated:       {summary['total_evaluated_cases']}")
        print(f"Trajectory Exact Match:{summary['trajectory_exact_match']['rate'] * 100:.1f}%")
        print(f"Macro F1 Score:        {summary['classification_metrics']['macro_f1'] * 100:.1f}%")
        print(f"Avg Hallucination:     {summary['llm_as_a_judge']['average_hallucination_score']:.2f} / 5.00")
        print(f"Avg Response Match:    {summary['llm_as_a_judge']['average_response_match_score']:.2f} / 5.00")
        print(f"Summary JSON Report:   evaluation/reports/evaluation_summary.json")
        print(f"Summary Markdown:      evaluation/reports/evaluation_summary.md")


if __name__ == "__main__":
    asyncio.run(main())
