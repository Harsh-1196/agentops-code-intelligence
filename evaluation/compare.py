"""Compare two evaluation benchmark summary runs and report regressions."""

import argparse
import json
import os
import sys
from typing import Any, Dict


def load_report(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_runs(baseline_path: str, candidate_path: str):
    base = load_report(baseline_path)
    cand = load_report(candidate_path)

    base_tem = base.get("trajectory_exact_match", {}).get("rate", 0.0)
    cand_tem = cand.get("trajectory_exact_match", {}).get("rate", 0.0)

    base_f1 = base.get("classification_metrics", {}).get("macro_f1", 0.0)
    cand_f1 = cand.get("classification_metrics", {}).get("macro_f1", 0.0)

    base_h = base.get("llm_as_a_judge", {}).get("average_hallucination_score", 0.0)
    cand_h = cand.get("llm_as_a_judge", {}).get("average_hallucination_score", 0.0)

    base_r = base.get("llm_as_a_judge", {}).get("average_response_match_score", 0.0)
    cand_r = cand.get("llm_as_a_judge", {}).get("average_response_match_score", 0.0)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPARISON: Baseline vs Candidate")
    print("=" * 60)
    print(f"{'Metric':<28} | {'Baseline':<10} | {'Candidate':<10} | {'Delta':<10}")
    print("-" * 65)

    def diff_line(name: str, b: float, c: float, is_pct: bool = False):
        d = c - b
        fmt = ".2%" if is_pct else ".2f"
        b_str = f"{b:{fmt}}"
        c_str = f"{c:{fmt}}"
        d_str = f"{'+' if d >= 0 else ''}{d:{fmt}}"
        print(f"{name:<28} | {b_str:<10} | {c_str:<10} | {d_str:<10}")

    diff_line("Trajectory Exact Match", base_tem, cand_tem, is_pct=True)
    diff_line("Macro F1 Score", base_f1, cand_f1, is_pct=True)
    diff_line("Hallucination Score", base_h, cand_h)
    diff_line("Response Match Score", base_r, cand_r)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AgentOps Benchmark Runs")
    parser.add_argument("baseline", help="Path to baseline evaluation_summary.json")
    parser.add_argument("candidate", help="Path to candidate evaluation_summary.json")
    args = parser.parse_args()
    compare_runs(args.baseline, args.candidate)
