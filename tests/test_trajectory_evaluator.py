"""Unit tests for trajectory exact-match evaluator."""

import pytest
from evaluation.evaluators.trajectory_evaluator import (
    TrajectoryExactMatchEvaluator,
    TrajectoryBatchResult,
)


def test_trajectory_exact_match_perfect():
    """Verify exact match returns True and zero missing/unexpected steps."""
    evaluator = TrajectoryExactMatchEvaluator()
    expected = [
        "orchestrator.classify",
        "researcher.execute",
        "researcher.retrieve_evidence",
        "writer.generate",
        "judge.evaluate",
        "orchestrator.approve"
    ]
    actual = list(expected)

    res = evaluator.evaluate_single(actual=actual, expected=expected, case_id="CASE-001")
    assert res.is_exact_match is True
    assert len(res.missing_steps) == 0
    assert len(res.unexpected_steps) == 0
    assert res.alignment_ratio == 1.0


def test_trajectory_exact_match_mismatch():
    """Verify mismatched trajectory identifies missing and unexpected steps."""
    evaluator = TrajectoryExactMatchEvaluator()
    expected = [
        "orchestrator.classify",
        "researcher.execute",
        "writer.generate",
        "judge.evaluate",
        "orchestrator.approve"
    ]
    # Actual skipped researcher.execute and added an unexpected step
    actual = [
        "orchestrator.classify",
        "rogue.step",
        "writer.generate",
        "judge.evaluate",
        "orchestrator.approve"
    ]

    res = evaluator.evaluate_single(actual=actual, expected=expected, case_id="CASE-002")
    assert res.is_exact_match is False
    assert "researcher.execute" in res.missing_steps
    assert "rogue.step" in res.unexpected_steps
    assert res.alignment_ratio < 1.0


def test_trajectory_batch_evaluation():
    """Verify batch trajectory exact match rate formula."""
    evaluator = TrajectoryExactMatchEvaluator()
    canonical = ["a", "b", "c"]

    pairs = [
        {"case_id": "1", "actual": ["a", "b", "c"], "expected": canonical},
        {"case_id": "2", "actual": ["a", "b", "c"], "expected": canonical},
        {"case_id": "3", "actual": ["a", "b", "z"], "expected": canonical},
        {"case_id": "4", "actual": ["a", "b"], "expected": canonical},
    ]

    batch = evaluator.evaluate_batch(pairs)
    assert batch.total_evaluated == 4
    assert batch.exact_matches == 2
    assert batch.mismatches == 2
    assert batch.trajectory_exact_match == 0.50
