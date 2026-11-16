"""Unit tests for finding-level Precision, Recall, and F1 calculations."""

import pytest
from evaluation.evaluators.classification_metrics import (
    ClassificationMetricsEvaluator,
    FindingMetricsResult,
    BatchClassificationMetrics,
)


def test_perfect_precision_and_recall():
    """Verify identical findings yield P=1.0, R=1.0, F1=1.0."""
    evaluator = ClassificationMetricsEvaluator(match_threshold=0.6)
    preds = [
        "Google ADK is used for multi-agent loops",
        "OpenTelemetry traces are exported to Jaeger"
    ]
    refs = list(preds)

    res = evaluator.evaluate_single(preds, refs, case_id="CASE-P1")
    assert res.true_positives == 2
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0


def test_partial_precision_and_recall():
    """Verify partial matches compute correct TP, FP, FN."""
    evaluator = ClassificationMetricsEvaluator(match_threshold=0.6)
    preds = [
        "Google ADK is used for multi-agent loops",  # TP match
        "Unrelated rogue claim about quantum computing"  # FP
    ]
    refs = [
        "Google ADK coordinates multi-agent loops",   # Matched by pred[0]
        "SQLite is used for storage"                 # FN
    ]

    res = evaluator.evaluate_single(preds, refs, case_id="CASE-P2")
    assert res.true_positives == 1
    assert res.false_positives == 1
    assert res.false_negatives == 1
    assert res.precision == 0.5
    assert res.recall == 0.5
    assert res.f1 == 0.5


def test_zero_denominator_safety():
    """Verify empty sets return 0.0 without division by zero errors."""
    evaluator = ClassificationMetricsEvaluator()
    res = evaluator.evaluate_single([], [], case_id="CASE-ZERO")
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0


def test_micro_and_macro_batch_averaging():
    """Verify micro vs macro batch calculations."""
    evaluator = ClassificationMetricsEvaluator()
    batch_data = [
        # Case 1: TP=2, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        {
            "case_id": "1",
            "predicted_findings": ["Fact A", "Fact B"],
            "reference_facts": ["Fact A", "Fact B"]
        },
        # Case 2: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        {
            "case_id": "2",
            "predicted_findings": ["Fact A", "False Claim"],
            "reference_facts": ["Fact A", "Missing Fact"]
        }
    ]

    res = evaluator.evaluate_batch(batch_data)
    assert res.total_cases == 2
    assert res.total_true_positives == 3
    assert res.total_false_positives == 1
    assert res.total_false_negatives == 1

    # Macro: avg(1.0, 0.5) = 0.75
    assert res.macro_precision == 0.75
    assert res.macro_recall == 0.75
    assert res.macro_f1 == 0.75

    # Micro: 3 / (3 + 1) = 0.75
    assert res.micro_precision == 0.75
    assert res.micro_recall == 0.75
