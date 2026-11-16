"""Unit tests for LLM-as-a-Judge response-match evaluator."""

import pytest
from evaluation.evaluators.response_match_evaluator import (
    ResponseMatchEvaluator,
    ResponseMatchEvaluationResult,
)


@pytest.mark.asyncio
async def test_response_match_evaluator_execution():
    """Verify response match evaluator returns structured scores and explanations."""
    evaluator = ResponseMatchEvaluator()

    task = "What port does Jaeger UI run on?"
    expected = "Jaeger UI runs on port 16686."
    candidate = "Jaeger UI is hosted on port 16686 in Docker Compose."

    res = await evaluator.evaluate(
        candidate_response=candidate,
        expected_response=expected,
        task=task,
        case_id="CASE-R1",
    )

    assert isinstance(res, ResponseMatchEvaluationResult)
    assert 0.0 <= res.response_match_score <= 5.0
    assert res.case_id == "CASE-R1"
    assert res.total_tokens > 0
    assert res.latency_ms > 0
    assert res.prompt_version == "response-match-rubric-v1.0.0"
