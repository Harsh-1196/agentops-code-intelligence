"""Unit tests for LLM-as-a-Judge hallucination evaluator."""

import pytest
from evaluation.evaluators.hallucination_evaluator import (
    HallucinationEvaluator,
    HallucinationEvaluationResult,
)


@pytest.mark.asyncio
async def test_hallucination_evaluator_execution():
    """Verify hallucination evaluator executes and returns structured telemetry."""
    evaluator = HallucinationEvaluator()

    context = "AgentOps uses Google ADK for multi-agent workflows."
    candidate = "AgentOps orchestrates agents using Google ADK."

    res = await evaluator.evaluate(
        candidate_response=candidate,
        reference_context=context,
        case_id="CASE-H1",
    )

    assert isinstance(res, HallucinationEvaluationResult)
    assert 0.0 <= res.hallucination_score <= 5.0
    assert 0.0 <= res.evidence_coverage <= 1.0
    assert res.case_id == "CASE-H1"
    assert res.total_tokens > 0
    assert res.latency_ms > 0
    assert res.prompt_version == "hallucination-rubric-v1.0.0"
